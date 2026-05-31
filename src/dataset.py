"""
dataset.py -- v37 CORAL-TFT Ordinal CDF Data Pipeline
=======================================================
Extends the v36 TFT data pipeline with CORAL ordinal encoding constants and
an expanded meteorological feature matrix verified from the updated
data/processed/ CSVs (Phase 1 inspection: 44 train cols / 43 test cols).

Key Changes from v36
--------------------
[v37] ADD  : ORDINAL_K=50, ORDINAL_DELTA_T=0.1, ORDINAL_THRESHOLDS (0.1–5.0)
             These constants are consumed by CORALTFTWrapper in model.py.
[v37] ADD  : TorchNormalizer(method="identity") as target_normalizer.
             CORAL thresholds compare against the raw [0, 5] drought score.
             Any per-window Z-score normalisation would destroy the absolute
             threshold semantics.  Identity normaliser preserves raw scale.
[v37] ADD  : Expanded TIME_VARYING_UNKNOWN_REALS from 8 (v36) → 37 features,
             incorporating all new columns confirmed in Phase 1:
             weekly stat summaries (max/min/std), 4-week lag features,
             PET / deficit / deficit_roll_cum_4w hydrology features.
[v37] RETAIN: time_idx = week_idx, region_id group, week_sin / week_cos
              known reals, WINDOW_SIZE=13, HORIZON=5, N_FOLDS=5.

Architecture Notes (unchanged from v36)
-----------------------------------------
* time_idx     : week_idx (0–781 for train; remapped 782–794 for test encoder).
* HORIZON      : 5-week future prediction window.
* WINDOW_SIZE  : 13-week historical encoder context (rigid).
* 5-Fold CV    : Rolling time-based validation; each fold reserves a clean
                 HORIZON-week block as the validation target.
* Inference    : Test encoder (13 weeks) + 5 future placeholder rows
                 per region → one prediction sample per region.

Column Categories (verified from data/processed/train_processed.csv)
----------------------------------------------------------------------
  time_varying_known_reals    : week_sin, week_cos        (calendar cycle, known ahead)
  time_varying_unknown_reals  : 37 meteorological + rolling + lag + hydrology features
  static_categoricals         : region_id
  target                      : score  (float, raw [0, 5] scale, NO normalisation)
"""

import os
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# ---------------------------------------------------------------------------
# Hyper-parameters (exported for train.py)
# ---------------------------------------------------------------------------
WINDOW_SIZE  : int = 13   # max_encoder_length — rigid 13-week historical horizon
HORIZON      : int = 5    # max_prediction_length — 5-week forecast window
N_FOLDS      : int = 5    # rolling time-based CV folds

# Inferred from processed CSV; hardcoded for determinism
T_TRAIN_MAX  : int = 781  # max week_idx in training data  (week 0 … 781 = 782 weeks)
T_TEST_OFFSET: int = T_TRAIN_MAX + 1   # test week_idx 0 → time_idx 782

# ---------------------------------------------------------------------------
# [v37] Ordinal CDF discretisation constants
# ---------------------------------------------------------------------------
ORDINAL_K      : int   = 50
ORDINAL_DELTA_T: float = 0.1

# Threshold array T = [0.1, 0.2, ..., 5.0]
# z_{ik} = 1  if  y_i >= t_k,  else 0
# Absolute-zero targets (y=0.0) produce all-zeros vector [0, 0, ..., 0].
# Extreme drought (y=4.5) produces ones through index 44: [1…1 (45), 0…0 (5)].
ORDINAL_THRESHOLDS: list[float] = [
    round(k * ORDINAL_DELTA_T, 1) for k in range(1, ORDINAL_K + 1)
]
# → [0.1, 0.2, 0.3, ..., 5.0]

# ---------------------------------------------------------------------------
# [v37] Identity normaliser — preserves raw [0, 5] drought score scale
# so that CORAL threshold comparisons operate on absolute values.
# ---------------------------------------------------------------------------
_TARGET_NORMALIZER = TorchNormalizer(method="identity")

# ---------------------------------------------------------------------------
# Feature column definitions (v37 — expanded from 8 to 33 unknown reals)
# ---------------------------------------------------------------------------

# Known seasonality signals: computed from calendar, available for any future week.
TIME_VARYING_KNOWN_REALS: list[str] = [
    "week_sin",           # sin(2π × week_of_year / 52)  — known for future weeks
    "week_cos",           # cos(2π × week_of_year / 52)  — known for future weeks
]

# All meteorological and derived features observed up to (and including) the
# current week only — unavailable for future decoder weeks.
# Source: Phase 1 inspection of data/processed/train_processed.csv (44 cols).
# 37 features total (verified from Phase 1: data/processed/train_processed.csv 44 cols)
# Breakdown: 5 core + 11 weekly stats + 7 derived daily + 3 rolling + 8 lag + 3 hydrology
TIME_VARYING_UNKNOWN_REALS: list[str] = [
    # ── Core weekly means (5) ────────────────────────────────────────────────
    "tmp",                    # weekly mean temperature
    "prec",                   # weekly precipitation total
    "humidity",               # weekly mean relative humidity
    "wind",                   # weekly mean wind speed
    "surf_pre",               # surface atmospheric pressure

    # ── Weekly statistical summaries (11, NEW in v37) ────────────────────
    "prec_week_max",          # weekly max rainfall
    "surf_pre_week_max",      # weekly max surface pressure
    "humidity_week_max",      # weekly max relative humidity
    "humidity_week_min",      # weekly min relative humidity
    "humidity_week_std",      # weekly humidity variability
    "tmp_week_max",           # weekly max temperature
    "tmp_week_min",           # weekly min temperature
    "tmp_week_std",           # weekly temperature variability
    "wind_week_max",          # weekly max wind speed
    "wind_week_min",          # weekly min wind speed
    "wind_week_std",          # weekly wind speed variability

    # ── Derived daily-scale summaries (7, NEW in v37) ────────────────────
    "tmp_max",                # daily max temperature (weekly agg)
    "tmp_min",                # daily min temperature (weekly agg)
    "tmp_range",              # daily temperature diurnal range
    "surf_tmp",               # surface temperature
    "wind_max",               # daily max wind speed (weekly agg)
    "wind_min",               # daily min wind speed (weekly agg)
    "wind_range",             # daily wind diurnal range

    # ── 4-week rolling aggregates (3) ────────────────────────────────────
    "tmp_roll_mean_4w",       # 4-week rolling mean temperature
    "humidity_roll_mean_4w",  # 4-week rolling mean humidity
    "prec_roll_sum_4w",       # 4-week rolling precipitation sum

    # ── Temporal lag features (8, NEW in v37) ────────────────────────────
    "tmp_lag1w",              # temperature 1 week ago
    "tmp_lag2w",              # temperature 2 weeks ago
    "humidity_lag1w",         # humidity 1 week ago
    "humidity_lag2w",         # humidity 2 weeks ago
    "prec_lag1w",             # precipitation 1 week ago
    "prec_lag2w",             # precipitation 2 weeks ago
    "wind_lag1w",             # wind speed 1 week ago
    "wind_lag2w",             # wind speed 2 weeks ago

    # ── PET / Water-Deficit hydrology features (3, NEW in v37) ──────────
    "pet",                    # potential evapotranspiration
    "deficit",                # atmospheric water deficit
    "deficit_roll_cum_4w",    # 4-week cumulative water deficit
]

STATIC_CATEGORICALS: list[str] = ["region_id"]

TARGET: str = "score"

# One full revolution per year in 52-week steps
WEEKLY_ANGLE_STEP: float = 2.0 * np.pi / 52.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_feature_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN in all feature columns with 0.0.

    NaNs arise naturally at early rows for lag and rolling features (boundary
    effect: e.g., tmp_lag1w is NaN in the very first week).  Setting to 0.0
    is a safe, conservative fill that avoids information leakage and is
    consistent with the decoder's unknown-real zeroing in inference.
    """
    all_feats = TIME_VARYING_KNOWN_REALS + TIME_VARYING_UNKNOWN_REALS
    for col in all_feats:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test processed CSVs.  Assign monotonically increasing
    time_idx per region, fill feature NaNs.

    Returns
    -------
    train_df : DataFrame
        Training data with time_idx = week_idx (0–781), score column present.
    test_df  : DataFrame
        Test data with time_idx = T_TRAIN_MAX + 1 + week_idx (782–794).
        A dummy score = 0.0 column is added (required by TimeSeriesDataSet).
    """
    train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test_processed.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}.\n"
            "Run `python src/preprocess.py` first."
        )

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # -- Sort chronologically per region ----------------------------------------
    train_df = (train_df
                .sort_values(["region_id", "week_idx"])
                .reset_index(drop=True))
    test_df  = (test_df
                .sort_values(["region_id", "week_idx"])
                .reset_index(drop=True))

    # -- Assign time_idx ---------------------------------------------------------
    # Training: time_idx = week_idx  (already 0–781, gap-free per region)
    train_df["time_idx"] = train_df["week_idx"].astype(int)

    # Test encoder: remap week_idx 0–12  →  time_idx 782–794 so the timeline
    # is continuous across the train→test boundary.
    test_df["time_idx"] = (T_TEST_OFFSET + test_df["week_idx"]).astype(int)
    test_df[TARGET]     = 0.0   # dummy placeholder — not used during inference

    # -- Drop rows missing score in training -------------------------------------
    before = len(train_df)
    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)
    dropped  = before - len(train_df)
    if dropped:
        print(f"  [dataset] Dropped {dropped:,} train rows with NaN score.")

    # -- Fill NaN in feature columns ---------------------------------------------
    train_df = _fill_feature_nans(train_df)
    test_df  = _fill_feature_nans(test_df)

    return train_df, test_df


def get_fold_boundaries(
    fold_k: int,
    n_folds: int = N_FOLDS,
    horizon: int = HORIZON,
) -> tuple[int, int, int]:
    """
    Compute rolling time-based split boundaries for CV fold k.

    Fold layout (T_TRAIN_MAX=781, HORIZON=5, N_FOLDS=5):
      fold 0 : train [0, 756]  →  val [757, 761]
      fold 1 : train [0, 761]  →  val [762, 766]
      fold 2 : train [0, 766]  →  val [767, 771]
      fold 3 : train [0, 771]  →  val [772, 776]
      fold 4 : train [0, 776]  →  val [777, 781]

    Parameters
    ----------
    fold_k : int  (0-based)

    Returns
    -------
    (train_end, val_start, val_end)
    """
    val_end   = T_TRAIN_MAX - (n_folds - 1 - fold_k) * horizon
    val_start = val_end - horizon + 1
    train_end = val_start - 1
    return int(train_end), int(val_start), int(val_end)


def build_training_dataset(fold_train_df: pd.DataFrame) -> TimeSeriesDataSet:
    """
    Construct a pytorch_forecasting TimeSeriesDataSet from one training fold's data.

    [v37] Uses TorchNormalizer(method="identity") for target_normalizer so the
    raw drought score in [0, 5] is preserved exactly.  CORAL ordinal thresholds
    [0.1, 0.2, ..., 5.0] are compared against this raw scale in CORALTFTWrapper.

    Parameters
    ----------
    fold_train_df : DataFrame
        Training rows for this fold  (time_idx ≤ train_end for the fold).
    """
    return TimeSeriesDataSet(
        data                       = fold_train_df,
        time_idx                   = "time_idx",
        target                     = TARGET,
        group_ids                  = STATIC_CATEGORICALS,
        max_encoder_length         = WINDOW_SIZE,
        max_prediction_length      = HORIZON,
        static_categoricals        = STATIC_CATEGORICALS,
        time_varying_known_reals   = TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals = TIME_VARYING_UNKNOWN_REALS,
        allow_missing_timesteps    = False,
        # [v37] Identity normaliser — preserves raw [0, 5] drought score scale.
        # CORAL threshold comparisons require absolute values; any Z-score
        # normalisation would shift the thresholds out of [0, 5] space.
        target_normalizer          = _TARGET_NORMALIZER,
    )


def build_val_dataset(
    training_dataset : TimeSeriesDataSet,
    fold_context_df  : pd.DataFrame,
    val_start        : int,
) -> TimeSeriesDataSet:
    """
    Construct the validation TimeSeriesDataSet for a fold.

    Parameters
    ----------
    training_dataset : TimeSeriesDataSet
        Fitted training dataset (categorical encoders applied).
    fold_context_df  : DataFrame
        All rows up to val_end (includes training context for encoder window).
    val_start        : int
        First validation time_idx; only decoder windows starting at or after
        this index are included (exactly one window per region per fold).
    """
    return TimeSeriesDataSet.from_dataset(
        training_dataset,
        data               = fold_context_df,
        stop_randomization = True,
        predict            = False,
        min_prediction_idx = val_start,
    )


def build_test_inference_dataset(
    training_dataset : TimeSeriesDataSet,
    test_df          : pd.DataFrame,
) -> tuple[TimeSeriesDataSet, list[str]]:
    """
    Construct the test inference TimeSeriesDataSet.

    Appends HORIZON dummy future rows per region after the 13-week encoder
    context rows in test_df, creating exactly ONE sample per region:
      encoder (time_idx 782–794) → decoder (time_idx 795–799).

    Week_sin / week_cos for the 5 future rows are extrapolated by advancing
    the last observed cyclic angle by WEEKLY_ANGLE_STEP (= 2π/52) per step.
    All unknown meteorological reals for future rows are set to 0.0 (they
    are masked in the TFT decoder anyway).

    Returns
    -------
    (pred_dataset, ordered_regions)
        pred_dataset    — TimeSeriesDataSet with one sample per region
        ordered_regions — sorted list of region_id strings in sample order
    """
    future_rows: list[dict] = []

    for region_id, grp in test_df.groupby("region_id", sort=True):
        grp_sorted = grp.sort_values("time_idx")
        last_row   = grp_sorted.iloc[-1]

        last_time_idx = int(last_row["time_idx"])
        last_angle    = float(np.arctan2(
            float(last_row["week_sin"]),
            float(last_row["week_cos"])
        ))

        for k in range(1, HORIZON + 1):
            new_angle = last_angle + WEEKLY_ANGLE_STEP * k

            row: dict = {col: last_row[col] for col in test_df.columns}
            row["time_idx"] = last_time_idx + k
            row["week_sin"] = float(np.sin(new_angle))
            row["week_cos"] = float(np.cos(new_angle))
            row[TARGET]     = 0.0
            # Zero out unknown reals — TFT decoder does not use these
            for col in TIME_VARYING_UNKNOWN_REALS:
                row[col] = 0.0
            future_rows.append(row)

    future_df    = pd.DataFrame(future_rows)
    inference_df = (
        pd.concat([test_df, future_df], ignore_index=True)
        .sort_values(["region_id", "time_idx"])
        .reset_index(drop=True)
    )

    ordered_regions: list[str] = sorted(inference_df["region_id"].unique().tolist())

    pred_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        data               = inference_df,
        predict            = True,   # one sample per group (last window)
        stop_randomization = True,
    )

    return pred_dataset, ordered_regions
