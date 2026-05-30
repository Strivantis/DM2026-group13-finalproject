"""
dataset.py -- v36 TFT Data Pipeline
=====================================
Provides TimeSeriesDataSet construction for pytorch_forecasting's
TemporalFusionTransformer. Replaces the v35 354-dimensional tabular
flattening pipeline with raw, sequential 13-week meteorological windows.

Architecture Notes
------------------
* time_idx     : week_idx (0–781 for train; remapped 782–794 for test encoder).
* HORIZON      : 5-week future prediction window.
* WINDOW_SIZE  : 13-week historical encoder context (rigid).
* 5-Fold CV    : Rolling time-based validation; each fold reserves a clean
                 HORIZON-week block as the validation target.
* Inference     : Test encoder (13 weeks) + 5 future placeholder rows
                  per region → one prediction sample per region.

Column Categories (from processed CSVs)
-----------------------------------------
  time_varying_known_reals    : week_sin, week_cos        (calendar cycle, known ahead)
  time_varying_unknown_reals  : tmp, prec, humidity, wind, surf_pre,
                                tmp_roll_mean_4w, humidity_roll_mean_4w,
                                prec_roll_sum_4w             (only observed in past)
  static_categoricals         : region_id
  target                      : score

Note: sunshine and evap are not present in the processed CSVs and are
therefore excluded despite the spec mention.
"""

import os
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# ---------------------------------------------------------------------------
# Hyper-parameters (exported for train.py)
# ---------------------------------------------------------------------------
WINDOW_SIZE : int = 13   # max_encoder_length — rigid 13-week historical horizon
HORIZON     : int = 5    # max_prediction_length — 5-week forecast window
N_FOLDS     : int = 5    # rolling time-based CV folds

# Inferred from processed CSV; hardcoded for determinism
T_TRAIN_MAX : int = 781  # max week_idx in training data  (week 0 … 781 = 782 weeks)
T_TEST_OFFSET: int = T_TRAIN_MAX + 1  # test week_idx 0 → time_idx 782

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------
TIME_VARYING_KNOWN_REALS: list[str] = [
    "week_sin",           # sin(2π × week_of_year / 52)  — known for future weeks
    "week_cos",           # cos(2π × week_of_year / 52)  — known for future weeks
]

TIME_VARYING_UNKNOWN_REALS: list[str] = [
    "tmp",                # weekly mean temperature
    "prec",               # weekly precipitation total
    "humidity",           # weekly mean relative humidity
    "wind",               # weekly mean wind speed
    "surf_pre",           # surface atmospheric pressure
    "tmp_roll_mean_4w",   # 4-week rolling mean temperature
    "humidity_roll_mean_4w",  # 4-week rolling mean humidity
    "prec_roll_sum_4w",   # 4-week rolling precipitation sum
]

STATIC_CATEGORICALS: list[str] = ["region_id"]

TARGET: str = "score"

# One full revolution per year in 52-week steps
WEEKLY_ANGLE_STEP: float = 2.0 * np.pi / 52.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_feature_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in rolling / lag feature columns with 0.0 (early-row boundary effect)."""
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
    Load train and test processed CSVs. Assign monotonically increasing
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

    # -- Sort chronologically per region ---------------------------------------
    train_df = (train_df
                .sort_values(["region_id", "week_idx"])
                .reset_index(drop=True))
    test_df  = (test_df
                .sort_values(["region_id", "week_idx"])
                .reset_index(drop=True))

    # -- Assign time_idx -------------------------------------------------------
    # Training: time_idx = week_idx  (already 0–781, gap-free per region)
    train_df["time_idx"] = train_df["week_idx"].astype(int)

    # Test encoder: remap week_idx 0–12  →  time_idx 782–794 so the timeline
    # is continuous across the train→test boundary.
    test_df["time_idx"] = (T_TEST_OFFSET + test_df["week_idx"]).astype(int)
    test_df[TARGET]     = 0.0   # dummy placeholder — not used during inference

    # -- Drop rows missing score in training -----------------------------------
    before = len(train_df)
    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)
    dropped  = before - len(train_df)
    if dropped:
        print(f"  [dataset] Dropped {dropped:,} train rows with NaN score.")

    # -- Fill NaN in feature columns -------------------------------------------
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
        train_end  — last training time_idx (inclusive)
        val_start  — first validation time_idx (first decoder target step)
        val_end    — last  validation time_idx (inclusive)
    """
    val_end   = T_TRAIN_MAX - (n_folds - 1 - fold_k) * horizon
    val_start = val_end - horizon + 1
    train_end = val_start - 1
    return int(train_end), int(val_start), int(val_end)


def build_training_dataset(fold_train_df: pd.DataFrame) -> TimeSeriesDataSet:
    """
    Construct a pytorch_forecasting TimeSeriesDataSet from one training fold's data.

    The TimeSeriesDataSet automatically creates sliding window samples:
      each sample = WINDOW_SIZE encoder steps + HORIZON decoder steps.

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
        # target_normalizer="auto" (EncoderNormalizer) — default, normalises
        # per encoder window to keep gradient scales stable across regions.
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

    Parameters
    ----------
    training_dataset : TimeSeriesDataSet
        Completed training dataset (used to reuse categorical encoders).
    test_df          : DataFrame
        13-week encoder context rows (time_idx 782–794) with dummy score.

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

        last_time_idx  = int(last_row["time_idx"])
        last_angle     = float(np.arctan2(
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
