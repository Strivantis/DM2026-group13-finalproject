"""
preprocess.py – Full Preprocessing Pipeline (v19)
=================================================
Functions
---------
  load_data              : load raw CSV from /data/
  impute_met_features    : ffill / bfill within region (applied BEFORE aggregation)
  handle_outliers        : Z-score clip (Z=3.5, applied BEFORE aggregation)
  align_labels_absolute  : weekly aggregation – ENRICHED (v19: mean/max/min/std for met;
                           sum/max for precip; dp_tmp & wb_tmp pruned from output)
  aggregate_test_weekly  : weekly aggregation for TEST (thin wrapper)
  preprocess_data        : rolling (4w ONLY) + lag (1/2w) features; min_periods=1
  export_processed       : save to /data/processed/
  add_drought_index      : PET / deficit 4w rolling features (used by dataset.py)

v19 Changes from v10
--------------------
  1. Enriched Weekly Statistics:
     Meteorological features (tmp, humidity, wind): previously only mean-aggregated.
     Now: mean (baseline) + max (peak extremes) + min (trough) + std (intra-week
     variability / climate shock magnitude):
       tmp_week_max, tmp_week_min, tmp_week_std
       humidity_week_max, humidity_week_min, humidity_week_std
       wind_week_max, wind_week_min, wind_week_std
     Precipitation features (prec, surf_pre): previously only sum-aggregated.
     Now: sum (accumulation) + max (single-day extreme event):
       prec_week_max, surf_pre_week_max
     Rationale: the std signal captures sudden intra-week temperature swings that are
     predictive of drought onset but invisible in the weekly mean alone.

  2. Adversarial Feature Pruning:
     dp_tmp (dew point) and wb_tmp (wet-bulb temperature) are explicitly EXCLUDED
     from the aggregation output. Both are severe collinear proxies of (tmp, humidity)
     and exhibit temporal covariate drift (instrument drift artefacts in the synthetic
     calendar). They remain in MET_COLS for imputation / outlier clipping only.
     Retaining `tmp` and `humidity` as stable baseline indicators is sufficient.

Catastrophic Flaws Fixed from v9 (retained from v10)
------------------------------------------------------
  FLAW 1 – Synthetic Calendar Grouping ("Ghost Weeks"):
    v9 used _date_to_components() which produced partial weeks at year-end.
    Fix: Absolute Index Grouping. week_idx = cumcount // 7 per region.

  FLAW 2 – Window Size > Test Horizon (Scale Collapse):
    v9 computed 8-week and 13-week rolling features causing domain shift.
    Fix: DELETE all 8w and 13w rolling features. Keep only 4-week rolling.

Data Constants (verified by verify_data.py)
-------------------------------------------
  Train: 2248 regions × 5480 daily rows each
    - Score non-NaN at row indices 6, 13, 20, ... (stride=7, first=6)
    - 5480 = 782 × 7 + 6  →  keep first 5474 rows (782 complete weeks)
  Test:  2248 regions × 91 daily rows = 13 × 7

Physical Aggregation Rules (v19)
---------------------------------
  prec, surf_pre  : sum   + max  (accumulation + extreme event)
  tmp, humidity,
  wind            : mean  + max  + min  + std  (intensive variables – climate shocks)
  tmp_max / tmp_min / tmp_range / wind_* / surf_tmp  : mean (daily derived, week-avg)
  score           : max   (exactly 1 non-NaN per 7-day block)
  dp_tmp, wb_tmp  : NOT included in output (adversarial collinear features)

Cyclic Time Encoding
--------------------
  doy   = day-of-year (1..366, synthetic 366-day calendar)
  ratio = doy / 365.25
  week_sin = sin(2π × ratio)
  week_cos = cos(2π × ratio)
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_BASE, "data")
_PROC = os.path.join(_DATA, "processed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MET_COLS = [
    "prec", "surf_pre", "humidity",
    "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

# Only 4-week rolling windows (8w and 13w DELETED – exceed test inference horizon)
ROLL_WINDOWS = [4]

# Drought index features – 4-week rolling only
DROUGHT_ROLL_WINDOWS = [4]
DROUGHT_FEAT_COLS = [
    "pet",
    "deficit",
    "deficit_roll_cum_4w",
]

# Training data constants (verified by verify_data.py)
_DAYS_PER_TRAIN  = 5480   # total daily rows per region
_KEEP_DAYS_TRAIN = 5474   # 782 × 7  (drop last 6 leftover days)
_WEEKS_PER_TRAIN = 782    # complete 7-day weeks in train
_DAYS_PER_TEST   = 91     # 13 × 7
_WEEKS_PER_TEST  = 13

# Cumulative day offset at start of each month (synthetic 366-day calendar)
# Jan=31, Feb=29, Mar=31, Apr=30, May=31, Jun=30,
# Jul=31, Aug=31, Sep=30, Oct=31, Nov=30, Dec=31
_MONTH_OFFSET = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_DAYS_PER_SYNTH_YEAR = 366   # synthetic calendar always has 366 days


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _parse_doy(date_str: str) -> int:
    """
    Parse 'YYYY-MM-DD' string → day-of-year (1..366).
    Works with any year (including 5-digit years like '23102-07-17').
    Only month and day are used; year is intentionally ignored.
    """
    parts = date_str.split("-")
    m   = int(parts[1])
    d   = int(parts[2])
    return _MONTH_OFFSET[m - 1] + d   # 1-indexed doy


def _parse_ordinal(date_str: str) -> int:
    """
    Parse 'YYYY-MM-DD' → absolute integer day ordinal using the synthetic
    366-day calendar.  Used to compute the temporal gap between train and test.

    ordinal = (year - 1) * 366 + (doy - 1)   [0-indexed from year 1, day 1]
    """
    parts = date_str.split("-")
    y   = int(parts[0])
    m   = int(parts[1])
    d   = int(parts[2])
    doy = _MONTH_OFFSET[m - 1] + d
    return (y - 1) * _DAYS_PER_SYNTH_YEAR + (doy - 1)


# ---------------------------------------------------------------------------
# 0. Data loading
# ---------------------------------------------------------------------------
def load_data(filename: str) -> pd.DataFrame:
    """Load raw CSV from /data/<filename>."""
    path = os.path.join(_DATA, filename)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. Imputation – forward/backward fill per region (applied on DAILY data)
# ---------------------------------------------------------------------------
def impute_met_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then backward-fill missing meteorological values within
    each region BEFORE aggregation.  Does NOT drop any rows.
    dp_tmp and wb_tmp are still imputed here (they exist in raw data),
    but their aggregated weekly columns are excluded from the output.
    """
    df = df.copy()
    for col in MET_COLS:
        if col in df.columns:
            df[col] = (
                df.groupby("region_id")[col]
                .transform(lambda s: s.ffill().bfill())
            )
    return df


# ---------------------------------------------------------------------------
# 2. Outlier handling – Z-score clip (applied on DAILY data)
# ---------------------------------------------------------------------------
def handle_outliers(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    """Clip met feature values beyond ±z_thresh standard deviations."""
    df = df.copy()
    for col in MET_COLS:
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 0:
                df[col] = df[col].clip(
                    lower=mu - z_thresh * sigma,
                    upper=mu + z_thresh * sigma,
                )
    return df


# ---------------------------------------------------------------------------
# 3a. Weekly aggregation – ABSOLUTE INDEX GROUPING (Train + Test)
# ---------------------------------------------------------------------------
def align_labels_absolute(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Absolute Index Grouping (v10 key fix; v19: enriched weekly statistics).

    Groups daily rows into 7-day blocks purely by row position
    (group_id = cumcount // 7 per region, after sorting by date).
    This guarantees zero ghost weeks.

    Train:  keeps first 5474 rows per region (782 complete weeks).
            Drops the last 6 leftover daily rows (no score assigned).
    Test:   91 rows × 13 weeks exactly – no truncation needed.

    v19 Enriched Aggregation
    -------------------------
    Precipitation (prec, surf_pre):
        sum    – weekly accumulation (original behaviour)
        max    – peak single-day rainfall / surface precip (extreme event)
                 → prec_week_max, surf_pre_week_max
    Meteorological (tmp, humidity, wind):
        mean   – weekly baseline average (retains original column name)
        max    – peak daily extreme          → tmp_week_max, humidity_week_max, wind_week_max
        min    – daily trough                → tmp_week_min, humidity_week_min, wind_week_min
        std    – intra-week variability      → tmp_week_std, humidity_week_std, wind_week_std
    Derived daily cols (tmp_max, tmp_min, tmp_range, wind_max, wind_min,
        wind_range, surf_tmp): mean over the 7-day block (unchanged from v10).
    score  : max – exactly 1 of 7 daily rows has a non-NaN score;
             max == that value and never creates a NaN.
    date   : last → renamed week_end_date (last daily date in the 7-day block).

    Adversarial Feature Pruning (v19)
    ----------------------------------
    dp_tmp (dew point) and wb_tmp (wet-bulb temperature) are intentionally
    EXCLUDED from the aggregation output. Both are severe collinear proxies of
    (tmp, humidity) and exhibit temporal covariate drift. They remain in
    MET_COLS for imputation / outlier clipping on the raw daily data.

    Outputs added per weekly row
    ----------------------------
    week_sin, week_cos  : cyclic time (doy / 365.25 of week_end_date)
    day_ordinal         : absolute ordinal of week_end_date
                          (integer, for gap computation in dataset.py)
    """
    df = df.copy().sort_values(["region_id", "date"], ignore_index=True)

    # Per-region 0-based row position
    cumcnt = df.groupby("region_id").cumcount()

    if is_train:
        # Keep only the first 5474 rows per region (= 782 complete 7-day weeks)
        keep_mask = cumcnt < _KEEP_DAYS_TRAIN
        df   = df[keep_mask].copy()
        cumcnt = cumcnt[keep_mask]

    # Assign week index: rows 0-6 → week 0, rows 7-13 → week 1, ...
    df["week_idx"] = (cumcnt // 7).values

    # --- Build named aggregation specification (v19: enriched weekly stats) ---
    # Uses pd.NamedAgg to produce multiple output columns from the same input column.
    named_agg = {}

    # Precipitation: weekly sum (accumulation) + intra-week max (extreme event capture)
    # dp_tmp and wb_tmp are intentionally NOT included here.
    if "prec" in df.columns:
        named_agg["prec"]          = pd.NamedAgg(column="prec", aggfunc="sum")
        named_agg["prec_week_max"] = pd.NamedAgg(column="prec", aggfunc="max")
    if "surf_pre" in df.columns:
        named_agg["surf_pre"]          = pd.NamedAgg(column="surf_pre", aggfunc="sum")
        named_agg["surf_pre_week_max"] = pd.NamedAgg(column="surf_pre", aggfunc="max")

    # Meteorological: mean + max + min + std  (dp_tmp & wb_tmp intentionally excluded)
    if "humidity" in df.columns:
        named_agg["humidity"]          = pd.NamedAgg(column="humidity", aggfunc="mean")
        named_agg["humidity_week_max"] = pd.NamedAgg(column="humidity", aggfunc="max")
        named_agg["humidity_week_min"] = pd.NamedAgg(column="humidity", aggfunc="min")
        named_agg["humidity_week_std"] = pd.NamedAgg(column="humidity", aggfunc="std")
    if "tmp" in df.columns:
        named_agg["tmp"]          = pd.NamedAgg(column="tmp", aggfunc="mean")
        named_agg["tmp_week_max"] = pd.NamedAgg(column="tmp", aggfunc="max")
        named_agg["tmp_week_min"] = pd.NamedAgg(column="tmp", aggfunc="min")
        named_agg["tmp_week_std"] = pd.NamedAgg(column="tmp", aggfunc="std")
    if "wind" in df.columns:
        named_agg["wind"]          = pd.NamedAgg(column="wind", aggfunc="mean")
        named_agg["wind_week_max"] = pd.NamedAgg(column="wind", aggfunc="max")
        named_agg["wind_week_min"] = pd.NamedAgg(column="wind", aggfunc="min")
        named_agg["wind_week_std"] = pd.NamedAgg(column="wind", aggfunc="std")

    # Derived daily columns: mean-averaged over the 7-day block (unchanged from v10)
    for _c in ["tmp_max", "tmp_min", "tmp_range", "surf_tmp",
               "wind_max", "wind_min", "wind_range"]:
        if _c in df.columns:
            named_agg[_c] = pd.NamedAgg(column=_c, aggfunc="mean")

    if "score" in df.columns:
        named_agg["score"] = pd.NamedAgg(column="score", aggfunc="max")

    # Output "date" as "week_end_date" directly via NamedAgg key (no separate rename needed)
    named_agg["week_end_date"] = pd.NamedAgg(column="date", aggfunc="last")

    weekly = (
        df.groupby(["region_id", "week_idx"], sort=False)
        .agg(**named_agg)
        .reset_index()
    )
    weekly.sort_values(["region_id", "week_idx"], inplace=True, ignore_index=True)

    # Safety fill: std columns may be NaN for perfectly constant feature groups (rare edge case)
    for _std_col in ["tmp_week_std", "humidity_week_std", "wind_week_std"]:
        if _std_col in weekly.columns:
            weekly[_std_col] = weekly[_std_col].fillna(0.0).astype(np.float32)

    # --- Cyclic time encoding (from week_end_date) ---
    doy_arr   = weekly["week_end_date"].map(_parse_doy).astype(np.float32)
    ratio_arr = doy_arr / 365.25
    weekly["week_sin"] = np.sin(2.0 * np.pi * ratio_arr).astype(np.float32)
    weekly["week_cos"] = np.cos(2.0 * np.pi * ratio_arr).astype(np.float32)

    # --- Absolute day ordinal (for temporal gap computation) ---
    weekly["day_ordinal"] = (
        weekly["week_end_date"].map(_parse_ordinal).astype(np.int64)
    )

    return weekly


# ---------------------------------------------------------------------------
# 3b. Weekly aggregation – TEST  (thin wrapper)
# ---------------------------------------------------------------------------
def aggregate_test_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly aggregation for test (91 days → 13 complete weeks, no truncation)."""
    return align_labels_absolute(df, is_train=False)


# ---------------------------------------------------------------------------
# 4. Rolling + lag feature engineering  (4w ONLY)
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling aggregates and lag features per region.

    Rolling (window=[4], min_periods=1):
      prec     → sum  : prec_roll_sum_4w
      tmp      → mean : tmp_roll_mean_4w
      humidity → mean : humidity_roll_mean_4w

    Lags (1w and 2w):
      tmp, humidity, prec, wind  (and score in train)

    8-week and 13-week rolling features are INTENTIONALLY OMITTED.
    They would require 8-13 weeks of warm-up data to stabilise, but the
    test set only has 13 weeks; features computed on partial windows at
    inference produce a distribution qualitatively different from training.

    v19: No rolling / lag features are added for the new enriched weekly
    statistics (tmp_week_max, etc.) as these are already intra-week
    temporal statistics computed at the aggregation stage.
    """
    df = df.copy().sort_values(["region_id", "week_idx"], ignore_index=True)

    roll_spec = [
        ("prec",     "sum",  "prec_roll_sum_{w}w"),
        ("tmp",      "mean", "tmp_roll_mean_{w}w"),
        ("humidity", "mean", "humidity_roll_mean_{w}w"),
    ]

    for base_col, func, tmpl in roll_spec:
        if base_col not in df.columns:
            continue
        for w in ROLL_WINDOWS:
            feat = tmpl.format(w=w)
            if func == "sum":
                df[feat] = (
                    df.groupby("region_id")[base_col]
                    .transform(lambda s: s.rolling(w, min_periods=1).sum())
                )
            else:
                df[feat] = (
                    df.groupby("region_id")[base_col]
                    .transform(lambda s: s.rolling(w, min_periods=1).mean())
                )

    lag_cols = ["tmp", "humidity", "prec", "wind"]
    if "score" in df.columns:
        lag_cols.append("score")

    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in [1, 2]:
            feat = f"{col}_lag{lag}w"
            df[feat] = df.groupby("region_id")[col].transform(
                lambda s: s.shift(lag)
            )

    return df


# ---------------------------------------------------------------------------
# 5. Export to /data/processed/
# ---------------------------------------------------------------------------
def export_processed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fmt: str = "csv",
) -> None:
    """Save processed DataFrames to /data/processed/ (raw files untouched)."""
    os.makedirs(_PROC, exist_ok=True)
    train_path = os.path.join(_PROC, "train_processed.csv")
    test_path  = os.path.join(_PROC, "test_processed.csv")
    if fmt == "csv":
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    print(f"  Exported train -> {train_path}  {train_df.shape}")
    print(f"  Exported test  -> {test_path}   {test_df.shape}")


# ---------------------------------------------------------------------------
# Drought Index Feature Engineering  (4w ONLY – used by src/dataset.py)
# ---------------------------------------------------------------------------
def add_drought_index(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    PET-based drought proxy features. 4-week rolling only.
    8w and 13w rolling variants DELETED (exceed test inference horizon).

    1. PET           = 0.55 * max(tmp, 0)
    2. deficit       = prec - PET
    3. deficit_roll_cum_4w = 4-week rolling cumulative deficit (min_periods=1)
    """
    df = df.copy()
    df["pet"]     = (0.55 * df["tmp"].clip(lower=0.0)).astype(np.float32)
    df["deficit"] = (df["prec"] - df["pet"]).astype(np.float32)

    for w in DROUGHT_ROLL_WINDOWS:
        col = f"deficit_roll_cum_{w}w"
        df[col] = (
            df.groupby("region_id")["deficit"]
            .transform(lambda s: s.rolling(window=w, min_periods=1).sum())
        ).astype(np.float32)

    if not is_train:
        for col in DROUGHT_FEAT_COLS:
            if col in df.columns:
                df[col] = (
                    df.groupby("region_id")[col]
                    .transform(lambda s: s.ffill().fillna(0))
                )
    return df


# ---------------------------------------------------------------------------
# CLI entry-point: regenerate processed CSVs
# ---------------------------------------------------------------------------
def main():
    import time
    t0 = time.time()

    print("=" * 75)
    print("Preprocessing Pipeline v19  (Enriched Weekly Stats + Adversarial Pruning)")
    print("Fixes: ghost-week grouping | 8w/13w scale collapse | dp_tmp/wb_tmp collinearity")
    print("New:   mean/max/min/std for tmp/humidity/wind  |  sum/max for prec/surf_pre")
    print("=" * 75)

    # 1. Load raw data
    print("\nLoading raw data ...")
    train_raw = load_data("train.csv")
    test_raw  = load_data("test.csv")
    print(f"  train: {train_raw.shape}  |  test: {test_raw.shape}")
    print(f"  train regions: {train_raw['region_id'].nunique()}")
    print(f"  test  regions: {test_raw['region_id'].nunique()}")

    # 2. Impute on daily data (before aggregation, preserves continuity)
    print("\nImputing met features (ffill/bfill per region on daily data) ...")
    train_raw = impute_met_features(train_raw)
    test_raw  = impute_met_features(test_raw)

    # 3. Outlier clip on daily data
    print("Outlier clipping (Z=3.5) ...")
    train_raw = handle_outliers(train_raw, z_thresh=3.5)
    test_raw  = handle_outliers(test_raw,  z_thresh=3.5)

    # 4. Absolute Index Weekly Aggregation (v19: enriched stats)
    print("\nAbsolute Index Grouping (cumcount // 7) with v19 enriched aggregation ...")
    train_w = align_labels_absolute(train_raw, is_train=True)
    test_w  = aggregate_test_weekly(test_raw)
    print(f"  train_w: {train_w.shape}  regions: {train_w['region_id'].nunique()}")
    print(f"  test_w : {test_w.shape}   regions: {test_w['region_id'].nunique()}")

    # v19: show new enriched columns
    new_cols = [c for c in train_w.columns
                if any(x in c for x in ["_week_max", "_week_min", "_week_std"])]
    print(f"  [v19] New enriched weekly columns ({len(new_cols)}): {new_cols}")

    # Verify dp_tmp and wb_tmp are absent from processed output
    adv_cols = [c for c in train_w.columns if c in ("dp_tmp", "wb_tmp")]
    if adv_cols:
        print(f"  *** WARNING: Adversarial columns still present: {adv_cols}")
    else:
        print("  v dp_tmp and wb_tmp correctly pruned from weekly output.")

    # Verify week counts are exact
    train_wk_counts = train_w.groupby("region_id").size()
    test_wk_counts  = test_w.groupby("region_id").size()
    print(f"  Weeks/train region : min={train_wk_counts.min()}  max={train_wk_counts.max()}  "
          f"(expected {_WEEKS_PER_TRAIN})")
    print(f"  Weeks/test  region : min={test_wk_counts.min()}   max={test_wk_counts.max()}  "
          f"(expected {_WEEKS_PER_TEST})")
    assert train_wk_counts.min() == _WEEKS_PER_TRAIN, \
        f"Train week count mismatch: min={train_wk_counts.min()}"
    assert test_wk_counts.min() == _WEEKS_PER_TEST, \
        f"Test week count mismatch: min={test_wk_counts.min()}"
    print("  WEEK COUNT ASSERTIONS PASSED")

    # 5. Cyclic feature range check
    assert "week_sin" in train_w.columns and "week_cos" in train_w.columns
    print(f"  week_sin range: [{train_w['week_sin'].min():.3f}, {train_w['week_sin'].max():.3f}]")
    print(f"  week_cos range: [{train_w['week_cos'].min():.3f}, {train_w['week_cos'].max():.3f}]")

    # 6. Rolling + lag features
    print("\nAdding rolling (4w only) & lag features ...")
    train_w = preprocess_data(train_w)
    test_w  = preprocess_data(test_w)
    print(f"  train_w: {train_w.shape}  |  test_w: {test_w.shape}")

    # 7. ZERO GHOST WEEKS assertion
    if "score" in train_w.columns:
        n_nan_scores = train_w["score"].isna().sum()
        print(f"\n  NaN scores in train_w (MUST BE 0): {n_nan_scores}")
        assert n_nan_scores == 0, \
            f"GHOST WEEKS DETECTED: {n_nan_scores} NaN scores. Aborting."
        print("  ZERO GHOST WEEKS CONFIRMED")

        score_vals = train_w["score"]
        print(f"  Score mean:         {score_vals.mean():.4f}")
        print(f"  Score zero fraction:{(score_vals == 0.0).mean():.2%}")
        print(f"  Score min / max:    {score_vals.min():.4f} / {score_vals.max():.4f}")

    # 8. Region count validation
    n_train_rgn = train_w["region_id"].nunique()
    n_test_rgn  = test_w["region_id"].nunique()
    assert n_train_rgn == 2248, f"Expected 2248 train regions, got {n_train_rgn}"
    assert n_test_rgn  == 2248, f"Expected 2248 test regions,  got {n_test_rgn}"
    print(f"\n  REGION COUNT PASSED: {n_train_rgn} train, {n_test_rgn} test")

    # 9. Export
    print("\nExporting processed data ...")
    export_processed(train_w, test_w, fmt="csv")

    elapsed = time.time() - t0
    print(f"\nTotal preprocessing time: {elapsed:.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
