"""
preprocess.py – Full Preprocessing Pipeline (v10)
=================================================
Functions
---------
  load_data              : load raw CSV from /data/
  impute_met_features    : ffill / bfill within region (applied BEFORE aggregation)
  handle_outliers        : Z-score clip (Z=3.5, applied BEFORE aggregation)
  align_labels_absolute  : weekly aggregation using ABSOLUTE INDEX grouping (TRAIN+TEST)
  aggregate_test_weekly  : weekly aggregation for TEST (thin wrapper)
  preprocess_data        : rolling (4w ONLY) + lag (1/2w) features; min_periods=1
  export_processed       : save to /data/processed/
  add_drought_index      : PET / deficit 4w rolling features (used by dataset.py)

Catastrophic Flaws Fixed from v9
---------------------------------
  FLAW 1 – Synthetic Calendar Grouping ("Ghost Weeks"):
    v9 used _date_to_components() which computed week_of_year from a 366-day
    synthetic calendar (53 weeks/year).  Because year boundaries don't align
    with 7-day multiples in this synthetic calendar, the groupby produced
    partial weeks at every year-end.  These ghost weeks had NaN scores
    (no daily row in that partial group was a "day 7") and silently polluted
    the training labels with ~15 NaN scores per region per year.

    Fix: Absolute Index Grouping.  After sorting by (region_id, date), assign
    week_idx = cumcount // 7.  This is purely positional and guarantees every
    group contains exactly 7 daily rows.

  FLAW 2 – Window Size > Test Horizon (Scale Collapse):
    v9 computed 8-week and 13-week rolling features.  The test set has only
    13 weeks of data.  For the 8-week and 13-week features at inference time
    (weeks 1-7 of test), min_periods=1 means the values are computed over
    fewer than 8/13 rows.  During training, those same features are always
    computed over full 8/13-row windows (782 weeks of history).  This creates
    a massive domain shift: the feature distribution at inference is
    qualitatively different from what the model saw during training.

    Fix: DELETE all 8-week and 13-week rolling features.  Keep ONLY 4-week
    rolling sums/means — a 4-week lookback is achievable in full from week 4
    onward, and the test set has 13 weeks, so feature stability begins at
    test week 4 (same as in training).

Data Constants (verified by verify_data.py)
-------------------------------------------
  Train: 2248 regions × 5480 daily rows each
    - Score non-NaN at row indices 6, 13, 20, ... (stride=7, first=6)
    - 5480 = 782 × 7 + 6  →  keep first 5474 rows (782 complete weeks)
    - Drop last 6 rows (no score, incomplete final week)
  Test:  2248 regions × 91 daily rows = 13 × 7 (zero leftover)

Physical Aggregation Rules
--------------------------
  prec, surf_pre  : sum   (precipitation accumulates over the week)
  all others      : mean  (temperature, humidity, wind — intensive variables)
  score           : max   (exactly 1 of 7 daily rows has a non-NaN score;
                           max == that value and never creates a NaN)

Cyclic Time Encoding
--------------------
  Uses the LAST day of each 7-day block for temporal position.
  Parses month and day from the date string (year is ignored — only
  intra-year position matters).
  doy   = day-of-year (1..366, synthetic 366-day calendar)
  ratio = doy / 365.25
  week_sin = sin(2π × ratio)
  week_cos = cos(2π × ratio)
  Both correctly handle the year wrap-around (sin/cos are periodic).
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
    Absolute Index Grouping (v10 key fix).

    Groups daily rows into 7-day blocks purely by row position
    (group_id = cumcount // 7 per region, after sorting by date).
    This guarantees zero ghost weeks.

    Train:  keeps first 5474 rows per region (782 complete weeks).
            Drops the last 6 leftover daily rows (no score assigned).
    Test:   91 rows × 13 weeks exactly – no truncation needed.

    Aggregation
    -----------
    prec, surf_pre          : sum   (precipitation accumulates)
    all other met cols      : mean  (intensive variables)
    score                   : max   (1 of 7 daily rows has non-NaN score;
                                     max == that value, never NaN)
    date                    : last  -> stored as week_end_date

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

    # --- Build aggregation dictionary ---
    sum_cols  = [c for c in ["prec", "surf_pre"] if c in df.columns]
    mean_cols = [
        c for c in [
            "humidity",
            "tmp", "dp_tmp", "wb_tmp", "surf_tmp",
            "tmp_max", "tmp_min", "tmp_range",
            "wind", "wind_max", "wind_min", "wind_range",
        ] if c in df.columns
    ]

    agg = {}
    for c in sum_cols:
        agg[c] = "sum"
    for c in mean_cols:
        agg[c] = "mean"
    if "score" in df.columns:
        agg["score"] = "max"    # 1 non-NaN per group → max == that value
    agg["date"] = "last"        # week_end_date = last daily date in the 7-day block

    weekly = (
        df.groupby(["region_id", "week_idx"], sort=False)
        .agg(agg)
        .reset_index()
    )
    weekly.rename(columns={"date": "week_end_date"}, inplace=True)
    weekly.sort_values(["region_id", "week_idx"], inplace=True, ignore_index=True)

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

    print("=" * 70)
    print("Preprocessing Pipeline v10  (Absolute Sequence Grouping)")
    print("Fixes: ghost-week grouping | 8w/13w feature scale collapse")
    print("Method: row_index // 7 per region | 4w-only rolling | doy/365.25 cyclic")
    print("=" * 70)

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

    # 4. Absolute Index Weekly Aggregation
    print("\nAbsolute Index Grouping (cumcount // 7) ...")
    train_w = align_labels_absolute(train_raw, is_train=True)
    test_w  = aggregate_test_weekly(test_raw)
    print(f"  train_w: {train_w.shape}  regions: {train_w['region_id'].nunique()}")
    print(f"  test_w : {test_w.shape}   regions: {test_w['region_id'].nunique()}")

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
    print("=" * 70)


if __name__ == "__main__":
    main()
