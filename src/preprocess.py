"""
preprocess.py – Full Preprocessing Pipeline (v5)
=================================================
Functions
---------
  load_data              : load raw CSV from /data/
  impute_met_features    : ffill / bfill within region (no dropna)
  handle_outliers        : Z-score clip (Z=3.5 default)
  align_labels_strategy_a: weekly aggregation for TRAIN – LEFT join, keeps all 2248 regions
  aggregate_test_weekly  : weekly aggregation for TEST
  preprocess_data        : rolling (4/8/13w) + lag (1/2w) features; min_periods=1
  export_processed       : save to /data/processed/

  add_drought_index      : PET / deficit rolling features (used by dataset.py)

Root Cause of v4 "Region Extinction Event"
------------------------------------------
  The old pipeline used pd.to_datetime(df['date'], errors='coerce') to parse date
  strings such as "3020-09-18" (year > 2262, beyond pandas ns-datetime limit).
  ALL dates became NaT → week_key = NaN for every row.  When the weekly weather
  aggregate was inner-joined against a score table keyed on valid week_keys, only
  the 133 regions whose NaN-keyed group happened to merge successfully survived.
  2115 regions were silently dropped.

  Fix: Custom string-based date parser that handles the dataset's synthetic calendar:
  - Feb 29 exists every year (366-day fixed calendar, no Gregorian leap rules)
  - Years can be > 9999 (5-digit years such as 23102)
  - Uses split('-') to correctly parse both 4-digit and 5-digit year strings
  Weekly aggregation uses a pure groupby (no secondary join) so ALL 2248 regions
  are always preserved.  Rolling windows use min_periods=1 to avoid dropping rows.

Custom Calendar
---------------
  The dataset uses a synthetic 366-day calendar where every year has the same
  month structure:
    Jan=31, Feb=29, Mar=31, Apr=30, May=31, Jun=30,
    Jul=31, Aug=31, Sep=30, Oct=31, Nov=30, Dec=31  (total = 366 days/year)
  This means Feb 29 is valid every year and there are NO Gregorian leap-year rules.
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

ROLL_WINDOWS = [4, 8, 13]   # weeks

# Drought index features (used by src/dataset.py)
DROUGHT_ROLL_WINDOWS = [4, 8, 13]
DROUGHT_FEAT_COLS = [
    "pet",
    "deficit",
    "deficit_roll_cum_4w",
    "deficit_roll_cum_8w",
    "deficit_roll_cum_13w",
]

# ---------------------------------------------------------------------------
# Custom calendar: 366-day fixed year, Feb always has 29 days
# ---------------------------------------------------------------------------
# Cumulative day offset at start of each month (0-based day-of-year before the month)
_MONTH_OFFSET = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
# Total days per year in this calendar
_DAYS_PER_YEAR = 366


def _date_to_components(date_str: str):
    """
    Parse 'YYYY-MM-DD' or 'YYYYY-MM-DD' into (year, month, day, doy, week_of_year,
    day_ordinal) using the dataset's synthetic 366-day calendar.

    Works for:
    - Regular 4-digit years (e.g., '3020-09-18')
    - 5-digit years (e.g., '23102-07-17')
    - Feb 29 on any year (not a valid Gregorian date for non-leap years)

    Returns
    -------
    (year: int, month: int, day: int, doy: int, woy: int, ordinal: int)
      doy : day-of-year 1..366
      woy : week-of-year 1..53  (floor-based: Jan 1-7 → W01, Jan 8-14 → W02 …)
      ordinal : days elapsed since day 0 of year 1 in this custom calendar
    """
    parts = date_str.split("-")
    y = int(parts[0])
    m = int(parts[1])
    d = int(parts[2])
    doy = _MONTH_OFFSET[m - 1] + d          # 1-indexed day of year
    woy = (doy - 1) // 7 + 1               # 1-indexed week of year (1-53)
    ordinal = (y - 1) * _DAYS_PER_YEAR + doy - 1   # days since custom epoch
    return y, m, d, doy, woy, ordinal


def _build_date_cache(dates) -> dict:
    """
    Build a lookup dict: date_str -> (week_key, year, month, week_of_year, ordinal)
    using the dataset's synthetic 366-day calendar.

    Only called once on unique date strings → fast regardless of dataset size.
    Handles:
    - Any 4-digit or 5-digit year (zero-padded to 6 digits in week_key for
      correct lexicographic sort order across 4-digit and 5-digit years)
    - Feb 29 on any year (dataset's synthetic calendar)
    """
    cache = {}
    for ds in dates:
        y, m, d, doy, woy, ordinal = _date_to_components(ds)
        # Zero-pad year to 6 digits so lexicographic sort == chronological sort
        # for years up to 999999 (dataset spans ~3004 to ~23102)
        week_key = f"{y:06d}-W{woy:02d}"
        cache[ds] = (week_key, y, m, woy, ordinal)
    return cache


# ---------------------------------------------------------------------------
# 0. Data loading
# ---------------------------------------------------------------------------
def load_data(filename: str) -> pd.DataFrame:
    """Load raw CSV from /data/<filename>."""
    path = os.path.join(_DATA, filename)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. Imputation – forward/backward fill per region (NO dropna)
# ---------------------------------------------------------------------------
def impute_met_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then backward-fill missing meteorological values within
    each region.  Does NOT drop any rows.
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
# 2. Outlier handling – Z-score clip
# ---------------------------------------------------------------------------
def handle_outliers(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    """Clip met feature values beyond ±z_thresh standard deviations.
    Never drops rows."""
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
# 3a. Weekly aggregation – TRAIN  (LEFT-join equivalent: all regions kept)
# ---------------------------------------------------------------------------
def align_labels_strategy_a(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily training rows to one row per (region_id, ISO-week).

    Meteorological features : mean of daily values in the week.
    score                   : mean of non-NaN daily scores in the week.
    week_end_date           : last date string in the group.

    All 2248 regions are always preserved (pure groupby, no secondary join).
    Uses custom 366-day calendar – safe for years > 9999 and Feb 29 every year.
    """
    df = df.copy()

    # Build date cache on unique date strings only (fast: ~5480 unique dates)
    unique_dates = df["date"].astype(str).unique()
    cache = _build_date_cache(unique_dates)

    # Use dict-map (faster than lambda for 12M rows — no per-element Python overhead)
    _wk  = {ds: v[0] for ds, v in cache.items()}
    _yr  = {ds: v[1] for ds, v in cache.items()}
    _mo  = {ds: v[2] for ds, v in cache.items()}
    _woy = {ds: v[3] for ds, v in cache.items()}
    _ord = {ds: v[4] for ds, v in cache.items()}

    date_col = df["date"].astype(str)
    df["week_key"]     = date_col.map(_wk)
    df["year"]         = date_col.map(_yr).astype(np.int64)
    df["month"]        = date_col.map(_mo).astype(np.int32)
    df["week_of_year"] = date_col.map(_woy).astype(np.int32)
    df["day_ordinal"]  = date_col.map(_ord).astype(np.float64)

    # Aggregation spec
    agg = {}
    for col in MET_COLS:
        if col in df.columns:
            agg[col] = "mean"
    if "score" in df.columns:
        agg["score"] = "mean"   # mean of non-NaN scored days

    agg.update({
        "month":        "first",
        "week_of_year": "first",
        "year":         "first",
        "day_ordinal":  "last",   # ordinal of last day in the week
        "date":         "last",   # → week_end_date
    })

    weekly = (
        df.groupby(["region_id", "week_key"], sort=False)
        .agg(agg)
        .reset_index()
    )
    weekly.rename(columns={"date": "week_end_date"}, inplace=True)
    weekly.sort_values(["region_id", "week_key"], inplace=True, ignore_index=True)
    return weekly


# ---------------------------------------------------------------------------
# 3b. Weekly aggregation – TEST  (no score column)
# ---------------------------------------------------------------------------
def aggregate_test_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Same weekly aggregation as align_labels_strategy_a, for test data."""
    return align_labels_strategy_a(df)


# ---------------------------------------------------------------------------
# 4. Rolling + lag feature engineering
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling aggregates and lag features per region.

    Rolling: prec → sum, tmp/humidity → mean  (windows 4/8/13w, min_periods=1)
    Lags:    tmp, humidity, prec, wind, score  (1w and 2w)
    """
    df = df.copy().sort_values(["region_id", "week_key"], ignore_index=True)

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
    print(f"  Exported train → {train_path}  {train_df.shape}")
    print(f"  Exported test  → {test_path}   {test_df.shape}")


# ---------------------------------------------------------------------------
# Drought Index Feature Engineering  (used by src/dataset.py)
# ---------------------------------------------------------------------------
def add_drought_index(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    PET-based drought proxy features.
    1. PET   = 0.55 * max(tmp, 0)
    2. deficit = prec - PET
    3. Rolling cumulative sum of deficit: 4w, 8w, 13w (min_periods=1)
    """
    df = df.copy()
    df["pet"] = (0.55 * df["tmp"].clip(lower=0.0)).astype(np.float32)
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
    import math
    t0 = time.time()

    print("=" * 65)
    print("Preprocessing Pipeline v5  (Region Extinction Fix)")
    print("Custom 366-day calendar: Feb 29 every year, years > 9999 OK")
    print("=" * 65)

    # 1. Load
    print("\nLoading raw data …")
    train_raw = load_data("train.csv")
    test_raw  = load_data("test.csv")
    print(f"  train: {train_raw.shape}  |  test: {test_raw.shape}")
    print(f"  train regions: {train_raw['region_id'].nunique()}")
    print(f"  test  regions: {test_raw['region_id'].nunique()}")

    # 2. Weekly aggregation FIRST (reduces 12.3M → ~1.76M rows)
    print("\nWeekly aggregation (custom 366-day calendar) …")
    train_w = align_labels_strategy_a(train_raw)
    test_w  = aggregate_test_weekly(test_raw)
    print(f"  train_w: {train_w.shape}  regions: {train_w['region_id'].nunique()}")
    print(f"  test_w : {test_w.shape}   regions: {test_w['region_id'].nunique()}")

    # 3. Impute on weekly data (7× faster than on raw daily rows)
    print("Imputing met features (ffill/bfill per region) …")
    train_w = impute_met_features(train_w)
    test_w  = impute_met_features(test_w)

    # 4. Outlier clip on weekly data
    print("Outlier clipping (Z=3.5) …")
    train_w = handle_outliers(train_w, z_thresh=3.5)
    test_w  = handle_outliers(test_w,  z_thresh=3.5)

    # 5. Rolling + lag features
    print("Adding rolling & lag features …")
    train_w = preprocess_data(train_w)
    test_w  = preprocess_data(test_w)
    print(f"  train_w: {train_w.shape}  |  test_w: {test_w.shape}")

    # 5b. Forward-fill any residual NaN scores (at most 1 partial week per region)
    if "score" in train_w.columns:
        before = train_w["score"].isna().sum()
        train_w["score"] = (
            train_w.groupby("region_id")["score"]
            .transform(lambda s: s.ffill().bfill())
        )
        after = train_w["score"].isna().sum()
        print(f"  Score NaN before/after region ffill: {before} → {after}")

    # 6. Validation
    n_train_regions = train_w["region_id"].nunique()
    n_test_regions  = test_w["region_id"].nunique()
    assert n_train_regions == 2248, (
        f"Expected 2248 train regions, got {n_train_regions}"
    )
    assert n_test_regions == 2248, (
        f"Expected 2248 test regions, got {n_test_regions}"
    )
    print(f"\n✓ VALIDATION PASSED: {n_train_regions} train regions, "
          f"{n_test_regions} test regions.")

    score_mean = train_w["score"].mean()
    print(f"  Score mean (weekly): {score_mean:.4f}")
    bias = math.log((score_mean / 5.0) / (1.0 - score_mean / 5.0))
    print(f"  Bias init (Sigmoid*5 head): {bias:.4f}")

    # 7. Export
    print("\nExporting processed data …")
    export_processed(train_w, test_w, fmt="csv")

    print(f"\nTotal preprocessing time: {time.time() - t0:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
