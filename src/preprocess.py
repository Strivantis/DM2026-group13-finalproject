"""
Drought Prediction - Data Preprocessing & Feature Engineering Pipeline
=======================================================================
Provides:
  - load_data()               : load train/test CSVs; parse dates safely
  - handle_outliers()         : configurable Z-score clipping
  - align_labels_strategy_a() : weekly aggregation (Strategy A – primary)
  - preprocess_data()         : full feature-engineering pipeline (reusable on any df)
  - export_processed()        : save final DataFrames to data/processed/

NOTE on date anomaly
--------------------
The dataset uses fictional years 3004-3020 which lie outside the
pandas Timestamp range (max ≈ 2262).  To avoid NaT / overflow errors we
keep dates as plain Python strings internally and extract temporal
features directly from the string components (year, month, ISO week-of-year).
A synthetic "time index" (integer day offset from the dataset's own epoch)
is derived where an ordinal date is needed for sorting / rolling windows.
"""

import os
import datetime
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# ---------------------------------------------------------------------------
# Meteorological feature columns (no label, no id, no date)
# ---------------------------------------------------------------------------
MET_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]


# ---------------------------------------------------------------------------
# Internal helpers for date string handling
# ---------------------------------------------------------------------------
_EPOCH = datetime.date(3004, 12, 31)  # first date seen in data → ordinal 0


def _date_str_to_ordinal(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' string to an integer ordinal since dataset epoch."""
    try:
        y, m, d = date_str.split("-")
        return (datetime.date(int(y), int(m), int(d)) - _EPOCH).days
    except Exception:
        return np.nan


def _date_str_to_iso_week(date_str: str):
    """Return (year, iso_week_number) from 'YYYY-MM-DD' string."""
    try:
        y, m, d = date_str.split("-")
        iso = datetime.date(int(y), int(m), int(d)).isocalendar()
        return iso[0], iso[1]   # iso_year, iso_week
    except Exception:
        return np.nan, np.nan


def _date_str_to_month(date_str: str) -> int:
    try:
        return int(date_str.split("-")[1])
    except Exception:
        return np.nan


def _date_str_to_year(date_str: str) -> int:
    try:
        return int(date_str.split("-")[0])
    except Exception:
        return np.nan


def _week_key(date_str: str) -> str:
    """
    Return an ISO-week key 'YYYY-Www' that can be used for grouping.
    Example: '3005-W03'
    """
    try:
        y, m, d = date_str.split("-")
        iso = datetime.date(int(y), int(m), int(d)).isocalendar()
        return f"{iso[0]:04d}-W{iso[1]:02d}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Data Loading & Casting
# ---------------------------------------------------------------------------
def load_data(filename: str) -> pd.DataFrame:
    """
    Load a CSV from /data.
    Dates are stored as strings (no pd.to_datetime) to avoid the
    pandas Timestamp overflow for years > 2262.
    A synthetic integer column ``day_ordinal`` (days since 3004-12-31)
    is added for sorting and rolling-window arithmetic.
    """
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, dtype=str)

    # Keep 'date' as plain string; derive a sortable integer ordinal
    df["day_ordinal"] = df["date"].apply(_date_str_to_ordinal)

    # Cast meteorological features to float
    for col in MET_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cast score to float if present
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Sort by region then day_ordinal (integer – no NaT risk)
    df = df.sort_values(["region_id", "day_ordinal"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Feature Imputation (forward/backward fill per region)
# ---------------------------------------------------------------------------
def impute_met_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing meteorological values via ffill → bfill,
    grouped by region_id to prevent cross-region leakage.
    """
    df = df.copy()
    df[MET_COLS] = (
        df.groupby("region_id")[MET_COLS]
        .transform(lambda s: s.ffill().bfill())
    )
    return df


# ---------------------------------------------------------------------------
# 3. Outlier Handling (Z-score clipping, configurable threshold)
# ---------------------------------------------------------------------------
def handle_outliers(
    df: pd.DataFrame,
    cols: list = None,
    z_thresh: float = 3.5,
    strategy: str = "clip",   # "clip" | "retain"
) -> pd.DataFrame:
    """
    Detect extreme outliers per meteorological column using Z-scores
    within each region_id.  strategy='clip' replaces outliers with the
    column-level boundary; strategy='retain' leaves them unchanged.
    """
    if cols is None:
        cols = MET_COLS
    if strategy == "retain":
        return df

    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        region_mean = df.groupby("region_id")[col].transform("mean")
        region_std  = df.groupby("region_id")[col].transform("std")
        z_scores    = (df[col] - region_mean) / region_std.replace(0, np.nan)

        lower = region_mean - z_thresh * region_std
        upper = region_mean + z_thresh * region_std

        df.loc[z_scores < -z_thresh, col] = lower[z_scores < -z_thresh]
        df.loc[z_scores >  z_thresh, col] = upper[z_scores >  z_thresh]
    return df


# ---------------------------------------------------------------------------
# 4. Label Alignment – Strategy A: Weekly Aggregation  (PRIMARY)
# ---------------------------------------------------------------------------
def align_labels_strategy_a(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily rows to the ISO-week frequency per region_id.

    For each week that contains a ground-truth ``score`` label we compute:
      • prec            → sum   (total precipitation)
      • tmp, tmp_max, tmp_min, tmp_range, surf_tmp → mean / max / min
      • humidity, dp_tmp, wb_tmp, surf_pre         → mean
      • wind, wind_max, wind_min, wind_range        → mean / max / min
      • score           → mean  (only one non-NaN per week ⇒ == that value)
      • week_of_year, month, year → derived from the score row's date string

    Rows without a ground-truth ``score`` are dropped (no interpolation).
    The result has one row per (region_id, ISO-week).
    """
    df = df.copy()

    # Attach ISO-week key to every row
    df["_week_key"] = df["date"].apply(_week_key)

    # Only keep weeks that have at least one score observation
    has_score = (
        df.groupby(["region_id", "_week_key"])["score"]
        .transform(lambda s: s.notna().any())
        .fillna(False)
        .astype(bool)
    )
    df_with_score = df[has_score].copy()

    agg_dict = {
        "prec":       "sum",
        "surf_pre":   "mean",
        "humidity":   "mean",
        "tmp":        "mean",
        "dp_tmp":     "mean",
        "wb_tmp":     "mean",
        "tmp_max":    "max",
        "tmp_min":    "min",
        "tmp_range":  "mean",
        "surf_tmp":   "mean",
        "wind":       "mean",
        "wind_max":   "max",
        "wind_min":   "min",
        "wind_range": "mean",
        "score":      "mean",   # one non-NaN per week → mean == that value
        # Keep the representative date string (last day of the week)
        "date":       "last",
        "day_ordinal": "last",
    }

    weekly = (
        df_with_score.groupby(["region_id", "_week_key"])
        .agg({k: v for k, v in agg_dict.items() if k in df_with_score.columns})
        .reset_index()
    )
    weekly = weekly.rename(columns={"date": "week_end_date", "_week_key": "week_key"})
    weekly = weekly.sort_values(["region_id", "day_ordinal"]).reset_index(drop=True)
    return weekly


# ---------------------------------------------------------------------------
# 5. Weekly Aggregation for Test set (no score column)
# ---------------------------------------------------------------------------
def aggregate_test_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate test set daily rows to ISO-week frequency per region_id.
    No score column → aggregate all 7 days of each week.
    """
    df = df.copy()
    df["_week_key"] = df["date"].apply(_week_key)

    agg_dict = {
        "prec":       "sum",
        "surf_pre":   "mean",
        "humidity":   "mean",
        "tmp":        "mean",
        "dp_tmp":     "mean",
        "wb_tmp":     "mean",
        "tmp_max":    "max",
        "tmp_min":    "min",
        "tmp_range":  "mean",
        "surf_tmp":   "mean",
        "wind":       "mean",
        "wind_max":   "max",
        "wind_min":   "min",
        "wind_range": "mean",
        "date":       "last",
        "day_ordinal": "last",
    }

    weekly = (
        df.groupby(["region_id", "_week_key"])
        .agg({k: v for k, v in agg_dict.items() if k in df.columns})
        .reset_index()
    )
    weekly = weekly.rename(columns={"date": "week_end_date", "_week_key": "week_key"})
    weekly = weekly.sort_values(["region_id", "day_ordinal"]).reset_index(drop=True)
    return weekly


# ---------------------------------------------------------------------------
# 6. Time-Series Feature Engineering  (reusable on train AND test)
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-engineering pipeline applied identically to train and test sets.

    Steps
    -----
    1. Extract calendar features (year, month, week_of_year) from the
       ``week_end_date`` string — no pd.to_datetime required.
    2. Rolling window statistics (4 / 8 / 13-week) grouped by region_id.
    3. Lag features (1-week and 2-week shift) grouped by region_id.

    Parameters
    ----------
    df : weekly-granularity DataFrame produced by align_labels_strategy_a()
         or aggregate_test_weekly().

    Returns
    -------
    df with all new engineered columns appended.
    """
    df = df.copy()
    date_col = "week_end_date"

    # ------------------------------------------------------------------
    # 6.1 Calendar features  (extracted from string – NaT-safe)
    # ------------------------------------------------------------------
    df["year"]        = df[date_col].apply(_date_str_to_year)
    df["month"]       = df[date_col].apply(_date_str_to_month)
    df["week_of_year"] = df[date_col].apply(lambda s: _date_str_to_iso_week(s)[1])

    # ------------------------------------------------------------------
    # 6.2 Rolling statistics (4 / 8 / 13 weeks, grouped by region_id)
    # ------------------------------------------------------------------
    roll_windows   = [4, 8, 13]
    roll_cols_sum  = ["prec"]
    roll_cols_mean = ["tmp", "humidity"]

    for win in roll_windows:
        for col in roll_cols_sum:
            if col in df.columns:
                df[f"{col}_roll_sum_{win}w"] = (
                    df.groupby("region_id")[col]
                    .transform(lambda s, w=win: s.rolling(w, min_periods=1).sum())
                )
        for col in roll_cols_mean:
            if col in df.columns:
                df[f"{col}_roll_mean_{win}w"] = (
                    df.groupby("region_id")[col]
                    .transform(lambda s, w=win: s.rolling(w, min_periods=1).mean())
                )

    # ------------------------------------------------------------------
    # 6.3 Lag features (1-week and 2-week shift, grouped by region_id)
    # ------------------------------------------------------------------
    lag_cols = ["tmp", "humidity", "prec", "wind"]
    for lag in [1, 2]:
        for col in lag_cols:
            if col in df.columns:
                df[f"{col}_lag{lag}w"] = (
                    df.groupby("region_id")[col]
                    .transform(lambda s, l=lag: s.shift(l))
                )
        if "score" in df.columns:
            df[f"score_lag{lag}w"] = (
                df.groupby("region_id")["score"]
                .transform(lambda s, l=lag: s.shift(l))
            )

    return df


# ---------------------------------------------------------------------------
# 7. Export processed DataFrames
# ---------------------------------------------------------------------------
def export_processed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fmt: str = "csv",   # "csv" | "parquet"
) -> None:
    """
    Save finalised train/test DataFrames to data/processed/.
    Does NOT overwrite the raw files in data/.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if fmt == "parquet":
        train_path = os.path.join(PROCESSED_DIR, "train_processed.parquet")
        test_path  = os.path.join(PROCESSED_DIR, "test_processed.parquet")
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path,   index=False)
    else:
        train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
        test_path  = os.path.join(PROCESSED_DIR, "test_processed.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,   index=False)

    print(f"  Exported train → {train_path}")
    print(f"  Exported test  → {test_path}")


# ---------------------------------------------------------------------------
# Main – demonstrate the pipeline end-to-end
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data …")
    train_raw = load_data("train.csv")
    test_raw  = load_data("test.csv")
    print(f"  train shape: {train_raw.shape}  |  test shape: {test_raw.shape}")

    # --- Imputation ---
    print("Imputing meteorological features …")
    train = impute_met_features(train_raw)
    test  = impute_met_features(test_raw)

    # --- Outlier handling ---
    print("Handling outliers (Z-score clipping, threshold=3.5) …")
    train = handle_outliers(train)
    test  = handle_outliers(test)

    # --- Weekly Aggregation ---
    print("\nWeekly aggregation (Strategy A) …")
    train_w = align_labels_strategy_a(train)
    test_w  = aggregate_test_weekly(test)

    # --- Feature engineering ---
    print("Feature engineering …")
    train_w = preprocess_data(train_w)
    test_w  = preprocess_data(test_w)

    print(f"\n  train_w shape : {train_w.shape}")
    print(f"  test_w  shape : {test_w.shape}")

    nan_feat = (
        train_w
        .drop(columns=["score", "score_lag1w", "score_lag2w"], errors="ignore")
        .isna().sum().sum()
    )
    print(f"  NaN in feature cols (excl. lag head): {nan_feat}")

    # --- Export ---
    print("\nExporting processed data …")
    export_processed(train_w, test_w, fmt="csv")

    print("\n✓ Preprocessing pipeline completed successfully.")
