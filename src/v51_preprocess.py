"""
preprocess.py – Full Preprocessing Pipeline (V29 / 45th)
=========================================================
Pipeline steps (daily → weekly → processed CSVs):
  1.  Load train.csv / test.csv.
  2.  Per-region ffill/bfill imputation on MET_COLS.
  3.  Global Z=3.5 outlier clip on MET_COLS.
  4.  Parallelised climatology pre-padding of test set (3 weeks × 21 days,
      joblib threads, all CPU cores).
  5.  Weekly aggregation (sum/mean/max per meteorological column).
  6.  4-week rolling and 1–2 week lag features.
  7.  Drought index: pet, deficit, deficit_roll_cum_4w from raw prec/tmp.
  8.  K-Means climate clustering (n=10) from raw {score_mean, score_zero_prob,
      score_std, tmp_mean, prec_mean}; cluster_id added as column.
  9.  Log1p on prec, prec_week_max, prec_roll_sum_4w (after drought index).
  10. _v29_normalized sentinel column: signals dataset.refine_features to
      skip duplicate drought-index computation and log1p.
  11. Strip test pre-padding; validate 13 weeks per region.
  12. Export train_processed.csv, test_processed.csv, region_stats.csv.

LightGBM is scale-invariant; raw absolute feature values are preserved.
"""

import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_BASE, "data")
_PROC = os.path.join(_DATA, "v51_processed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MET_COLS = [
    "prec", "surf_pre", "humidity",
    "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

ROLL_WINDOWS = [4]
DROUGHT_ROLL_WINDOWS = [4]

DROUGHT_FEAT_COLS = [
    "pet",
    "deficit",
    "deficit_roll_cum_4w",
]

# Columns that are log1p-transformed in V29 preprocess (to prevent double-apply
# by dataset.refine_features when _v29_normalized sentinel is present)
LOG1P_COLS = ["prec", "prec_week_max", "prec_roll_sum_4w"]


_DAYS_PER_TRAIN  = 5480
_KEEP_DAYS_TRAIN = 5474
_WEEKS_PER_TRAIN = 782
_DAYS_PER_TEST   = 91
_WEEKS_PER_TEST  = 13

_MONTH_OFFSET = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_DAYS_IN_MONTH_COMMON = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_PER_SYNTH_YEAR = 366

# ---------------------------------------------------------------------------
# Date helpers (unchanged from V20)
# ---------------------------------------------------------------------------
def _parse_doy(date_str: str) -> int:
    parts = date_str.split("-")
    m = int(parts[1])
    d = int(parts[2])
    return _MONTH_OFFSET[m - 1] + d

def _parse_ordinal(date_str: str) -> int:
    parts = date_str.split("-")
    y   = int(parts[0])
    m   = int(parts[1])
    d   = int(parts[2])
    doy = _MONTH_OFFSET[m - 1] + d
    return (y - 1) * _DAYS_PER_SYNTH_YEAR + (doy - 1)

def _get_days_in_month(region_id, year, month, region_leap_remainder):
    if month == 2:
        if region_id in region_leap_remainder:
            return 29 if (year % 4 == region_leap_remainder[region_id]) else 28
        return 29 if (year % 4 == 1) else 28
    return _DAYS_IN_MONTH_COMMON[month - 1]

def _subtract_days(region_id, date_str, subtract_days, region_leap_remainder):
    parts = date_str.split("-")
    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
    d -= subtract_days
    while d <= 0:
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        d += _get_days_in_month(region_id, y, m, region_leap_remainder)
    return f"{y}-{m:02d}-{d:02d}"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(filename: str) -> pd.DataFrame:
    path = os.path.join(_DATA, filename)
    return pd.read_csv(path)

def impute_met_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MET_COLS:
        if col in df.columns:
            df[col] = df.groupby("region_id")[col].transform(
                lambda s: s.ffill().bfill()
            )
    return df

def handle_outliers(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    df = df.copy()
    # [V51 修正] 只 Clip 溫度特徵，絕對不 Clip 降雨(prec, surf_pre)與風速(wind)
    cols_to_clip = ["tmp", "dp_tmp", "wb_tmp", "tmp_max", "tmp_min", "surf_tmp"]
    
    for col in cols_to_clip:
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 0:
                df[col] = df[col].clip(
                    lower=mu - z_thresh * sigma,
                    upper=mu + z_thresh * sigma,
                )
    return df

# ---------------------------------------------------------------------------
# V29 – Parallelised Climatology Pre-Padding
# ---------------------------------------------------------------------------
def _pad_one_region(region, group, baseline_dict, global_mean_vals, met_cols,
                    pad_days, region_leap_remainder):
    """Pad a single test-set region. Called in parallel."""
    group = group.sort_values("date").copy()
    start_date_str = group["date"].iloc[0]

    pad_dates_str = [
        _subtract_days(region, start_date_str, i, region_leap_remainder)
        for i in range(1, pad_days + 1)
    ][::-1]
    pad_doys = [_parse_doy(d) for d in pad_dates_str]

    pad_df = pd.DataFrame({"region_id": region, "date": pad_dates_str, "doy": pad_doys})

    region_baseline = baseline_dict.get(region, pd.DataFrame(columns=["region_id", "doy"] + met_cols))
    pad_df = pad_df.merge(region_baseline, on=["region_id", "doy"], how="left")
    # Fill missing climatology with global fallback
    for col, gval in global_mean_vals.items():
        if col in pad_df.columns:
            pad_df[col] = pad_df[col].fillna(gval)
    pad_df = pad_df.drop(columns=["doy"])

    return pd.concat([pad_df, group], ignore_index=True)


def apply_climatology_padding(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    pad_weeks: int = 3,
) -> pd.DataFrame:
    pad_days = pad_weeks * 7

    train_doy = train_raw.copy()
    train_doy["doy"] = train_doy["date"].apply(_parse_doy)

    print("   [V29] Detecting per-region leap-year cycles ...")
    region_leap_remainder = {}
    has_feb29 = train_doy["date"].str.endswith("-02-29")
    if has_feb29.any():
        for item in (
            train_doy[has_feb29][["region_id", "date"]]
            .drop_duplicates("region_id")
            .itertuples()
        ):
            y = int(item.date.split("-")[0])
            region_leap_remainder[item.region_id] = y % 4

    baseline = (
        train_doy.groupby(["region_id", "doy"])[MET_COLS].mean().reset_index()
    )
    global_mean_vals = train_raw[MET_COLS].mean().to_dict()

    # Pre-split baseline by region to avoid O(N) filter per worker
    baseline_dict = {rid: grp for rid, grp in baseline.groupby("region_id")}

    groups = list(test_raw.groupby("region_id"))
    n_jobs = min(os.cpu_count() or 8, 32)
    print(f"   [V29] Padding {len(groups)} test regions in parallel (n_jobs={n_jobs}) ...")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_pad_one_region)(
            region, group, baseline_dict, global_mean_vals,
            MET_COLS, pad_days, region_leap_remainder,
        )
        for region, group in groups
    )

    return pd.concat(results, ignore_index=True)

# ---------------------------------------------------------------------------
# Weekly aggregation (unchanged from V20)
# ---------------------------------------------------------------------------
def align_labels_absolute(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy().sort_values(["region_id", "date"], ignore_index=True)
    cumcnt = df.groupby("region_id").cumcount()

    if is_train:
        keep_mask = cumcnt < _KEEP_DAYS_TRAIN
        df     = df[keep_mask].copy()
        cumcnt = cumcnt[keep_mask]

    df["week_idx"] = (cumcnt // 7).values

    named_agg = {}
    if "prec" in df.columns:
        named_agg["prec"]          = pd.NamedAgg(column="prec",   aggfunc="sum")
        named_agg["prec_week_max"] = pd.NamedAgg(column="prec",   aggfunc="max")
    if "surf_pre" in df.columns:
        named_agg["surf_pre"]          = pd.NamedAgg(column="surf_pre", aggfunc="sum")
        named_agg["surf_pre_week_max"] = pd.NamedAgg(column="surf_pre", aggfunc="max")
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

    for _c in ["tmp_max", "tmp_min", "tmp_range", "surf_tmp",
               "wind_max", "wind_min", "wind_range"]:
        if _c in df.columns:
            named_agg[_c] = pd.NamedAgg(column=_c, aggfunc="mean")

    if "score" in df.columns:
        named_agg["score"] = pd.NamedAgg(column="score", aggfunc="max")

    named_agg["week_end_date"] = pd.NamedAgg(column="date", aggfunc="last")

    weekly = (
        df.groupby(["region_id", "week_idx"], sort=False)
        .agg(**named_agg)
        .reset_index()
    )
    weekly.sort_values(["region_id", "week_idx"], inplace=True, ignore_index=True)

    for _std_col in ["tmp_week_std", "humidity_week_std", "wind_week_std"]:
        if _std_col in weekly.columns:
            weekly[_std_col] = weekly[_std_col].fillna(0.0).astype(np.float32)

    doy_arr   = weekly["week_end_date"].map(_parse_doy).astype(np.float32)
    ratio_arr = doy_arr / 365.25
    weekly["week_sin"]     = np.sin(2.0 * np.pi * ratio_arr).astype(np.float32)
    weekly["week_cos"]     = np.cos(2.0 * np.pi * ratio_arr).astype(np.float32)
    weekly["day_ordinal"]  = weekly["week_end_date"].map(_parse_ordinal).astype(np.int64)

    return weekly

def aggregate_test_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return align_labels_absolute(df, is_train=False)

# ---------------------------------------------------------------------------
# Rolling + lag features (unchanged from V20)
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
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
                df[feat] = df.groupby("region_id")[base_col].transform(
                    lambda s: s.rolling(w, min_periods=1).sum()
                )
            else:
                df[feat] = df.groupby("region_id")[base_col].transform(
                    lambda s: s.rolling(w, min_periods=1).mean()
                )

    lag_cols = ["tmp", "humidity", "prec", "wind"]
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in [1, 2]:
            df[f"{col}_lag{lag}w"] = df.groupby("region_id")[col].transform(
                lambda s: s.shift(lag)
            )

    return df

# ---------------------------------------------------------------------------
# Drought Index (also imported by dataset.py – signature unchanged)
# ---------------------------------------------------------------------------
def add_drought_index(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()
    
    # ---------------------------------------------------------------------
    # [原版保留] 基礎蒸發與赤字
    # ---------------------------------------------------------------------
    df["pet"]     = (0.55 * df["tmp"].clip(lower=0.0)).astype(np.float32)
    df["deficit"] = (df["prec"] - df["pet"]).astype(np.float32)

    for w in DROUGHT_ROLL_WINDOWS:
        col = f"deficit_roll_cum_{w}w"
        df[col] = (
            df.groupby("region_id")["deficit"]
            .transform(lambda s: s.rolling(window=w, min_periods=1).sum())
            .astype(np.float32)
        )

    # ---------------------------------------------------------------------
    # [V51 新增] 戰略特徵擴增 (針對 Cluster 1 & 5 的高分段預測)
    # ---------------------------------------------------------------------
    # 1. 非線性乾燥指數 (Aridity Index) - 捕捉高溫低濕的極端蒸發
    df["aridity_index"] = (df["tmp"] / (df["humidity"] + 1.0)).astype(np.float32)
    
    # 2. 熱衝擊指數 (Heat Shock) - 捕捉週內極端高溫與平均溫度的落差
    if "tmp_week_max" in df.columns:
        df["heat_shock"] = (df["tmp_week_max"] - df["tmp"]).astype(np.float32)
        
    # 3. 區域氣候異常度 (Temperature Anomaly)
    # 計算該地區長期的平均與標準差 (利用 transform)
    # 注意：在 Test set 推論時，這會使用 Test set 13 週的區域均值，
    # 雖然不如歷史均值準確，但在沒有 leakage 的情況下是最佳近似。
    region_tmp_mean = df.groupby("region_id")["tmp"].transform("mean")
    region_tmp_std  = df.groupby("region_id")["tmp"].transform("std").replace(0, 1.0)
    df["tmp_anomaly"] = ((df["tmp"] - region_tmp_mean) / region_tmp_std).astype(np.float32)


    # 處理 Test set 前幾週缺失值 (ffill)
    if not is_train:
        # 將 V51 新增的特徵也加入 ffill 清單
        v51_new_cols = ["aridity_index", "heat_shock", "tmp_anomaly"]
        for col in DROUGHT_FEAT_COLS + v51_new_cols:
            if col in df.columns:
                df[col] = df.groupby("region_id")[col].transform(
                    lambda s: s.ffill().fillna(0)
                )
    return df

# ---------------------------------------------------------------------------
# V29 – K-Means Climate Clustering
# ---------------------------------------------------------------------------
def compute_climate_clusters(
    train_w: pd.DataFrame,
    n_clusters: int = 10,
) -> pd.DataFrame:
    """
    Cluster 2248 regions into n_clusters climate ecosystems.
    Uses RAW (pre-normalisation) values so clusters reflect true climate.

    Features: score_mean, score_zero_prob, score_std, tmp_mean, prec_mean (log1p)
    Returns a DataFrame with region_id, cluster_id, and cluster feature values.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    def _zero_prob(x):
        valid = x.dropna()
        return float((valid == 0.0).mean()) if len(valid) > 0 else 0.5

    region_stats = (
        train_w.groupby("region_id")
        .agg(
            score_mean     =("score", lambda x: float(x.dropna().mean()) if x.notna().any() else 0.0),
            score_zero_prob=("score", _zero_prob),
            score_std      =("score", lambda x: float(x.dropna().std()) if x.notna().sum() > 1 else 0.0),
            tmp_mean       =("tmp",   "mean"),
            prec_mean      =("prec",  "mean"),
        )
        .reset_index()
        .fillna(0.0)
    )

    cluster_feats = ["score_mean", "score_zero_prob", "score_std", "tmp_mean", "prec_mean"]
    X = region_stats[cluster_feats].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    region_stats["cluster_id"] = kmeans.fit_predict(X_scaled).astype(np.int8)

    # Print cluster distribution
    cluster_dist = region_stats["cluster_id"].value_counts().sort_index()
    print(f"   [V29] Cluster distribution (10 ecosystems):")
    for cid, cnt in cluster_dist.items():
        print(f"          Cluster {cid:2d}: {cnt:4d} regions")

    return region_stats[["region_id", "cluster_id"] + cluster_feats]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_processed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fmt: str = "csv",
) -> None:
    os.makedirs(_PROC, exist_ok=True)
    train_path = os.path.join(_PROC, "train_processed.csv")
    test_path  = os.path.join(_PROC, "test_processed.csv")
    if fmt == "csv":
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    print(f"  Exported train -> {train_path}  {train_df.shape}")
    print(f"  Exported test  -> {test_path}   {test_df.shape}")

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(_PROC, exist_ok=True)

    print("=" * 75)
    print("Preprocessing Pipeline V29")
    print("  Per-region normalisation | K-Means clustering | Parallel padding")
    print("=" * 75)

    # ---- 1. Load raw data ---------------------------------------------------
    print("\nLoading raw data ...")
    train_raw = load_data("train.csv")
    test_raw  = load_data("test.csv")
    print(f"  train_raw: {train_raw.shape}  |  test_raw: {test_raw.shape}")

    # ---- 2. Impute meteorological features ---------------------------------
    print("\nImputing met features (ffill/bfill per region) ...")
    train_raw = impute_met_features(train_raw)
    test_raw  = impute_met_features(test_raw)

    # ---- 3. Outlier clipping -----------------------------------------------
    print("Outlier clipping (Z=3.5, global) ...")
    train_raw = handle_outliers(train_raw, z_thresh=3.5)
    test_raw  = handle_outliers(test_raw,  z_thresh=3.5)

    # ---- 4. Climatology pre-padding for test set ---------------------------
    print("\n[V29] Climatology pre-padding: test set (21 days, parallelised) ...")
    test_raw = apply_climatology_padding(train_raw, test_raw, pad_weeks=3)

    # ---- 5. Weekly aggregation ---------------------------------------------
    print("\nWeekly aggregation ...")
    train_w = align_labels_absolute(train_raw, is_train=True)
    test_w  = aggregate_test_weekly(test_raw)
    print(f"  train_w: {train_w.shape}  |  test_w: {test_w.shape}")

    # ---- 6. Rolling + lag features -----------------------------------------
    print("\nAdding rolling (4w) & lag features ...")
    train_w = preprocess_data(train_w)
    test_w  = preprocess_data(test_w)

    # ---- 7. Drought Index (uses RAW prec and tmp) ---------------------------
    print("\nComputing drought index (pet, deficit, deficit_roll_cum_4w) ...")
    train_w = add_drought_index(train_w, is_train=True)
    test_w  = add_drought_index(test_w,  is_train=False)

    # ---- 8. K-Means climate clustering (uses RAW values) -------------------
    print("\n[V29] K-Means climate clustering (n=10) from raw climate stats ...")
    cluster_df  = compute_climate_clusters(train_w, n_clusters=10)
    cluster_map = dict(zip(cluster_df["region_id"], cluster_df["cluster_id"]))
    train_w["cluster_id"] = train_w["region_id"].map(cluster_map).astype(np.int8)
    test_w["cluster_id"]  = (
        test_w["region_id"].map(cluster_map).fillna(0).astype(np.int8)
    )
    region_stats_path = os.path.join(_PROC, "region_stats.csv")
    cluster_df.to_csv(region_stats_path, index=False)
    print(f"   Region stats -> {region_stats_path}")

    # ---- 9. Log1p precipitation (moved here from dataset.refine_features) --
    print("\n[V29] Applying log1p to precipitation columns ...")
    for col in LOG1P_COLS:
        for df_ in (train_w, test_w):
            if col in df_.columns:
                df_[col] = np.log1p(df_[col].clip(lower=0.0)).astype(np.float32)
    print(f"   log1p applied to: {[c for c in LOG1P_COLS]}")

    # ---- 10. Add V29 sentinel column ----------------------------------------
    train_w["_v29_normalized"] = np.int8(1)
    test_w["_v29_normalized"]  = np.int8(1)

    # ---- 11. Strip test pre-padding -----------------------------------------
    print("\n[V29] Stripping test pre-padding (restoring to strict 13 weeks) ...")
    test_w = test_w[test_w["week_idx"] >= 3].copy()
    test_w["week_idx"] = test_w["week_idx"] - 3
    test_w.reset_index(drop=True, inplace=True)

    # ---- 12. Validation -----------------------------------------------------
    n_train_rgn = train_w["region_id"].nunique()
    n_test_rgn  = test_w["region_id"].nunique()
    test_wk_counts = test_w.groupby("region_id").size()

    assert n_train_rgn == 2248, f"Expected 2248 train regions, got {n_train_rgn}"
    assert n_test_rgn  == 2248, f"Expected 2248 test regions, got {n_test_rgn}"
    assert test_wk_counts.min() == _WEEKS_PER_TEST, (
        f"Test week count mismatch: min={test_wk_counts.min()}"
    )
    print(f"\n  REGION COUNT PASSED: {n_train_rgn} train, {n_test_rgn} test")
    print(f"  Test weeks per region: min={test_wk_counts.min()}  "
          f"max={test_wk_counts.max()}")

    # Sentinel present
    assert "_v29_normalized" in train_w.columns
    assert "_v29_normalized" in test_w.columns
    print("  _v29_normalized sentinel: present in both CSVs")

    # cluster_id present
    assert "cluster_id" in train_w.columns
    assert "cluster_id" in test_w.columns
    n_clusters_found = train_w["cluster_id"].nunique()
    print(f"  cluster_id: {n_clusters_found} distinct clusters in train")

    # ---- 13. Export ---------------------------------------------------------
    print("\nExporting processed data ...")
    export_processed(train_w, test_w, fmt="csv")

    elapsed = time.time() - t0
    print(f"\nTotal preprocessing time: {elapsed:.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
