"""
v54_preprocess.py – Full Preprocessing Pipeline (V54)
=========================================================
[FIXED] Removed Target Leakage in K-Means (Now purely physical climate clustering).
[ADDED] Std in K-Means for better ecosystem separation.
[FIXED] Retained valid rolling features (tmp/humidity) and removed unused lag features.
"""

import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_BASE, "data")
_PROC = os.path.join(_DATA, "v54_processed") 

MET_COLS = [
    "prec", "surf_pre", "humidity",
    "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]

ROLL_WINDOWS = [4]
DROUGHT_ROLL_WINDOWS = [4, 8] 

DROUGHT_FEAT_COLS = [
    "pet",
    "water_shortage",
    "water_shortage_roll_cum_4w",
    "water_shortage_roll_cum_8w", 
]

LOG1P_COLS = ["prec", "prec_week_max", "prec_roll_sum_4w"]

_DAYS_PER_TRAIN  = 5480
_KEEP_DAYS_TRAIN = 5474
_WEEKS_PER_TRAIN = 782
_DAYS_PER_TEST   = 91
_WEEKS_PER_TEST  = 13

_MONTH_OFFSET = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_DAYS_IN_MONTH_COMMON = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_PER_SYNTH_YEAR = 366

GLOBAL_TRAIN_TMP_STATS = None

# ---------------------------------------------------------------------------
# Date helpers & Pre-padding
# ---------------------------------------------------------------------------
def _parse_doy(date_str: str) -> int:
    parts = date_str.split("-")
    return _MONTH_OFFSET[int(parts[1]) - 1] + int(parts[2])

def _parse_ordinal(date_str: str) -> int:
    parts = date_str.split("-")
    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
    return (y - 1) * _DAYS_PER_SYNTH_YEAR + (_MONTH_OFFSET[m - 1] + d - 1)

def _get_days_in_month(region_id, year, month, region_leap_remainder):
    if month == 2:
        if region_id in region_leap_remainder: return 29 if (year % 4 == region_leap_remainder[region_id]) else 28
        return 29 if (year % 4 == 1) else 28
    return _DAYS_IN_MONTH_COMMON[month - 1]

def _subtract_days(region_id, date_str, subtract_days, region_leap_remainder):
    y, m, d = map(int, date_str.split("-"))
    d -= subtract_days
    while d <= 0:
        m -= 1
        if m == 0: m, y = 12, y - 1
        d += _get_days_in_month(region_id, y, m, region_leap_remainder)
    return f"{y}-{m:02d}-{d:02d}"

def load_data(filename: str) -> pd.DataFrame: 
    return pd.read_csv(os.path.join(_DATA, filename))

def impute_met_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    prec_cols = ["prec", "surf_pre"]
    for col in prec_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
            
    continuous_cols = [c for c in MET_COLS if c not in prec_cols]
    for col in continuous_cols:
        if col in df.columns:
            df[col] = df.groupby("region_id")[col].transform(lambda s: s.ffill(limit=3))
            df[col] = df.groupby("region_id")[col].transform(lambda s: s.fillna(s.mean()))
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
                
    return df

def handle_outliers(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    df = df.copy()
    for col in ["tmp", "dp_tmp", "wb_tmp", "tmp_max", "tmp_min", "surf_tmp"]:
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 0: 
                df[col] = df[col].clip(lower=mu - z_thresh * sigma, upper=mu + z_thresh * sigma)
    return df

def _pad_one_region(region, group, baseline_dict, global_mean_vals, met_cols, pad_days, region_leap_remainder):
    group = group.sort_values("date").copy()
    pad_dates_str = [_subtract_days(region, group["date"].iloc[0], i, region_leap_remainder) for i in range(1, pad_days + 1)][::-1]
    pad_df = pd.DataFrame({"region_id": region, "date": pad_dates_str, "doy": [_parse_doy(d) for d in pad_dates_str]})
    pad_df = pad_df.merge(baseline_dict.get(region, pd.DataFrame(columns=["region_id", "doy"] + met_cols)), on=["region_id", "doy"], how="left")
    for col, gval in global_mean_vals.items():
        if col in pad_df.columns: 
            pad_df[col] = pad_df[col].fillna(gval)
    return pd.concat([pad_df.drop(columns=["doy"]), group], ignore_index=True)

def apply_climatology_padding(train_raw, test_raw, pad_weeks=7):
    train_doy = train_raw.copy()
    train_doy["doy"] = train_doy["date"].apply(_parse_doy)
    
    region_leap_remainder = {}
    has_feb29 = train_doy["date"].str.endswith("-02-29")
    if has_feb29.any():
        for item in train_doy[has_feb29][["region_id", "date"]].drop_duplicates("region_id").itertuples():
            region_leap_remainder[item.region_id] = int(item.date.split("-")[0]) % 4
            
    baseline = train_doy.groupby(["region_id", "doy"])[MET_COLS].mean().reset_index()
    baseline_dict = {rid: grp for rid, grp in baseline.groupby("region_id")}
    
    results = Parallel(n_jobs=min(os.cpu_count() or 8, 32), prefer="threads")(
        delayed(_pad_one_region)(region, group, baseline_dict, train_raw[MET_COLS].mean().to_dict(), MET_COLS, pad_weeks * 7, region_leap_remainder) 
        for region, group in test_raw.groupby("region_id")
    )
    return pd.concat(results, ignore_index=True)

def align_labels_absolute(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy().sort_values(["region_id", "date"], ignore_index=True)
    cumcnt = df.groupby("region_id").cumcount()
    if is_train:
        keep_mask = cumcnt < _KEEP_DAYS_TRAIN
        df, cumcnt = df[keep_mask].copy(), cumcnt[keep_mask]
    df["week_idx"] = (cumcnt // 7).values
    
    aggs = {"week_end_date": pd.NamedAgg(column="date", aggfunc="last")}
    for c in ["prec", "surf_pre"]:
        if c in df.columns: 
            aggs[c] = pd.NamedAgg(column=c, aggfunc="sum")
            aggs[f"{c}_week_max"] = pd.NamedAgg(column=c, aggfunc="max")
    for c in ["humidity", "tmp", "wind"]:
        if c in df.columns: 
            aggs[c] = pd.NamedAgg(column=c, aggfunc="mean")
            aggs[f"{c}_week_max"] = pd.NamedAgg(column=c, aggfunc="max")
            aggs[f"{c}_week_min"] = pd.NamedAgg(column=c, aggfunc="min")
            aggs[f"{c}_week_std"] = pd.NamedAgg(column=c, aggfunc="std")
    for c in ["tmp_max", "tmp_min", "tmp_range", "surf_tmp", "wind_max", "wind_min", "wind_range"]:
        if c in df.columns: 
            aggs[c] = pd.NamedAgg(column=c, aggfunc="mean")
    if "score" in df.columns: 
        aggs["score"] = pd.NamedAgg(column="score", aggfunc="max")

    weekly = df.groupby(["region_id", "week_idx"], sort=False).agg(**aggs).reset_index().sort_values(["region_id", "week_idx"], ignore_index=True)
    
    for c in ["tmp_week_std", "humidity_week_std", "wind_week_std"]:
        if c in weekly.columns: 
            weekly[c] = weekly[c].fillna(0.0).astype(np.float32)
            
    ratio_arr = weekly["week_end_date"].map(_parse_doy).astype(np.float32) / 365.25
    weekly["week_sin"] = np.sin(2.0 * np.pi * ratio_arr).astype(np.float32)
    weekly["week_cos"] = np.cos(2.0 * np.pi * ratio_arr).astype(np.float32)
    weekly["day_ordinal"] = weekly["week_end_date"].map(_parse_ordinal).astype(np.int64)
    
    return weekly

def aggregate_test_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return align_labels_absolute(df, is_train=False)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["region_id", "week_idx"], ignore_index=True)
    
    # [FIXED] Retained valid rolling features and removed unused lag features.
    for base_col, func, tmpl in [("prec", "sum", "prec_roll_sum_{w}w"), ("tmp", "mean", "tmp_roll_mean_{w}w"), ("humidity", "mean", "humidity_roll_mean_{w}w")]:
        if base_col in df.columns:
            for w in ROLL_WINDOWS: 
                if func == "sum":
                    df[tmpl.format(w=w)] = df.groupby("region_id")[base_col].transform(lambda s: s.rolling(w, min_periods=1).sum())
                else:
                    df[tmpl.format(w=w)] = df.groupby("region_id")[base_col].transform(lambda s: s.rolling(w, min_periods=1).mean())
                    
    return df

# ---------------------------------------------------------------------------
# V54 Strategic Features (Data-Driven Interventions)
# ---------------------------------------------------------------------------
def add_drought_index(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    global GLOBAL_TRAIN_TMP_STATS
    df = df.copy()
    
    df["pet"] = (0.55 * df["tmp"].clip(lower=0.0)).astype(np.float32)
    df["water_shortage"] = (df["pet"] - df["prec"]).clip(lower=0.0).astype(np.float32)

    for w in DROUGHT_ROLL_WINDOWS:
        col = f"water_shortage_roll_cum_{w}w"
        df[col] = (
            df.groupby("region_id")["water_shortage"]
            .transform(lambda s: s.rolling(window=w, min_periods=1).sum())
            .astype(np.float32)
        )

    df["shortage_lag1w"] = df.groupby("region_id")["water_shortage"].shift(1)
    df["shortage_lag2w"] = df.groupby("region_id")["water_shortage"].shift(2)

    if is_train:
        stats = df.groupby("region_id")["tmp"].agg(["mean", "std"]).reset_index()
        stats["std"] = stats["std"].replace(0, 1.0)
        GLOBAL_TRAIN_TMP_STATS = stats.set_index("region_id").to_dict("index")
        region_tmp_mean = df["region_id"].map(lambda rid: GLOBAL_TRAIN_TMP_STATS[rid]["mean"])
        region_tmp_std  = df["region_id"].map(lambda rid: GLOBAL_TRAIN_TMP_STATS[rid]["std"])
    else:
        if GLOBAL_TRAIN_TMP_STATS is None:
             raise ValueError("GLOBAL_TRAIN_TMP_STATS is not initialized. Process train data first.")
        region_tmp_mean = df["region_id"].map(lambda rid: GLOBAL_TRAIN_TMP_STATS.get(rid, {"mean": df[df["region_id"]==rid]["tmp"].mean()})["mean"])
        region_tmp_std = df["region_id"].map(lambda rid: GLOBAL_TRAIN_TMP_STATS.get(rid, {"std": 1.0})["std"])

    df["tmp_anomaly"] = ((df["tmp"] - region_tmp_mean) / region_tmp_std).astype(np.float32)

    df["cross_shortage_x_anomaly"] = (df["water_shortage_roll_cum_4w"] * np.clip(df["tmp_anomaly"], 0.0, None)).astype(np.float32)

    prec_roll_max_2w = df.groupby("region_id")["prec"].transform(lambda s: s.rolling(2, min_periods=1).max())
    df["is_dry_spell"] = (prec_roll_max_2w < 0.5).astype(np.int8)

    momentum = (df["water_shortage"] * 4.0 + 
                df["shortage_lag1w"].fillna(0) * 2.0 + 
                df["shortage_lag2w"].fillna(0) * 1.0)
    df["shortage_momentum"] = (momentum ** 2).astype(np.float32)
    
    df["aridity_index"] = (df["tmp"] / (df["humidity"] + 1.0)).astype(np.float32)
    if "tmp_week_max" in df.columns:
        df["heat_shock"] = (df["tmp_week_max"] - df["tmp"]).astype(np.float32)

    if not is_train:
        v54_new_cols = ["cross_shortage_x_anomaly", "is_dry_spell", "shortage_momentum", "aridity_index", "heat_shock", "tmp_anomaly"]
        for col in DROUGHT_FEAT_COLS + v54_new_cols:
            if col in df.columns:
                df[col] = df.groupby("region_id")[col].transform(lambda s: s.ffill().fillna(0))
                
    df.drop(columns=["shortage_lag1w", "shortage_lag2w"], inplace=True)
    return df

# ---------------------------------------------------------------------------
# Clustering & Export 
# ---------------------------------------------------------------------------
def compute_climate_clusters(train_w: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # [FIXED 1] Removed Target Leakage. Uses physical features with mean and std.
    region_stats = (
        train_w.groupby("region_id")
        .agg(
            tmp_mean       =("tmp", "mean"),
            tmp_std        =("tmp", "std"),
            prec_mean      =("prec", "mean"),
            prec_std       =("prec", "std"),
            humidity_mean  =("humidity", "mean"),
            humidity_std   =("humidity", "std"),
            wind_mean      =("wind", "mean"),
            wind_std       =("wind", "std"),
        )
        .reset_index()
        .fillna(0.0)
    )

    cluster_feats = [
        "tmp_mean", "tmp_std", "prec_mean", "prec_std", 
        "humidity_mean", "humidity_std", "wind_mean", "wind_std"
    ]
    X = region_stats[cluster_feats].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    region_stats["cluster_id"] = kmeans.fit_predict(X_scaled).astype(np.int8)

    cluster_dist = region_stats["cluster_id"].value_counts().sort_index()
    print(f"   [V54] Physical Climate Cluster distribution (10 ecosystems):")
    for cid, cnt in cluster_dist.items():
        print(f"          Cluster {cid:2d}: {cnt:4d} regions")

    return region_stats[["region_id", "cluster_id"] + cluster_feats]

def export_processed(train_df: pd.DataFrame, test_df: pd.DataFrame, fmt: str = "csv") -> None:
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
    print("Preprocessing Pipeline V54 (Leakage Free & Lean)")
    print("  Physical Climate Clustering | Precise Imputation | 7-Week Padding")
    print("=" * 75)

    print("\nLoading raw data ...")
    train_raw = load_data("train.csv")
    test_raw  = load_data("test.csv")

    print("\nImputing met features (No phantom rain, safe temporal ffill) ...")
    train_raw = impute_met_features(train_raw)
    test_raw  = impute_met_features(test_raw)

    print("Outlier clipping (Z=3.5, global, tmp only) ...")
    train_raw = handle_outliers(train_raw, z_thresh=3.5)
    test_raw  = handle_outliers(test_raw,  z_thresh=3.5)

    print("\n[V54] Climatology pre-padding: test set (49 days, parallelised) ...")
    test_raw = apply_climatology_padding(train_raw, test_raw, pad_weeks=7)

    print("\nWeekly aggregation ...")
    train_w = align_labels_absolute(train_raw, is_train=True)
    test_w  = aggregate_test_weekly(test_raw)

    print("\nAdding rolling (4w) features ...")
    train_w = preprocess_data(train_w)
    test_w  = preprocess_data(test_w)

    print("\nComputing drought index & V54 features ...")
    train_w = add_drought_index(train_w, is_train=True)
    test_w  = add_drought_index(test_w,  is_train=False)

    print("\n[V54] Physical K-Means climate clustering (n=10) ...")
    cluster_df  = compute_climate_clusters(train_w, n_clusters=10)
    cluster_map = dict(zip(cluster_df["region_id"], cluster_df["cluster_id"]))
    train_w["cluster_id"] = train_w["region_id"].map(cluster_map).astype(np.int8)
    test_w["cluster_id"]  = test_w["region_id"].map(cluster_map).fillna(0).astype(np.int8)
    
    region_stats_path = os.path.join(_PROC, "region_stats.csv")
    cluster_df.to_csv(region_stats_path, index=False)
    print(f"   Region stats -> {region_stats_path}")

    print("\n[V54] Applying log1p to precipitation columns ...")
    for col in LOG1P_COLS:
        for df_ in (train_w, test_w):
            if col in df_.columns:
                df_[col] = np.log1p(df_[col].clip(lower=0.0)).astype(np.float32)

    train_w["_v54_processed"] = np.int8(1)
    test_w["_v54_processed"]  = np.int8(1)

    print("\n[V54] Stripping test pre-padding (restoring to strict 13 weeks) ...")
    test_w = test_w[test_w["week_idx"] >= 7].copy()
    test_w["week_idx"] = test_w["week_idx"] - 7
    test_w.reset_index(drop=True, inplace=True)

    n_train_rgn = train_w["region_id"].nunique()
    n_test_rgn  = test_w["region_id"].nunique()
    test_wk_counts = test_w.groupby("region_id").size()

    assert n_train_rgn == 2248, f"Expected 2248 train regions, got {n_train_rgn}"
    assert n_test_rgn  == 2248, f"Expected 2248 test regions, got {n_test_rgn}"
    assert test_wk_counts.min() == _WEEKS_PER_TEST, f"Test week count mismatch: min={test_wk_counts.min()}"
    
    print("\nExporting processed data ...")
    export_processed(train_w, test_w, fmt="csv")

    elapsed = time.time() - t0
    print(f"\nTotal preprocessing time: {elapsed:.1f}s")
    print("=" * 75)

if __name__ == "__main__":
    main()