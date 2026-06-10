"""
v54_dataset.py - Tabular dataset construction (v54 - Time CV & Data-Driven Features).

Flat feature matrix layout per sample:
  Effective: X features × 13 weeks = week-major columns
  + X explicit trend deltas (w13 - w1) = total dimensions.

CV Strategy (v54):
  Cross-Seasonal 4-Fold Time-based CV.
  Mimics exactly the Kaggle public/private LB evaluation.
"""

import numpy as np
import pandas as pd
from src.v54_preprocess import add_drought_index # 確保引用 v54 的 preprocess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 13
HORIZON     = 5

# Removed highly collinear or leaked seasonal proxies
DROP_COLS = ["wb_tmp", "dp_tmp", "surf_tmp", "wind_max", "dow_sin"]

# Log1p targets
PREC_COLS = ["prec", "prec_week_max", "prec_roll_sum_4w"]

# [V54 修改] 嚴格對齊 V54 的新特徵，並徹底移除 TE 特徵 (region_mean_score, region_zero_prob)
FEATURE_COLS = [
    # --- 基礎氣象 ---
    "prec", "surf_pre", "humidity",
    "tmp", "tmp_max", "tmp_min", "tmp_range",
    "wind", "wind_min", "wind_range",
    
    # --- 週級統計 ---
    "tmp_week_max", "tmp_week_min", "tmp_week_std",
    "humidity_week_max", "humidity_week_min", "humidity_week_std",
    "wind_week_max", "wind_week_min", "wind_week_std",
    "prec_week_max", "surf_pre_week_max", "tmp_roll_mean_4w",
    "humidity_roll_mean_4w",
    
    # --- 時間特徵 ---
    "week_sin", "week_cos",
    
    # --- 乾旱指標 (V52/V53 延續) ---
    "pet", 
    "prec_roll_sum_4w",
    "water_shortage", 
    "water_shortage_roll_cum_4w",
    "aridity_index", "heat_shock", "tmp_anomaly",
    
    # --- V54 核彈特徵 ---
    "water_shortage_roll_cum_8w", 
    "cross_shortage_x_anomaly",
    "is_dry_spell",
    "shortage_momentum",
    
    # --- 氣候分群 (有益特徵) ---
    "cluster_id" 
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_flat_col_names(feat_cols: list, window: int = WINDOW_SIZE) -> list:
    names = []
    for w in range(1, window + 1):
        for feat in feat_cols:
            names.append(f"{feat}_w{w}")
    for feat in feat_cols:
        names.append(f"{feat}_delta")
    return names

def refine_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()
    # [V54 修改] 更新哨兵欄位名稱
    already_preprocessed = "_v54_processed" in df.columns

    if not already_preprocessed:
        df = add_drought_index(df, is_train=is_train)

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    if not already_preprocessed:
        for col in PREC_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

    # [修正] 確保 cluster_id 即使包含 "id" 字眼也不會被濾掉
    present_feats = [c for c in FEATURE_COLS if c in df.columns]

    if is_train:
        df = df.dropna(subset=present_feats).reset_index(drop=True)
    else:
        for col in present_feats:
            df[col] = df.groupby("region_id")[col].transform(lambda s: s.ffill().fillna(0))

    return df

# ---------------------------------------------------------------------------
# Tabular dataset builders
# ---------------------------------------------------------------------------
def build_tabular_dataset(region_groups: list, feat_cols: list, window: int = WINDOW_SIZE, horizon: int = HORIZON):
    X_rows, y_rows, region_id_lst = [], [], []

    for group, i_min, i_max in region_groups:
        group = group.reset_index(drop=True)
        n = len(group)
        cols_present = [c for c in feat_cols if c in group.columns]
        X_mat = group[cols_present].values.astype(np.float32)
        y_arr = group["score"].values.astype(np.float32) if "score" in group.columns else None
        rid = group["region_id"].iloc[0] if "region_id" in group.columns else 0

        for i in range(i_min, i_max + 1):
            end_feat = i + window
            tgt_start, tgt_end = end_feat, end_feat + horizon
            if tgt_end > n: break

            window_block = X_mat[i:end_feat]
            flat_row = window_block.reshape(-1)
            delta_row = window_block[-1] - window_block[0]
            X_rows.append(np.concatenate([flat_row, delta_row]))

            if y_arr is not None:
                y_rows.append(y_arr[tgt_start:tgt_end])
            region_id_lst.append(rid)

    n_out = window * len(feat_cols) + len(feat_cols)
    if not X_rows: return np.empty((0, n_out), dtype=np.float32), None, np.empty(0)

    return np.array(X_rows, dtype=np.float32), (np.array(y_rows, dtype=np.float32) if y_rows else None), np.array(region_id_lst)

def build_tabular_test(test_df: pd.DataFrame, feat_cols: list, window: int = WINDOW_SIZE):
    X_rows, region_ids = [], []
    for region_id, group in test_df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < window:
            pad_df = pd.concat([group.iloc[[0]]] * (window - n) + [group], ignore_index=True)
            group = pad_df

        win_df = group.iloc[-window:]
        cols_present = [c for c in feat_cols if c in win_df.columns]
        window_block = win_df[cols_present].values.astype(np.float32)
        
        flat_row = window_block.reshape(-1)
        delta_row = window_block[-1] - window_block[0]
        X_rows.append(np.concatenate([flat_row, delta_row]))
        region_ids.append(region_id)

    return np.array(X_rows, dtype=np.float32), np.array(region_ids)

# ---------------------------------------------------------------------------
# CV Fold Builder (v54: Cross-Seasonal Time CV)
# ---------------------------------------------------------------------------
def build_time_seasonal_cv_folds(df: pd.DataFrame, n_splits: int = 4, season_step: int = 13):
    """
    Time-based CV with Cross-Seasonal spacing.
    Ensures no data leakage by strictly separating train and validation sets by time.
    """
    folds = []
    max_week = int(df["week_idx"].max())
    
    for f in range(n_splits):
        val_end = max_week - f * season_step
        val_start = val_end - HORIZON + 1
        train_end = val_start - 1
        
        print(f"  [CV Fold {f}] Train ends week: {train_end} | Val predicts weeks: {val_start} to {val_end}")
        
        train_groups = []
        val_groups = []
        
        for _, group in df.groupby("region_id"):
            group = group.reset_index(drop=True)
            n = len(group)
            train_i_max, val_i = -1, -1
            
            for i in range(n - WINDOW_SIZE - HORIZON + 1):
                tgt_start_idx = i + WINDOW_SIZE
                tgt_end_idx = i + WINDOW_SIZE + HORIZON - 1
                
                w_start = group["week_idx"].iloc[tgt_start_idx]
                w_end = group["week_idx"].iloc[tgt_end_idx]
                
                if w_end <= train_end:
                    train_i_max = i
                elif w_start == val_start and w_end == val_end:
                    val_i = i 
                    
            if train_i_max >= 0:
                train_groups.append((group, 0, train_i_max))
            if val_i >= 0:
                val_groups.append((group, val_i, val_i))
                
        folds.append((train_groups, val_groups))
        
    return folds

# ---------------------------------------------------------------------------
# General Helpers
# ---------------------------------------------------------------------------
def build_full_train_groups(df: pd.DataFrame):
    train_groups = []
    for _, group in df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n >= WINDOW_SIZE + HORIZON:
            train_groups.append((group, 0, n - WINDOW_SIZE - HORIZON))
    return train_groups