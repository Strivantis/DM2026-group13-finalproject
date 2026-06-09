"""
dataset.py - Tabular dataset construction (v52 - Cross-Seasonal Time CV).

Flat feature matrix layout per sample:
  Effective: 23 features × 13 weeks = 299 week-major columns
  + 23 explicit trend deltas (w13 - w1) = 322 total dimensions.

CV Strategy (v52):
  Cross-Seasonal 4-Fold Time-based CV.
  Mimics exactly the Kaggle public/private LB evaluation by predicting
  exactly one 5-week horizon per region per fold, spaced 13 weeks apart.
"""

import numpy as np
import pandas as pd
from src.preprocess import add_drought_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 13
HORIZON     = 5

# Removed highly collinear or leaked seasonal proxies
DROP_COLS = ["wb_tmp", "dp_tmp", "surf_tmp", "wind_max", "dow_sin"]

# Log1p targets
PREC_COLS = ["prec", "prec_week_max", "prec_roll_sum_4w"]

# 32 declared features (incl. V51 additions) -> 27 effective after DROP_COLS
FEATURE_COLS = [
    "prec", "surf_pre", "humidity",
    "tmp", "tmp_max", "tmp_min", "tmp_range",
    "wind", "wind_min", "wind_range",
    
    "tmp_week_max", "tmp_week_min", "tmp_week_std",
    "humidity_week_max", "humidity_week_min", "humidity_week_std",
    "wind_week_max", "wind_week_min", "wind_week_std",
    "prec_week_max", "surf_pre_week_max",
    
    "week_sin", "week_cos",
    
    "pet", "deficit",
    "prec_roll_sum_4w", "deficit_roll_cum_4w",
    
    "aridity_index", "heat_shock", "tmp_anomaly",  # V51 features
    
    "region_mean_score", "region_zero_prob",       # TE features
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
    already_preprocessed = "_v29_normalized" in df.columns

    if not already_preprocessed:
        df = add_drought_index(df, is_train=is_train)

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    if not already_preprocessed:
        for col in PREC_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

    present_feats = [c for c in FEATURE_COLS if c in df.columns and "region" not in c]

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
# CV Fold Builder (v52: Cross-Seasonal Time CV)
# ---------------------------------------------------------------------------
def build_time_seasonal_cv_folds(df: pd.DataFrame, n_splits: int = 4, season_step: int = 13):
    """
    Time-based CV with Cross-Seasonal spacing.
    Ensures no data leakage by strictly separating train and validation sets by time.
    Validation evaluates EXACTLY one 5-week horizon per region (mimicking LB).
    """
    folds = []
    max_week = int(df["week_idx"].max())
    
    for f in range(n_splits):
        # Val window targets exactly HORIZON (5) weeks.
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
                    val_i = i # Exact match for Kaggle horizon simulation
                    
            if train_i_max >= 0:
                train_groups.append((group, 0, train_i_max))
            if val_i >= 0:
                val_groups.append((group, val_i, val_i)) # Only 1 validation sample per region!
                
        folds.append((train_groups, val_groups))
        
    return folds

# ---------------------------------------------------------------------------
# TE Leakage Prevention Helper
# ---------------------------------------------------------------------------
def extract_training_targets_for_te(train_groups):
    """
    Extracts scores ONLY from the training timeframe to prevent future leakage
    when calculating region_mean_score in train.py.
    """
    scores, rids = [], []
    for group, _, i_max in train_groups:
        max_idx = i_max + WINDOW_SIZE + HORIZON
        rid = group["region_id"].iloc[0]
        fold_scores = group["score"].iloc[:max_idx].dropna().values
        scores.extend(fold_scores)
        rids.extend([rid] * len(fold_scores))
    
    return pd.DataFrame({"region_id": rids, "score": scores})

def build_full_train_groups(df: pd.DataFrame):
    train_groups = []
    for _, group in df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n >= WINDOW_SIZE + HORIZON:
            train_groups.append((group, 0, n - WINDOW_SIZE - HORIZON))
    return train_groups