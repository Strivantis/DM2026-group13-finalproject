"""
dataset.py – Tabular dataset construction for drought score forecasting (v47).

Flat feature matrix layout per sample:
  FEATURE_COLS declares 29 features; DROP_COLS prunes 6 at runtime.
  Effective: 23 features × 13 weeks = 299 week-major columns (feat_w1 … feat_w13)
  + 23 explicit trend deltas (feat_w13 – feat_w1)
  = 322 total dimensions

CV: 5-Fold StratifiedGroupKFold.
  Strata: K-Means cluster_id (10 climate ecosystems) when present in df,
          otherwise 10-quantile bins of per-region mean score.
  Group:  region_id (never split across folds).
"""

import numpy as np
import pandas as pd

from src.preprocess import add_drought_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 13   # look-back weeks (matches test-set horizon)
HORIZON     = 5    # forecast horizon (weeks)

# Adversarial / collinear columns pruned before model input.
# week_sin/cos: seasonal proxy leaks target (train/test month distributions differ).
# Extreme-variability stats: near-perfect adversarial AUC against score label.
# DROP_COLS = [
#     "wb_tmp", "dp_tmp", "surf_tmp", "wind_max", "dow_sin",
#     "prec_week_max", "surf_pre_week_max",
#     "tmp_week_std", "humidity_week_std",
#     "week_cos", "week_sin"
# ]

# # Precipitation columns that receive log1p transform (only in non-V29 path)
# PREC_COLS = ["prec", "prec_roll_sum_4w"]

# [v45_30k] 拿掉昨天加的那些內鬼，只保留最原本的 5 個物理衝突特徵
DROP_COLS = ["wb_tmp", "dp_tmp", "surf_tmp", "wind_max", "dow_sin"]

# [v45_30k] 把 prec_week_max 加回 log1p 的轉換名單中
PREC_COLS = ["prec", "prec_week_max", "prec_roll_sum_4w"]

# 29-feature input set (order preserved for flat-matrix alignment)
FEATURE_COLS = [
    # base weather (10)
    "prec", "surf_pre", "humidity",
    "tmp", "tmp_max", "tmp_min", "tmp_range",
    "wind", "wind_min", "wind_range",
    # intra-week statistics (11)
    "tmp_week_max", "tmp_week_min", "tmp_week_std",
    "humidity_week_max", "humidity_week_min", "humidity_week_std",
    "wind_week_max", "wind_week_min", "wind_week_std",
    "prec_week_max", "surf_pre_week_max",
    # cyclical calendar (2)
    "week_sin", "week_cos",
    # drought proxy (4)
    "pet", "deficit",
    "prec_roll_sum_4w", "deficit_roll_cum_4w",
    # target encoding (2) – injected by train.py per fold, not in CSVs
    "region_mean_score", "region_zero_prob",
]   # 29 declared; 6 overlap DROP_COLS → 23 effective | flat dim = 23 × 13 + 23 = 322


# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------
def make_flat_col_names(feat_cols: list, window: int = WINDOW_SIZE) -> list:
    """
    Return the 406 column names for the flat feature matrix.

    Layout: [f0_w1, f0_w2, …, f0_w13, f1_w1, …, f28_w13,
             f0_delta, f1_delta, …, f28_delta]

    Parameters
    ----------
    feat_cols : list[str]   feature column names (len = F)
    window    : int         context window size (default WINDOW_SIZE)

    Returns
    -------
    list[str]  length = window * len(feat_cols) + len(feat_cols)
    """
    names = []
    for w in range(1, window + 1):
        for feat in feat_cols:
            names.append(f"{feat}_w{w}")
    for feat in feat_cols:
        names.append(f"{feat}_delta")
    return names


# ---------------------------------------------------------------------------
# Feature refinement
# ---------------------------------------------------------------------------
def refine_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Prepare a processed DataFrame for model consumption.

    When `_v29_normalized` sentinel column is present (V29+ preprocessed CSV):
      - drought index (pet, deficit, deficit_roll_cum_4w) is already computed
        from raw values in preprocess.py → skip recomputation.
      - log1p was already applied to prec columns → skip.
    When sentinel is absent (legacy CSV):
      - compute drought index here, then apply log1p.

    In both paths:
      - drop adversarial / collinear columns (DROP_COLS).
      - train: drop rows where any non-TE FEATURE_COL is NaN.
      - test:  ffill then zero-fill per region.

    region_mean_score / region_zero_prob are NOT added here; injected by
    train.py per fold for leakage-free target encoding.

    Parameters
    ----------
    df       : pd.DataFrame  weekly preprocessed data
    is_train : bool          True → drop NaN rows; False → ffill/fill

    Returns
    -------
    pd.DataFrame  clean copy
    """
    df = df.copy()

    already_preprocessed = "_v29_normalized" in df.columns

    if not already_preprocessed:
        df = add_drought_index(df, is_train=is_train)

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    if not already_preprocessed:
        for col in PREC_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

    present_feats = [
        c for c in FEATURE_COLS
        if c in df.columns and c not in ("region_mean_score", "region_zero_prob")
    ]

    if is_train:
        df = df.dropna(subset=present_feats).reset_index(drop=True)
    else:
        for col in present_feats:
            df[col] = (
                df.groupby("region_id")[col]
                .transform(lambda s: s.ffill().fillna(0))
            )

    return df


# ---------------------------------------------------------------------------
# Tabular dataset builders
# ---------------------------------------------------------------------------
def build_tabular_dataset(
    region_groups: list,
    feat_cols: list,
    window: int = WINDOW_SIZE,
    horizon: int = HORIZON,
):
    """
    Build flat training matrices from a list of region groups.

    For each valid window [i, i+window) → flatten all window rows into a
    (window × F,) vector, then append F delta values (w13 – w1).

    Parameters
    ----------
    region_groups : list of (group_df, i_min, i_max) or 4-tuple with gap
    feat_cols     : list[str]  F feature column names
    window        : int        context window size
    horizon       : int        forecast horizon

    Returns
    -------
    X          : np.ndarray  shape (N, window*F + F)  float32
    y          : np.ndarray  shape (N, horizon)        float32
    region_ids : np.ndarray  shape (N,)
    """
    X_rows        = []
    y_rows        = []
    region_id_lst = []

    for entry in region_groups:
        group, i_min, i_max = entry[0], entry[1], entry[2]
        group  = group.reset_index(drop=True)
        n      = len(group)

        cols_present = [c for c in feat_cols if c in group.columns]
        X_mat = group[cols_present].values.astype(np.float32)   # (n, F)
        y_arr = (
            group["score"].values.astype(np.float32)
            if "score" in group.columns else None
        )
        rid = group["region_id"].iloc[0] if "region_id" in group.columns else 0

        for i in range(i_min, i_max + 1):
            end_feat  = i + window
            tgt_start = end_feat
            tgt_end   = tgt_start + horizon
            if tgt_end > n:
                break

            window_block = X_mat[i:end_feat]          # (window, F)
            flat_row     = window_block.reshape(-1)   # (window*F,)
            delta_row    = window_block[-1] - window_block[0]  # (F,)
            X_rows.append(np.concatenate([flat_row, delta_row]))

            if y_arr is not None:
                y_rows.append(y_arr[tgt_start:tgt_end])

            region_id_lst.append(rid)

    n_out = window * len(feat_cols) + len(feat_cols)

    if not X_rows:
        return (
            np.empty((0, n_out), dtype=np.float32),
            None,
            np.empty(0),
        )

    X          = np.array(X_rows, dtype=np.float32)
    y          = np.array(y_rows, dtype=np.float32) if y_rows else None
    region_ids = np.array(region_id_lst)

    return X, y, region_ids


def build_tabular_test(
    test_df: pd.DataFrame,
    feat_cols: list,
    window: int = WINDOW_SIZE,
):
    """
    Build one flat feature row per test region from the last `window` weeks.

    Pads at the front with the first row when a region has fewer than
    `window` historical weeks.

    Parameters
    ----------
    test_df   : pd.DataFrame  refined test data with TE columns injected
    feat_cols : list[str]     F feature column names
    window    : int           context window size

    Returns
    -------
    X          : np.ndarray  shape (n_regions, window*F + F)  float32
    region_ids : np.ndarray  shape (n_regions,)
    """
    X_rows     = []
    region_ids = []

    for region_id, group in test_df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n     = len(group)

        if n < window:
            pad_n  = window - n
            pad_df = pd.concat([group.iloc[[0]]] * pad_n + [group], ignore_index=True)
            group  = pad_df

        win_df       = group.iloc[-window:]
        cols_present = [c for c in feat_cols if c in win_df.columns]

        window_block = win_df[cols_present].values.astype(np.float32)
        flat_row     = window_block.reshape(-1)
        delta_row    = window_block[-1] - window_block[0]
        X_rows.append(np.concatenate([flat_row, delta_row]))
        region_ids.append(region_id)

    return np.array(X_rows, dtype=np.float32), np.array(region_ids)


# ---------------------------------------------------------------------------
# CV fold builder
# ---------------------------------------------------------------------------
def build_stratified_group_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
):
    """
    5-Fold StratifiedGroupKFold CV.

    Group  = region_id (all windows of a region stay together).
    Strata = K-Means cluster_id (10 climate ecosystems) if the column
             exists in df; otherwise 10-quantile bins of per-region mean score.

    Returns
    -------
    folds : list of (train_groups, val_groups)
        Each element is a list of (group_df, i_min, i_max).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    region_ids = df["region_id"].unique()
    n_regions  = len(region_ids)

    if "cluster_id" in df.columns:
        cluster_series = (
            df.drop_duplicates("region_id")
            .set_index("region_id")["cluster_id"]
        )
        strat_bins = np.array(
            [int(cluster_series.get(rid, 0)) for rid in region_ids],
            dtype=int,
        )
        print(f"  [V29] CV strata: cluster_id "
              f"({len(np.unique(strat_bins))} climate clusters)")
    else:
        region_mean_series = df.groupby("region_id")["score"].mean()
        region_means_arr   = np.array(
            [float(region_mean_series.get(rid, 0.0)) for rid in region_ids],
            dtype=np.float64,
        )
        strat_bins = pd.qcut(
            region_means_arr, q=10, labels=False, duplicates="drop"
        ).astype(int)
        print("  [Fallback] CV strata: 10-quantile bins of per-region mean score")

    sgkf    = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    dummy_X = np.zeros((n_regions, 1))
    folds   = []

    for train_reg_idx, val_reg_idx in sgkf.split(
        dummy_X, y=strat_bins, groups=region_ids
    ):
        train_region_set = set(region_ids[train_reg_idx])
        val_region_set   = set(region_ids[val_reg_idx])

        train_df_fold = df[df["region_id"].isin(train_region_set)]
        val_df_fold   = df[df["region_id"].isin(val_region_set)]

        train_groups = []
        for _, group in train_df_fold.groupby("region_id"):
            group = group.reset_index(drop=True)
            n     = len(group)
            if n < WINDOW_SIZE + HORIZON:
                continue
            train_groups.append((group, 0, n - WINDOW_SIZE - HORIZON))

        val_groups = []
        for _, group in val_df_fold.groupby("region_id"):
            group = group.reset_index(drop=True)
            n     = len(group)
            if n < WINDOW_SIZE + HORIZON:
                continue
            val_groups.append((group, 0, n - WINDOW_SIZE - HORIZON))

        folds.append((train_groups, val_groups))

    return folds


# ---------------------------------------------------------------------------
# Full-train group builder (used for final inference model)
# ---------------------------------------------------------------------------
def build_full_train_groups(df: pd.DataFrame):
    """
    Build region groups from ALL rows for final model training after CV.

    Returns
    -------
    list of (group_df, 0, i_max)
    """
    train_groups = []
    for _, group in df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n     = len(group)
        if n < WINDOW_SIZE + HORIZON:
            continue
        train_groups.append((group, 0, n - WINDOW_SIZE - HORIZON))
    return train_groups
