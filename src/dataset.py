"""
DroughtDataset
==============
Custom data builder for drought score multi-step forecasting.

Design
------
- V31 PARADIGM SHIFT: 3D Temporal Sequence Tensors for Deep Learning.
  The complete 13-week context window is reshaped into a chronological
  3D time-series tensor (B, 13, 27) where:
    - Dimension 0 (B)  : batch axis
    - Dimension 1 (13) : sequence length (13 weekly time steps)
    - Dimension 2 (27) : clean feature vector (v26 27-feature set)

  Tabular flattening (v23–v29) and explicit trend deltas (v27) are ABOLISHED.
  Deep learning handles feature interactions inside Conv1d and BiLSTM layers.

- Multi-step target: the next H=5 weekly `score` values (Y), stored as a
  (N_samples, 5) matrix — one column per target week.

- Region boundaries are strictly respected — windows NEVER cross regions.
- Feature pruning, log1p precipitation transform, and Drought Index
  (PET-based deficit) are applied in refine_features().

v31 Changes (Deep Learning Revival — Parallel Hybrid Sequence Net)
------------------------------------------------------------------
  REVIVE: build_sequence_dataset() and build_sequence_test() return
    3D NumPy arrays of shape (N, 13, 27) for direct PyTorch consumption.
  ABOLISH: build_tabular_dataset() flat-row construction with v27 deltas.
           make_flat_col_names() 378-dim wide matrix.
  RETAIN:  refine_features(), build_stratified_group_cv_folds(),
           FEATURE_COLS, WINDOW_SIZE, HORIZON, all CV helpers.

v27 Changes (The Tweedie-Hurdle Paradigm) [SUPERSEDED by v31]
--------------------------------------------------------------
  378-dim flat layout (351 flat + 27 explicit w13-w1 deltas).

v26 Changes (The Clean Slate -- Feature Purge)
----------------------------------------------
  FEATURE_COLS: 39 -> 27 (12 tokens purged).
  Retained: raw weekly baseline + intra-week stats + cyclic calendar
            + drought proxy + target encoding.

v23 Changes (13-Week Full Tabular Flattening) [SUPERSEDED by v31]
------------------------------------------------------------------
  Flattened ALL 13 rows into a wide 2D matrix (N, 351).

v22 Changes (Tabular Flattening + LightGBM) [SUPERSEDED by v31]
-----------------------------------------------------------------
  Paradigm shift from PyTorch 3D datasets to flat LightGBM tables.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocess import add_drought_index, DROUGHT_FEAT_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 13   # look-back (weeks) -- v16: capped at test-set horizon (13w)
HORIZON = 5        # forecast horizon (weeks)

# Walk-Forward Validation parameters (kept for backward compat imports)
WF_FOLD_WEEKS  = 5   # weeks per fold
WF_NUM_FOLDS   = 3   # number of folds

# v11 legacy fallback gap (kept for import compat; UNUSED in V31 path)
GAP_WEEKS = 0

# v20 legacy constants (kept for backward compat imports)
N_TS_FOLDS     = 5
TS_VAL_HORIZON = 5
TS_SHIFT_WEEKS = 26

# ---------------------------------------------------------------------------
# V22 Adversarial Feature Pruning (expanded from v21)
# ---------------------------------------------------------------------------
# dp_tmp  : collinearity > 0.9999 with tmp  (pruned at preprocess.py level, v19)
# wb_tmp  : collinearity > 0.9999 with tmp  (same)
# surf_tmp: collinearity > 0.99 with tmp    (v16 guard)
# wind_max: collinearity > 0.95 with wind   (v22 new)
# dow_sin : lowest permutation importance   (v22 new; may be absent in older CSVs)
DROP_COLS = ["wb_tmp", "dp_tmp", "surf_tmp", "wind_max", "dow_sin"]

# Precipitation columns to log1p-transform (handle right-skew)
# v26: prec_roll_sum_4w, prec_lag1w, prec_lag2w removed (cross-week bleed purge)
PREC_COLS = [
    "prec",
    "prec_week_max",
]

# Feature columns fed to the model (order matters for alignment)
# v26 Clean Slate: 27 features
#   base weather (10) + enriched weekly stats (11) + cyclic calendar (2)
#   + drought proxy (2) + target encoding (2) = 27
#
# PURGED vs v25 (12 tokens removed):
#   prec_roll_sum_4w, tmp_roll_mean_4w, humidity_roll_mean_4w  (3 cross-week rolling)
#   tmp_lag1w, humidity_lag1w, prec_lag1w, wind_lag1w           (4 lag-1)
#   tmp_lag2w, humidity_lag2w, prec_lag2w, wind_lag2w           (4 lag-2)
#   deficit_roll_cum_4w                                          (1 cross-week rolling)
FEATURE_COLS = [
    # --- base weather (10) [v22: wind_max removed] ---
    "prec", "surf_pre", "humidity",
    "tmp", "tmp_max", "tmp_min", "tmp_range",
    "wind", "wind_min", "wind_range",
    # --- enriched weekly statistics (11) [v19: intra-week climate shocks] ---
    "tmp_week_max", "tmp_week_min", "tmp_week_std",
    "humidity_week_max", "humidity_week_min", "humidity_week_std",
    "wind_week_max", "wind_week_min", "wind_week_std",
    "prec_week_max", "surf_pre_week_max",
    # --- cyclical calendar (2) [v9: replaces month + week_of_year] ---
    "week_sin", "week_cos",
    # --- drought proxy index (2) [v26: deficit_roll_cum_4w purged] ---
    "pet", "deficit",
    # --- target encoding (2) [v9: leakage-free region stats, injected in train.py] ---
    "region_mean_score", "region_zero_prob",
]   # total = 27 features  |  3D tensor shape per sample = (13, 27)


# ---------------------------------------------------------------------------
# V23 Flat Column Names: retained for backward compat imports only.
# ---------------------------------------------------------------------------
def make_flat_col_names(feat_cols: list, window: int = WINDOW_SIZE) -> list:
    """
    [v23/v27 legacy — retained for backward compat; not used in v31 pipeline]
    Generate flat column names: feat_w1 ... feat_w13 + feat_delta (378 total).
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
    1. Compute Drought Index (PET/deficit) BEFORE log1p (uses raw prec).
    2. Drop adversarial / collinear columns (wb_tmp, dp_tmp, surf_tmp,
       wind_max, dow_sin — v22 expanded pruning).
    3. Apply log1p to precipitation columns (v26: prec and prec_week_max only).
    4. Handle NaN rows:
       - Train: drop rows where any FEATURE_COL is NaN.
       - Test:  forward-fill then zero-fill so we keep all rows.

    NOTE: `region_mean_score` and `region_zero_prob` are NOT added here.
    They are injected by train.py (leakage-free, per-fold) AFTER this call.

    Returns a clean copy of df.
    """
    df = df.copy()

    # Step 0 - drought index (must be before log1p of prec)
    df = add_drought_index(df, is_train=is_train)

    # Step 1 - drop collinear / adversarial columns (v22: expanded)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Step 2 - log1p precipitation (clip negatives to 0 first)
    # v26: only prec and prec_week_max (lag/rolling prec cols purged)
    for col in PREC_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Step 3 - NaN handling (only on features present in df, excl. TE cols)
    present_feats = [
        c for c in FEATURE_COLS
        if c in df.columns and c not in ("region_mean_score", "region_zero_prob")
    ]

    if is_train:
        # Step 3a - drop NaN rows (v26: no lag cols so minimal head-NaN rows)
        df = df.dropna(subset=present_feats).reset_index(drop=True)
    else:
        # Step 3b - ffill then zero-fill within each region
        for col in present_feats:
            df[col] = (
                df.groupby("region_id")[col]
                .transform(lambda s: s.ffill().fillna(0))
            )

    return df


# ---------------------------------------------------------------------------
# V31 Primary: 3D Sequence Dataset Builders (B, 13, 27)
# ---------------------------------------------------------------------------
def build_sequence_dataset(
    region_groups: list,
    feat_cols: list,
    window: int = WINDOW_SIZE,
    horizon: int = HORIZON,
):
    """
    V31 3D Sequence Builder: reshape each sliding window into a chronological
    3D temporal tensor of shape (N_samples, window, F).

    For each window [i, i+window), the block of shape (window, F) is kept as-is
    (no flattening, no delta appending). The deep layers (Conv1d, BiLSTM)
    handle temporal interactions internally.

    v31: window=13, F=27 → each sample is a (13, 27) tensor.

    Parameters
    ----------
    region_groups : list of (group_df, i_min, i_max) OR 4-tuple with gap
        Each entry defines one region's data and valid window start indices.
    feat_cols : list of str
        Feature column names (must exist in group_df after TE augmentation).
    window  : int, context window size (default 13)
    horizon : int, forecast horizon (default 5)

    Returns
    -------
    X          : np.ndarray of shape (N_samples, window, F), float32
                 i.e. (N_samples, 13, 27) for v31 27-feature set
    y          : np.ndarray of shape (N_samples, horizon), float32, or None
    region_ids : np.ndarray of shape (N_samples,)
    """
    X_rows        = []
    y_rows        = []
    region_id_lst = []

    for entry in region_groups:
        if len(entry) == 4:
            group, i_min, i_max, _gap = entry
        else:
            group, i_min, i_max = entry

        group = group.reset_index(drop=True)
        n     = len(group)

        # Extract column arrays
        cols_present = [c for c in feat_cols if c in group.columns]
        X_mat = group[cols_present].values.astype(np.float32)   # (n, F)

        y_arr = (
            group["score"].values.astype(np.float32)
            if "score" in group.columns
            else None
        )

        rid = group["region_id"].iloc[0] if "region_id" in group.columns else 0

        for i in range(i_min, i_max + 1):
            end_feat  = i + window          # exclusive end of context window
            tgt_start = end_feat
            tgt_end   = tgt_start + horizon

            if tgt_end > n:
                break

            # V31 KEY: keep the (window, F) block as 3D — no flattening
            window_block = X_mat[i:end_feat]    # (13, 27)
            X_rows.append(window_block)

            if y_arr is not None:
                y_rows.append(y_arr[tgt_start:tgt_end])  # (H,)

            region_id_lst.append(rid)

    n_feat = len(feat_cols)

    if not X_rows:
        return (
            np.empty((0, window, n_feat), dtype=np.float32),
            None,
            np.empty(0),
        )

    X          = np.array(X_rows, dtype=np.float32)              # (N, 13, 27)
    y          = np.array(y_rows, dtype=np.float32) if y_rows else None   # (N, H)
    region_ids = np.array(region_id_lst)

    return X, y, region_ids


def build_sequence_test(
    test_df: pd.DataFrame,
    feat_cols: list,
    window: int = WINDOW_SIZE,
):
    """
    V31 3D Sequence Test Extractor.

    For each region in test_df, take the LAST `window` rows (padding at front
    if the region has fewer than `window` historical rows), and return the
    block as a 3D array of shape (n_regions, window, F).

    v31: window=13, F=27 → each row is a (13, 27) temporal block.

    Parameters
    ----------
    test_df   : pd.DataFrame with refined features + TE columns
    feat_cols : list of str — must match the training feature order
    window    : int, context window size (default 13)

    Returns
    -------
    X          : np.ndarray of shape (n_regions, window, F), float32
                 i.e. (2248, 13, 27) for the competition test set
    region_ids : np.ndarray of shape (n_regions,)
    """
    X_rows     = []
    region_ids = []

    for region_id, group in test_df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n     = len(group)

        # Pad at front if region is shorter than window
        if n < window:
            pad_n  = window - n
            pad_df = pd.concat([group.iloc[[0]]] * pad_n + [group], ignore_index=True)
            group  = pad_df

        # Take the last `window` rows
        win_df       = group.iloc[-window:]
        cols_present = [c for c in feat_cols if c in win_df.columns]

        # V31 KEY: keep the (window, F) block as 3D — no flattening
        window_block = win_df[cols_present].values.astype(np.float32)  # (13, 27)

        X_rows.append(window_block)
        region_ids.append(region_id)

    X = np.array(X_rows, dtype=np.float32)    # (2248, 13, 27)
    return X, np.array(region_ids)


# ---------------------------------------------------------------------------
# V31 PyTorch Dataset wrapper
# ---------------------------------------------------------------------------
class DroughtSequenceDataset(Dataset):
    """
    V31 PyTorch Dataset wrapping pre-built 3D NumPy arrays.

    Parameters
    ----------
    X : np.ndarray  (N, 13, 27)  — temporal sequence tensors
    y : np.ndarray  (N, 5)       — multi-step targets, or None for inference
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.from_numpy(X)                          # (N, 13, 27) float32
        self.y = torch.from_numpy(y) if y is not None else None  # (N, 5) float32

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ---------------------------------------------------------------------------
# V21 Primary CV: 5-Fold StratifiedGroupKFold builder (RETAINED for V31)
# ---------------------------------------------------------------------------
def build_stratified_group_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
):
    """
    V21/V22/V23/V26/V27/V31 Primary CV: 5-Fold StratifiedGroupKFold.

    Strategy:
    ---------
    - Group  = region_id (one group per region, never split across folds).
    - Strata = 10-quantile bins of each region's historical mean drought score.
    - Train  = 80% of regions per fold (geography unseen during validation).
    - Val    = 20% of regions per fold (completely held-out geography).

    This forces the model to generalise climate physics rather than memorise
    region-specific baselines.

    Parameters
    ----------
    df       : pd.DataFrame with refined features
    n_splits : int, number of CV folds (default 5)

    Returns
    -------
    folds : list of (train_groups, val_groups)
        train_groups : list of (group_df, i_min, i_max)
        val_groups   : list of (group_df, val_i, val_i)
    """
    from sklearn.model_selection import StratifiedGroupKFold

    region_ids = df["region_id"].unique()
    n_regions  = len(region_ids)

    # ------------------------------------------------------------------
    # Build stratification labels — 10-quantile bins of per-region mean score
    # ------------------------------------------------------------------
    region_mean_series = df.groupby("region_id")["score"].mean()
    region_means_arr   = np.array(
        [float(region_mean_series.get(rid, 0.0)) for rid in region_ids],
        dtype=np.float64,
    )
    strat_bins = pd.qcut(
        region_means_arr, q=10, labels=False, duplicates="drop"
    ).astype(int)

    sgkf    = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    dummy_X = np.zeros((n_regions, 1))

    folds = []

    for train_reg_idx, val_reg_idx in sgkf.split(
        dummy_X, y=strat_bins, groups=region_ids
    ):
        train_region_set = set(region_ids[train_reg_idx])
        val_region_set   = set(region_ids[val_reg_idx])

        train_df_fold = df[df["region_id"].isin(train_region_set)]
        val_df_fold   = df[df["region_id"].isin(val_region_set)]

        # ---- Train: ALL sliding windows with gap=0 (data-maximizing) ----
        train_groups = []
        for _, group in train_df_fold.groupby("region_id"):
            group = group.reset_index(drop=True)
            n = len(group)

            if n < WINDOW_SIZE + HORIZON:
                continue

            train_i_max = n - WINDOW_SIZE - HORIZON
            train_groups.append((group, 0, train_i_max))

        # ---- Val: ALL sliding windows across the entire timeline ----
        val_groups = []
        for _, group in val_df_fold.groupby("region_id"):
            group = group.reset_index(drop=True)
            n = len(group)

            if n < WINDOW_SIZE + HORIZON:
                continue

            val_i_max = n - WINDOW_SIZE - HORIZON
            val_groups.append((group, 0, val_i_max))

        folds.append((train_groups, val_groups))

    return folds


# ---------------------------------------------------------------------------
# Helper — build full-train group list (train on ALL data for final model)
# ---------------------------------------------------------------------------
def build_full_train_groups(df: pd.DataFrame, actual_gaps: dict = None):
    """
    Build region groups using ALL rows (no held-out validation period).
    Used for final model training after CV is complete.

    V21/V22/V23/V26/V27/V31: gap=0 throughout.  actual_gaps parameter retained
    for backward compat but is ignored.
    """
    train_groups = []
    for _, group in df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)

        if n < WINDOW_SIZE + HORIZON:
            continue

        train_i_max = n - WINDOW_SIZE - HORIZON
        if train_i_max >= 0:
            train_groups.append((group, 0, train_i_max))
    return train_groups


# ---------------------------------------------------------------------------
# V31 backward-compat shims for tabular builders
# (kept so that adversarial_witchhunt.py, eda.py import without crashing)
# ---------------------------------------------------------------------------
def build_tabular_dataset(
    region_groups: list,
    feat_cols: list,
    window: int = WINDOW_SIZE,
    horizon: int = HORIZON,
):
    """
    [v31 backward-compat shim]
    Delegates to build_sequence_dataset() and returns (X_flat, y, region_ids)
    where X_flat has shape (N, window*F) — a 2D reshaping of the 3D block.
    This ensures stale imports (adversarial_witchhunt.py) do not crash.
    """
    X_3d, y, region_ids = build_sequence_dataset(
        region_groups, feat_cols, window, horizon
    )
    if X_3d.shape[0] == 0:
        return X_3d.reshape(0, window * len(feat_cols)), y, region_ids
    N, W, F = X_3d.shape
    return X_3d.reshape(N, W * F), y, region_ids


def build_tabular_test(
    test_df: pd.DataFrame,
    feat_cols: list,
    window: int = WINDOW_SIZE,
):
    """
    [v31 backward-compat shim]
    Delegates to build_sequence_test() and returns (X_flat, region_ids)
    where X_flat has shape (n_regions, window*F).
    """
    X_3d, region_ids = build_sequence_test(test_df, feat_cols, window)
    N, W, F = X_3d.shape
    return X_3d.reshape(N, W * F), region_ids


# ---------------------------------------------------------------------------
# Legacy PyTorch DroughtDataset stub  (V22+: raises NotImplementedError)
# Retained so that any stale imports from older train.py versions do not crash
# at module-load time.  Attempting to instantiate will raise clearly.
# ---------------------------------------------------------------------------
class DroughtDataset:
    """
    V22+ STUB: Original PyTorch DroughtDataset is ABOLISHED.

    V31 introduces DroughtSequenceDataset (3D temporal tensor).
    Use build_sequence_dataset() / build_sequence_test() for the v31 path.

    Retained for import compatibility only.  Instantiation raises NotImplementedError.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DroughtDataset (v21 PyTorch 3D sliding window) is abolished. "
            "V31: use DroughtSequenceDataset / build_sequence_dataset() instead."
        )


# ---------------------------------------------------------------------------
# Legacy stubs retained for backward compat imports
# ---------------------------------------------------------------------------
def compute_actual_gaps(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """V21/V22/V23 STUB: Gap computation abolished. Returns all-zero dict."""
    all_ids = set(train_df["region_id"].unique()) | set(test_df["region_id"].unique())
    return {rid: 0 for rid in all_ids}


def build_temporal_shift_cv_folds(
    df: pd.DataFrame,
    actual_gaps: dict = None,
    n_folds: int = N_TS_FOLDS,
    val_horizon: int = TS_VAL_HORIZON,
    shift_weeks: int = TS_SHIFT_WEEKS,
):
    """V21/V22/V23 STUB: Temporal Shift CV abolished. Delegates to StratifiedGroupKFold."""
    return build_stratified_group_cv_folds(df, n_splits=n_folds)


def build_region_group_cv_folds(
    df: pd.DataFrame,
    actual_gaps: dict = None,
    n_splits: int = 5,
):
    """V18/V19 builder — now identical to V21+ primary builder."""
    return build_stratified_group_cv_folds(df, n_splits=n_splits)


def build_single_fold(df: pd.DataFrame, actual_gaps: dict = None):
    """V15 legacy stub. Returns (train_groups, val_groups) from first fold."""
    folds = build_stratified_group_cv_folds(df, n_splits=5)
    if folds:
        return folds[0]
    return [], []


def build_gap_replay_folds(df: pd.DataFrame, actual_gaps: dict = None):
    """V14 legacy stub. Returns StratifiedGroupKFold folds."""
    return build_stratified_group_cv_folds(df, n_splits=WF_NUM_FOLDS)


def build_walk_forward_folds(df: pd.DataFrame):
    """Legacy stub. Delegates to build_stratified_group_cv_folds()."""
    return build_stratified_group_cv_folds(df, n_splits=WF_NUM_FOLDS)
