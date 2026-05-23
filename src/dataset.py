"""
DroughtDataset
==============
Custom tabular data builder for drought score multi-step forecasting.

Design
------
- V23 PARADIGM SHIFT: 13-Week Full Tabular Flattening.
  The complete 13-week context window (all 27 features x 13 weeks) is
  flattened into a wide 2D spreadsheet of shape (N_samples, 351).
  Column names follow the pattern `<feature>_w1` through `<feature>_w13`
  so LightGBM can construct split paths over the full temporal dimension.

- Multi-step target: the next H=5 weekly `score` values (Y), stored as a
  (N_samples, 5) matrix - one column per target week.

- Region boundaries are strictly respected - windows NEVER cross regions.
- Feature pruning, log1p precipitation transform, and Drought Index
  (PET-based deficit) are applied in refine_features().

v27 Changes (The Tweedie-Hurdle Paradigm)
------------------------------------------
EXPAND: Append 27 explicit trend differential features (w13 - w1) onto the
  351-dimensional flat matrix, upgrading the input dimension to 378.

  For each sequence row, the direct physical derivative is computed as:
    feature_delta = feature_w13 - feature_w1
  for all 27 pure meteorological and intra-week statistics.

  Column naming: <feat>_delta  (27 columns appended after the 351 flat cols).

- build_tabular_dataset() now returns X of shape (N_samples, 378) in v27.
- build_tabular_test() now returns X of shape (2248, 378) in v27.

v26 Changes (The Clean Slate -- Feature Purge)
----------------------------------------------
PURGE: All lag features (tmp_lag1w/2w, humidity_lag1w/2w, prec_lag1w/2w,
  wind_lag1w/2w -- 8 tokens) and all cross-week rolling aggregates
  (prec_roll_sum_4w, tmp_roll_mean_4w, humidity_roll_mean_4w,
  deficit_roll_cum_4w -- 4 tokens) removed.

  Rationale: With 13-Week Full Tabular Flattening, week 12 features ARE
  the 1-week lag of week 13; lag columns are perfectly redundant and create
  artificial Train/Test distribution boundaries (adversarial AUC=1.0).
  Cross-week rolling sums overflow boundaries during weeks 1-3 of each
  sequence, causing additional domain shift.

RETAIN: raw weekly baseline meteorological variables + intra-week
  aggregation summaries (_week_max, _week_min, _week_std).
  These statistics never bleed across week indices.

- FEATURE_COLS: 39 -> 27  (12 tokens purged)
- Flat feature dimension: 507 -> 351  (27 x 13)

v23 Changes (13-Week Full Tabular Flattening)
----------------------------------------------
EXPAND: Flatten ALL 13 rows of each sliding window into a single wide
  feature vector.  Column naming: <feat>_w1 through <feat>_w13.

- build_tabular_dataset() now returns X of shape (N, 351) in v26.
- build_tabular_test() now returns X of shape (2248, 351) in v26.

v22 Changes (Tabular Flattening + LightGBM)
--------------------------------------------
PARADIGM SHIFT: Abolish 3D sliding-window PyTorch Datasets.  Replace with
  build_tabular_dataset() that extracted ONLY the 13th row (index 12) from
  each (i, i+13) window.  No DataLoader, no tensor wrappers.

- NEW DROP_COLS pruning additions (v22 adversarial guard):
    wind_max   : collinearity > 0.95 with baseline `wind`.
    dow_sin    : lowest permutation importance, seasonal encoding artefact.
  (dp_tmp, wb_tmp were already pruned in v19/v21 via DROP_COLS.)

v21 Changes (Pure Continuous-Time Prediction + StratifiedGroupKFold CV)
------------------------------------------------------------------------
ABOLISHED: all Gap mechanisms.  V21 gap = 0.
NEW: build_stratified_group_cv_folds() (5-Fold StratifiedGroupKFold).

v19 Changes (Tweedie Paradigm Shift - Enriched Features + Time-Decay)
----------------------------------------------------------------------
FEATURE_COLS expanded: 29 to 40 enriched weekly statistics.

v16 Changes (Preprocessing Overhaul - Sequence Grouping & 13-Week Bounds)
--------------------------------------------------------------------------
WINDOW_SIZE reduced 26 to 13.  8w/13w rolling features removed.
"""

import numpy as np
import pandas as pd

from src.preprocess import add_drought_index, DROUGHT_FEAT_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 13   # look-back (weeks) -- v16: capped at test-set horizon (13w)
HORIZON = 5        # forecast horizon (weeks)

# Walk-Forward Validation parameters (kept for backward compat imports)
WF_FOLD_WEEKS  = 5   # weeks per fold
WF_NUM_FOLDS   = 3   # number of folds

# v11 legacy fallback gap (kept for import compat; UNUSED in V22 path)
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

# Feature columns fed to the model (order matters for scaler alignment)
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
]   # total = 27 features  |  flat dim = 27 x 13 = 351


# ---------------------------------------------------------------------------
# V23 Flat Column Names: <feat>_w1 ... <feat>_w13  (351 total in v26)
# V27 Delta Column Names: <feat>_delta              (27 total in v27)
# Combined: 351 + 27 = 378 total columns in v27
# ---------------------------------------------------------------------------
def make_flat_col_names(feat_cols: list, window: int = WINDOW_SIZE) -> list:
    """
    Generate the flat column names used by the v23+ wide matrix.
    v26: 27 features x 13 weeks = 351 columns.
    v27: additionally appends 27 delta columns -> 378 total.
    Order: feat_w1, feat_w2, ..., feat_w13 for each feat in feat_cols.
    i.e.  [f0_w1, f0_w2, ..., f0_w13, f1_w1, ..., f26_w13]
    Week-major ordering matches numpy reshape(window * n_feats):
      row.reshape(-1) after stacking window rows =
        [f0_w1,..,f26_w1, f0_w2,..,f26_w2, ...]
    For week_idx w (1-indexed), feat f -> f_w{w}.
    v27 delta columns appended as: [f0_delta, f1_delta, ..., f26_delta]
    """
    names = []
    for w in range(1, window + 1):
        for feat in feat_cols:
            names.append(f"{feat}_w{w}")
    # v27: append explicit trend delta column names
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
       wind_max, dow_sin - v22 expanded pruning).
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
# V23 Primary: Tabular Dataset Builders (13-Week Full Flattening)
# ---------------------------------------------------------------------------
def build_tabular_dataset(
    region_groups: list,
    feat_cols: list,
    window: int = WINDOW_SIZE,
    horizon: int = HORIZON,
):
    """
    V23 Tabular Flattening: flatten ALL `window` rows of each sliding window
    into a single wide feature vector of length (window * len(feat_cols)).
    v26: 27 features x 13 weeks = 351 columns.

    V27 EXPAND: Append 27 explicit trend differential features (w13 - w1)
    directly after the 351-dim flat vector, yielding 378 dimensions total.
    Delta computation:  feature_delta = feature_w13 - feature_w1
    for each of the 27 features (feat_cols order is preserved).

    For each window [i, i+window), the full matrix of shape (window, F) is
    flattened row-major (week-major) into a 1D vector of shape (window*F,),
    then the F delta values are concatenated -> (window*F + F,) = (378,) in v27.
    Column naming convention: feat_w1, feat_w2, ..., feat_w13, feat_delta.

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
    X          : np.ndarray of shape (N_samples, window * len(feat_cols) + len(feat_cols)), float32
                 i.e. (N_samples, 378) with v27 27-feature set + 27 deltas
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

            # V23 KEY: flatten ALL window rows into a single wide row
            # X_mat[i:end_feat] has shape (window, F) -> reshape to (window*F,)
            window_block = X_mat[i:end_feat]          # (window, F)
            flat_row     = window_block.reshape(-1)   # (window*F,) = (351,) in v26

            # V27 KEY: compute explicit trend deltas (w13 - w1) for all F features
            # window_block[-1] = week 13 values (shape F,)
            # window_block[0]  = week 1  values (shape F,)
            delta_row = window_block[-1] - window_block[0]   # (F,) = (27,) in v27

            # Horizontal concatenation: 351 + 27 = 378 dimensions
            full_row = np.concatenate([flat_row, delta_row])  # (378,)
            X_rows.append(full_row)

            if y_arr is not None:
                y_rows.append(y_arr[tgt_start:tgt_end])  # (H,)

            region_id_lst.append(rid)

    n_flat_cols = len(feat_cols)
    n_out = window * n_flat_cols + n_flat_cols  # 351 + 27 = 378

    if not X_rows:
        return (
            np.empty((0, n_out), dtype=np.float32),
            None,
            np.empty(0),
        )

    X = np.array(X_rows, dtype=np.float32)                           # (N, 378)
    y = np.array(y_rows, dtype=np.float32) if y_rows else None       # (N, H)
    region_ids = np.array(region_id_lst)

    return X, y, region_ids


def build_tabular_test(
    test_df: pd.DataFrame,
    feat_cols: list,
    window: int = WINDOW_SIZE,
):
    """
    V23 Tabular Test Extractor.

    For each region in test_df, take the LAST `window` rows (padding at front
    if the region has fewer than `window` historical rows), then flatten ALL
    `window` rows into a single wide feature vector of length (window * F).
    v26: 27 features x 13 weeks = 351.

    V27 EXPAND: Append 27 explicit trend differential features (w13 - w1)
    directly after the 351-dim flat vector, yielding 378 dimensions total.
    Delta computation:  feature_delta = feature_w13 - feature_w1.

    This gives exactly one row per region (2248 rows for the competition).

    Parameters
    ----------
    test_df   : pd.DataFrame with refined features + TE columns
    feat_cols : list of str - must match the scaler's fitted column order
    window    : int, context window size (default 13)

    Returns
    -------
    X          : np.ndarray of shape (n_regions, window * len(feat_cols) + len(feat_cols)), float32
                 i.e. (2248, 378) with v27 27-feature set + 27 deltas
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

        # V23 KEY: flatten all window rows into a single wide row
        window_block = win_df[cols_present].values.astype(np.float32)  # (window, F)
        flat_row     = window_block.reshape(-1)                         # (window*F,)

        # V27 KEY: compute explicit trend deltas (w13 - w1) for all F features
        delta_row = window_block[-1] - window_block[0]                  # (F,)

        # Horizontal concatenation: 351 + 27 = 378 dimensions
        full_row = np.concatenate([flat_row, delta_row])                # (378,)

        X_rows.append(full_row)
        region_ids.append(region_id)

    X = np.array(X_rows, dtype=np.float32)
    return X, np.array(region_ids)


# ---------------------------------------------------------------------------
# V21 Primary CV: 5-Fold StratifiedGroupKFold builder (RETAINED for V27)
# ---------------------------------------------------------------------------
def build_stratified_group_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
):
    """
    V21/V22/V23/V26/V27 Primary CV: 5-Fold StratifiedGroupKFold.

    Strategy:
    ---------
    - Group  = region_id (one group per region, never split across folds).
    - Strata = 10-quantile bins of each region's historical mean drought score.
    - Train  = 80% of regions per fold (geography unseen during validation).
    - Val    = 20% of regions per fold (completely held-out geography).

    This forces the model to generalise climate physics rather than memorise
    region-specific baselines.

    Train sample construction (per region):
    ----------------------------------------
    ALL valid sliding windows with gap = 0:
      X[i : i+WINDOW_SIZE]  ->  Y[i+WINDOW_SIZE : i+WINDOW_SIZE+HORIZON]
    i ranges from 0 to n - WINDOW_SIZE - HORIZON (inclusive).

    Val sample construction (per region):
    ---------------------------------------
    ALL valid sliding windows (same treatment as training regions).

    Note: gap = 0 throughout (V21/V22/V23/V26/V27 paradigm).

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
    # Build stratification labels -- 10-quantile bins of per-region mean score
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
                continue  # region too short for even one sample

            train_i_max = n - WINDOW_SIZE - HORIZON
            train_groups.append((group, 0, train_i_max))

        # ---- Val: ALL sliding windows across the entire timeline ----
        val_groups = []
        for _, group in val_df_fold.groupby("region_id"):
            group = group.reset_index(drop=True)
            n = len(group)

            if n < WINDOW_SIZE + HORIZON:
                continue  # region too short for even one sample

            val_i_max = n - WINDOW_SIZE - HORIZON
            val_groups.append((group, 0, val_i_max))

        folds.append((train_groups, val_groups))

    return folds


# ---------------------------------------------------------------------------
# Helper - build full-train group list (train on ALL data for final model)
# ---------------------------------------------------------------------------
def build_full_train_groups(df: pd.DataFrame, actual_gaps: dict = None):
    """
    Build region groups using ALL rows (no held-out validation period).
    Used for final model training after CV is complete.

    V21/V22/V23/V26/V27: gap=0 throughout.  actual_gaps parameter retained for
    backward compat but is ignored.
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
# Legacy PyTorch DroughtDataset stub  (V22+: raises NotImplementedError)
# Retained so that any stale imports from older train.py versions do not crash
# at module-load time.  Attempting to instantiate will raise clearly.
# ---------------------------------------------------------------------------
class DroughtDataset:
    """
    V22+ STUB: PyTorch DroughtDataset is ABOLISHED.

    The V22/V23/V26/V27 pipeline uses build_tabular_dataset() and build_tabular_test()
    for pure NumPy/Pandas tabular matrices.  LightGBM does not use DataLoaders.

    Retained for import compatibility only.  Instantiation raises NotImplementedError.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DroughtDataset (PyTorch 3D sliding window) is abolished in V22+. "
            "Use build_tabular_dataset() / build_tabular_test() instead."
        )


# ---------------------------------------------------------------------------
# Legacy stubs retained for backward compat imports
# ---------------------------------------------------------------------------
def compute_actual_gaps(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    V21/V22/V23 STUB: Gap computation is abolished.  Returns a dict mapping every
    region_id to 0.  Retained for import compatibility with older train.py.
    """
    all_ids = set(train_df["region_id"].unique()) | set(test_df["region_id"].unique())
    return {rid: 0 for rid in all_ids}


def build_temporal_shift_cv_folds(
    df: pd.DataFrame,
    actual_gaps: dict = None,
    n_folds: int = N_TS_FOLDS,
    val_horizon: int = TS_VAL_HORIZON,
    shift_weeks: int = TS_SHIFT_WEEKS,
):
    """
    V21/V22/V23 STUB: Temporal Shift CV is abolished.
    Delegates to build_stratified_group_cv_folds().
    Retained for import compatibility.
    """
    return build_stratified_group_cv_folds(df, n_splits=n_folds)


def build_region_group_cv_folds(
    df: pd.DataFrame,
    actual_gaps: dict = None,
    n_splits: int = 5,
):
    """
    V18/V19 builder - now identical to V21/V22/V23 primary builder.
    Retained for import compatibility.
    """
    return build_stratified_group_cv_folds(df, n_splits=n_splits)


def build_single_fold(df: pd.DataFrame, actual_gaps: dict = None):
    """
    V15 legacy stub.  Returns (train_groups, val_groups) from first fold of
    StratifiedGroupKFold.

    V21/V22/V23/V26/V27: gap=0. actual_gaps ignored.
    """
    folds = build_stratified_group_cv_folds(df, n_splits=5)
    if folds:
        return folds[0]
    return [], []


def build_gap_replay_folds(df: pd.DataFrame, actual_gaps: dict = None):
    """
    V14 legacy stub. Returns StratifiedGroupKFold folds.
    V21/V22/V23/V26/V27: gap=0. actual_gaps ignored.
    """
    return build_stratified_group_cv_folds(df, n_splits=WF_NUM_FOLDS)


def build_walk_forward_folds(df: pd.DataFrame):
    """
    Legacy stub. Delegates to build_stratified_group_cv_folds().
    """
    return build_stratified_group_cv_folds(df, n_splits=WF_NUM_FOLDS)
