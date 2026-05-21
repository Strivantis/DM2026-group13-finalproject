"""
DroughtDataset
==============
Custom PyTorch Dataset for drought score multi-step forecasting.

Design
------
- Sliding window (step=1) of W=13 weekly rows as input (X).
- Multi-step target: the next H=5 weekly `score` values (Y).
- Region boundaries are strictly respected – windows NEVER cross regions.
- Feature pruning, log1p precipitation transform, and Drought Index
  (PET-based deficit 4-week rolling sum) are applied here.

v21 Changes (Pure Continuous-Time Prediction + StratifiedGroupKFold CV)
------------------------------------------------------------------------
PARADIGM SHIFT: The Dataset Gap (train/test time discontinuity) was NEVER
  a valid modelling concern at the input→prediction level.
  V21 returns to the purest possible formulation:
    Given 13 consecutive weeks X, predict the IMMEDIATELY FOLLOWING 5 weeks Y.
    Gap = 0 between end of X and start of Y.

- ABOLISHED all Gap mechanisms: `gap_size`, `actual_gap`, `effective_gap`,
  `compute_actual_gaps`, `build_temporal_shift_cv_folds`, `build_single_fold`,
  `build_gap_replay_folds`.  All Gap-related constants removed.

- NEW primary CV: `build_stratified_group_cv_folds()` (5-Fold StratifiedGroupKFold).
    Group  = region_id (each region is one group).
    Strata = 10-quantile bins of each region's historical mean_score.
    Train  = 80% of regions per fold  (geographically unseen at validation).
    Val    = 20% of regions per fold  (completely held-out geography).
  Forces the model to generalise weather physics, not memorise regions.

- DroughtDataset.__getitem__ returns a 4-tuple (removing gap_size):
    (X, y, target_time, group_id)
  group_id retained for diagnostic purposes (last context row index).

v19 Changes (Tweedie Paradigm Shift – Enriched Features + Time-Decay)
----------------------------------------------------------------------
- FEATURE_COLS expanded from 29 → 40 features to incorporate v19 enriched
  weekly statistics produced by preprocess.py:
    tmp_week_max, tmp_week_min, tmp_week_std     (intra-week temperature shocks)
    humidity_week_max, humidity_week_min, humidity_week_std
    wind_week_max, wind_week_min, wind_week_std
    prec_week_max, surf_pre_week_max             (extreme precipitation events)

- PREC_COLS updated: added prec_week_max for log1p precipitation transform
  (right-skewed positive distribution consistent with weekly rainfall peaks).

- DROP_COLS retained as guard: dp_tmp, wb_tmp, surf_tmp are excluded both at
  the preprocess.py aggregation level (v19 new) and here (defensive pruning).

v16 Changes (Preprocessing Overhaul – Sequence Grouping & 13-Week Bounds)
--------------------------------------------------------------------------
- WINDOW_SIZE reduced from 26 → 13 (hard cap at the test-set horizon).
- FEATURE_COLS: removed ALL 8-week and 13-week rolling features.
- PREC_COLS: removed 8w/13w entries.
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

# v11 legacy fallback gap (kept for import compat; UNUSED in V21 path)
GAP_WEEKS = 0

# v20 legacy constants (kept for backward compat imports)
N_TS_FOLDS     = 5
TS_VAL_HORIZON = 5
TS_SHIFT_WEEKS = 26

# Drop collinear / adversarial columns
# dp_tmp and wb_tmp are now pruned at the preprocess.py level (v19);
# they are kept here as a defensive guard in case older CSVs are loaded.
DROP_COLS = ["wb_tmp", "dp_tmp", "surf_tmp"]

# Precipitation columns to log1p-transform (handle right-skew)
# v16: removed prec_roll_sum_8w, prec_roll_sum_13w (features deleted)
# v19: added prec_week_max (extreme event peak – also right-skewed)
PREC_COLS = [
    "prec",
    "prec_week_max",
    "prec_roll_sum_4w",
    "prec_lag1w", "prec_lag2w",
]

# Feature columns fed to the model (order matters for scaler alignment)
# v19: 40 features = 11 base weather + 11 enriched weekly stats + 2 cyclic
#      + 3 rolling-4w + 4 lag1 + 4 lag2 + 3 drought-4w + 2 TE
#
# Count breakdown:
#   base weather (11) + enriched weekly stats (11) + cyclic calendar (2)
#   + rolling 4w (3) + lag-1 (4) + lag-2 (4) + drought 4w (3)
#   + target encoding (2) = 40
FEATURE_COLS = [
    # --- base weather (11) ---
    "prec", "surf_pre", "humidity",
    "tmp", "tmp_max", "tmp_min", "tmp_range",
    "wind", "wind_max", "wind_min", "wind_range",
    # --- enriched weekly statistics (11) [v19: intra-week climate shocks] ---
    "tmp_week_max", "tmp_week_min", "tmp_week_std",
    "humidity_week_max", "humidity_week_min", "humidity_week_std",
    "wind_week_max", "wind_week_min", "wind_week_std",
    "prec_week_max", "surf_pre_week_max",
    # --- cyclical calendar (2) [v9: replaces month + week_of_year] ---
    "week_sin", "week_cos",
    # --- rolling aggregates 4w only (3) [v16: removed 8w and 13w] ---
    "prec_roll_sum_4w",  "tmp_roll_mean_4w",  "humidity_roll_mean_4w",
    # --- lag-1 (4) ---
    "tmp_lag1w", "humidity_lag1w", "prec_lag1w", "wind_lag1w",
    # --- lag-2 (4) ---
    "tmp_lag2w", "humidity_lag2w", "prec_lag2w", "wind_lag2w",
    # --- drought proxy index 4w only (3) [v16: removed 8w and 13w] ---
    "pet", "deficit",
    "deficit_roll_cum_4w",
    # --- target encoding (2) [v9: leakage-free region stats, injected in train.py] ---
    "region_mean_score", "region_zero_prob",
]   # total = 40 features


# ---------------------------------------------------------------------------
# Feature refinement
# ---------------------------------------------------------------------------
def refine_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    1. Compute Drought Index (PET/deficit 4w) BEFORE log1p (uses raw prec).
    2. Drop adversarial / collinear columns (wb_tmp, dp_tmp, surf_tmp).
    3. Apply log1p to all precipitation-related columns.
    4. Handle NaN rows:
       - Train: drop rows where any FEATURE_COL is NaN (lag head: first 2/region).
       - Test:  forward-fill then zero-fill lag NaN so we keep all rows.

    NOTE: `region_mean_score` and `region_zero_prob` are NOT added here.
    They are injected by train.py (leakage-free, per-fold) AFTER this call.

    Returns a clean copy of df.
    """
    df = df.copy()

    # Step 0 – drought index (must be before log1p of prec)
    df = add_drought_index(df, is_train=is_train)

    # Step 1 – drop collinear / adversarial columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Step 2 – log1p precipitation (clip negatives to 0 first)
    for col in PREC_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Step 3 – NaN handling (only on features present in df, excl. TE cols)
    present_feats = [
        c for c in FEATURE_COLS
        if c in df.columns and c not in ("region_mean_score", "region_zero_prob")
    ]

    if is_train:
        # Step 3a – drop NaN rows (the first 2 rows per region due to lag-2)
        df = df.dropna(subset=present_feats).reset_index(drop=True)
    else:
        # Step 3b – ffill then zero-fill within each region
        for col in present_feats:
            df[col] = (
                df.groupby("region_id")[col]
                .transform(lambda s: s.ffill().fillna(0))
            )

    return df


# ---------------------------------------------------------------------------
# Region-aware sliding-window dataset  (v21: pure gap=0)
# ---------------------------------------------------------------------------
class DroughtDataset(Dataset):
    """
    V21 Pure Sliding Window Dataset (Hurdle Dual-Head)
    ---------------------------------------------------
    Parameters
    ----------
    region_groups : list of (group_df, i_min, i_max)  OR
                   (group_df, i_min, i_max, _unused_gap)
        Each entry defines one region's data and the [inclusive] range of
        valid sequence start indices.  Gap is always 0 in V21.
    window  : look-back length (weeks) — v16: default=13
    horizon : forecast horizon (weeks)
    scaler  : fitted sklearn scaler; applied if not None

    __getitem__ returns a 4-tuple:
        (X, y, target_time, group_id)

    group_id
    --------
    For each training sample starting at window index i, the group_id is
    defined as (i + window - 1) — the index of the last row in the context
    window within this region's weekly time series.
    """

    def __init__(
        self,
        region_groups: list,
        window: int = WINDOW_SIZE,
        horizon: int = HORIZON,
        scaler=None,
    ):
        self.window = window
        self.horizon = horizon

        sequences    = []
        targets      = []
        target_times = []   # (H, 2) week_sin/cos for each target week
        group_ids    = []   # chronological week position (last context row index)

        for entry in region_groups:
            # Support both old 4-tuple (group, i_min, i_max, _gap) and
            # new 3-tuple (group, i_min, i_max) for backward compat.
            if len(entry) == 4:
                group, i_min, i_max, _gap = entry
            else:
                group, i_min, i_max = entry

            feat_cols = [c for c in FEATURE_COLS if c in group.columns]
            X = group[feat_cols].values.astype(np.float32)

            # week_sin / week_cos arrays for target_time construction
            wsin_col = group["week_sin"].values.astype(np.float32) \
                if "week_sin" in group.columns else None
            wcos_col = group["week_cos"].values.astype(np.float32) \
                if "week_cos" in group.columns else None

            y_all = (
                group["score"].values.astype(np.float32)
                if "score" in group.columns
                else None
            )

            for i in range(i_min, i_max + 1):
                end_feat  = i + window
                # V21: target starts IMMEDIATELY after X ends (gap = 0)
                tgt_start = end_feat
                tgt_end   = tgt_start + horizon
                if tgt_end > len(group):
                    break

                sequences.append(X[i:end_feat])   # (W, F)

                if y_all is not None:
                    targets.append(y_all[tgt_start:tgt_end])  # (H,)

                # target_time: week_sin/cos of the H target weeks -> (H, 2)
                if wsin_col is not None and wcos_col is not None:
                    tt = np.stack([
                        wsin_col[tgt_start:tgt_end],
                        wcos_col[tgt_start:tgt_end],
                    ], axis=-1)   # (H, 2)
                else:
                    tt = np.zeros((horizon, 2), dtype=np.float32)
                target_times.append(tt)

                # group_id: index of the last context row = end_feat - 1
                group_ids.append(np.array([float(end_feat - 1)], dtype=np.float32))

        self.sequences    = np.array(sequences,    dtype=np.float32)   # (N, W, F)
        self.target_times = np.array(target_times, dtype=np.float32)   # (N, H, 2)
        self.group_ids    = np.array(group_ids,    dtype=np.float32)   # (N, 1)
        self.targets = (
            np.array(targets, dtype=np.float32) if targets else None   # (N, H)
        )

        # Apply scaler (fitted externally on train data)
        if scaler is not None and len(self.sequences) > 0:
            N, W, F = self.sequences.shape
            flat = self.sequences.reshape(-1, F)
            flat = scaler.transform(flat)
            self.sequences = flat.reshape(N, W, F).astype(np.float32)

    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx):
        x           = torch.tensor(self.sequences[idx],    dtype=torch.float32)
        target_time = torch.tensor(self.target_times[idx], dtype=torch.float32)
        group_id    = torch.tensor(self.group_ids[idx],    dtype=torch.float32)  # (1,)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            y = torch.empty(0)
        return x, y, target_time, group_id


# ---------------------------------------------------------------------------
# V21 Primary CV: 5-Fold StratifiedGroupKFold builder
# ---------------------------------------------------------------------------
def build_stratified_group_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
):
    """
    V21 Primary CV: 5-Fold StratifiedGroupKFold.

    Strategy:
    ---------
    - Group  = region_id (one group per region, never split across folds).
    - Strata = 10-quantile bins of each region's historical mean drought score.
    - Train  = 80% of regions per fold (geography unseen during validation).
    - Val    = 20% of regions per fold (completely held-out geography).

    This forces the model to generalise climate physics rather than memorise
    region-specific baselines.  20% held-out geography is the gold standard
    for spatial time-series cross-validation.

    Train sample construction (per region):
    ----------------------------------------
    ALL valid sliding windows with gap = 0:
      X[i : i+WINDOW_SIZE]  →  Y[i+WINDOW_SIZE : i+WINDOW_SIZE+HORIZON]
    i ranges from 0 to n - WINDOW_SIZE - HORIZON (inclusive).

    Val sample construction (per region):
    ---------------------------------------
    LAST HORIZON rows as Val_Y.  Val_X = WINDOW_SIZE rows immediately before.
    i.e. val_x_start = n - WINDOW_SIZE - HORIZON
         val_x_end   = n - HORIZON
         val_y_start = n - HORIZON
         val_y_end   = n

    Note: gap = 0 throughout (V21 paradigm).

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
        # v21.1 FIX: Use ALL historical windows for validation regions,
        # identical to training region treatment.  Previously this was
        # limited to a single last-window snapshot (Snapshot Bias).
        # Expected ~340k sequences once all windows are included.
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
# Helper – build full-train group list (train on ALL data for final model)
# ---------------------------------------------------------------------------
def build_full_train_groups(df: pd.DataFrame, actual_gaps: dict = None):
    """
    Build region groups using ALL rows (no held-out validation period).
    Used for final model training after CV is complete.

    V21: gap=0 throughout.  actual_gaps parameter retained for backward compat
    but is ignored.
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
# Legacy stubs retained for backward compat imports
# ---------------------------------------------------------------------------
def compute_actual_gaps(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    V21 STUB: Gap computation is abolished.  Returns a dict mapping every
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
    V21 STUB: Temporal Shift CV is abolished.
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
    V18/V19 builder – now identical to V21 primary builder.
    Retained for import compatibility.
    """
    return build_stratified_group_cv_folds(df, n_splits=n_splits)


def build_single_fold(df: pd.DataFrame, actual_gaps: dict = None):
    """
    V15 legacy stub.  Returns (train_groups, val_groups) from first fold of
    StratifiedGroupKFold.

    V21: gap=0. actual_gaps ignored.
    """
    folds = build_stratified_group_cv_folds(df, n_splits=5)
    if folds:
        return folds[0]
    return [], []


def build_gap_replay_folds(df: pd.DataFrame, actual_gaps: dict = None):
    """
    V14 legacy stub. Returns StratifiedGroupKFold folds.
    V21: gap=0. actual_gaps ignored.
    """
    return build_stratified_group_cv_folds(df, n_splits=WF_NUM_FOLDS)


def build_walk_forward_folds(df: pd.DataFrame):
    """
    Legacy stub. Delegates to build_stratified_group_cv_folds().
    """
    return build_stratified_group_cv_folds(df, n_splits=WF_NUM_FOLDS)
