"""
DroughtDataset
==============
Custom PyTorch Dataset for drought score multi-step forecasting.

Design
------
- Sliding window (step=1) of W=26 weekly rows as input (X).
- Multi-step target: the next H=5 weekly `score` values (Y).
- Region boundaries are strictly respected – windows NEVER cross regions.
- Feature pruning, log1p precipitation transform, and Drought Index
  (PET-based deficit rolling sums) are applied here.

v9 Changes
----------
- FEATURE_COLS updated: `month` and `week_of_year` replaced with
  `week_sin` and `week_cos` (cyclical encoding from preprocess.py).
- Added `region_mean_score` and `region_zero_prob` for target encoding.
  These are computed leakage-free in train.py (fold-specific) and
  dynamically injected into region group DataFrames before Dataset creation.
- Total feature count: 37 (up from 35).

v11 Changes
-----------
- WINDOW_SIZE increased from 13 to 26 (half a year of context).
- GAP_WEEKS = 4 introduced for Gap-Aware Walk-Forward CV.
  Training fold strictly ends at val_start - GAP_WEEKS (not val_start-1)
  to simulate the real-world 4-week prediction gap between train and test.
- Bounds check: gracefully skip or limit regions where val_start < WINDOW_SIZE.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocess import add_drought_index, DROUGHT_FEAT_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 26   # look-back (weeks) -- v11: increased from 13 to 26
HORIZON = 5        # forecast horizon (weeks)

# Walk-Forward Validation parameters
WF_FOLD_WEEKS  = 5   # weeks per fold
WF_NUM_FOLDS   = 3   # number of folds  → 15 total hold-out weeks

# v11: Gap between end of train and start of validation (simulates real deployment)
GAP_WEEKS = 4

# Drop collinear temp proxies (keep tmp, tmp_max, tmp_min, tmp_range)
DROP_COLS = ["wb_tmp", "dp_tmp", "surf_tmp"]

# Precipitation columns to log1p-transform (handle right-skew)
PREC_COLS = [
    "prec",
    "prec_roll_sum_4w", "prec_roll_sum_8w", "prec_roll_sum_13w",
    "prec_lag1w", "prec_lag2w",
]

# Feature columns fed to the model (order matters for scaler alignment)
# v9: `month` and `week_of_year` replaced by `week_sin` and `week_cos`;
#     `region_mean_score` and `region_zero_prob` added (target encoding).
FEATURE_COLS = [
    # --- base weather (11) ---
    "prec", "surf_pre", "humidity",
    "tmp", "tmp_max", "tmp_min", "tmp_range",
    "wind", "wind_max", "wind_min", "wind_range",
    # --- cyclical calendar (2) [v9: replaces month + week_of_year] ---
    "week_sin", "week_cos",
    # --- rolling aggregates (9) ---
    "prec_roll_sum_4w",  "tmp_roll_mean_4w",  "humidity_roll_mean_4w",
    "prec_roll_sum_8w",  "tmp_roll_mean_8w",  "humidity_roll_mean_8w",
    "prec_roll_sum_13w", "tmp_roll_mean_13w", "humidity_roll_mean_13w",
    # --- lag-1 (4) ---
    "tmp_lag1w", "humidity_lag1w", "prec_lag1w", "wind_lag1w",
    # --- lag-2 (4) ---
    "tmp_lag2w", "humidity_lag2w", "prec_lag2w", "wind_lag2w",
    # --- drought proxy index (5) ---
    "pet", "deficit",
    "deficit_roll_cum_4w", "deficit_roll_cum_8w", "deficit_roll_cum_13w",
    # --- target encoding (2) [v9: leakage-free region stats, injected in train.py] ---
    "region_mean_score", "region_zero_prob",
]   # total = 37 features


# ---------------------------------------------------------------------------
# Feature refinement
# ---------------------------------------------------------------------------
def refine_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    1. Compute Drought Index (PET/deficit) BEFORE log1p (uses raw prec).
    2. Drop collinear temperature proxies.
    3. Apply log1p to all precipitation-related columns.
    4. Handle NaN rows:
       - Train: drop rows where any FEATURE_COL is NaN (lag head).
       - Test:  forward-fill then zero-fill lag NaN so we keep all rows.

    NOTE: `region_mean_score` and `region_zero_prob` are NOT added here.
    They are injected by train.py (leakage-free, per-fold) AFTER this call.

    Returns a clean copy of df.
    """
    df = df.copy()

    # Step 0 – drought index (must be before log1p of prec)
    df = add_drought_index(df, is_train=is_train)

    # Step 1 – drop collinear columns
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
        # Step 3a – drop NaN rows (usually the first 2 rows per region)
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
# Region-aware sliding-window dataset
# ---------------------------------------------------------------------------
class DroughtDataset(Dataset):
    """
    Parameters
    ----------
    region_groups : list of (group_df, i_min, i_max)
        Each entry defines one region's data and the [inclusive] range of
        valid sequence start indices.  Sequences outside this range are
        silently skipped.
    window  : look-back length (weeks)
    horizon : forecast horizon (weeks)
    scaler  : fitted sklearn scaler; applied if not None
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

        sequences, targets = [], []

        for group, i_min, i_max in region_groups:
            feat_cols = [c for c in FEATURE_COLS if c in group.columns]
            X = group[feat_cols].values.astype(np.float32)
            y = (
                group["score"].values.astype(np.float32)
                if "score" in group.columns
                else None
            )

            for i in range(i_min, i_max + 1):
                end_feat = i + window
                end_tgt = end_feat + horizon
                if end_tgt > len(group):
                    break
                sequences.append(X[i:end_feat])             # (W, F)
                if y is not None:
                    targets.append(y[end_feat:end_tgt])     # (H,)

        self.sequences = np.array(sequences, dtype=np.float32)  # (N, W, F)
        self.targets = (
            np.array(targets, dtype=np.float32) if targets else None  # (N, H)
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
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            y = torch.empty(0)
        return x, y


# ---------------------------------------------------------------------------
# Helper – Walk-Forward Cross-Validation fold builder
# ---------------------------------------------------------------------------
def build_walk_forward_folds(df: pd.DataFrame):
    """
    Implement Gap-Aware Walk-Forward (Time-Series) Cross-Validation.

    The last WF_NUM_FOLDS * WF_FOLD_WEEKS = 15 weeks of each region are
    reserved for validation and are split into 3 non-overlapping 5-week folds:

        Fold 0 (most recent) : rows [-5:]
        Fold 1               : rows [-10:-5]
        Fold 2 (oldest)      : rows [-15:-10]

    v11 Gap-Aware: For each fold k, training strictly ends at
        val_start - GAP_WEEKS
    rather than val_start - 1.  This prevents the model from seeing the
    4-week "buffer zone" immediately before the validation period, simulating
    the real-world 4-week deployment gap between the training cutoff and the
    first test week.

    Returns
    -------
    folds : list of 3 tuples (train_groups, val_groups)
        Each element is ready to be passed to DroughtDataset.
    """
    total_hold = WF_NUM_FOLDS * WF_FOLD_WEEKS   # 15 weeks

    folds = []
    for fold_k in range(WF_NUM_FOLDS):
        # val_start_from_end: how many rows from the END is the val period start
        #   fold 0: rows [-5:]       → val_start_from_end = 5
        #   fold 1: rows [-10:-5]    → val_start_from_end = 10
        #   fold 2: rows [-15:-10]   → val_start_from_end = 15
        val_end_from_end   = fold_k * WF_FOLD_WEEKS            # 0,  5, 10
        val_start_from_end = val_end_from_end + WF_FOLD_WEEKS  # 5, 10, 15

        train_groups, val_groups = [], []

        for _, group in df.groupby("region_id"):
            group = group.reset_index(drop=True)
            n = len(group)

            # Absolute indices
            val_start = n - val_start_from_end  # inclusive
            val_end   = n - val_end_from_end    # exclusive  (=n for fold 0)

            # v11: Need at least WINDOW_SIZE rows of history before val_start
            if val_start < WINDOW_SIZE:
                # Not enough history – skip region for this fold
                continue

            # --- v11 Gap-Aware train: last training sequence target must end
            #     at least GAP_WEEKS before val_start.
            #     Effective training cutoff row = val_start - GAP_WEEKS
            #     train_i_max: largest start index i such that
            #       i + WINDOW_SIZE + HORIZON <= val_start - GAP_WEEKS
            #     => train_i_max = val_start - GAP_WEEKS - WINDOW_SIZE - HORIZON
            train_cutoff = val_start - GAP_WEEKS  # exclusive upper bound for train targets
            train_i_max  = train_cutoff - WINDOW_SIZE - HORIZON
            if train_i_max >= 0:
                train_groups.append((group, 0, train_i_max))

            # --- val: sequences whose FIRST target row >= val_start ----------
            val_i_min = val_start - WINDOW_SIZE
            # last sequence must not run past val_end
            val_i_max = val_end - WINDOW_SIZE - HORIZON
            if val_i_min >= 0 and val_i_min <= val_i_max:
                val_groups.append((group, val_i_min, val_i_max))

        folds.append((train_groups, val_groups))

    return folds


# ---------------------------------------------------------------------------
# Helper – build full-train group list (train on ALL data for final model)
# ---------------------------------------------------------------------------
def build_full_train_groups(df: pd.DataFrame):
    """
    Build region groups using ALL rows (no held-out validation period).
    Used for final model training after walk-forward CV is complete.
    """
    train_groups = []
    for _, group in df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        train_i_max = n - WINDOW_SIZE - HORIZON
        if train_i_max >= 0:
            train_groups.append((group, 0, train_i_max))
    return train_groups
