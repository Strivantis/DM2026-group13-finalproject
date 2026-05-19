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

v14 Changes
-----------
- PARADIGM SHIFT: Absolute CV replaced with "Relative Gap-Replay" CV.
  - actual_gap is computed per-region from the real calendar distance
    between the last train week and the first test week.
  - Validation split replicates this exact gap: Val_X ends `actual_gap`
    weeks before Val_Y starts (last 5 weeks of historical train data).
  - All sliding-window (Train_X, Train_Y) pairs strictly enforce
    Distance(End_of_X, Start_of_Y) == actual_gap.
  - Zero Data Waste Fallback: if history too short, shrink actual_gap
    to the maximum feasible size that yields ≥1 (X, Y) pair.
- __getitem__ now returns (X, y, target_time, gap_size):
    target_time : (5, 2)  week_sin/cos of the 5 future target weeks
    gap_size    : (1,)    actual_gap / 100.0  (normalised scalar)
- build_gap_replay_folds() replaces build_walk_forward_folds().
- build_full_train_groups() unchanged (used for final model training).
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

# Walk-Forward Validation parameters (kept for backward compat imports)
WF_FOLD_WEEKS  = 5   # weeks per fold
WF_NUM_FOLDS   = 3   # number of folds  → 15 total hold-out weeks

# v11 (legacy – no longer used as fixed gap; kept for import compat)
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
# Region-aware sliding-window dataset  (v14: Gap-Replay)
# ---------------------------------------------------------------------------
class DroughtDataset(Dataset):
    """
    v14 Gap-Replay Dataset
    ----------------------
    Parameters
    ----------
    region_groups : list of (group_df, i_min, i_max, actual_gap)
        Each entry defines one region's data, the [inclusive] range of
        valid sequence start indices, and the actual deployment gap for
        that region.  The `actual_gap` is used to:
          1. Determine the correct target window offset from X.
          2. Build the `gap_size` tensor returned in __getitem__.
          3. Provide `target_time` (week_sin/cos of the 5 target weeks).
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

        sequences    = []
        targets      = []
        target_times = []   # (5, 2) week_sin/cos for each target week
        gap_sizes    = []   # scalar actual_gap / 100.0

        for entry in region_groups:
            # Support both old 3-tuple (group, i_min, i_max) and new
            # 4-tuple (group, i_min, i_max, actual_gap) for backward compat.
            if len(entry) == 4:
                group, i_min, i_max, actual_gap = entry
            else:
                group, i_min, i_max = entry
                actual_gap = GAP_WEEKS  # legacy fallback

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
                end_feat = i + window
                # v14: target window starts `actual_gap` weeks after X ends
                tgt_start = end_feat + actual_gap
                tgt_end   = tgt_start + horizon
                if tgt_end > len(group):
                    break

                sequences.append(X[i:end_feat])   # (W, F)

                if y_all is not None:
                    targets.append(y_all[tgt_start:tgt_end])  # (H,)

                # target_time: week_sin/cos of the 5 target weeks -> (H, 2)
                if wsin_col is not None and wcos_col is not None:
                    tt = np.stack([
                        wsin_col[tgt_start:tgt_end],
                        wcos_col[tgt_start:tgt_end],
                    ], axis=-1)   # (H, 2)
                else:
                    tt = np.zeros((horizon, 2), dtype=np.float32)
                target_times.append(tt)

                # gap_size: normalised scalar (1,)
                gap_sizes.append(np.array([actual_gap / 100.0], dtype=np.float32))

        self.sequences    = np.array(sequences,    dtype=np.float32)   # (N, W, F)
        self.target_times = np.array(target_times, dtype=np.float32)   # (N, H, 2)
        self.gap_sizes    = np.array(gap_sizes,    dtype=np.float32)   # (N, 1)
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
        gap_size    = torch.tensor(self.gap_sizes[idx],    dtype=torch.float32)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            y = torch.empty(0)
        return x, y, target_time, gap_size


# ---------------------------------------------------------------------------
# Helper – compute per-region actual gap from train/test DataFrames
# ---------------------------------------------------------------------------
def compute_actual_gaps(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Compute the actual deployment gap (in weeks) for each region_id.

    The gap is defined as: (first test row index) - (last train row index)
    using the `abs_week` column if present, otherwise falling back to
    counting from sorted position within each region.

    Returns
    -------
    gaps : dict  region_id -> int  (number of weeks gap, >= 1)
    """
    def _abs_week(df):
        """Convert week_key 'YYYYNN-WXX' to an absolute integer week."""
        def parse(wk):
            if pd.isna(wk):
                return np.nan
            parts = str(wk).split("-W")
            year = int(parts[0]) if parts[0].isdigit() else int(parts[0][-6:])
            week = int(parts[1])
            return year * 53 + week
        return df["week_key"].apply(parse)

    if "week_key" in train_df.columns and "week_key" in test_df.columns:
        tr_abs = _abs_week(train_df)
        te_abs = _abs_week(test_df)

        train_max = train_df.assign(_aw=tr_abs).groupby("region_id")["_aw"].max()
        test_min  = test_df.assign(_aw=te_abs).groupby("region_id")["_aw"].min()
    else:
        # Fallback: use the row count within each region's contiguous block
        # The sorted order determines the implied week indices.
        train_max = (
            train_df.groupby("region_id").apply(lambda g: len(g) - 1)
        )
        test_min  = train_max + 1   # assume test starts right after train

    gaps = {}
    for rid in train_max.index:
        if rid in test_min.index:
            gap = int(test_min[rid] - train_max[rid])
            gaps[rid] = max(gap, 1)   # clamp to minimum 1
        else:
            gaps[rid] = GAP_WEEKS     # fallback for missing test regions
    return gaps


# ---------------------------------------------------------------------------
# Helper – Walk-Forward Gap-Replay Fold builder  (v14)
# ---------------------------------------------------------------------------
def build_gap_replay_folds(
    df: pd.DataFrame,
    actual_gaps: dict,
):
    """
    v14 Relative Gap-Replay Walk-Forward CV.

    For each region:
      - Val_Y  = last 5 rows of historical train data.
      - Val_X  (length WINDOW_SIZE) ends exactly `actual_gap` weeks before
        Val_Y starts.
      - Train (X, Y) pairs are generated by sliding backwards from this
        validation cut-point; every pair enforces the same `actual_gap`
        distance between End_of_X and Start_of_Y.
      - Zero Data Waste Fallback: if the history is too short for
        `WINDOW_SIZE + actual_gap + HORIZON`, shrink the gap dynamically
        to the largest value that yields ≥1 pair.

    Returns
    -------
    folds : list of 1 tuple (train_groups, val_groups)
        train_groups / val_groups are lists of
        (group_df, i_min, i_max, effective_gap).

    Note: We return a single "fold" here (the Replay fold).
    To keep the 3-fold CV structure, we replicate the same fold 3 times
    with slight backward-shifted val windows (similar to v11).
    """
    folds = []

    # We still run 3 folds where each fold uses last 5, prev-5, prev-10 as val.
    # Val windows (from the end):
    #   fold 0: rows [-5:]        val_start_from_end=5
    #   fold 1: rows [-10:-5]     val_start_from_end=10
    #   fold 2: rows [-15:-10]    val_start_from_end=15
    for fold_k in range(WF_NUM_FOLDS):
        val_end_from_end   = fold_k * WF_FOLD_WEEKS            # 0,  5, 10
        val_start_from_end = val_end_from_end + WF_FOLD_WEEKS  # 5, 10, 15

        train_groups, val_groups = [], []

        for _, group in df.groupby("region_id"):
            group = group.reset_index(drop=True)
            n     = len(group)
            rid   = group["region_id"].iloc[0]

            # Retrieve per-region actual gap; fall back to global default
            actual_gap = actual_gaps.get(rid, GAP_WEEKS)

            # ----------------------------------------------------------------
            # Absolute indices of the validation window
            # ----------------------------------------------------------------
            val_start = n - val_start_from_end   # inclusive
            val_end   = n - val_end_from_end     # exclusive  (=n for fold 0)

            # ----------------------------------------------------------------
            # Zero Data Waste: shrink gap if needed
            # Minimum requirement: WINDOW_SIZE + actual_gap + HORIZON <= n
            #   → actual_gap <= n - WINDOW_SIZE - HORIZON
            # ----------------------------------------------------------------
            min_required = WINDOW_SIZE + actual_gap + HORIZON
            if min_required > n:
                max_feasible_gap = n - WINDOW_SIZE - HORIZON
                if max_feasible_gap < 1:
                    # Region has fewer rows than even WINDOW_SIZE+HORIZON+1
                    continue
                effective_gap = max_feasible_gap
            else:
                effective_gap = actual_gap

            # ----------------------------------------------------------------
            # Val split: Val_X ends `effective_gap` weeks before val_start,
            # Val_Y = rows [val_start : val_end]
            # Val_X window: rows [val_x_start : val_x_end]  len=WINDOW_SIZE
            # ----------------------------------------------------------------
            # Val_Y first row == val_start
            # End of Val_X must be: val_start - effective_gap
            val_x_end   = val_start - effective_gap   # exclusive; Val_X rows [.., val_x_end)
            val_x_start = val_x_end - WINDOW_SIZE     # inclusive

            if val_x_start < 0 or val_x_end < WINDOW_SIZE:
                # Not enough history for a valid val window
                continue

            # val_i is the single start index for the validation window
            val_i = val_x_start
            # Verify target fits within group
            val_tgt_start = val_x_end + effective_gap
            val_tgt_end   = val_tgt_start + HORIZON
            if val_tgt_end > n:
                continue

            val_groups.append((group, val_i, val_i, effective_gap))

            # ----------------------------------------------------------------
            # Train: sliding window backwards from the val cut-point.
            # Each (X, Y) pair: X=[i : i+WINDOW_SIZE], Y=[i+W+gap : i+W+gap+H]
            # Constraint: end of Y must be <= val_x_start
            #   i.e.  i + WINDOW_SIZE + effective_gap + HORIZON <= val_x_start
            #         i <= val_x_start - WINDOW_SIZE - effective_gap - HORIZON
            # ----------------------------------------------------------------
            train_i_max = val_x_start - WINDOW_SIZE - effective_gap - HORIZON
            if train_i_max >= 0:
                train_groups.append((group, 0, train_i_max, effective_gap))

        folds.append((train_groups, val_groups))

    return folds


# ---------------------------------------------------------------------------
# Backward compat alias  (kept so train.py imports still work in a mixed env)
# ---------------------------------------------------------------------------
def build_walk_forward_folds(df: pd.DataFrame):
    """
    Legacy stub that calls build_gap_replay_folds with a zero-gap dict
    (all regions get GAP_WEEKS=4).  Provided only for backward compatibility.
    New code should call build_gap_replay_folds() directly.
    """
    fallback_gaps = {rid: GAP_WEEKS for rid in df["region_id"].unique()}
    return build_gap_replay_folds(df, fallback_gaps)


# ---------------------------------------------------------------------------
# Helper – build full-train group list (train on ALL data for final model)
# ---------------------------------------------------------------------------
def build_full_train_groups(df: pd.DataFrame, actual_gaps: dict = None):
    """
    Build region groups using ALL rows (no held-out validation period).
    Used for final model training after walk-forward CV is complete.

    v14: accepts optional actual_gaps dict; falls back to GAP_WEEKS.
    """
    train_groups = []
    for _, group in df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n     = len(group)
        rid   = group["region_id"].iloc[0]

        if actual_gaps is not None:
            actual_gap = actual_gaps.get(rid, GAP_WEEKS)
        else:
            actual_gap = GAP_WEEKS

        # Zero Data Waste Fallback
        if n < WINDOW_SIZE + actual_gap + HORIZON:
            effective_gap = max(n - WINDOW_SIZE - HORIZON, 1)
        else:
            effective_gap = actual_gap

        train_i_max = n - WINDOW_SIZE - effective_gap - HORIZON
        if train_i_max >= 0:
            train_groups.append((group, 0, train_i_max, effective_gap))
    return train_groups
