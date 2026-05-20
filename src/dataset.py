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

v16 Changes (Preprocessing Overhaul – Sequence Grouping & 13-Week Bounds)
--------------------------------------------------------------------------
- WINDOW_SIZE reduced from 26 → 13 (hard cap at the test-set horizon).
  Using a window larger than the test horizon causes domain shift: the model
  never sees a 26-week context at inference time (test has only 13 weeks).
  Setting WINDOW_SIZE=13 ensures the inference context is always achievable.

- FEATURE_COLS updated: removed ALL 8-week and 13-week rolling features
  (prec_roll_sum_8w/13w, tmp_roll_mean_8w/13w, humidity_roll_mean_8w/13w,
  deficit_roll_cum_8w/13w) that cause training-vs-inference distribution shift.
  These features require 8-13 weeks of history to stabilise, but the model
  sees them fully-computed during training (782 weeks) and partially-computed
  (min_periods=1 truncation) at inference — a catastrophic domain shift.
  Only 4-week rolling features are kept: they stabilise from week 4 onward,
  which is achievable in both training and the 13-week test set.

- PREC_COLS updated accordingly (removed 8w/13w entries).

- compute_actual_gaps() updated to prefer the `day_ordinal` column (produced
  by v10 preprocess.py) over the legacy `week_key` string column.  Uses:
    gap_weeks = round((test_min_ordinal - train_max_ordinal) / 7)

  Total feature count: 29 (down from 37).

v15 Changes
-----------
- PARADIGM SHIFT 2: 3-Fold Walk-Forward CV abolished in favour of a
  Single-Fold data-maximizing strategy (build_single_fold).
  - Val_Y = last 5 weeks of each region's historical data (fixed, no rotation).
  - Train set uses ALL available sliding windows ending strictly before
    Val_Y (no data left on the table from folding).
  - Scaler is fit on the entire maximized training split.
- build_gap_replay_folds() and build_walk_forward_folds() are retained
  as backward-compat stubs but are no longer called by train.py.

v14 Changes
-----------
- PARADIGM SHIFT: Absolute CV replaced with "Relative Gap-Replay" CV.
  - actual_gap is computed per-region from the real calendar distance
    between the last train week and the first test week.
  - __getitem__ returns (X, y, target_time, gap_size).
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

# v11 legacy fallback gap (kept for import compat; unused in SingleFold path)
GAP_WEEKS = 4

# Drop collinear temp proxies (keep tmp, tmp_max, tmp_min, tmp_range)
DROP_COLS = ["wb_tmp", "dp_tmp", "surf_tmp"]

# Precipitation columns to log1p-transform (handle right-skew)
# v16: removed prec_roll_sum_8w, prec_roll_sum_13w (features deleted)
PREC_COLS = [
    "prec",
    "prec_roll_sum_4w",
    "prec_lag1w", "prec_lag2w",
]

# Feature columns fed to the model (order matters for scaler alignment)
# v16: removed ALL 8w and 13w rolling features; WINDOW_SIZE capped at 13.
# Count breakdown:
#   base weather (11) + cyclic calendar (2) + rolling 4w (3)
#   + lag-1 (4) + lag-2 (4) + drought 4w (3) + target encoding (2) = 29
FEATURE_COLS = [
    # --- base weather (11) ---
    "prec", "surf_pre", "humidity",
    "tmp", "tmp_max", "tmp_min", "tmp_range",
    "wind", "wind_max", "wind_min", "wind_range",
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
]   # total = 29 features


# ---------------------------------------------------------------------------
# Feature refinement
# ---------------------------------------------------------------------------
def refine_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    1. Compute Drought Index (PET/deficit 4w) BEFORE log1p (uses raw prec).
    2. Drop collinear temperature proxies (wb_tmp, dp_tmp, surf_tmp).
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
# Region-aware sliding-window dataset  (v14+: Gap-Replay)
# ---------------------------------------------------------------------------
class DroughtDataset(Dataset):
    """
    v14 Gap-Replay Dataset
    ----------------------
    Parameters
    ----------
    region_groups : list of (group_df, i_min, i_max, actual_gap)
        Each entry defines one region's data, the [inclusive] range of
        valid sequence start indices, and the actual deployment gap.
    window  : look-back length (weeks) — v16: default=13
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
        target_times = []   # (H, 2) week_sin/cos for each target week
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

                # target_time: week_sin/cos of the H target weeks -> (H, 2)
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

    Priority:
    1. day_ordinal column (v10 preprocessed data): precise calendar-based gap.
       gap_days  = test_min_ordinal - train_max_ordinal
       gap_weeks = max(round(gap_days / 7), 1)

    2. week_key column (v9 legacy): string-based week key conversion.

    3. Fallback: assume test starts immediately after train (gap = 1 week).

    Returns
    -------
    gaps : dict  region_id -> int  (number of weeks gap, >= 1)
    """
    # ------------------------------------------------------------------
    # Path 1: day_ordinal (v10+ preprocess)
    # ------------------------------------------------------------------
    if "day_ordinal" in train_df.columns and "day_ordinal" in test_df.columns:
        train_max = train_df.groupby("region_id")["day_ordinal"].max()
        test_min  = test_df.groupby("region_id")["day_ordinal"].min()

        gaps = {}
        for rid in train_max.index:
            if rid in test_min.index:
                gap_days  = float(test_min[rid]) - float(train_max[rid])
                gap_weeks = max(int(round(gap_days / 7)), 1)
                gaps[rid] = gap_weeks
            else:
                gaps[rid] = GAP_WEEKS
        return gaps

    # ------------------------------------------------------------------
    # Path 2: week_key (v9 legacy)
    # ------------------------------------------------------------------
    if "week_key" in train_df.columns and "week_key" in test_df.columns:
        def _abs_week(df):
            def parse(wk):
                if pd.isna(wk):
                    return np.nan
                parts = str(wk).split("-W")
                year = int(parts[0]) if parts[0].isdigit() else int(parts[0][-6:])
                week = int(parts[1])
                return year * 53 + week
            return df["week_key"].apply(parse)

        tr_abs = _abs_week(train_df)
        te_abs = _abs_week(test_df)

        train_max = train_df.assign(_aw=tr_abs).groupby("region_id")["_aw"].max()
        test_min  = test_df.assign(_aw=te_abs).groupby("region_id")["_aw"].min()

        gaps = {}
        for rid in train_max.index:
            if rid in test_min.index:
                gap = int(test_min[rid] - train_max[rid])
                gaps[rid] = max(gap, 1)
            else:
                gaps[rid] = GAP_WEEKS
        return gaps

    # ------------------------------------------------------------------
    # Path 3: Positional fallback (gap = 1 week)
    # ------------------------------------------------------------------
    train_max = train_df.groupby("region_id").apply(lambda g: len(g) - 1)
    test_min  = train_max + 1   # assume test starts immediately after train

    gaps = {}
    for rid in train_max.index:
        if rid in test_min.index:
            gap = int(test_min[rid] - train_max[rid])
            gaps[rid] = max(gap, 1)
        else:
            gaps[rid] = GAP_WEEKS
    return gaps


# ---------------------------------------------------------------------------
# v15: Single-Fold Data-Maximizing Split builder
# ---------------------------------------------------------------------------
def build_single_fold(
    df: pd.DataFrame,
    actual_gaps: dict,
):
    """
    v15 Single-Fold Data-Maximizing CV.

    For each region:
      - Val_Y  = last 5 rows (HORIZON) of historical train data (fixed).
      - Val_X  (length WINDOW_SIZE) ends exactly `actual_gap` weeks before
        Val_Y starts.
      - Train (X, Y) pairs use ALL available sliding windows; each pair
        enforces the same `actual_gap` distance between End_of_X and
        Start_of_Y, and the target window must end strictly before Val_Y
        begins (no leakage into validation).
      - Zero Data Waste Fallback: if history is too short for
        `WINDOW_SIZE + actual_gap + HORIZON`, shrink the gap dynamically
        to the largest value that yields ≥1 pair.

    v16: WINDOW_SIZE=13 allows more training samples per region.

    Returns
    -------
    train_groups : list of (group_df, i_min, i_max, effective_gap)
    val_groups   : list of (group_df, val_i, val_i, effective_gap)
    """
    train_groups = []
    val_groups   = []

    for _, group in df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n     = len(group)
        rid   = group["region_id"].iloc[0]

        # Retrieve per-region actual gap; fall back to global default
        actual_gap = actual_gaps.get(rid, GAP_WEEKS)

        # Val_Y = last HORIZON rows: indices [n-HORIZON : n]
        val_y_start = n - HORIZON  # inclusive index of first val target row

        # --------------------------------------------------------------------
        # Zero Data Waste: shrink gap if needed
        # --------------------------------------------------------------------
        min_required = WINDOW_SIZE + actual_gap + HORIZON
        if min_required > n:
            max_feasible_gap = n - WINDOW_SIZE - HORIZON
            if max_feasible_gap < 1:
                continue
            effective_gap = max_feasible_gap
        else:
            effective_gap = actual_gap

        # --------------------------------------------------------------------
        # Val split: Val_X ends `effective_gap` weeks before val_y_start
        # --------------------------------------------------------------------
        val_x_end   = val_y_start - effective_gap
        val_x_start = val_x_end - WINDOW_SIZE

        if val_x_start < 0 or val_x_end < WINDOW_SIZE:
            continue

        val_tgt_start = val_x_end + effective_gap   # == val_y_start
        val_tgt_end   = val_tgt_start + HORIZON
        if val_tgt_end > n:
            continue

        val_groups.append((group, val_x_start, val_x_start, effective_gap))

        # --------------------------------------------------------------------
        # Train: ALL sliding windows with target strictly before val_y_start.
        # i + WINDOW_SIZE + effective_gap + HORIZON <= val_y_start
        # --------------------------------------------------------------------
        train_i_max = val_y_start - WINDOW_SIZE - effective_gap - HORIZON
        if train_i_max >= 0:
            train_groups.append((group, 0, train_i_max, effective_gap))

    return train_groups, val_groups


# ---------------------------------------------------------------------------
# Helper – Walk-Forward Gap-Replay Fold builder  (v14 – retained as stub)
# ---------------------------------------------------------------------------
def build_gap_replay_folds(
    df: pd.DataFrame,
    actual_gaps: dict,
):
    """
    v14 Relative Gap-Replay Walk-Forward CV.
    Retained for backward compatibility. New code uses build_single_fold().
    """
    folds = []

    for fold_k in range(WF_NUM_FOLDS):
        val_end_from_end   = fold_k * WF_FOLD_WEEKS
        val_start_from_end = val_end_from_end + WF_FOLD_WEEKS

        train_groups, val_groups = [], []

        for _, group in df.groupby("region_id"):
            group = group.reset_index(drop=True)
            n     = len(group)
            rid   = group["region_id"].iloc[0]

            actual_gap = actual_gaps.get(rid, GAP_WEEKS)

            val_start = n - val_start_from_end
            val_end   = n - val_end_from_end

            min_required = WINDOW_SIZE + actual_gap + HORIZON
            if min_required > n:
                max_feasible_gap = n - WINDOW_SIZE - HORIZON
                if max_feasible_gap < 1:
                    continue
                effective_gap = max_feasible_gap
            else:
                effective_gap = actual_gap

            val_x_end   = val_start - effective_gap
            val_x_start = val_x_end - WINDOW_SIZE

            if val_x_start < 0 or val_x_end < WINDOW_SIZE:
                continue

            val_i = val_x_start
            val_tgt_start = val_x_end + effective_gap
            val_tgt_end   = val_tgt_start + HORIZON
            if val_tgt_end > n:
                continue

            val_groups.append((group, val_i, val_i, effective_gap))

            train_i_max = val_x_start - WINDOW_SIZE - effective_gap - HORIZON
            if train_i_max >= 0:
                train_groups.append((group, 0, train_i_max, effective_gap))

        folds.append((train_groups, val_groups))

    return folds


# ---------------------------------------------------------------------------
# Backward compat alias
# ---------------------------------------------------------------------------
def build_walk_forward_folds(df: pd.DataFrame):
    """
    Legacy stub. New code uses build_single_fold().
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
    v16: WINDOW_SIZE=13, so more samples per region.
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
