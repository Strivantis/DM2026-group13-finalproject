"""
DroughtDataset
==============
Custom PyTorch Dataset for drought score multi-step forecasting.

Design
------
- Sliding window (step=1) of W=13 weekly rows as input (X).
- Multi-step target: the next H=5 weekly `score` values (Y).
- Region boundaries are strictly respected – windows NEVER cross regions.
- Feature pruning and log1p precipitation transform are applied here.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 13   # look-back (weeks)
HORIZON = 5        # forecast horizon (weeks)
VAL_WEEKS = 10     # hold-out weeks per region for validation

# Drop collinear temp proxies (keep tmp, tmp_max, tmp_min, tmp_range)
DROP_COLS = ["wb_tmp", "dp_tmp", "surf_tmp"]

# Precipitation columns to log1p-transform (handle right-skew)
PREC_COLS = [
    "prec",
    "prec_roll_sum_4w", "prec_roll_sum_8w", "prec_roll_sum_13w",
    "prec_lag1w", "prec_lag2w",
]

# Feature columns fed to the model (order matters for scaler alignment)
FEATURE_COLS = [
    # --- base weather (11) ---
    "prec", "surf_pre", "humidity",
    "tmp", "tmp_max", "tmp_min", "tmp_range",
    "wind", "wind_max", "wind_min", "wind_range",
    # --- calendar (2) ---
    "month", "week_of_year",
    # --- rolling aggregates (9) ---
    "prec_roll_sum_4w",  "tmp_roll_mean_4w",  "humidity_roll_mean_4w",
    "prec_roll_sum_8w",  "tmp_roll_mean_8w",  "humidity_roll_mean_8w",
    "prec_roll_sum_13w", "tmp_roll_mean_13w", "humidity_roll_mean_13w",
    # --- lag-1 (4) ---
    "tmp_lag1w", "humidity_lag1w", "prec_lag1w", "wind_lag1w",
    # --- lag-2 (4) ---
    "tmp_lag2w", "humidity_lag2w", "prec_lag2w", "wind_lag2w",
]   # total = 30 features


# ---------------------------------------------------------------------------
# Feature refinement
# ---------------------------------------------------------------------------
def refine_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    1. Drop collinear temperature proxies.
    2. Apply log1p to all precipitation-related columns.
    3. Handle NaN rows:
       - Train: drop rows where any FEATURE_COL is NaN (lag head).
       - Test:  forward-fill then zero-fill lag NaN so we keep all rows.

    Returns a clean copy of df.
    """
    df = df.copy()

    # Step 1 – drop collinear columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Step 2 – log1p precipitation (clip negatives to 0 first)
    for col in PREC_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    present_feats = [c for c in FEATURE_COLS if c in df.columns]

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
# Helper – build train / val region group lists
# ---------------------------------------------------------------------------
def build_train_val_groups(df: pd.DataFrame):
    """
    Split each region's timeline into train and validation index ranges.

    Validation: the last VAL_WEEKS rows of each region.
    Train:      everything before val, such that NO target window overlaps
                with the validation period.

    Guarantee:
      train sequence i → targets at rows [i+W, i+W+H)
      i + W + H - 1 < val_start_row   ⟹   i ≤ val_start_row - W - H

    Returns (train_groups, val_groups), each a list of (group_df, i_min, i_max).
    """
    train_groups, val_groups = [], []

    for _, group in df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        val_start = n - VAL_WEEKS

        # --- train ---
        train_i_max = val_start - WINDOW_SIZE - HORIZON
        if train_i_max >= 0:
            train_groups.append((group, 0, train_i_max))

        # --- val ---
        # sequences where the FIRST target row >= val_start
        val_i_min = val_start - WINDOW_SIZE
        val_i_max = n - WINDOW_SIZE - HORIZON
        if val_i_min >= 0 and val_i_min <= val_i_max:
            val_groups.append((group, val_i_min, val_i_max))

    return train_groups, val_groups
