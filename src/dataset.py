"""
src/dataset.py  — v40 Native PyTorch Sliding-Window Dataset
============================================================
* NO manual Z-score / MinMax / Ratio scaling — PatchTST RevIN handles it.
* DroughtDataset : sliding windows of (13, n_features) → X,  (5,) → Y  (train mode)
* TestDataset    : per-region 13-week windows → X  (inference mode)

Feature schema (n_features = 40, derived from exploratory Phase-0 audit):
  Excludes: region_id, week_idx, score, week_end_date
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Canonical feature column list  (n_features = 40)
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    # Precipitation
    "prec", "prec_week_max",
    # Surface pressure
    "surf_pre", "surf_pre_week_max",
    # Humidity
    "humidity", "humidity_week_max", "humidity_week_min", "humidity_week_std",
    # Temperature
    "tmp", "tmp_week_max", "tmp_week_min", "tmp_week_std",
    # Wind
    "wind", "wind_week_max", "wind_week_min", "wind_week_std",
    # Derived temperature / wind
    "tmp_max", "tmp_min", "tmp_range",
    "surf_tmp",
    "wind_max", "wind_min", "wind_range",
    # Seasonality encoding
    "week_sin", "week_cos", "day_ordinal",
    # Rolling aggregates
    "prec_roll_sum_4w", "tmp_roll_mean_4w", "humidity_roll_mean_4w",
    # Lag features
    "tmp_lag1w", "tmp_lag2w",
    "humidity_lag1w", "humidity_lag2w",
    "prec_lag1w", "prec_lag2w",
    "wind_lag1w", "wind_lag2w",
    # Evapotranspiration & deficit
    "pet", "deficit", "deficit_roll_cum_4w",
]

N_FEATURES: int = len(FEATURE_COLS)   # 40
CONTEXT_LEN: int = 13                 # historical input weeks
HORIZON_LEN: int = 5                  # target forecast weeks
WINDOW_LEN:  int = CONTEXT_LEN + HORIZON_LEN  # 18


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def refine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal DataFrame cleaning:
      - Forward-fill missing values (lag/rolling cols at series start)
      - Back-fill any remaining NaN
      - Zero-fill as last resort
    Does NOT apply any scalar normalization (RevIN in PatchTST handles it).
    """
    df = df.copy()
    df[FEATURE_COLS] = (
        df[FEATURE_COLS]
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    return df


def build_region_index(df: pd.DataFrame) -> dict[str, int]:
    """Return sorted mapping: region_id → integer index."""
    regions = sorted(df["region_id"].unique())
    return {rid: i for i, rid in enumerate(regions)}


# ---------------------------------------------------------------------------
# Training Dataset  (sliding window over historical time series)
# ---------------------------------------------------------------------------

class DroughtDataset(Dataset):
    """
    Sliding-window dataset for model training / validation.

    For each region the dataframe is sorted by `week_idx`.  Every valid
    contiguous segment of length WINDOW_LEN (18) produces one sample:

        x  : FloatTensor  (CONTEXT_LEN, N_FEATURES)  = first 13 weeks
        y  : FloatTensor  (HORIZON_LEN,)              = next 5 weeks of score
        rid: int                                       = region integer index

    Parameters
    ----------
    df            : pd.DataFrame   — full train_processed DataFrame
    region_index  : dict           — mapping {region_id: int}
    val_split     : bool           — if True, use ONLY the last val_windows windows
                                     per region; if False, use all preceding windows
    val_windows   : int            — number of trailing time-windows reserved for val
    """

    def __init__(
        self,
        df: pd.DataFrame,
        region_index: dict[str, int],
        val_split: bool = False,
        val_windows: int = 75,
    ) -> None:
        super().__init__()
        df = refine_features(df)

        # Pre-allocate storage for index tuples (region_idx, start_position_in_arr)
        self._index: list[tuple[int, int]] = []
        self._data: list[np.ndarray] = []   # per-region (T, N_FEATURES+1) arrays

        for rid, grp in df.groupby("region_id", sort=False):
            grp = grp.sort_values("week_idx").reset_index(drop=True)
            T = len(grp)
            if T < WINDOW_LEN:
                continue  # skip regions too short

            feat_arr = grp[FEATURE_COLS].values.astype(np.float32)  # (T, 40)
            score_arr = grp["score"].values.astype(np.float32)       # (T,)
            combined = np.concatenate(
                [feat_arr, score_arr[:, None]], axis=1
            )  # (T, 41)

            max_start = T - WINDOW_LEN      # last valid start index (inclusive)
            n_val_windows = min(val_windows, max_start + 1)
            cutoff = max_start - n_val_windows  # last TRAIN-only start

            if val_split:
                valid_starts = range(cutoff + 1, max_start + 1)
            else:
                valid_starts = range(0, cutoff + 1)

            rid_idx = region_index[rid]
            arr_idx = len(self._data)
            self._data.append(combined)

            for s in valid_starts:
                self._index.append((arr_idx, s, rid_idx))

    # -- Dataset interface ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        arr_idx, start, rid_idx = self._index[idx]
        combined = self._data[arr_idx]

        x = combined[start : start + CONTEXT_LEN, :N_FEATURES]          # (13, 40)
        y = combined[start + CONTEXT_LEN : start + WINDOW_LEN, N_FEATURES]  # (5,)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            rid_idx,
        )


# ---------------------------------------------------------------------------
# Test / Inference Dataset  (one sample = one region's 13-week block)
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    """
    Each test region has exactly CONTEXT_LEN (13) weeks of observations.
    Returns one sample per region for inference.

    Returns
    -------
    x   : FloatTensor  (CONTEXT_LEN, N_FEATURES)
    rid : int  (region integer index)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        region_index: dict[str, int],
    ) -> None:
        super().__init__()
        df = refine_features(df)

        self._X: list[np.ndarray] = []
        self._rid_indices: list[int] = []
        self._region_ids: list[str] = []

        for rid, grp in df.groupby("region_id", sort=False):
            grp = grp.sort_values("week_idx").reset_index(drop=True)
            feat = grp[FEATURE_COLS].values.astype(np.float32)  # (13, 40)

            # Pad / trim to exactly CONTEXT_LEN rows
            if len(feat) < CONTEXT_LEN:
                pad = np.zeros((CONTEXT_LEN - len(feat), N_FEATURES), dtype=np.float32)
                feat = np.concatenate([pad, feat], axis=0)
            else:
                feat = feat[:CONTEXT_LEN]

            self._X.append(feat)
            self._rid_indices.append(region_index.get(rid, 0))
            self._region_ids.append(rid)

    # -- Dataset interface ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._X[idx]),   # (13, 40)
            self._rid_indices[idx],
        )

    @property
    def ordered_region_ids(self) -> list[str]:
        """Region IDs in the same order as dataset samples."""
        return self._region_ids
