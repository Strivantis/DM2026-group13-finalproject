"""
preprocess.py – Proxy Feature Engineering Utilities
====================================================
Provides Unsupervised Drought Index (simplified SPEI-like) features:
  1. PET  = max(0.55 * tmp, 0)          (Hamon simplified PET)
  2. deficit = prec - PET               (moisture balance)
  3. Rolling cumulative sums of deficit: 4w, 8w, 13w
     – computed **within each region_id** to prevent leakage
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Drought Index Feature Engineering
# ---------------------------------------------------------------------------
DROUGHT_ROLL_WINDOWS = [4, 8, 13]   # weeks

DROUGHT_FEAT_COLS = [
    "pet",
    "deficit",
    "deficit_roll_cum_4w",
    "deficit_roll_cum_8w",
    "deficit_roll_cum_13w",
]


def add_drought_index(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Compute PET-based drought proxy features and append them to ``df``.

    Steps
    -----
    1. PET (simplified Hamon): PET = max(0.55 * tmp, 0)
    2. Deficit = prec - PET    (log1p-transformed prec must NOT be used here;
       we work on the original column before log1p is applied in refine_features)
    3. Rolling cumulative sum of deficit over 4, 8, 13 weeks,
       strictly grouped by region_id.

    Parameters
    ----------
    df       : DataFrame **before** log1p transformation of precipitation.
    is_train : If True, NaN rows created by rolling are left as-is (they will
               be dropped by refine_features).  If False, forward-fill within
               region to keep all rows.

    Returns
    -------
    df with new columns appended in-place on a copy.
    """
    df = df.copy()

    # --- 1. PET (Hamon simplified) ----------------------------------------
    # tmp column holds weekly mean temperature (°C).
    # Hargreaves-inspired proxy: PET ≈ 0.55 * max(T, 0)
    df["pet"] = (0.55 * df["tmp"].clip(lower=0.0)).astype(np.float32)

    # --- 2. Moisture deficit ------------------------------------------------
    # Use raw prec (before log1p) for physical meaning.
    # prec might already be log1p'd if called after refine_features – guard:
    raw_prec = df["prec"]   # caller is responsible for ordering
    df["deficit"] = (raw_prec - df["pet"]).astype(np.float32)

    # --- 3. Rolling cumulative sums by region, causal (no look-ahead) ------
    for w in DROUGHT_ROLL_WINDOWS:
        col = f"deficit_roll_cum_{w}w"
        df[col] = (
            df.groupby("region_id")["deficit"]
            .transform(lambda s: s.rolling(window=w, min_periods=1).sum())
        ).astype(np.float32)

    # --- 4. Handle NaN for test set -----------------------------------------
    if not is_train:
        for col in DROUGHT_FEAT_COLS:
            if col in df.columns:
                df[col] = (
                    df.groupby("region_id")[col]
                    .transform(lambda s: s.ffill().fillna(0))
                )

    return df
