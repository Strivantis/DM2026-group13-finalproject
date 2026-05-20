"""
Phase 1 Data Verification  (v16 preprocessing overhaul)
========================================================
Skeptically verify user's claims about train.csv & test.csv.
Strategy: read ONLY the key columns (region_id, date, score) to avoid
loading all 12 M rows of meteorological data into memory.
We then group by region_id, verify day-counts, and check score cadence.
"""

import pandas as pd
import numpy as np
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_BASE, "data")

print("=" * 70)
print("Phase 1: Data Verification")
print("=" * 70)

# =========================================================================
# TRAIN
# =========================================================================
print("\n--- TRAIN.CSV (reading region_id + date + score only) ---")
train_path = os.path.join(_DATA, "train.csv")

train_cols = ["region_id", "date", "score"]
train_slim = pd.read_csv(train_path, usecols=train_cols)
print(f"  Loaded slim train shape : {train_slim.shape}")

# --- Region count
n_train_regions = train_slim["region_id"].nunique()
print(f"  Unique region_id       : {n_train_regions}")

# --- Row count per region
train_counts = train_slim.groupby("region_id").size()
print(f"  Rows per region - min  : {train_counts.min()}")
print(f"  Rows per region - max  : {train_counts.max()}")
print(f"  Rows per region - mode : {int(train_counts.mode().iloc[0])}")

# Pick 3 regions for detailed inspection
sample_rids = train_slim["region_id"].unique()[:3].tolist()

for rid in sample_rids:
    grp = (
        train_slim[train_slim["region_id"] == rid]
        .sort_values("date")
        .reset_index(drop=True)
    )
    n_days = len(grp)
    print(f"\n  region_id = {rid}")
    print(f"    Total rows   : {n_days}")
    print(f"    Date range   : {grp['date'].iloc[0]}  ->  {grp['date'].iloc[-1]}")

    # Score non-NaN positions
    score_col = grp["score"]
    non_nan_idx = score_col.dropna().index.tolist()
    n_scored = len(non_nan_idx)
    print(f"    Non-NaN score count : {n_scored}")

    idx_arr = np.array(non_nan_idx)
    if len(idx_arr) >= 2:
        strides     = np.diff(idx_arr)
        bad_strides = strides[strides != 7]
        first_idx   = idx_arr[0]
        print(f"    First non-NaN index : {first_idx}  (day-7-of-week pattern => expect 6)")
        print(f"    Stride violations   : {len(bad_strides)}  (expected 0)")
        print(f"    Unique strides seen : {np.unique(strides).tolist()}")
        if len(bad_strides) == 0:
            print(f"    OK Score stride = 7 (consistent 7-day cadence)")
        else:
            print(f"    FAIL Score stride INCONSISTENT -- sample bad={bad_strides[:5]}")
    else:
        print(f"    (not enough scored rows to check stride)")

    # Verify 7-divisibility at n_days level
    complete_weeks = n_days // 7
    leftover       = n_days %  7
    print(f"    Complete 7-day weeks : {complete_weeks}  (leftover days = {leftover})")

# =========================================================================
# TEST
# =========================================================================
print("\n--- TEST.CSV (reading region_id + date columns only) ---")
test_path = os.path.join(_DATA, "test.csv")

# detect if test has a score column
_header = pd.read_csv(test_path, nrows=1)
test_cols = ["region_id", "date"]
if "score" in _header.columns:
    test_cols.append("score")

test_slim = pd.read_csv(test_path, usecols=test_cols)
print(f"  Loaded slim test shape  : {test_slim.shape}")

n_test_regions = test_slim["region_id"].nunique()
print(f"  Unique region_id        : {n_test_regions}")

test_counts = test_slim.groupby("region_id").size()
print(f"  Rows per region - min   : {test_counts.min()}")
print(f"  Rows per region - max   : {test_counts.max()}")
print(f"  Rows per region - mode  : {int(test_counts.mode().iloc[0])}")

sample_rids_t = test_slim["region_id"].unique()[:3].tolist()
for rid in sample_rids_t:
    grp = (
        test_slim[test_slim["region_id"] == rid]
        .sort_values("date")
        .reset_index(drop=True)
    )
    n_days = len(grp)
    complete_weeks = n_days // 7
    leftover       = n_days %  7
    print(f"\n  region_id = {rid}")
    print(f"    Total rows   : {n_days}")
    print(f"    Date range   : {grp['date'].iloc[0]}  ->  {grp['date'].iloc[-1]}")
    print(f"    Complete 7-day weeks : {complete_weeks}  (leftover = {leftover})")
    if n_days == 91:
        print(f"    OK 91-day (13-week) check PASSED")
    else:
        print(f"    UNEXPECTED day count = {n_days}")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print(f"  Train regions   : {n_train_regions}")
print(f"  Train days/rgn  : min={train_counts.min()}  max={train_counts.max()}")
print(f"  Test  regions   : {n_test_regions}")
print(f"  Test  days/rgn  : min={test_counts.min()}   max={test_counts.max()}")

# Derived constants for preprocess
actual_days  = int(train_counts.mode().iloc[0])
complete_wks = actual_days // 7
keep_days    = complete_wks * 7
leftover     = actual_days % 7
test_days    = int(test_counts.mode().iloc[0])
test_wks     = test_days // 7
print(f"\n  Derived constants for preprocess.py:")
print(f"    actual_days per region  = {actual_days}")
print(f"    complete 7-day weeks    = {complete_wks}  "
      f"(keep first {keep_days} days, drop {leftover} leftover)")
print(f"    test days per region    = {test_days}  ->  {test_wks} weeks")
print("=" * 70)
