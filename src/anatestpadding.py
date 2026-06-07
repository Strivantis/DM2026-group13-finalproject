"""
analyze_padding.py - Rigorous Train vs Test Distribution Analysis
"""

import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# The REAL ghosts that are actually in dataset.py's FEATURE_COLS
REAL_GHOSTS = [
    "tmp_week_std",
    "surf_pre_week_max",
    "humidity_week_std",
    "prec_week_max",
    "surf_pre",
    "tmp" # We add tmp to see if absolute temperature shifted
]

def main():
    print("=" * 80)
    print("Train vs Test Feature Distribution Analysis (The REAL Ghosts)")
    print("=" * 80)

    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))

    # Just analyzing the overlapping columns
    cols = [c for c in REAL_GHOSTS if c in train.columns and c in test.columns]

    header = f"| {'Feature':<20} | {'Set':<5} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Zero %':>7} |"
    print(header)
    print("-" * len(header))

    for c in cols:
        for name, df in [("Train", train), ("Test", test)]:
            vals = df[c].dropna().values
            mean_v = np.mean(vals)
            std_v  = np.std(vals)
            min_v  = np.min(vals)
            max_v  = np.max(vals)
            zero_p = np.mean(vals == 0.0) * 100
            
            print(f"| {c:<20} | {name:<5} | {mean_v:8.2f} | {std_v:8.2f} | {min_v:8.2f} | {max_v:8.2f} | {zero_p:6.1f}% |")
        print("-" * len(header))

if __name__ == "__main__":
    main()