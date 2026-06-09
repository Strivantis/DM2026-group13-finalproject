"""
analyze_submissions.py – High-Resolution Submission Diagnostics

Provides decile statistics, physical drought interval bucketing,
and a visual KDE density plot to compare prediction distributions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    # "Baseline (v37)": os.path.join(ROOT, "submission_37th.csv"),
    # "V45 Raw": os.path.join(ROOT, "submission_45th.csv"),
    # "V45 Optimized": os.path.join(ROOT, "submission_45thopt.csv"),
    # "V45 L2-Stacked": os.path.join(ROOT, "submission_45thstacked.csv"),
    "V45.1 Raw Rounded": os.path.join(ROOT, "submission_45th_30k_ABraw_rounded.csv"),
    # "V45.1 30k Naked": os.path.join(ROOT, "submission_45th_30k_naked.csv"),
    # "V45.1 30k 25% Zero": os.path.join(ROOT, "submission_45th_30k_25perzero.csv"),
    # "V47 30k Pruned": os.path.join(ROOT, "submission_47th.csv"),
    # "V48 Tweedie": os.path.join(ROOT, "submission_48th.csv"),
    # "V48 Tweedie Snapped": os.path.join(ROOT, "submission_48th_snap.csv"),
    "V49 Multi": os.path.join(ROOT, "submission_49th.csv"),
    
    # "V49 Post-Argmax": os.path.join(ROOT, "submission_49th_A_arg.csv"),
    # "V49 Post-Optimized": os.path.join(ROOT, "submission_49th_B_opt.csv"),
    # "V49 Post-Stacked": os.path.join(ROOT, "submission_49th_C_stack.csv"),
    "V49 Multi Post-Meta": os.path.join(ROOT, "submission_49th_meta.csv"),
    "V50 (A w/o 0)": os.path.join(ROOT, "submission_50th.csv")
}

def get_interval_bins(vals):
    n = len(vals)
    return {
        "Exact 0.0": np.sum(vals == 0.0) / n,
        "0.0+ to 0.5": np.sum((vals > 0.0) & (vals <= 0.5)) / n,
        "0.5  to 1.0": np.sum((vals > 0.5) & (vals <= 1.0)) / n,
        "1.0  to 2.0": np.sum((vals > 1.0) & (vals <= 2.0)) / n,
        "2.0  to 3.0": np.sum((vals > 2.0) & (vals <= 3.0)) / n,
        "3.0  to 4.0": np.sum((vals > 3.0) & (vals <= 4.0)) / n,
        "4.0  to 5.0": np.sum((vals > 4.0) & (vals <= 5.0)) / n,
    }

def main():
    print("=" * 110)
    print(f"{'High-Resolution Submission Distribution Analytics':^110}")
    print("=" * 110)

    all_data = {}
    
    # 1. Print Percentiles (Deciles)
    print("\n[1] Percentile Breakdown (Where is the mass concentrated?)")
    header = f"  {'Model':<18} | {'Min':>5} | {'10%':>5} | {'25%':>5} | {'50%':>5} | {'75%':>5} | {'90%':>5} | {'95%':>5} | {'99%':>5} | {'Max':>5}"
    print(header)
    print("  " + "-" * 105)
    
    for name, path in FILES.items():
        if not os.path.exists(path):
            continue
        vals = pd.read_csv(path).iloc[:, 1:].values.flatten()
        all_data[name] = vals
        
        p = np.percentile(vals, [0, 10, 25, 50, 75, 90, 95, 99, 100])
        print(f"  {name:<18} | {p[0]:5.2f} | {p[1]:5.2f} | {p[2]:5.2f} | {p[3]:5.2f} | {p[4]:5.2f} | {p[5]:5.2f} | {p[6]:5.2f} | {p[7]:5.2f} | {p[8]:5.2f}")

    # 2. Print Physical Drought Bins
    print("\n[2] Physical Drought Intervals (Spotting the 'Hedging' effect)")
    bin_header = f"  {'Model':<18} | {'==0.0':>8} | {'0~0.5':>8} | {'0.5~1':>8} | {'1~2':>8} | {'2~3':>8} | {'3~4':>8} | {'4~5':>8}"
    print(bin_header)
    print("  " + "-" * 105)
    
    for name, vals in all_data.items():
        bins = get_interval_bins(vals)
        print(f"  {name:<18} | {bins['Exact 0.0']:7.1%} | {bins['0.0+ to 0.5']:7.1%} | {bins['0.5  to 1.0']:7.1%} | {bins['1.0  to 2.0']:7.1%} | {bins['2.0  to 3.0']:7.1%} | {bins['3.0  to 4.0']:7.1%} | {bins['4.0  to 5.0']:7.1%}")

    # 3. Generate Visual Density Plot
    print("\n[3] Generating Density Plot (KDE)...")
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    for name, vals in all_data.items():
        # Using KDE to smooth the distribution, but adding a rug or histogram for the exact zeros
        sns.kdeplot(vals, label=name, bw_adjust=0.5, linewidth=2, alpha=0.8)
        
    plt.title("Prediction Distribution Density (Notice the spikes near 0.0)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Drought Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(-0.2, 5.2)
    plt.legend(title="Submissions")
    plt.tight_layout()
    
    plot_path = os.path.join(ROOT, "submission_kde_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"    Saved -> {plot_path}")
    print("=" * 110)

if __name__ == "__main__":
    main()