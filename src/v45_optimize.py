"""
optimize.py – Post-Training Threshold Optimization and Distribution Analysis

Execution:
    python src/optimize.py

Inputs:
    data/processed/train_processed.csv
    data/processed/test_processed.csv
    data/processed/region_stats.csv
    models/lgbm_a_fold{k}_week{w}.pkl
    models/lgbm_b_fold{k}_week{w}.pkl
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_stratified_group_cv_folds,
    build_tabular_dataset,
    build_tabular_test,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
)
from src.train import (
    _compute_te_stats,
    _augment_groups_with_te,
    _merge_te_to_df,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")
N_FOLDS       = 5

# Baseline files for post-optimization comparison
SUB_45TH_PATH = os.path.join(ROOT, "submission_45th.csv")
SUB_37TH_PATH = os.path.join(ROOT, "submission_37th.csv")
SUB_OPT_PATH  = os.path.join(ROOT, "submission_45thopt.csv")


# ---------------------------------------------------------------------------
# Core Optimization Routine (Grid Search)
# ---------------------------------------------------------------------------
def optimize_threshold(probs: np.ndarray, preds: np.ndarray, y_true: np.ndarray) -> tuple:
    """
    Exhaustive 1D grid search for the optimal zero-gating threshold T in [0.01, 0.99].
    Minimizes Mean Absolute Error (MAE).
    """
    best_t = 0.5
    best_mae = float('inf')
    
    # Grid search from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        gated_preds = np.where(probs < t, 0.0, preds)
        mae = float(np.mean(np.abs(gated_preds - y_true)))
        if mae < best_mae:
            best_mae = mae
            best_t = t
            
    return best_t, best_mae


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    
    print("=" * 80)
    print("Post-Training Optimizer  (Threshold Tuning & Distribution Analysis)")
    print("=" * 80)

    # 1. Load Data & Cluster Mappings
    print("\n[1] Loading processed datasets and cluster mapping...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    region_stats = pd.read_csv(os.path.join(PROCESSED_DIR, "region_stats.csv"))
    
    cluster_map = dict(zip(region_stats["region_id"], region_stats["cluster_id"]))
    
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)

    # 2. Reconstruct Folds
    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    
    oof_records = []
    test_preds_a = []
    test_preds_b = []

    print("\n[2] Reconstructing OOF Predictions via pre-trained models...")
    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):
        # TE Leakage-Free Reconstruction
        train_region_ids_fold = {entry[0]["region_id"].iloc[0] for entry in raw_train_groups}
        train_df_fold_regions = train_df[train_df["region_id"].isin(train_region_ids_fold)]
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(train_df_fold_regions)
        
        aug_val_groups = _augment_groups_with_te(raw_val_groups, te_map_fold, gm_fold, gzp_fold)
        feat_cols = [c for c in FEATURE_COLS if c in aug_val_groups[0][0].columns]
        
        # Build Validation Matrices
        X_val_np, y_val_np, val_region_ids = build_tabular_dataset(aug_val_groups, feat_cols)
        X_val_np = np.asfortranarray(X_val_np)
        
        # Build Test Matrices
        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test_np, test_region_ids = build_tabular_test(test_df_fold, feat_cols)
        X_test_np = np.asfortranarray(X_test_np)
        
        fold_test_pred_l1 = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)
        fold_test_prob    = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)
        
        for week_idx in range(HORIZON):
            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            ckpt_b = os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl")
            
            with open(ckpt_a, "rb") as fh: model_a = pickle.load(fh)
            with open(ckpt_b, "rb") as fh: model_b = pickle.load(fh)
                
            val_l1_w   = model_a.predict(X_val_np)
            val_prob_w = model_b.predict_proba(X_val_np)[:, 1]
            y_val_w    = y_val_np[:, week_idx]
            
            fold_test_pred_l1[:, week_idx] = model_a.predict(X_test_np)
            fold_test_prob[:, week_idx]    = model_b.predict_proba(X_test_np)[:, 1]
            
            # Store OOF granular data
            for i in range(len(val_region_ids)):
                oof_records.append({
                    "region_id": val_region_ids[i],
                    "cluster_id": cluster_map.get(val_region_ids[i], 0),
                    "fold": fold_k,
                    "week": week_idx,
                    "y_true": y_val_w[i],
                    "prob": val_prob_w[i],
                    "pred_l1": val_l1_w[i]
                })

        test_preds_a.append({"preds": fold_test_pred_l1, "region_ids": test_region_ids})
        test_preds_b.append({"probs": fold_test_prob, "region_ids": test_region_ids})

    oof_df = pd.DataFrame(oof_records)
    
    # 3. Optimization Topologies
    print("\n[3] Threshold Optimization Results")
    
    # Baseline (Static 0.5)
    oof_df["pred_baseline"] = np.where(oof_df["prob"] < 0.5, 0.0, oof_df["pred_l1"])
    baseline_mae = np.mean(np.abs(oof_df["pred_baseline"] - oof_df["y_true"]))
    print(f"  [Baseline] Static Threshold 0.50  --> MAE: {baseline_mae:.5f}")
    
    # Global Optimization
    global_t, global_mae = optimize_threshold(oof_df["prob"].values, oof_df["pred_l1"].values, oof_df["y_true"].values)
    print(f"  [Global]   Optimal Threshold {global_t:.2f}  --> MAE: {global_mae:.5f}")
    
    # Per-Week Optimization
    week_thresholds = {}
    oof_df["pred_per_week"] = oof_df["pred_l1"].copy()
    for w in range(HORIZON):
        w_df = oof_df[oof_df["week"] == w]
        t_w, _ = optimize_threshold(w_df["prob"].values, w_df["pred_l1"].values, w_df["y_true"].values)
        week_thresholds[w] = t_w
        mask = (oof_df["week"] == w) & (oof_df["prob"] < t_w)
        oof_df.loc[mask, "pred_per_week"] = 0.0
        
    pw_mae = np.mean(np.abs(oof_df["pred_per_week"] - oof_df["y_true"]))
    print(f"  [Per-Week] MAE: {pw_mae:.5f}")
    for w, t in week_thresholds.items():
        print(f"             Week {w+1} Threshold: {t:.2f}")

    # Per-Cluster Optimization
    cluster_thresholds = {}
    oof_df["pred_per_cluster"] = oof_df["pred_l1"].copy()
    clusters = sorted(oof_df["cluster_id"].unique())
    for c in clusters:
        c_df = oof_df[oof_df["cluster_id"] == c]
        if len(c_df) > 0:
            t_c, _ = optimize_threshold(c_df["prob"].values, c_df["pred_l1"].values, c_df["y_true"].values)
            cluster_thresholds[c] = t_c
            mask = (oof_df["cluster_id"] == c) & (oof_df["prob"] < t_c)
            oof_df.loc[mask, "pred_per_cluster"] = 0.0

    pc_mae = np.mean(np.abs(oof_df["pred_per_cluster"] - oof_df["y_true"]))
    print(f"  [Per-Cluster] MAE: {pc_mae:.5f}")
    
    # 4. Generate Optimized Submission (Using best strategy, likely Per-Cluster)
    print("\n[4] Generating Optimized Test Submission (Per-Cluster Topology)")
    preds_a_stack = np.stack([fp["preds"] for fp in test_preds_a], axis=0)
    probs_b_stack = np.stack([fp["probs"] for fp in test_preds_b], axis=0)
    
    l1_median = np.median(preds_a_stack, axis=0)
    prob_mean = np.mean(probs_b_stack, axis=0)
    
    final_preds = np.copy(l1_median)
    for i, rid in enumerate(test_preds_a[0]["region_ids"]):
        cid = cluster_map.get(rid, 0)
        t_c = cluster_thresholds.get(cid, 0.5)
        for w in range(HORIZON):
            if prob_mean[i, w] < t_c:
                final_preds[i, w] = 0.0
                
    final_preds = np.clip(final_preds, 0.0, 5.0)
    
    rows = []
    for i, rid in enumerate(test_preds_a[0]["region_ids"]):
        rows.append({
            "region_id": rid,
            "pred_week1": final_preds[i, 0],
            "pred_week2": final_preds[i, 1],
            "pred_week3": final_preds[i, 2],
            "pred_week4": final_preds[i, 3],
            "pred_week5": final_preds[i, 4],
        })
    sub_opt = pd.DataFrame(rows)
    sub_opt.to_csv(SUB_OPT_PATH, index=False)
    
    # 5. Distribution Diagnostics & Baseline Comparison
    print("\n[5] Submission Distribution Diagnostics")
    
    def _print_stats(name, df_sub):
        if df_sub is None:
            print(f"  {name:<15} | File not found")
            return
        vals = df_sub.iloc[:, 1:].values.flatten()
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        z_frac = np.mean(vals == 0.0)
        p50 = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        p99 = np.percentile(vals, 99)
        print(f"  {name:<15} | Mean: {mean_v:.4f} | Std: {std_v:.4f} | Zero: {z_frac:.1%} | P50: {p50:.2f} | P90: {p90:.2f} | P99: {p99:.2f}")

    sub_37th = pd.read_csv(SUB_37TH_PATH) if os.path.exists(SUB_37TH_PATH) else None
    sub_45th = pd.read_csv(SUB_45TH_PATH) if os.path.exists(SUB_45TH_PATH) else None
    
    print("  " + "-" * 105)
    _print_stats("Kaggle Baseline", sub_37th)
    _print_stats("V45 Raw (T=0.5)", sub_45th)
    _print_stats("V45 Optimized", sub_opt)
    print("  " + "-" * 105)

    elapsed = time.time() - t0
    print(f"\nOptimization completed in {elapsed:.1f}s. Saved -> {SUB_OPT_PATH}")

if __name__ == "__main__":
    main()