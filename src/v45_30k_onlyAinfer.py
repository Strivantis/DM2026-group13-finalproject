"""
v45_30k_infer.py - Flawless Inference Script for "Naked" Model A (30k Trees)

Strategy:
  1. No Model B, No Stacking, No Historical Priors.
  2. Perfect reconstruction of Fold-local Target Encoding to ensure exact feature match.
  3. 5-Fold Median Ensembling for Robustness.
  4. Micro-Dust Filter: np.where(pred < 0.05, 0.0, pred) to clean L1 regression noise.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

# Ensure we can import from src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_stratified_group_cv_folds,
    build_tabular_test,
    FEATURE_COLS,
    HORIZON
)
from src.train import (
    _compute_te_stats,
    _merge_te_to_df,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")
SUB_OUTPUT    = os.path.join(ROOT, "submission_45th_30k_25perzero.csv")

N_FOLDS       = 5
MICRO_DUST_TH = 0.05   # The "Clean Zero" threshold


def main():
    t0 = time.time()
    print("=" * 85)
    print("Flawless Inference: 30k 'Naked' Model A (Pure L1 Conditional Median)")
    print("=" * 85)

    # 1. Load Data
    print("\n[1] Loading Processed Data...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)

    # 2. Reconstruct Folds for Exact TE Match
    print("\n[2] Reconstructing Folds to guarantee bit-exact Target Encoding...")
    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    
    # Store predictions: shape (5_folds, 2248_regions, 5_weeks)
    n_test_regions = test_df["region_id"].nunique()
    all_fold_preds = np.zeros((N_FOLDS, n_test_regions, HORIZON), dtype=np.float32)
    master_region_order = None

    # 3. Inference Loop
    print("\n[3] Executing 30k Trees Inference...")
    for fold_k, (raw_train_groups, _) in enumerate(folds):
        
        # --- TE Leakage-Free Reconstruction ---
        train_region_ids_fold = {entry[0]["region_id"].iloc[0] for entry in raw_train_groups}
        train_df_fold_regions = train_df[train_df["region_id"].isin(train_region_ids_fold)]
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(train_df_fold_regions)
        
        # Apply exactly the same TE map to the test set
        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        
        # Resolve features
        feat_cols = [c for c in FEATURE_COLS if c in test_df_fold.columns]
        
        # Build 406-dim Matrix
        X_test_np, test_region_ids = build_tabular_test(test_df_fold, feat_cols)
        X_test_np = np.asfortranarray(X_test_np)
        
        if fold_k == 0:
            master_region_order = test_region_ids
        else:
            # Absolute safety check: ensure region order never mismatches across folds
            assert np.array_equal(master_region_order, test_region_ids), "Region order mismatch!"
            
        print(f"  --> Fold {fold_k} | X_test shape: {X_test_np.shape} | Predict...", end="", flush=True)

        for week_idx in range(HORIZON):
            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            if not os.path.exists(ckpt_a):
                raise FileNotFoundError(f"CRITICAL ERROR: Missing Model A -> {ckpt_a}")
                
            with open(ckpt_a, "rb") as fh:
                model_a = pickle.load(fh)
                
            # Raw L1 Prediction
            all_fold_preds[fold_k, :, week_idx] = model_a.predict(X_test_np)
            
        print(" Done.")

    # # 4. Ensemble & Post-Processing (naked version)
    # print("\n[4] Ensembling & Micro-Dust Filtering...")
    
    # # Take the median across 5 folds (most robust against outlier trees)
    # final_preds = np.median(all_fold_preds, axis=0)
    
    # # The Micro-Dust Filter (Clean the L1 noise)
    # final_preds = np.where(final_preds < MICRO_DUST_TH, 0.0, final_preds)
    
    # # Physical Limits Check
    # final_preds = np.clip(final_preds, 0.0, 5.0)

    # 4. Ensemble & Post-Processing
    print("\n[4] Ensembling & Quantile Thresholding...")
    
    # 取 5 個 Fold 的中位數
    final_preds = np.median(all_fold_preds, axis=0)
    
    # 【改動這裡】尋找全域的第 25.5 百分位數，作為動態微塵濾網
    magic_threshold = np.percentile(final_preds, 25.5)
    print(f"  [Quantile Magic] 25.5th Percentile Threshold found at: {magic_threshold:.4f}")
    
    # 將低於這個門檻的所有微小預測值，無情斬為 0.0
    final_preds = np.where(final_preds <= magic_threshold, 0.0, final_preds)
    
    # 物理極限保護
    final_preds = np.clip(final_preds, 0.0, 5.0)

    # 5. Format Submission
    print("\n[5] Formatting submission.csv ...")
    rows = []
    for i, rid in enumerate(master_region_order):
        rows.append({
            "region_id": rid,
            "pred_week1": final_preds[i, 0],
            "pred_week2": final_preds[i, 1],
            "pred_week3": final_preds[i, 2],
            "pred_week4": final_preds[i, 3],
            "pred_week5": final_preds[i, 4],
        })
        
    submission = pd.DataFrame(rows)
    submission.to_csv(SUB_OUTPUT, index=False)

    # 6. Diagnostics
    print("\n" + "=" * 85)
    print("Final Diagnostics (submission_30k_25%zero.csv)")
    print("=" * 85)
    
    vals = final_preds.flatten()
    print(f"  Mean       : {np.mean(vals):.4f}")
    print(f"  Zero Rate  : {np.mean(vals == 0.0):.1%}")
    print(f"  P50        : {np.percentile(vals, 50):.2f}")
    print(f"  P90        : {np.percentile(vals, 90):.2f}")
    print(f"  P99        : {np.percentile(vals, 99):.2f}")
    print(f"  Max        : {np.max(vals):.2f}")
    
    print(f"\n  Saved -> {SUB_OUTPUT}")
    print(f"  Completed in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()