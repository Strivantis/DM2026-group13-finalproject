"""
infer.py
Decoupled Inference Script for Dual-Tree Hurdle.
Supports both "SOFT" (Expected Value) and "HARD" (Threshold-based) gating strategies.
"""

import os
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VERSION = "v54_3"  # 統一管理版本號

# 閘門策略設定
# "SOFT": 期望值計算 (Score * Prob)，保留中低分段平滑過渡
# "HARD": 傳統閾值截斷 (Prob < Threshold -> 0)
GATING_STRATEGY = "HARD"  # "SOFT" or "HARD"

# Hard Gate 專用參數 (僅在 GATING_STRATEGY == "HARD" 時生效)
USE_AUTO_THRESHOLD = False  # 是否使用訓練過程中自動優化的閾值 
MANUAL_THRESHOLD   = 0.5

APPLY_ROUNDING     = True  # 是否將最終預測值四捨五入到整數 (0, 1, 2, 3, 4, 5)

# ---------------------------------------------------------------------------
# Setup Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models", f"{VERSION}_models")

def main():
    print("=" * 80)
    print(f" {VERSION} INFERENCE & ENSEMBLE SUBMISSION BUILDER")
    print("=" * 80)

    raw_preds_path = os.path.join(MODELS_DIR, f"{VERSION}_raw_test_preds.pkl")
    if not os.path.exists(raw_preds_path):
        raise FileNotFoundError(f"Missing {raw_preds_path}. Run training script first.")

    print(f"Loading raw predictions from: {raw_preds_path}")
    with open(raw_preds_path, "rb") as f:
        data = pickle.load(f)
    
    preds_a_stack = data["preds_a_stack"]  
    probs_b_stack = data["probs_b_stack"]  
    test_region_ids = data["region_ids"]   
    
    # 取得 Train 算出的最佳閾值 (如果有的話)
    auto_best_threshold = data.get("best_threshold", 0.5) 

    n_folds = preds_a_stack.shape[0]
    n_regions = preds_a_stack.shape[1]
    
    print(f"Loaded {n_folds} folds for {n_regions} regions.")

    # 1. Asymmetric Ensemble Aggregation
    print("\nExecuting Asymmetric Ensemble...")
    l1_median = np.median(preds_a_stack, axis=0) 
    prob_mean = np.mean(probs_b_stack,  axis=0)

    print(f"  Model A (Severity Median) mean : {l1_median.mean():.4f}")
    print(f"  Model B (Probability Mean) mean: {prob_mean.mean():.4f}")

    # 2. Apply Gate Strategy
    if GATING_STRATEGY == "SOFT":
        print("\n[Mode: SOFT] Applying Expected Value Gate (Score * Prob)...")
        final_preds = l1_median * prob_mean
        mode_str = "SOFT"
    else:
        if USE_AUTO_THRESHOLD:
            active_threshold = auto_best_threshold
            mode_str = "AUTO"
            print(f"\n[Mode: HARD-AUTO] Using OOF-Optimized Threshold: {active_threshold:.2f}")
        else:
            active_threshold = MANUAL_THRESHOLD
            mode_str = f"MANUAL_{active_threshold:.2f}"
            print(f"\n[Mode: HARD-MANUAL] OVERRIDE! Using Manual Threshold: {active_threshold:.2f}")
            
        print(f"Applying Hard Hurdle Gate (Threshold = {active_threshold:.2f})...")
        final_preds = np.where(prob_mean < active_threshold, 0.0, l1_median)

    # 確保預測值不超出合理物理區間
    final_preds = np.clip(final_preds, 0.0, 5.0)

    # 3. Optional Post-Processing
    if APPLY_ROUNDING:
        print("Applying integer rounding to final predictions...")
        final_preds = np.round(final_preds)

    # 4. Diagnostics
    print(f"  Post-gate & round mean: {final_preds.mean():.4f}")
    print(f"  Exact-zero fraction   : {(final_preds == 0.0).mean():.2%}")

    # 5. Build Submission
    print("\nFormatting submission.csv...")
    rows = []
    for i, rid in enumerate(test_region_ids):
        row = {"region_id": rid}
        for w in range(5):
            row[f"pred_week{w+1}"] = float(final_preds[i, w])
        rows.append(row)
        
    submission = pd.DataFrame(rows)
    
    sub_path = os.path.join(ROOT, f"submission_{VERSION}_{mode_str}.csv")
    submission.to_csv(sub_path, index=False)
    
    print(f"Submission saved to: {sub_path} ({len(submission)} rows)")
    print("\nPreview:")
    print(submission.head())
    print("\n" + "=" * 80)
    print("✅ Ready to submit to Kaggle!")

if __name__ == "__main__":
    main()