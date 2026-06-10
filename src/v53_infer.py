"""
v52_infer.py
Decoupled Inference Script for v52 Dual-Tree Hurdle.
 Allows using the Auto-Tuned Best Threshold from training, OR manual override.
"""

import os
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models", "v52_models")

# ---------------------------------------------------------------------------
# Configuration (Your "Regret Pill")
# ---------------------------------------------------------------------------
# 設定 USE_AUTO_THRESHOLD = True  -> 讀取 v52_train 算出來的 OOF 最佳閾值
# 設定 USE_AUTO_THRESHOLD = False -> 強制使用下方的 MANUAL_THRESHOLD
USE_AUTO_THRESHOLD = False  
MANUAL_THRESHOLD   = 0.50

APPLY_ROUNDING     = False 

def main():
    print("=" * 80)
    print(" v52 INFERENCE & ENSEMBLE SUBMISSION BUILDER")
    print("=" * 80)

    raw_preds_path = os.path.join(MODELS_DIR, "v52_raw_test_preds.pkl")
    if not os.path.exists(raw_preds_path):
        raise FileNotFoundError(f"Missing {raw_preds_path}. Run v52_train.py first.")

    print(f"Loading raw predictions from: {raw_preds_path}")
    with open(raw_preds_path, "rb") as f:
        data = pickle.load(f)
    
    preds_a_stack = data["preds_a_stack"]  
    probs_b_stack = data["probs_b_stack"]  
    test_region_ids = data["region_ids"]   
    
    # 取得 Train 算出的最佳閾值
    auto_best_threshold = data.get("best_threshold", 0.5) 

    n_folds = preds_a_stack.shape[0]
    n_regions = preds_a_stack.shape[1]
    
    print(f"Loaded {n_folds} folds for {n_regions} regions.")

    # 1. 決定最終要使用的閾值
    if USE_AUTO_THRESHOLD:
        active_threshold = auto_best_threshold
        print(f"\n[Mode: AUTO] Using OOF-Optimized Threshold: {active_threshold:.2f}")
    else:
        active_threshold = MANUAL_THRESHOLD
        print(f"\n[Mode: MANUAL] OVERRIDE! Using Manual Threshold: {active_threshold:.2f}")
        print(f"             (The OOF-Optimized was: {auto_best_threshold:.2f})")

    # 2. Asymmetric Ensemble
    print("\nExecuting Asymmetric Ensemble...")
    l1_median = np.median(preds_a_stack, axis=0) 
    prob_mean = np.mean(probs_b_stack,  axis=0)

    print(f"  Model A (Severity) mean : {l1_median.mean():.4f}")
    print(f"  Model B (Prob>0)   mean : {prob_mean.mean():.4f}")

    # 3. Hurdle Gate
    print(f"\nApplying Hurdle Gate (Threshold = {active_threshold:.2f})...")
    final_preds = np.where(prob_mean < active_threshold, 0.0, l1_median)
    final_preds = np.clip(final_preds, 0.0, 5.0)

    # 4. Optional Post-Processing
    if APPLY_ROUNDING:
        print("Applying integer rounding to final predictions...")
        final_preds = np.round(final_preds)

    # 5. Diagnostics
    print(f"  Post-gate & round mean: {final_preds.mean():.4f}")
    print(f"  Exact-zero fraction   : {(final_preds == 0.0).mean():.2%}")

    # 6. Build Submission
    print("\nFormatting submission.csv...")
    rows = []
    for i, rid in enumerate(test_region_ids):
        row = {"region_id": rid}
        for w in range(5):
            row[f"pred_week{w+1}"] = float(final_preds[i, w])
        rows.append(row)
        
    submission = pd.DataFrame(rows)
    
    # 動態命名檔案，方便你比較不同閾值的結果
    mode_str = "AUTO" if USE_AUTO_THRESHOLD else f"MANUAL_{active_threshold:.2f}"
    sub_path = os.path.join(ROOT, f"submission_v52_{mode_str}.csv")
    submission.to_csv(sub_path, index=False)
    
    print(f"Submission saved to: {sub_path} ({len(submission)} rows)")
    print("\nPreview:")
    print(submission.head())
    print("\n" + "=" * 80)
    print("✅ Ready to submit to Kaggle!")

if __name__ == "__main__":
    main()