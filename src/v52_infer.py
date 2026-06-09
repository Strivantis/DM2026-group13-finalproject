"""
v52_infer.py
Decoupled Inference Script for v52 Dual-Tree Hurdle.
Allows instant threshold tuning and submission generation without retraining.
"""

import os
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models/v52_2_models")

# ---------------------------------------------------------------------------
# Configuration (You can quickly tune this without retraining!)
# ---------------------------------------------------------------------------
HURDLE_THRESHOLD = 0.5   # P(nonzero) < threshold -> Force 0.0
APPLY_ROUNDING   = True  # Whether to round to nearest integer (Kaggle scores are integers)

def main():
    print("=" * 80)
    print(" V52 INFERENCE & ENSEMBLE SUBMISSION BUILDER")
    print("=" * 80)

    raw_preds_path = os.path.join(MODELS_DIR, "v52_2_raw_test_preds.pkl")
    if not os.path.exists(raw_preds_path):
        raise FileNotFoundError(f"Missing {raw_preds_path}. Run v52_train.py first.")

    print(f"Loading raw predictions from: {raw_preds_path}")
    with open(raw_preds_path, "rb") as f:
        data = pickle.load(f)
    
    preds_a_stack = data["preds_a_stack"]  # Shape: (n_folds, n_regions, 5)
    probs_b_stack = data["probs_b_stack"]  # Shape: (n_folds, n_regions, 5)
    test_region_ids = data["region_ids"]   # Shape: (n_regions,)
    
    n_folds = preds_a_stack.shape[0]
    n_regions = preds_a_stack.shape[1]
    
    print(f"Loaded {n_folds} folds for {n_regions} regions.")

    # 1. Asymmetric Ensemble
    print("\nExecuting Asymmetric Ensemble...")
    # Model A -> Median (robust to fold outliers)
    l1_median = np.median(preds_a_stack, axis=0) 
    # Model B -> Mean (probability averaging)
    prob_mean = np.mean(probs_b_stack,  axis=0)

    print(f"  Model A (Severity) mean : {l1_median.mean():.4f}")
    print(f"  Model B (Prob>0)   mean : {prob_mean.mean():.4f}")

    # 2. Hurdle Gate
    print(f"\nApplying Hurdle Gate (Threshold = {HURDLE_THRESHOLD})...")
    final_preds = np.where(prob_mean < HURDLE_THRESHOLD, 0.0, l1_median)
    final_preds = np.clip(final_preds, 0.0, 5.0)

    # 3. Optional Post-Processing (Rounding)
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
    sub_path = os.path.join(ROOT, "submission_52th_2.csv")
    submission.to_csv(sub_path, index=False)
    
    print(f"Submission saved to: {sub_path} ({len(submission)} rows)")
    print("\nPreview:")
    print(submission.head())
    print("\n" + "=" * 80)
    print("✅ Ready to submit to Kaggle!")

if __name__ == "__main__":
    main()