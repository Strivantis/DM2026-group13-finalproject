"""
inference.py
Decoupled Inference Script for Dual-Tree Hurdle.
Supports "SOFT" (Expected Value) and "HARD" (Threshold-based) gating strategies.
"""

import os
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VERSION = "v52"

# Gating strategy: "SOFT" (Score * Prob) or "HARD" (Prob < Threshold -> 0)
GATING_STRATEGY = "HARD"

# Hard Gate parameters
USE_AUTO_THRESHOLD = False
MANUAL_THRESHOLD   = 0.5

APPLY_ROUNDING     = True

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models", f"{VERSION}_models")

def main():
    print(f"--- [inference.py] Pipeline Init | Version: {VERSION} ---")

    raw_preds_path = os.path.join(MODELS_DIR, f"{VERSION}_raw_test_preds.pkl")
    if not os.path.exists(raw_preds_path):
        raise FileNotFoundError(f"Missing {raw_preds_path}. Execute training phase first.")

    print(f"I/O: Loading raw predictions from {raw_preds_path}")
    with open(raw_preds_path, "rb") as f:
        data = pickle.load(f)
    
    preds_a_stack = data["preds_a_stack"]  
    probs_b_stack = data["probs_b_stack"]  
    test_region_ids = data["region_ids"]   
    auto_best_threshold = data.get("best_threshold", 0.5) 

    n_folds = preds_a_stack.shape[0]
    n_regions = preds_a_stack.shape[1]
    
    print(f"Data: Loaded shape (folds={n_folds}, regions={n_regions})")

    # 1. Asymmetric Ensemble Aggregation
    print("Ensemble: Executing asymmetric aggregation")
    l1_median = np.median(preds_a_stack, axis=0) 
    prob_mean = np.mean(probs_b_stack,  axis=0)

    print(f"Metrics: Model A (Severity Median) mean = {l1_median.mean():.4f}")
    print(f"Metrics: Model B (Probability Mean) mean = {prob_mean.mean():.4f}")

    # 2. Gate Strategy Application
    if GATING_STRATEGY == "SOFT":
        print("Gating: Applying SOFT strategy (Expected Value)")
        final_preds = l1_median * prob_mean
        mode_str = "SOFT"
    else:
        if USE_AUTO_THRESHOLD:
            active_threshold = auto_best_threshold
            mode_str = "AUTO"
            print(f"Gating: Applying HARD-AUTO strategy (OOF threshold={active_threshold:.2f})")
        else:
            active_threshold = MANUAL_THRESHOLD
            mode_str = f"MANUAL_{active_threshold:.2f}"
            print(f"Gating: Applying HARD-MANUAL strategy (Manual threshold={active_threshold:.2f})")
            
        final_preds = np.where(prob_mean < active_threshold, 0.0, l1_median)

    # Bound predictions to physical range [0.0, 5.0]
    final_preds = np.clip(final_preds, 0.0, 5.0)

    # 3. Post-Processing
    if APPLY_ROUNDING:
        print("Post-processing: Applying integer rounding")
        final_preds = np.round(final_preds)

    # 4. Diagnostics
    print(f"Diagnostics: Post-gate mean = {final_preds.mean():.4f}")
    print(f"Diagnostics: Exact-zero fraction = {(final_preds == 0.0).mean():.2%}")

    # 5. Output Construction
    print("I/O: Formatting submission payload")
    rows = []
    for i, rid in enumerate(test_region_ids):
        row = {"region_id": rid}
        for w in range(5):
            row[f"pred_week{w+1}"] = float(final_preds[i, w])
        rows.append(row)
        
    submission = pd.DataFrame(rows)
    sub_path = os.path.join(ROOT, f"submission_{VERSION}_{mode_str}.csv")
    submission.to_csv(sub_path, index=False)
    
    print(f"I/O: Submission exported to {sub_path} ({len(submission)} rows)")
    print("Preview:")
    print(submission.head())
    print("--- [inference.py] Pipeline Complete ---")

if __name__ == "__main__":
    main()