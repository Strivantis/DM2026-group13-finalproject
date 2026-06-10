"""
v52_analyze_preds.py
讀取原始預測快取，尋找完美匹配真實分佈 (82% 零值) 的最佳閾值。
"""

import os
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models", "v52_models") # 確保路徑對應你存 pkl 的地方

def main():
    raw_preds_path = os.path.join(MODELS_DIR, "v52_raw_test_preds.pkl")
    if not os.path.exists(raw_preds_path):
        print(f"找不到檔案: {raw_preds_path}")
        return

    print("載入 Raw Predictions...")
    with open(raw_preds_path, "rb") as f:
        data = pickle.load(f)
    
    preds_a_stack = data["preds_a_stack"]  # (4, 2248, 5)
    probs_b_stack = data["probs_b_stack"]  # (4, 2248, 5)
    
    # 進行 Ensemble
    l1_median = np.median(preds_a_stack, axis=0)
    prob_mean = np.mean(probs_b_stack, axis=0)

    # 目標零值比例：根據 Validation Set 的先驗分佈，大約是 82.5%
    TARGET_ZERO_FRAC = 0.825
    
    print("\n" + "="*60)
    print(" 閾值掃描 (Threshold Scanning)")
    print("="*60)
    print(f"{'Threshold':<12} | {'Zero Fraction':<15} | {'Mean Score':<12}")
    print("-" * 60)

    best_threshold = 0.5
    min_diff = 1.0

    # 掃描從 0.5 到 0.95 的閾值
    for thresh in np.arange(0.50, 0.96, 0.05):
        # 模擬 Hurdle Gate
        final_preds = np.where(prob_mean < thresh, 0.0, l1_median)
        final_preds = np.clip(final_preds, 0.0, 5.0)
        final_preds = np.round(final_preds) # 模擬提交時的 round

        zero_frac = (final_preds == 0.0).mean()
        mean_score = final_preds.mean()
        
        print(f"{thresh:<12.2f} | {zero_frac:<15.2%} | {mean_score:<12.4f}")

        # 尋找最接近目標 82.5% 的閾值
        if abs(zero_frac - TARGET_ZERO_FRAC) < min_diff:
            min_diff = abs(zero_frac - TARGET_ZERO_FRAC)
            best_threshold = thresh

    print("="*60)
    print(f"💡 建議行動: 將 v52_infer.py 的 HURDLE_THRESHOLD 設為 {best_threshold:.2f}")
    print(f"   這樣能讓你的預測分佈最接近真實世界的 82.5% 無旱災機率！")
    print("="*60)

if __name__ == "__main__":
    main()