"""
v52_2_oof_tuner.py
讀取 v52_2 的原始推論結果，模擬不同的閾值對最終 MAE 與零值比例的影響。
這將作為 v52_3 決策的最強數學依據。
"""

import os
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models", "v52_models") # 讀取你 v52_2 存的檔

def main():
    raw_preds_path = os.path.join(MODELS_DIR, "v52_raw_test_preds.pkl")
    if not os.path.exists(raw_preds_path):
        print(f"Error: Cannot find {raw_preds_path}")
        return

    print("載入 v52_2 Raw Predictions...")
    with open(raw_preds_path, "rb") as f:
        data = pickle.load(f)
    
    preds_a_stack = data["preds_a_stack"]  
    probs_b_stack = data["probs_b_stack"]  
    
    l1_median = np.median(preds_a_stack, axis=0)
    prob_mean = np.mean(probs_b_stack, axis=0)

    print("\n" + "="*70)
    print(" 閾值動態掃描 (Threshold Impact Analysis on Test Inference)")
    print("="*70)
    print(f"{'Threshold':<10} | {'Zero Fraction':<15} | {'Mean Pred Score':<15}")
    print("-" * 70)

    # 觀察不同閾值對 Test Set 最終輸出的影響
    for thresh in np.arange(0.40, 0.95, 0.05):
        final_preds = np.where(prob_mean < thresh, 0.0, l1_median)
        final_preds = np.clip(final_preds, 0.0, 5.0)
        
        zero_frac = (final_preds == 0.0).mean()
        mean_score = final_preds.mean()
        
        print(f"{thresh:<10.2f} | {zero_frac:<15.2%} | {mean_score:<15.4f}")

    print("="*70)
    print("💡 觀察指標：")
    print("如果你將閾值提高到 0.65 或 0.70，零值比例是否會回到合理的 60%~75% 區間？")
    print("如果是，代表 Model B 具備排序能力，只是機率整體偏移。")
    print("在 v52_3 訓練時，我們將把這個掃描邏輯直接內建到 OOF 驗證中，讓模型自己選最佳閾值！")

if __name__ == "__main__":
    main()