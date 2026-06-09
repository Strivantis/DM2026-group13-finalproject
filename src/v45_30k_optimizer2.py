"""
v45_shift_stretch_opt.py - Single-Parameter Shift & Stretch Optimization

This script:
1. Reconstructs the exact 5-Fold CV splits.
2. Loads the saved V45_30k PKL files (Model A & Model B).
3. Generates the TRUE OOF continuous predictions.
4. Uses 1D scalar optimization to find the best alpha for:
   y_adjusted = y_pred + alpha * (y_pred ** power)
5. Applies the optimal alpha + np.round() to the submission file.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.optimize as opt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 引用你現有的 dataset 工具
from src.dataset import (
    refine_features, build_stratified_group_cv_folds, 
    build_tabular_dataset, FEATURE_COLS, WINDOW_SIZE, HORIZON
)

# =====================================================================
# Configuration (請確認路徑與檔名格式)
# =====================================================================
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models/v45_30k/") 
# 這裡是 V45_30k 原始的浮點數 submission 檔案 (還沒做過任何 np.round 的版本)
INPUT_SUB     = os.path.join(ROOT, "submission_45th_30k_ABraw.csv") 
OUTPUT_SUB    = os.path.join(ROOT, "submission_45th_30k_ShiftStretch.csv")

# 拉伸函數的次方 (1.2 是基於對高分段低估的經驗法則，你也可以試試 1.5)
STRETCH_POWER = 1.2 

# =====================================================================
# Target Encoding Helpers (確保與訓練時完全一致)
# =====================================================================
def _zero_prob(x): return (x == 0.0).mean()

def _compute_te_stats(df: pd.DataFrame):
    te_stats = df.groupby("region_id")["score"].agg(
        region_mean_score="mean", region_zero_prob=_zero_prob
    ).reset_index()
    gm = float(te_stats["region_mean_score"].mean())
    gzp = float(te_stats["region_zero_prob"].mean())
    te_map = {row["region_id"]: (float(row["region_mean_score"]), float(row["region_zero_prob"])) 
              for _, row in te_stats.iterrows()}
    return te_map, gm, gzp

def _augment_groups_with_te(groups, te_map, gm, gzp):
    result = []
    for entry in groups:
        g, i_min, i_max = entry[0].copy(), entry[1], entry[2]
        rid = g["region_id"].iloc[0]
        m, z = te_map.get(rid, (gm, gzp))
        g["region_mean_score"] = np.float32(m)
        g["region_zero_prob"]  = np.float32(z)
        result.append((g, i_min, i_max))
    return result

# =====================================================================
# Optimization Logic (單一參數 alpha 最佳化)
# =====================================================================
def optimize_shift_stretch(oof_preds, y_true, power=STRETCH_POWER):
    print("\n[3] Optimizing Shift & Stretch Alpha via minimize_scalar...")
    
    def loss_fn(alpha):
        # [關鍵修正] 確保 oof_preds 沒有負數，避免負數開 1.2 次方產生 nan
        safe_preds = np.clip(oof_preds, 0.0, None)
        
        # 1. 套用平移拉伸公式
        y_adjusted = safe_preds + alpha * (safe_preds ** power)
        
        # 2. 神聖的四捨五入與截斷 (這是 Kaggle 計分的核心)
        y_rounded = np.round(y_adjusted).clip(0.0, 5.0)
        
        # 3. 回傳整數 MAE
        return float(np.mean(np.abs(y_true - y_rounded)))

    # 使用有界的一維最佳化 (Brent method)
    # alpha 通常是一個極小的正數或負數。我們設定搜尋範圍 -0.15 到 0.25
    result = opt.minimize_scalar(
        loss_fn,
        bounds=(-0.15, 0.25),
        method='bounded'
    )
    
    best_alpha = result.x
    
    print(f"  -> Baseline MAE (Pure np.round)  : {loss_fn(0.0):.6f}")
    print(f"  -> Optimized MAE (Shift+Round)   : {result.fun:.6f}")
    print(f"  -> Optimal Alpha                 : {best_alpha:.6f}")
    
    return best_alpha

# =====================================================================
# Main Execution
# =====================================================================
def main():
    print("=" * 80)
    print(" TRUE OOF RECONSTRUCTION & SHIFT-STRETCH OPTIMIZER ")
    print("=" * 80)

    # 1. Load Data & CV
    print("\n[1] Loading data and building exact CV splits...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    train_df = refine_features(train_raw, is_train=True)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    
    # Random seed 42 確保切出來的 Fold 跟訓練時一模一樣
    folds = build_stratified_group_cv_folds(train_df, n_splits=5) 
    
    all_oof_preds = []
    all_y_true = []

    # 2. Reconstruct OOF
    print("\n[2] Reconstructing True OOF Predictions from PKLs...")
    for fold_k, (train_groups, val_groups) in enumerate(folds):
        print(f"  Processing Fold {fold_k}...")
        
        # TE Setup
        train_rids = {e[0]["region_id"].iloc[0] for e in train_groups}
        train_df_fold = train_df[train_df["region_id"].isin(train_rids)]
        te_map, gm, gzp = _compute_te_stats(train_df_fold)
        aug_val_groups = _augment_groups_with_te(val_groups, te_map, gm, gzp)
        
        # Build Val Matrix
        X_val, y_val, _ = build_tabular_dataset(aug_val_groups, FEATURE_COLS)
        X_val = np.asfortranarray(X_val)
        
        fold_preds = np.zeros_like(y_val, dtype=np.float32)
        
        for week_idx in range(HORIZON):
            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            ckpt_b = os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl")
            
            with open(ckpt_a, "rb") as fa, open(ckpt_b, "rb") as fb:
                model_a = pickle.load(fa)
                model_b = pickle.load(fb)
                
            pred_a = model_a.predict(X_val)
            prob_b = model_b.predict_proba(X_val)[:, 1]
            
            # 雙頭架構的標準 Hurdle 邏輯 (V45_30k)
            fold_preds[:, week_idx] = np.where(prob_b < 0.5, 0.0, pred_a)
            
        all_oof_preds.append(fold_preds.flatten())
        all_y_true.append(y_val.flatten())

    full_oof_preds = np.concatenate(all_oof_preds)
    full_y_true = np.concatenate(all_y_true)

    # 3. Optimize
    best_alpha = optimize_shift_stretch(full_oof_preds, full_y_true)

    # 4. Apply to Submission
    print("\n[4] Applying Optimal Alpha to Submission...")
    sub_df = pd.read_csv(INPUT_SUB)
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    for col in pred_cols:
        # 套用公式
        adjusted = sub_df[col] + best_alpha * (sub_df[col] ** STRETCH_POWER)
        # 神聖四捨五入
        sub_df[col] = np.round(adjusted).clip(0.0, 5.0)
        
    sub_df.to_csv(OUTPUT_SUB, index=False)
    print(f"\n✅ SUCCESS! File saved to {OUTPUT_SUB}")
    
if __name__ == "__main__":
    main()