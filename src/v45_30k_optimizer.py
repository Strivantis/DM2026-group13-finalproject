"""
v50_true_oof_optimizer.py - True OOF Reconstruction & Ordinal Threshold Optimization

This script:
1. Reconstructs the exact 5-Fold CV splits.
2. Loads the saved V45.1 Dual-Tree PKL files (Model A & Model B).
3. Generates the TRUE OOF predictions for all 137M rows.
4. Uses Nelder-Mead to find the mathematical optimal thresholds for MAE.
5. Applies these thresholds to the continuous submission file.
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
# Configuration (請確認這裡的路徑與檔名格式符合你的 V45.1)
# =====================================================================
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models/v45_30k/") # 如果你的 pkl 放在 models/v45_30k，請改這裡
INPUT_SUB     = os.path.join(ROOT, "submission_45th_30k_ABraw.csv") # 你的 0.8428 浮點數原檔
OUTPUT_SUB    = os.path.join(ROOT, "submission_45th_30k_optimized.csv")

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
# Optimization Logic
# =====================================================================
def optimize_thresholds(oof_preds, y_true):
    print("\n[3] Optimizing Thresholds via Nelder-Mead...")
    
    def loss_fn(thresholds):
        # 確保閾值是單調遞增的
        if not all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1)):
            return np.inf
        # 將連續預測轉為 0~5 的整數並計算 MAE
        preds_int = np.digitize(oof_preds, thresholds)
        return np.mean(np.abs(y_true - preds_int))

    # 初始猜測：標準的 .5 切分點
    initial_thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
    
    result = opt.minimize(
        loss_fn, 
        initial_thresholds, 
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-3, 'fatol': 1e-4}
    )
    
    best_thresholds = result.x
    print(f"  -> Initial MAE (Standard Round): {loss_fn([0.5, 1.5, 2.5, 3.5, 4.5]):.4f}")
    print(f"  -> Optimized MAE               : {result.fun:.4f}")
    print(f"  -> Optimal Thresholds          : {np.round(best_thresholds, 4)}")
    return best_thresholds

# =====================================================================
# Main Execution
# =====================================================================
def main():
    print("=" * 80)
    print(" TRUE OOF RECONSTRUCTION & ORDINAL OPTIMIZER ")
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
        
        # Build Val Matrix (406 dimensions)
        X_val, y_val, _ = build_tabular_dataset(aug_val_groups, FEATURE_COLS)
        X_val = np.asfortranarray(X_val)
        
        fold_preds = np.zeros_like(y_val, dtype=np.float32)
        
        for week_idx in range(HORIZON):
            # 路徑請確認是否符合你的檔案命名習慣
            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            ckpt_b = os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl")
            
            with open(ckpt_a, "rb") as fa, open(ckpt_b, "rb") as fb:
                model_a = pickle.load(fa)
                model_b = pickle.load(fb)
                
            pred_a = model_a.predict(X_val)
            prob_b = model_b.predict_proba(X_val)[:, 1]
            
            # 雙頭架構的標準 Hurdle 邏輯
            fold_preds[:, week_idx] = np.where(prob_b < 0.5, 0.0, pred_a)
            
        all_oof_preds.append(fold_preds.flatten())
        all_y_true.append(y_val.flatten())

    full_oof_preds = np.concatenate(all_oof_preds)
    full_y_true = np.concatenate(all_y_true)

    # 3. Optimize
    best_thresholds = optimize_thresholds(full_oof_preds, full_y_true)

    # 4. Apply to Submission
    print("\n[4] Applying Optimal Thresholds to Submission...")
    sub_df = pd.read_csv(INPUT_SUB)
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    
    for col in pred_cols:
        sub_df[col] = np.digitize(sub_df[col], best_thresholds)
        
    sub_df.to_csv(OUTPUT_SUB, index=False)
    print(f"\n✅ SUCCESS! File saved to {OUTPUT_SUB}")
    
if __name__ == "__main__":
    main()