"""
v54_oof_tuner.py
不用重新訓練！直接載入已訓練的模型，生成 OOF 並尋找最佳 Hurdle 閾值。
並將 best_threshold 補回 test_preds.pkl 中。
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 依據你的版本號進行管理
VERSION = "v54_3" 

PROCESSED_DIR = os.path.join(ROOT, "data", "v54_processed")
MODELS_DIR    = os.path.join(ROOT, "models", f"{VERSION}_models")

# 引入你的 Dataset 模組
from src.v54_dataset import (
    refine_features,
    build_time_seasonal_cv_folds,    
    build_tabular_dataset,
    FEATURE_COLS,
    HORIZON,
)

def main():
    print("=" * 80)
    print(f" 🚀 {VERSION} OOF THRESHOLD AUTO-TUNER PATCH")
    print("=" * 80)

    # 1. 載入已有的測試集預報檔案
    raw_preds_path = os.path.join(MODELS_DIR, f"{VERSION}_raw_test_preds.pkl")
    if not os.path.exists(raw_preds_path):
        raise FileNotFoundError(f"找不到 {raw_preds_path}，請確認版本號是否正確。")
        
    with open(raw_preds_path, "rb") as f:
        test_preds_data = pickle.load(f)

    # 2. 載入訓練集以取得 Validation 特徵與真實 score
    print("Loading processed train data to reconstruct validation folds...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    train_df  = refine_features(train_raw, is_train=True).dropna(subset=["score"]).reset_index(drop=True)
    
    folds = build_time_seasonal_cv_folds(train_df, n_splits=4, season_step=13)
    
    all_oof_true   = []
    all_oof_preds  = []
    all_oof_probs  = []

    # 3. 進入 Folds 快速推論 (不訓練，只 Predict)
    print("\nGenerating OOF Predictions from saved checkpoints...")
    for fold_k, (_, raw_val_groups) in enumerate(folds):
        print(f"  Processing Fold {fold_k + 1} / 4 ...")
        
        feat_cols = [c for c in FEATURE_COLS if c in raw_val_groups[0][0].columns]
        X_val_np, y_val_np, _ = build_tabular_dataset(raw_val_groups, feat_cols)
        X_val_np = np.asfortranarray(X_val_np)
        
        fold_true  = np.zeros_like(y_val_np)
        fold_preds = np.zeros_like(y_val_np)
        fold_probs = np.zeros_like(y_val_np)

        for week_idx in range(HORIZON):
            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            ckpt_b = os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl")
            
            with open(ckpt_a, "rb") as fh: model_a = pickle.load(fh)
            with open(ckpt_b, "rb") as fh: model_b = pickle.load(fh)
            
            fold_true[:, week_idx]  = y_val_np[:, week_idx]
            fold_preds[:, week_idx] = model_a.predict(X_val_np)
            fold_probs[:, week_idx] = model_b.predict_proba(X_val_np)[:, 1]
            
        all_oof_true.append(fold_true.ravel())
        all_oof_preds.append(fold_preds.ravel())
        all_oof_probs.append(fold_probs.ravel())
        
        del X_val_np, y_val_np, model_a, model_b
        gc.collect()

    # 展平所有 OOF 樣本以進行全局搜尋
    oof_true  = np.concatenate(all_oof_true)
    oof_preds = np.concatenate(all_oof_preds)
    oof_probs = np.concatenate(all_oof_probs)

    # 4. 暴力尋找最佳閾值
    print("\nOptimizing Hurdle Threshold...")
    thresholds = np.arange(0.05, 0.95, 0.01) # 步長 0.01 精細搜尋
    best_threshold = 0.50
    best_mae = float('inf')

    for th in thresholds:
        # 模擬 Hurdle Gate 邏輯
        gated_preds = np.where(oof_probs < th, 0.0, oof_preds)
        gated_preds = np.clip(gated_preds, 0.0, 5.0)
        
        mae = np.mean(np.abs(gated_preds - oof_true))
        if mae < best_mae:
            best_mae = mae
            best_threshold = th

    print("-" * 60)
    print(f"🔥 OPTIMIZATION COMPLETE")
    print(f"  Best OOF Threshold : {best_threshold:.4f}")
    print(f"  Minimum OOF MAE    : {best_mae:.4f}")
    print("-" * 60)

    # 5. 把最佳閾值寫入原本的測試集 pkl 檔案中（時空回溯補丁）
    test_preds_data["best_threshold"] = float(best_threshold)
    
    with open(raw_preds_path, "wb") as f:
        pickle.dump(test_preds_data, f)
        
    print(f"成功將 best_threshold={best_threshold:.4f} 補回 {raw_preds_path}！")
    print("現在你可以直接執行 infer.py 並開啟 AUTO 模式了！")
    print("=" * 80)

if __name__ == "__main__":
    main()