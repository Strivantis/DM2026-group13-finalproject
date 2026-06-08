"""
v49_postprocess_all_strategies.py - Multi-Strategy Decision Pipeline for V49 Multiclass Models

This script loads the original v49 models (lgbm_multi_fold{k}_week{w}.pkl), Reconstructs 
the 6D probability space on OOF and Test sets, and executes 3 distinct post-processing pathways:
  Strategy A: Argmax (Maximum A Posteriori) - Eliminates expectation smoothing.
  Strategy B: Nelder-Mead Weight Optimization - Optimizes class weights to minimize L1 loss.
  Strategy C: L2 Probability Stacking (Meta-LGBM L1) - Non-linear mapping from 6D probabilities to MAE target.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.optimize as opt
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.dataset import (
    refine_features, build_stratified_group_cv_folds, 
    build_tabular_dataset, build_tabular_test, FEATURE_COLS, HORIZON
)

# =====================================================================
# Path Configuration (對齊原版 v49 輸出結構)
# =====================================================================
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")  # 原版 v49 pkl 存放根目錄

SUB_A_ARGMAX  = os.path.join(ROOT, "submission_v49_post_A_argmax.csv")
SUB_B_OPT     = os.path.join(ROOT, "submission_v49_post_B_optimized.csv")
SUB_C_STACK   = os.path.join(ROOT, "submission_v49_post_C_stacked.csv")

# =====================================================================
# Target Encoding Helpers (與訓練期完全一致)
# =====================================================================
def _zero_prob(x): 
    return (x == 0.0).mean()

def _compute_te_stats(df: pd.DataFrame) -> tuple:
    te_stats = (
        df.groupby("region_id")["score"]
        .agg(region_mean_score="mean", region_zero_prob=_zero_prob)
        .reset_index()
    )
    global_mean      = float(te_stats["region_mean_score"].mean())
    global_zero_prob = float(te_stats["region_zero_prob"].mean())
    te_map = {
        row["region_id"]: (float(row["region_mean_score"]), float(row["region_zero_prob"]))
        for _, row in te_stats.iterrows()
    }
    return te_map, global_mean, global_zero_prob

def _augment_groups_with_te(groups, te_map, global_mean, global_zero_prob):
    result = []
    for entry in groups:
        group, i_min, i_max = entry[0], entry[1], entry[2]
        g   = group.copy()
        rid = g["region_id"].iloc[0]
        mean_s, zero_p = te_map.get(rid, (global_mean, global_zero_prob))
        g["region_mean_score"] = np.float32(mean_s)
        g["region_zero_prob"]  = np.float32(zero_p)
        result.append((g, i_min, i_max))
    return result

def _merge_te_to_df(df, te_map, global_mean, global_zero_prob):
    df = df.copy()
    df["region_mean_score"] = df["region_id"].map(
        lambda rid: te_map.get(rid, (global_mean, global_zero_prob))[0]
    ).astype(np.float32)
    df["region_zero_prob"] = df["region_id"].map(
        lambda rid: te_map.get(rid, (global_mean, global_zero_prob))[1]
    ).astype(np.float32)
    return df

# =====================================================================
# Optimization Helper for Strategy B
# =====================================================================
def optimize_prob_weights(oof_probs, y_true):
    def loss_fn(weights):
        expected = np.sum(oof_probs * weights, axis=1)
        rounded = np.clip(np.round(expected), 0, 5)
        return np.mean(np.abs(y_true - rounded))

    initial_weights = np.arange(6, dtype=np.float64)  # 初始值為標準期望值 [0, 1, 2, 3, 4, 5]
    result = opt.minimize(
        loss_fn, initial_weights, method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-3, 'fatol': 1e-4}
    )
    return result.x, result.fun

# =====================================================================
# Main Pipeline
# =====================================================================
def main():
    print("=" * 90)
    print(" 🛠️  V49 MULTI-STRATEGY POST-PROCESSING PIPELINE ")
    print("=" * 90)

    # 1. 資料加載與與驗證集切分
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    train_df["score"] = np.round(train_df["score"]).astype(int)

    folds = build_stratified_group_cv_folds(train_df, n_splits=5)
    
    all_oof_probs = []
    all_y_true = []
    
    n_test_regions = test_df["region_id"].nunique()
    all_test_probs = np.zeros((5, n_test_regions, HORIZON, 6), dtype=np.float32)
    master_region_order = None

    # 2. 核心機率矩陣重構
    print("\n[1] Extracting 6D Probability Matrices from v49 pkl checkpoints...")
    for fold_k, (train_groups, val_groups) in enumerate(folds):
        print(f"  -> Reconstructing Fold {fold_k + 1} / 5 ...")
        
        train_rids = {e[0]["region_id"].iloc[0] for e in train_groups}
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(train_df[train_df["region_id"].isin(train_rids)])
        aug_val = _augment_groups_with_te(val_groups, te_map_fold, gm_fold, gzp_fold)
        
        X_val, y_val, _ = build_tabular_dataset(aug_val, FEATURE_COLS)
        X_val = np.asfortranarray(X_val)

        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test, test_rids = build_tabular_test(test_df_fold, FEATURE_COLS)
        X_test = np.asfortranarray(X_test)
        if fold_k == 0: master_region_order = test_rids
        
        fold_val_probs = np.zeros((X_val.shape[0], HORIZON, 6), dtype=np.float32)
        
        for week_idx in range(HORIZON):
            # 讀取非 fast 版本的原版模型
            ckpt = os.path.join(MODELS_DIR, f"lgbm_multi_fold{fold_k}_week{week_idx}.pkl")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"找不到指定的模型檔案: {ckpt}，請確認檔名格式。")
                
            with open(ckpt, "rb") as fh: 
                model = pickle.load(fh)
            
            fold_val_probs[:, week_idx, :] = model.predict_proba(X_val)
            all_test_probs[fold_k, :, week_idx, :] = model.predict_proba(X_test)
            
        all_oof_probs.append(fold_val_probs)
        all_y_true.append(y_val)
        
    full_oof_probs = np.concatenate(all_oof_probs, axis=0)  # Shape: (N, HORIZON, 6)
    full_y_true = np.concatenate(all_y_true, axis=0)        # Shape: (N, HORIZON)
    
    flat_oof_probs = full_oof_probs.reshape(-1, 6)
    flat_y_true = full_y_true.reshape(-1)
    
    test_probs_mean = np.mean(all_test_probs, axis=0)       # Shape: (n_test_regions, HORIZON, 6)

    print("\n" + "="*50)
    print(" STRATEGY EXECUTION & COMPARISON SUMMARY ")
    print("="*50)

    # -----------------------------------------------------------------------
    # 策略 A：最大後驗機率 (Argmax)
    # -----------------------------------------------------------------------
    oof_argmax = np.argmax(full_oof_probs, axis=2)
    mae_a = np.mean(np.abs(full_y_true - oof_argmax))
    print(f"🏆 Strategy A [Argmax]          OOF MAE: {mae_a:.4f}")
    
    final_argmax = np.argmax(test_probs_mean, axis=2)
    rows_a = [{"region_id": rid, **{f"pred_week{w+1}": int(final_argmax[i, w]) for w in range(HORIZON)}} for i, rid in enumerate(master_region_order)]
    pd.DataFrame(rows_a).to_csv(SUB_A_ARGMAX, index=False)

    # -----------------------------------------------------------------------
    # 策略 B：Nelder-Mead 權重最佳化
    # -----------------------------------------------------------------------
    opt_weights, mae_b = optimize_prob_weights(flat_oof_probs, flat_y_true)
    print(f"🏆 Strategy B [Weight Opt]      OOF MAE: {mae_b:.4f}")
    print(f"   -> Learned Weight Vector: {np.round(opt_weights, 3)}")
    
    final_opt_expected = np.sum(test_probs_mean * opt_weights, axis=2)
    final_opt_rounded = np.clip(np.round(final_opt_expected), 0, 5)
    rows_b = [{"region_id": rid, **{f"pred_week{w+1}": int(final_opt_rounded[i, w]) for w in range(HORIZON)}} for i, rid in enumerate(master_region_order)]
    pd.DataFrame(rows_b).to_csv(SUB_B_OPT, index=False)

    # -----------------------------------------------------------------------
    # 策略 C：L2 類別機率堆疊 (Meta-LGBM L1)
    # -----------------------------------------------------------------------
    meta_learner = LGBMRegressor(
        objective="regression_l1",
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        n_estimators=600,
        min_child_samples=50,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    meta_learner.fit(flat_oof_probs, flat_y_true)
    oof_stack_preds = meta_learner.predict(flat_oof_probs)
    oof_stack_rounded = np.clip(np.round(oof_stack_preds), 0, 5)
    mae_c = np.mean(np.abs(flat_y_true - oof_stack_rounded))
    print(f"🏆 Strategy C [L2 Stacking]     OOF MAE: {mae_c:.4f}")
    
    flat_test_probs = test_probs_mean.reshape(-1, 6)
    flat_test_preds = meta_learner.predict(flat_test_probs)
    final_stack_preds = flat_test_preds.reshape(n_test_regions, HORIZON)
    final_stack_rounded = np.clip(np.round(final_stack_preds), 0, 5)
    rows_c = [{"region_id": rid, **{f"pred_week{w+1}": int(final_stack_rounded[i, w]) for w in range(HORIZON)}} for i, rid in enumerate(master_region_order)]
    pd.DataFrame(rows_c).to_csv(SUB_C_STACK, index=False)

    print("\n" + "="*90)
    print(f"✅ Pipeline Completed Successfully.")
    print(f"   Generated Submissions:")
    print(f"   1. {SUB_A_ARGMAX}")
    print(f"   2. {SUB_B_OPT}")
    print(f"   3. {SUB_C_STACK}")
    print("=" * 90)

if __name__ == "__main__":
    main()