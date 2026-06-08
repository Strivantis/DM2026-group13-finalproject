"""
export_meta_features.py - Export V45.1 OOF and Test Predictions for Stacking
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.dataset import (
    refine_features, build_stratified_group_cv_folds, 
    build_tabular_dataset, build_tabular_test, 
    FEATURE_COLS, HORIZON, WINDOW_SIZE
)

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models/v45_30k/") # 如果你的 pkl 放在 models/v45_30k，請改這裡
OOF_OUTPUT    = os.path.join(ROOT, "v45.1_meta_train_oof.csv")
TEST_OUTPUT   = os.path.join(ROOT, "v45.1_meta_test_preds.csv")

# TE Helpers
def _zero_prob(x): return (x == 0.0).mean()
def _compute_te_stats(df):
    te_stats = df.groupby("region_id")["score"].agg(
        region_mean_score="mean", region_zero_prob=_zero_prob
    ).reset_index()
    gm, gzp = float(te_stats["region_mean_score"].mean()), float(te_stats["region_zero_prob"].mean())
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

def _merge_te_to_df(df, te_map, gm, gzp):
    df = df.copy()
    df["region_mean_score"] = df["region_id"].map(lambda rid: te_map.get(rid, (gm, gzp))[0]).astype(np.float32)
    df["region_zero_prob"] = df["region_id"].map(lambda rid: te_map.get(rid, (gm, gzp))[1]).astype(np.float32)
    return df

def main():
    print("=" * 60)
    print(" EXPORTING V45.1 META-FEATURES FOR STACKING ")
    print("=" * 60)

    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    
    folds = build_stratified_group_cv_folds(train_df, n_splits=5) 
    oof_records = []
    
    # 準備存放 Test Set 預測矩陣 (Fold, Region, Week)
    n_test_regions = test_df["region_id"].nunique()
    test_preds_a_all = np.zeros((5, n_test_regions, HORIZON), dtype=np.float32)
    test_probs_b_all = np.zeros((5, n_test_regions, HORIZON), dtype=np.float32)
    master_test_rids = None

    for fold_k, (train_groups, val_groups) in enumerate(folds):
        print(f"\nProcessing Fold {fold_k}...")
        
        # 1. TE Setup
        train_rids = {e[0]["region_id"].iloc[0] for e in train_groups}
        train_df_fold = train_df[train_df["region_id"].isin(train_rids)]
        te_map, gm, gzp = _compute_te_stats(train_df_fold)
        
        # 2. 處理 Validation (OOF)
        aug_val_groups = _augment_groups_with_te(val_groups, te_map, gm, gzp)
        X_val, y_val, val_rids = build_tabular_dataset(aug_val_groups, FEATURE_COLS)
        X_val = np.asfortranarray(X_val)
        
        # 3. 處理 Test Set
        test_df_fold = _merge_te_to_df(test_df, te_map, gm, gzp)
        X_test, test_rids = build_tabular_test(test_df_fold, FEATURE_COLS)
        X_test = np.asfortranarray(X_test)
        if fold_k == 0: master_test_rids = test_rids
        
        # OOF 時間軸對齊字典
        region_dfs = {g_df["region_id"].iloc[0]: g_df for g_df, _, _ in aug_val_groups}
        rid_counts = {rid: 0 for rid in val_rids}
        row_to_rel_idx = [rid_counts.update({rid: rid_counts[rid]+1}) or rid_counts[rid]-1 for rid in val_rids]
        
        for week_idx in range(HORIZON):
            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            ckpt_b = os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl")
            
            with open(ckpt_a, "rb") as fa, open(ckpt_b, "rb") as fb:
                model_a = pickle.load(fa)
                model_b = pickle.load(fb)
                
            # OOF 推論
            pred_a_val = model_a.predict(X_val)
            prob_b_val = model_b.predict_proba(X_val)[:, 1]
            final_val  = np.where(prob_b_val < 0.5, 0.0, pred_a_val) # 加入決策
            
            # Test 推論
            test_preds_a_all[fold_k, :, week_idx] = model_a.predict(X_test)
            test_probs_b_all[fold_k, :, week_idx] = model_b.predict_proba(X_test)[:, 1]
            
            for i in range(len(val_rids)):
                rid = val_rids[i]
                rel_idx = row_to_rel_idx[i]
                try:
                    target_date = region_dfs[rid]["week_end_date"].iloc[WINDOW_SIZE + rel_idx + week_idx]
                except IndexError: target_date = "Unknown"
                
                oof_records.append({
                    "region_id": rid,
                    "target_date": target_date,
                    "fold": fold_k,
                    "horizon_week": week_idx + 1,
                    "true_score": y_val[i, week_idx],
                    "model_a_l1_pred": pred_a_val[i],
                    "model_b_prob": prob_b_val[i],
                    "final_hurdle_pred": final_val[i] # 新增最終決策
                })

    # =========================================================
    # 輸出 OOF 檔案
    # =========================================================
    print("\n[1/2] Saving Train OOF Meta-Features...")
    oof_df = pd.DataFrame(oof_records).sort_values(by=["region_id", "target_date", "horizon_week"])
    oof_df.to_csv(OOF_OUTPUT, index=False)

    # =========================================================
    # 輸出 Test 檔案 (取 5 Fold 的平均/中位數)
    # =========================================================
    print("[2/2] Saving Test Meta-Features...")
    test_l1_median = np.median(test_preds_a_all, axis=0)
    test_prob_mean = np.mean(test_probs_b_all, axis=0)
    test_final = np.where(test_prob_mean < 0.5, 0.0, test_l1_median)
    
    test_records = []
    for i, rid in enumerate(master_test_rids):
        for w in range(HORIZON):
            test_records.append({
                "region_id": rid,
                "horizon_week": w + 1,
                "model_a_l1_pred": test_l1_median[i, w],
                "model_b_prob": test_prob_mean[i, w],
                "final_hurdle_pred": test_final[i, w]
            })
            
    test_meta_df = pd.DataFrame(test_records)
    test_meta_df.to_csv(TEST_OUTPUT, index=False)

    print(f"\n✅ SUCCESS! Give these two files to your teammate:")
    print(f"  1. {OOF_OUTPUT}")
    print(f"  2. {TEST_OUTPUT}")

if __name__ == "__main__":
    main()