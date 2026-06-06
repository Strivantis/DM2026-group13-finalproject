"""
stacking.py – Level-2 Meta-Model Stacking & Submission Diagnostics

Architecture:
  Level-1: 25 Pairs of Dual-Tree Hurdle models (Model A + Model B).
  Level-2: LightGBM Regressor (MAE optimized) learning to blend L1 outputs non-linearly
           based on context (cluster_id, week, region base rates).
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_stratified_group_cv_folds,
    build_tabular_dataset,
    build_tabular_test,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
)
from src.train import (
    _compute_te_stats,
    _augment_groups_with_te,
    _merge_te_to_df,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")
N_FOLDS       = 5

FILES_TO_COMPARE = {
    "Baseline (v37)": os.path.join(ROOT, "submission_37th.csv"),
    "V45 Raw (T=0.5)": os.path.join(ROOT, "submission_45th.csv"),
    "V45 Optimized": os.path.join(ROOT, "submission_45thopt.csv"),
    "V45 L2-Stacked": os.path.join(ROOT, "submission_45thstacked.csv")
}


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    
    print("=" * 95)
    print("Level-2 Meta-Model Stacking (Non-linear Dual-Tree Blending)")
    print("=" * 95)

    # 1. Load Data
    print("\n[1] Loading Base Data & Climate Ecosystems...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    region_stats = pd.read_csv(os.path.join(PROCESSED_DIR, "region_stats.csv"))
    
    cluster_map = dict(zip(region_stats["region_id"], region_stats["cluster_id"]))
    
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)

    # Global TE for Test Set Level-2 Features
    global_te_map, global_mean, global_zp = _compute_te_stats(train_df)

    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    
    oof_records = []
    test_preds_a = []
    test_preds_b = []
    test_region_order = []

    # 2. Extract Level-1 OOF & Test Predictions
    print("\n[2] Reconstructing Level-1 OOF Predictions...")
    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):
        train_region_ids_fold = {entry[0]["region_id"].iloc[0] for entry in raw_train_groups}
        train_df_fold_regions = train_df[train_df["region_id"].isin(train_region_ids_fold)]
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(train_df_fold_regions)
        
        aug_val_groups = _augment_groups_with_te(raw_val_groups, te_map_fold, gm_fold, gzp_fold)
        feat_cols = [c for c in FEATURE_COLS if c in aug_val_groups[0][0].columns]
        
        X_val_np, y_val_np, val_region_ids = build_tabular_dataset(aug_val_groups, feat_cols)
        X_val_np = np.asfortranarray(X_val_np)
        
        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test_np, test_region_ids = build_tabular_test(test_df_fold, feat_cols)
        X_test_np = np.asfortranarray(X_test_np)
        
        if fold_k == 0:
            test_region_order = test_region_ids
            
        fold_test_pred_l1 = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)
        fold_test_prob    = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)
        
        for week_idx in range(HORIZON):
            with open(os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl"), "rb") as fh: model_a = pickle.load(fh)
            with open(os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl"), "rb") as fh: model_b = pickle.load(fh)
                
            val_l1_w   = model_a.predict(X_val_np)
            val_prob_w = model_b.predict_proba(X_val_np)[:, 1]
            y_val_w    = y_val_np[:, week_idx]
            
            fold_test_pred_l1[:, week_idx] = model_a.predict(X_test_np)
            fold_test_prob[:, week_idx]    = model_b.predict_proba(X_test_np)[:, 1]
            
            # Contextual Features for Meta-Model
            for i in range(len(val_region_ids)):
                rid = val_region_ids[i]
                r_mean, r_zp = te_map_fold.get(rid, (gm_fold, gzp_fold))
                oof_records.append({
                    "region_id": rid,
                    "fold": fold_k,
                    "week": week_idx,
                    "cluster_id": cluster_map.get(rid, 0),
                    "region_zero_prob": r_zp,
                    "region_mean_score": r_mean,
                    "pred_l1": val_l1_w[i],
                    "prob": val_prob_w[i],
                    "y_true": y_val_w[i]
                })

        test_preds_a.append(fold_test_pred_l1)
        test_preds_b.append(fold_test_prob)

    oof_df = pd.DataFrame(oof_records)
    
    # 3. Build Test Meta-Features
    print("\n[3] Building Level-2 Meta-Features...")
    preds_a_stack = np.stack(test_preds_a, axis=0) # (5, 2248, 5)
    probs_b_stack = np.stack(test_preds_b, axis=0) # (5, 2248, 5)
    
    test_l1_median = np.median(preds_a_stack, axis=0)
    test_prob_mean = np.mean(probs_b_stack, axis=0)
    
    test_meta_records = []
    for i, rid in enumerate(test_region_order):
        r_mean, r_zp = global_te_map.get(rid, (global_mean, global_zp))
        for w in range(HORIZON):
            test_meta_records.append({
                "region_id": rid,
                "week": w,
                "cluster_id": cluster_map.get(rid, 0),
                "region_zero_prob": r_zp,
                "region_mean_score": r_mean,
                "pred_l1": test_l1_median[i, w],
                "prob": test_prob_mean[i, w],
            })
    test_meta_df = pd.DataFrame(test_meta_records)

    # Formatting Categorical columns for LightGBM
    meta_features = ["pred_l1", "prob", "region_zero_prob", "region_mean_score", "week", "cluster_id"] # <--- 補上
    cat_features  = ["week", "cluster_id"]
    
    for df in [oof_df, test_meta_df]:
        for c in cat_features:
            df[c] = df[c].astype("category")

    # 4. Train Meta-Model (Level-2 Stacking)
    print("\n[4] Training Meta-Model (LGBMRegressor: regression_l1)")
    meta_test_preds = np.zeros(len(test_meta_df))
    meta_oof_preds  = np.zeros(len(oof_df))
    
    meta_maes = []
    
    for fold_k in range(N_FOLDS):
        train_idx = oof_df["fold"] != fold_k
        val_idx   = oof_df["fold"] == fold_k
        
        X_tr, y_tr = oof_df.loc[train_idx, meta_features], oof_df.loc[train_idx, "y_true"]
        X_va, y_va = oof_df.loc[val_idx, meta_features],   oof_df.loc[val_idx, "y_true"]
        
        # Meta-Model Configuration (Small depth to prevent overfitting OOF)
        meta_model = lgb.LGBMRegressor(
            objective="regression_l1",
            max_depth=4,                # Shallow depth for Meta-model
            num_leaves=15,
            learning_rate=0.01,
            n_estimators=3000,
            colsample_bytree=0.8,
            random_state=42 + fold_k,
            device="cpu",               # Level-2 is tiny, CPU is faster here
            n_jobs=-1
        )
        
        meta_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        val_preds = meta_model.predict(X_va)
        meta_oof_preds[val_idx] = val_preds
        mae_fold = np.mean(np.abs(val_preds - y_va))
        meta_maes.append(mae_fold)
        
        print(f"  Level-2 Fold {fold_k} | Best Iter: {meta_model.best_iteration_:<4} | MAE: {mae_fold:.5f}")
        
        # Add to test predictions (Ensemble average)
        meta_test_preds += meta_model.predict(test_meta_df[meta_features]) / N_FOLDS

    print(f"\n  [Level-2 OOF MAE] {np.mean(meta_maes):.5f} ± {np.std(meta_maes):.5f}")

    # 5. Format Stacked Submission
    print("\n[5] Formatting Stacked Submission...")
    test_meta_df["stacked_pred"] = np.clip(meta_test_preds, 0.0, 5.0)
    
    stacked_pivot = test_meta_df.pivot(index="region_id", columns="week", values="stacked_pred")
    # Reorder to match submission exactly
    stacked_pivot = stacked_pivot.reindex(test_region_order)
    
    sub_stacked = pd.DataFrame({"region_id": test_region_order})
    for w in range(HORIZON):
        sub_stacked[f"pred_week{w+1}"] = stacked_pivot[w].values
        
    sub_stacked.to_csv(FILES_TO_COMPARE["V45 L2-Stacked"], index=False)
    
    # 6. Distribution Diagnostics
    print("\n" + "=" * 95)
    print("Final Submission Distribution Diagnostics")
    print("=" * 95)
    
    header = f"  {'Model / Version':<22} | {'Mean':>6} | {'Std':>6} | {'Zero %':>7} | {'P50':>5} | {'P90':>5} | {'P99':>5} | {'Max':>5}"
    print(header)
    print("  " + "-" * 93)

    for name, filename in FILES_TO_COMPARE.items():
        if not os.path.exists(filename):
            print(f"  {name:<22} | [File not found: {filename}]")
            continue
            
        df = pd.read_csv(filename)
        vals = df.iloc[:, 1:].values.flatten()
        
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        z_frac = np.mean(vals == 0.0)
        p50 = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        p99 = np.percentile(vals, 99)
        max_v = np.max(vals)
        
        print(f"  {name:<22} | {mean_v:6.4f} | {std_v:6.4f} | {z_frac:7.1%} | {p50:5.2f} | {p90:5.2f} | {p99:5.2f} | {max_v:5.2f}")
        
    print("  " + "=" * 93)

    elapsed = time.time() - t0
    print(f"\nStacking completed in {elapsed:.1f}s.")

if __name__ == "__main__":
    main()