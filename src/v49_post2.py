"""
v50_meta_stacking.py - Ultimate Dual-Engine Fusion (V45.1 L1 + V49 Probabilities)

Architecture:
  Level 1 (Base Models):
    - V45.1 (LGBMRegressor) -> Continuous robust L1 predictions.
    - V49 (LGBMClassifier) -> Sensitive 6-dimensional probability distributions.
  Level 2 (Meta-Model):
    - LightGBM (objective=regression_l1) -> Learns the optimal non-linear blending
      of the 7 meta-features (1 L1 + 6 Probs) to minimize expected MAE.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features, build_stratified_group_cv_folds, 
    build_tabular_dataset, build_tabular_test, FEATURE_COLS, HORIZON
)

# =====================================================================
# Configuration Paths
# =====================================================================
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# V45.1 Meta-Features (確保這些檔案已經存在專案根目錄下)
V45_OOF_PATH  = os.path.join(ROOT, "v45.1_meta_train_oof.csv")
V45_TEST_PATH = os.path.join(ROOT, "v45.1_meta_test_preds.csv")

# V49 PKL Directory (請確認這是你原本 v49 模型存放的路徑)
V49_MODELS_DIR = os.path.join(ROOT, "models")

SUB_OUTPUT    = os.path.join(ROOT, "submission_49th_meta_stacked.csv")

# =====================================================================
# Target Encoding Helpers (為了重現 V49 機率所需)
# =====================================================================
def _zero_prob(x): return (x == 0.0).mean()

def _compute_te_stats(df: pd.DataFrame) -> tuple:
    te_stats = df.groupby("region_id")["score"].agg(region_mean_score="mean", region_zero_prob=_zero_prob).reset_index()
    gm, gzp = float(te_stats["region_mean_score"].mean()), float(te_stats["region_zero_prob"].mean())
    te_map = {row["region_id"]: (float(row["region_mean_score"]), float(row["region_zero_prob"])) for _, row in te_stats.iterrows()}
    return te_map, gm, gzp

def _augment_groups_with_te(groups, te_map, gm, gzp):
    result = []
    for entry in groups:
        g, i_min, i_max = entry[0].copy(), entry[1], entry[2]
        m, z = te_map.get(g["region_id"].iloc[0], (gm, gzp))
        g["region_mean_score"] = np.float32(m)
        g["region_zero_prob"]  = np.float32(z)
        result.append((g, i_min, i_max))
    return result

def _merge_te_to_df(df, te_map, gm, gzp):
    df = df.copy()
    df["region_mean_score"] = df["region_id"].map(lambda rid: te_map.get(rid, (gm, gzp))[0]).astype(np.float32)
    df["region_zero_prob"] = df["region_id"].map(lambda rid: te_map.get(rid, (gm, gzp))[1]).astype(np.float32)
    return df

# =====================================================================
# Main Execution
# =====================================================================
def main():
    print("=" * 80)
    print(" 🚀 V50 ULTIMATE META-L1 STACKING (V45.1 + V49) ")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Step 1: 載入 V45.1 的 OOF 與 Test 特徵
    # -----------------------------------------------------------------
    print("\n[1/4] Loading V45.1 Base Predictions...")
    if not os.path.exists(V45_OOF_PATH) or not os.path.exists(V45_TEST_PATH):
        raise FileNotFoundError(f"Missing V45.1 files. Check paths:\n{V45_OOF_PATH}\n{V45_TEST_PATH}")
        
    v45_oof = pd.read_csv(V45_OOF_PATH)
    v45_test = pd.read_csv(V45_TEST_PATH)
    
    # 為了嚴格對齊，我們等一下會在迴圈內動態對齊 v45.1 的資料
    
    # -----------------------------------------------------------------
    # Step 2: 從 V49 PKL 重建 6D 機率特徵
    # -----------------------------------------------------------------
    print("\n[2/4] Reconstructing V49 6D Probability Space...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    train_df["score"] = np.round(train_df["score"]).astype(int)

    folds = build_stratified_group_cv_folds(train_df, n_splits=5)
    
    meta_X_train_list = []
    meta_y_train_list = []
    meta_groups_list = []
    
    n_test_regions = test_df["region_id"].nunique()
    v49_test_probs_all = np.zeros((5, n_test_regions, HORIZON, 6), dtype=np.float32)
    master_region_order = None

    for fold_k, (train_groups, val_groups) in enumerate(folds):
        print(f"  -> Processing Fold {fold_k + 1} / 5 ...")
        
        train_rids = {e[0]["region_id"].iloc[0] for e in train_groups}
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(train_df[train_df["region_id"].isin(train_rids)])
        aug_val = _augment_groups_with_te(val_groups, te_map_fold, gm_fold, gzp_fold)
        
        X_val, y_val, val_rids = build_tabular_dataset(aug_val, FEATURE_COLS)
        X_val = np.asfortranarray(X_val)

        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test, test_rids = build_tabular_test(test_df_fold, FEATURE_COLS)
        X_test = np.asfortranarray(X_test)
        if fold_k == 0: master_region_order = test_rids
        
        for week_idx in range(HORIZON):
            # 讀取 V49 PKL
            ckpt = os.path.join(V49_MODELS_DIR, f"lgbm_multi_fold{fold_k}_week{week_idx}.pkl")
            with open(ckpt, "rb") as fh: 
                model_v49 = pickle.load(fh)
            
            # 推論 V49 機率
            v49_val_prob = model_v49.predict_proba(X_val)   # Shape: (N, 6)
            v49_test_probs_all[fold_k, :, week_idx, :] = model_v49.predict_proba(X_test)
            
            # 從 V45 OOF 檔案中提取對應的 L1 預測值
            # 確保對齊 fold 和 week
            v45_fold_week = v45_oof[(v45_oof["fold"] == fold_k) & (v45_oof["horizon_week"] == week_idx + 1)]
            
            # 由於我們需要對齊 v45 OOF 的 region_id 順序，使用 map
            l1_preds = np.zeros(len(val_rids))
            # 為了提速，建立一個臨時字典
            rid_to_l1 = dict(zip(v45_fold_week["region_id"], v45_fold_week["model_a_l1_pred"]))
            for i, rid in enumerate(val_rids):
                # 如果某個 region 出現多次，這是一種簡化對齊方式 (實戰中通常足夠精確)
                l1_preds[i] = rid_to_l1.get(rid, 0.0)
            
            # 串接特徵: V45_L1 (1維) + V49_Probs (6維) = 7維 Meta 特徵
            meta_features = np.column_stack([l1_preds, v49_val_prob])
            
            meta_X_train_list.append(meta_features)
            meta_y_train_list.append(y_val[:, week_idx])
            meta_groups_list.append(val_rids)

    # -----------------------------------------------------------------
    # Step 3: 訓練 Meta-Model (L2 Level)
    # -----------------------------------------------------------------
    print("\n[3/4] Training Level-2 Meta-Regressor...")
    X_meta_train = np.vstack(meta_X_train_list)
    y_meta_train = np.concatenate(meta_y_train_list)
    groups_meta = np.concatenate(meta_groups_list)
    
    oof_meta_preds = np.zeros_like(y_meta_train, dtype=np.float64)
    gkf = GroupKFold(n_splits=5)
    
    meta_models = []
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_meta_train, y_meta_train, groups=groups_meta)):
        X_tr, y_tr = X_meta_train[tr_idx], y_meta_train[tr_idx]
        X_va, y_va = X_meta_train[va_idx], y_meta_train[va_idx]
        
        # 使用極淺的樹防止 Meta-Model 過擬合
        meta_model = LGBMRegressor(
            objective="regression_l1",
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        
        meta_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                # 使用簡單的 early stopping 即可，因為特徵只有 7 維，收斂極快
                # lightgbm.early_stopping is removed to avoid API warnings in newer versions. Let it run fully.
            ]
        )
        
        oof_meta_preds[va_idx] = meta_model.predict(X_va)
        meta_models.append(meta_model)
        
    oof_rounded = np.clip(np.round(oof_meta_preds), 0, 5)
    meta_mae = np.mean(np.abs(y_meta_train - oof_rounded))
    print(f"🏆 Meta-Model OOF Rounded MAE: {meta_mae:.4f}")

    # -----------------------------------------------------------------
    # Step 4: 推論 Test Set 並儲存
    # -----------------------------------------------------------------
    print("\n[4/4] Generating Final Blended Predictions...")
    
    # 計算 V49 在 Test 上的平均 6D 機率
    v49_test_probs_mean = np.mean(v49_test_probs_all, axis=0) # Shape: (2248, 5, 6)
    
    final_rows = []
    
    for i, rid in enumerate(master_region_order):
        row_data = {"region_id": rid}
        for w in range(HORIZON):
            # 取得 V45 的 L1 預測
            v45_val = v45_test.loc[(v45_test["region_id"] == rid) & (v45_test["horizon_week"] == w+1), "model_a_l1_pred"].values[0]
            
            # 取得 V49 的 6D 機率
            v49_probs = v49_test_probs_mean[i, w, :]
            
            # 組合 7D 特徵
            test_meta_feat = np.concatenate([[v45_val], v49_probs]).reshape(1, -1)
            
            # 透過 5 個 Meta-Model 進行預測並取中位數
            preds = [m.predict(test_meta_feat)[0] for m in meta_models]
            final_cont = np.median(preds)
            final_int = int(np.clip(np.round(final_cont), 0, 5))
            
            row_data[f"pred_week{w+1}"] = final_int
            
        final_rows.append(row_data)

    sub_df = pd.DataFrame(final_rows)
    sub_df.to_csv(SUB_OUTPUT, index=False)
    
    print(f"\n✅ SUCCESS! Ultimate Fusion Submissions saved to:")
    print(f"   -> {SUB_OUTPUT}")

if __name__ == "__main__":
    main()