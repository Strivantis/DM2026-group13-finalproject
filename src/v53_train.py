"""
v53_train.py
Cross-Seasonal Time CV (v53) + Decoupled Dual-Tree Hurdle
[MAJOR UPGRADE] Huber Regression + Enhanced Boundary Classifier + OOF Auto-Tuner
"""

import os
import sys
import time
import random
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
import gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.v53_dataset import (
    refine_features,
    build_time_seasonal_cv_folds,
    extract_training_targets_for_te,
    build_tabular_dataset,
    build_tabular_test,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(ROOT, "data", "v53_processed")
MODELS_DIR    = os.path.join(ROOT, "models","v53_models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_FOLDS         = 4  
N_FLAT_FEATURES = WINDOW_SIZE * len(FEATURE_COLS) + len(FEATURE_COLS) 

# ---------------------------------------------------------------------------
# Per-fold leakage-free target encoding
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Binned error diagnostics
# ---------------------------------------------------------------------------
INTERVAL_LABELS = [
    "Interval 0  [Absolute Zero       y == 0.0      ]",
    "Interval 1  [Mild Drought        0.0 < y <= 1.0]",
    "Interval 2  [Moderate Drought    1.0 < y <= 2.0]",
    "Interval 3  [Severe Drought      2.0 < y <= 3.0]",
    "Interval 4  [Extreme Drought     3.0 < y <= 4.0]",
    "Interval 5  [Exceptional Drought 4.0 < y <= 5.0]",
]

def _interval_mask(y_true: np.ndarray, idx: int) -> np.ndarray:
    if idx == 0: return y_true == 0.0
    if idx == 1: return (y_true > 0.0) & (y_true <= 1.0)
    if idx == 2: return (y_true > 1.0) & (y_true <= 2.0)
    if idx == 3: return (y_true > 2.0) & (y_true <= 3.0)
    if idx == 4: return (y_true > 3.0) & (y_true <= 4.0)
    if idx == 5: return (y_true > 4.0) & (y_true <= 5.0)
    raise ValueError(f"Unknown idx: {idx}")

def print_binned_error_matrix(y_true: np.ndarray, y_pred: np.ndarray, fold_k: int, log_fn) -> None:
    log_fn("")
    log_fn("  " + "=" * 80)
    log_fn(f"  BINNED ERROR MATRIX  --  Fold {fold_k}")
    log_fn("  " + "=" * 80)
    log_fn(f"  {'Interval':<50}  {'Count':>7}  {'AvgTrue':>8}  {'AvgPred':>8}  {'MAE':>8}")
    log_fn("  " + "-" * 78)
    for idx, label in enumerate(INTERVAL_LABELS):
        mask = _interval_mask(y_true, idx)
        n    = int(mask.sum())
        if n == 0:
            log_fn(f"  {label:<50}  {n:>7}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}")
            continue
        avg_true = float(y_true[mask].mean())
        avg_pred = float(y_pred[mask].mean())
        mae      = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
        log_fn(f"  {label:<50}  {n:>7,}  {avg_true:>8.4f}  {avg_pred:>8.4f}  {mae:>8.4f}")
    log_fn("  " + "=" * 80)
    log_fn("")

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 90)
    log("Drought Forecasting Pipeline v53 (Time CV | Huber + Enhanced Boundary)")
    log("Model A: LGBMRegressor (Huber)  |  Model B: LGBMClassifier (Binary)")
    log(f"CV: {N_FOLDS}-Fold Time Seasonal CV (Strict no future leakage)")
    log("=" * 90)

    # -- 1. Load processed data -----------------------------------------------
    log("\nLoading processed data ...")
    try:
        train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
        test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}\n  --> Run `python src/v53_preprocess.py` first.")

    log(f"  train: {train_raw.shape}  |  test: {test_raw.shape}")

    # -- 2. Feature refinement ------------------------------------------------
    log("\nRefining features ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    log(f"  train: {train_df.shape}  |  test: {test_df.shape}")

    # -- 3. Drop NaN score rows -----------------------------------------------
    before    = len(train_df)
    train_df  = train_df.dropna(subset=["score"]).reset_index(drop=True)
    dropped   = before - len(train_df)
    if dropped: log(f"  Dropped {dropped:,} NaN-score rows.")

    # -- 4. Build CV folds ----------------------------------------------------
    log(f"\n{'='*90}")
    log(f"Building {N_FOLDS}-Fold Cross-Seasonal CV")
    log(f"{'='*90}")

    folds = build_time_seasonal_cv_folds(train_df, n_splits=N_FOLDS, season_step=13)
    log(f"  Folds built: {len(folds)}")

    # -- 5. Dual-Tree Hurdle Training Loop ------------------------------------
    log(f"\n{'='*90}")
    log("Dual-Tree Hurdle Training (V53 HEAVYWEIGHT)")
    log(f"{'='*90}")

    fold_results      = []
    fold_test_preds_a = []
    fold_test_preds_b = []
    
    # [V53 新增] 儲存全局 OOF 陣列，用於稍後的 Threshold 最佳化
    oof_val_probs_all = []
    oof_val_preds_a_all = []
    y_val_all = []

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):
        log(f"\n{'='*90}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}")
        log(f"  train_groups: {len(raw_train_groups):,}  |  val_groups: {len(raw_val_groups):,}")
        log(f"{'='*90}")

        fold_t0 = time.time()

        safe_train_df_for_te = extract_training_targets_for_te(raw_train_groups)
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(safe_train_df_for_te)
        log(f"  [TE] Computed from {len(safe_train_df_for_te):,} historical records.")

        aug_train_groups = _augment_groups_with_te(raw_train_groups, te_map_fold, gm_fold, gzp_fold)
        aug_val_groups   = _augment_groups_with_te(raw_val_groups, te_map_fold, gm_fold, gzp_fold)

        feat_cols = [c for c in FEATURE_COLS if c in aug_train_groups[0][0].columns]
        
        X_train_np, y_train_np, _ = build_tabular_dataset(aug_train_groups, feat_cols)
        X_val_np,   y_val_np,   _ = build_tabular_dataset(aug_val_groups,   feat_cols)
        X_train_np = np.asfortranarray(X_train_np)
        X_val_np   = np.asfortranarray(X_val_np)

        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test_np, test_region_ids = build_tabular_test(test_df_fold, feat_cols)
        X_test_np = np.asfortranarray(X_test_np)

        fold_val_preds_l1 = np.zeros_like(y_val_np)
        fold_val_probs    = np.zeros_like(y_val_np)
        fold_val_final    = np.zeros_like(y_val_np) # 預設以 0.5 切割供觀察
        fold_test_pred_l1 = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)
        fold_test_prob    = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)

        for week_idx in range(HORIZON):
            y_train_w = y_train_np[:, week_idx]
            y_val_w   = y_val_np[:,   week_idx]
            y_train_b = (y_train_w > 0.0).astype(int)
            y_val_b   = (y_val_w   > 0.0).astype(int)

            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            ckpt_b = os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl")

            log(f"\n  --- Fold {fold_k} | Week {week_idx + 1} ---")

            # --- Model A [終極覺醒版] True Hurdle Regression ---
            # 1. 建立遮罩：只取真實分數 > 0 的資料來訓練 Model A
            pos_mask = y_train_w > 0.0
            
            # 2. 確保有足夠的乾旱樣本才訓練
            if pos_mask.sum() > 100:
                model_a = LGBMRegressor(
                    objective        = "regression", # [修正] 放棄 Huber，用 MSE 狠狠懲罰誤差
                    max_depth        = 7,           
                    num_leaves       = 63,          
                    colsample_bytree = 0.5,         
                    learning_rate    = 0.03,        
                    subsample        = 0.8,
                    min_child_samples= 100,          # 降低約束，讓它能捕捉極端值
                    reg_alpha        = 0.0,         
                    reg_lambda       = 1.0,         
                    n_estimators     = 10000,       
                    device           = "gpu",
                    random_state     = 42,
                    n_jobs           = -1,
                    verbose          = -1,
                )
                
                # 3. 訓練時，X 和 y 都只餵入 >0 的資料！
                model_a.fit(
                    X_train_np[pos_mask], y_train_w[pos_mask],
                    # Evaluation 依然看全部的 Val Set，這樣 MAE 才是真實的
                    eval_set    = [(X_val_np, y_val_w)], 
                    eval_metric = "mae",
                    callbacks   = [lightgbm.early_stopping(stopping_rounds=300, verbose=False)], 
                )
            else:
                # 萬一該週完全沒有乾旱資料 (極罕見防呆)
                model_a = None
                
            # 4. 預測時，對全部的 Validation 和 Test 進行預測
            if model_a is not None:
                val_l1_w  = model_a.predict(X_val_np)
                test_l1_w = model_a.predict(X_test_np)
                log(f"    [Model A] best_iter={model_a.best_iteration_} | val_MAE={float(np.mean(np.abs(val_l1_w - y_val_w))):.4f}")
                with open(ckpt_a, "wb") as fh: pickle.dump(model_a, fh)
            else:
                val_l1_w  = np.zeros_like(y_val_w)
                test_l1_w = np.zeros(X_test_np.shape[0], dtype=np.float32)
                log(f"    [Model A] SKIPPED (No positive samples)")

            fold_val_preds_l1[:, week_idx] = val_l1_w
            fold_test_pred_l1[:, week_idx] = test_l1_w.astype(np.float32)

            # --- Model B [V53 升級] Enhanced Boundary Classifier ---
            model_b = LGBMClassifier(
                objective        = "binary",
                max_depth        = 8,           # 加深，切出微弱降雨的細微邊界
                num_leaves       = 127,         # 大幅增廣葉子數量
                colsample_bytree = 0.5,         
                learning_rate    = 0.02,        
                subsample        = 0.8,
                min_child_samples= 100,         # 降低約束，讓模型敢給出極端的機率 (接近0或1)
                reg_alpha        = 0.5,         
                reg_lambda       = 2.0,         
                n_estimators     = 20000,       
                device           = "gpu",
                random_state     = 42,
                n_jobs           = -1,
                verbose          = -1,
            )
            model_b.fit(
                X_train_np, y_train_b,
                eval_set    = [(X_val_np, y_val_b)],
                eval_metric = "binary_logloss",
                callbacks   = [lightgbm.early_stopping(stopping_rounds=300, verbose=False)],
            )
            val_prob_w  = model_b.predict_proba(X_val_np)[:, 1]
            test_prob_w = model_b.predict_proba(X_test_np)[:, 1]
            log(f"    [Model B] best_iter={model_b.best_iteration_}")
            with open(ckpt_b, "wb") as fh: pickle.dump(model_b, fh)
            fold_val_probs[:, week_idx]    = val_prob_w
            fold_test_prob[:, week_idx]    = test_prob_w.astype(np.float32)

            # 預設觀察閘門 (後續由 Auto-Tuner 決定最佳值)
            oof_final_w = np.where(val_prob_w < 0.5, 0.0, val_l1_w)
            fold_val_final[:, week_idx] = oof_final_w

        fold_elapsed = time.time() - fold_t0
        week_maes_final = [float(np.mean(np.abs(fold_val_final[:, w] - y_val_np[:, w]))) for w in range(HORIZON)]
        mean_fold_mae_final = float(np.mean(week_maes_final))

        log(f"\n  [Fold {fold_k}] Mean MAE (Thresh=0.5): {mean_fold_mae_final:.4f} ({fold_elapsed:.1f}s)")
        print_binned_error_matrix(y_val_np.ravel(), fold_val_final.ravel(), fold_k, log)

        log(f"  [Diagnostics Fold {fold_k}] Model B Probability range: Min={fold_val_probs.min():.4f}, Max={fold_val_probs.max():.4f}")
        log(f"  [Diagnostics Fold {fold_k}] Model A Score       range: Min={fold_val_preds_l1.min():.4f}, Max={fold_val_preds_l1.max():.4f}")

        fold_results.append((fold_k, week_maes_final, mean_fold_mae_final))
        fold_test_preds_a.append({"preds": fold_test_pred_l1, "region_ids": test_region_ids})
        fold_test_preds_b.append({"probs": fold_test_prob,    "region_ids": test_region_ids})
        
        # 收集 OOF 資料以供自動尋優
        oof_val_probs_all.append(fold_val_probs.ravel())
        oof_val_preds_a_all.append(fold_val_preds_l1.ravel())
        y_val_all.append(y_val_np.ravel())

        del X_train_np, y_train_np, X_val_np, y_val_np, model_a, model_b
        gc.collect()

    # -- 6. OOF Threshold Auto-Tuning [V53 終極進化] ---------------------------
    log(f"\n{'='*90}")
    log("V53 Auto-Tuning: Optimizing Hurdle Threshold based on Global OOF MAE")
    log(f"{'='*90}")
    
    global_probs = np.concatenate(oof_val_probs_all)
    global_preds = np.concatenate(oof_val_preds_a_all)
    global_y     = np.concatenate(y_val_all)
    
    best_thresh = 0.5
    best_global_mae = 999.0
    
    log(f"  {'Threshold':<10} | {'Global OOF MAE':<15} | {'Zero Fraction':<15}")
    log("-" * 50)
    
    for thresh in np.arange(0.30, 0.96, 0.02):
        gated_preds = np.where(global_probs < thresh, 0.0, global_preds)
        current_mae = np.mean(np.abs(gated_preds - global_y))
        zero_frac   = (gated_preds == 0.0).mean()
        
        log(f"  {thresh:<10.2f} | {current_mae:<15.4f} | {zero_frac:<15.2%}")
        
        if current_mae < best_global_mae:
            best_global_mae = current_mae
            best_thresh = thresh

    log("-" * 50)
    log(f"🔥 BEST THRESHOLD DETERMINED: {best_thresh:.2f} (Global MAE: {best_global_mae:.4f})")

    # -- 7. Summary and Asymmetric Ensemble -----------------------------------
    log(f"\n{'='*90}")
    log(f"Asymmetric Ensemble (applying Best Threshold = {best_thresh:.2f})")
    log(f"{'='*90}")

    preds_a_stack = np.stack([fp["preds"] for fp in fold_test_preds_a], axis=0)
    probs_b_stack = np.stack([fp["probs"] for fp in fold_test_preds_b], axis=0)
    test_region_ids = fold_test_preds_a[0]["region_ids"]

    sample_a_preds = preds_a_stack[:, 0, 0]
    sample_b_probs = probs_b_stack[:, 0, 0]
    log(f"\n[Test Set Sample 0, Week 1 Diagnostic]")
    log(f"  4 Folds Model A preds: {np.round(sample_a_preds, 3)}")
    log(f"  4 Folds Model B probs: {np.round(sample_b_probs, 3)}")

    # 儲存推論需要的 raw preds 與 自動尋找出的最佳 Threshold
    raw_preds_path = os.path.join(MODELS_DIR, "v53_raw_test_preds.pkl")
    with open(raw_preds_path, "wb") as f:
        pickle.dump({
            "preds_a_stack": preds_a_stack,
            "probs_b_stack": probs_b_stack,
            "region_ids": test_region_ids,
            "best_threshold": float(best_thresh) # 供 v53_infer.py 讀取
        }, f)
    log(f"\nSaved raw test predictions & Best Threshold to {raw_preds_path}")

    log_path = os.path.join(ROOT, "_training_log_53rd.txt")
    with open(log_path, "w") as fh: fh.write("\n".join(log_lines))
    print(f"Training log -> {log_path}")

    return {"best_global_mae": best_global_mae}

if __name__ == "__main__":
    results = main()