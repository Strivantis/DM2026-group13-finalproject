"""
v54_train.py
Cross-Seasonal Time CV (v54) + Decoupled Dual-Tree Hurdle
[CLEANUP] Removed Target Encoding, Leveraging Data-Driven Nuclear Features.
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

# [V54 修正] 確保引入的是 v54_dataset，這才包含核彈特徵且不包含 TE
from src.v54_dataset import (
    refine_features,
    build_time_seasonal_cv_folds,    
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
PROCESSED_DIR = os.path.join(ROOT, "data", "v54_processed") 
MODELS_DIR    = os.path.join(ROOT, "models","v54_models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_FOLDS         = 4  
N_FLAT_FEATURES = WINDOW_SIZE * len(FEATURE_COLS) + len(FEATURE_COLS) 

# [V54 淨化] 徹底移除了 _compute_te_stats 等所有 Target Encoding 相關的輔助函數。
# 因為在 Time-Fold 且面對 Test Set 嚴重 Temporal Shift 時，依賴歷史平均分是致命的毒藥。

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
    log_fn(f"  BINNED ERROR MATRIX  --  Fold {fold_k}  (OOF Hurdle-Gated)")
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
    log("Drought Forecasting Pipeline v54 (Cross-Seasonal Time CV | Data-Driven Interventions)")
    log("Model A: LGBMRegressor (MSE/L2)  |  Model B: LGBMClassifier (Binary)")
    log("Test gate: np.median(A, axis=folds)  |  np.mean(B, axis=folds)")
    log(f"CV: {N_FOLDS}-Fold Time Seasonal CV (Strict no future leakage)")
    log("=" * 90)

    # -- 1. Load processed data -----------------------------------------------
    log("\nLoading processed data ...")
    try:
        train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
        test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}\n  --> Run `python src/v54_preprocess.py` first.")

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
    log("Dual-Tree Hurdle Training (V54 CORE)")
    log(f"{'='*90}")

    fold_results      = []
    fold_test_preds_a = []
    fold_test_preds_b = []

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):
        log(f"\n{'='*90}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}")
        log(f"  train_groups: {len(raw_train_groups):,}  |  val_groups: {len(raw_val_groups):,}")
        log(f"{'='*90}")

        fold_t0 = time.time()

        # [V54 淨化] 不再進行任何 TE 特徵的計算與綁定
        feat_cols = [c for c in FEATURE_COLS if c in raw_train_groups[0][0].columns]
        
        X_train_np, y_train_np, _ = build_tabular_dataset(raw_train_groups, feat_cols)
        X_val_np,   y_val_np,   _ = build_tabular_dataset(raw_val_groups,   feat_cols)
        X_train_np = np.asfortranarray(X_train_np)
        X_val_np   = np.asfortranarray(X_val_np)

        X_test_np, test_region_ids = build_tabular_test(test_df, feat_cols)
        X_test_np = np.asfortranarray(X_test_np)

        fold_val_preds_l1 = np.zeros_like(y_val_np)
        fold_val_probs    = np.zeros_like(y_val_np)
        fold_val_final    = np.zeros_like(y_val_np)
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

            # --- Model A: Severity Regression ---
            model_a = LGBMRegressor(
                objective        = "regression",  # [維持] MSE 強迫重視極端值
                max_depth        = 7,
                num_leaves       = 63,
                colsample_bytree = 0.5,         
                learning_rate    = 0.02,          # [配合 MSE] 穩定收斂
                subsample        = 0.8,
                min_child_samples= 100,           # [維持] 允許切出極端高分葉子    
                reg_alpha        = 0.0,         
                reg_lambda       = 1.0,         
                n_estimators     = 10000,       
                device           = "gpu",
                random_state     = 42,
                n_jobs           = -1,
                verbose          = -1,
            )
            model_a.fit(
                X_train_np, y_train_w,
                eval_set    = [(X_val_np, y_val_w)],
                eval_metric = "mae",              # [注意] 指標依然看 MAE，對齊 Kaggle
                callbacks   = [lightgbm.early_stopping(stopping_rounds=300, verbose=False)], 
            )
            val_l1_w  = model_a.predict(X_val_np)
            test_l1_w = model_a.predict(X_test_np)
            log(f"    [Model A] best_iter={model_a.best_iteration_} | val_MAE={float(np.mean(np.abs(val_l1_w - y_val_w))):.4f}")
            with open(ckpt_a, "wb") as fh: pickle.dump(model_a, fh)
            fold_val_preds_l1[:, week_idx] = val_l1_w
            fold_test_pred_l1[:, week_idx] = test_l1_w.astype(np.float32)

            # --- Model B: Drought Probability Classification ---
            model_b = LGBMClassifier(
                objective        = "binary",
                max_depth        = 7,           # [退回穩健版] 防止過度激進的布林特徵導致過擬合
                num_leaves       = 85,          
                colsample_bytree = 0.5,         
                learning_rate    = 0.02,        
                subsample        = 0.8,
                min_child_samples= 500,         # [退回穩健版] 強制葉子有足夠統計量，壓制機率膨脹
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

            # 預設 0.5 觀察閾值
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

        del X_train_np, y_train_np, X_val_np, y_val_np, model_a, model_b
        gc.collect()

    # -- 6. Summary and Asymmetric Ensemble -----------------------------------
    log(f"\n{'='*90}")
    mean_cv_maes = [r[2] for r in fold_results]
    log(f"  Overall CV MAE (Thresh=0.5): {np.mean(mean_cv_maes):.4f}  +/-  {np.std(mean_cv_maes):.4f}")

    preds_a_stack = np.stack([fp["preds"] for fp in fold_test_preds_a], axis=0)
    probs_b_stack = np.stack([fp["probs"] for fp in fold_test_preds_b], axis=0)
    
    test_region_ids = fold_test_preds_a[0]["region_ids"]

    sample_a_preds = preds_a_stack[:, 0, 0]
    sample_b_probs = probs_b_stack[:, 0, 0]
    log(f"\n[Test Set Sample 0, Week 1 Diagnostic]")
    log(f"  4 Folds Model A preds: {np.round(sample_a_preds, 3)}")
    log(f"  4 Folds Model B probs: {np.round(sample_b_probs, 3)}")

    # 儲存推論需要的 raw preds
    raw_preds_path = os.path.join(MODELS_DIR, "v54_raw_test_preds.pkl")
    with open(raw_preds_path, "wb") as f:
        pickle.dump({
            "preds_a_stack": preds_a_stack,
            "probs_b_stack": probs_b_stack,
            "region_ids": test_region_ids
        }, f)
    log(f"\nSaved raw test predictions to {raw_preds_path}")

    log_path = os.path.join(ROOT, "_training_log_54th.txt")
    with open(log_path, "w") as fh: fh.write("\n".join(log_lines))
    print(f"Training log -> {log_path}")

    return {"overall_cv_mae": np.mean(mean_cv_maes)}

if __name__ == "__main__":
    results = main()