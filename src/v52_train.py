"""
v52_train.py
Cross-Seasonal Time CV (v52) + Decoupled Dual-Tree Hurdle Execution Pipeline.
Optimized for multi-horizon drought forecasting.
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

from src.v52_dataset import (
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
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(ROOT, "data", "v51_processed") 
MODELS_DIR    = os.path.join(ROOT, "models", "v52_models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_FOLDS         = 4  
N_FLAT_FEATURES = WINDOW_SIZE * len(FEATURE_COLS) + len(FEATURE_COLS) 

# ---------------------------------------------------------------------------
# Target Encoding (Leakage-Free Per-Fold Execution)
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
# Diagnostics & Matrix Logging
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
    raise ValueError(f"Unknown index bounds: {idx}")

def print_binned_error_matrix(y_true: np.ndarray, y_pred: np.ndarray, fold_k: int, log_fn) -> None:
    log_fn(f"Diagnostics: Binned Error Analysis Matrix -- Fold {fold_k} (OOF Evaluated)")
    log_fn(f"  {'Interval Class':<50}  {'Count':>7}  {'AvgTrue':>8}  {'AvgPred':>8}  {'MAE':>8}")
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

# ---------------------------------------------------------------------------
# Training Pipeline Core
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log("--- [v52_train.py] Pipeline Init ---")
    log(f"Configuration: Cross-Seasonal Time CV Setup | Folds = {N_FOLDS} | Target Horizon = {HORIZON}")
    log("Architecture: Dual-Tree Hurdle (Model A: L1 Regressor | Model B: Binary Classifier)")

    # 1. Load Data
    log("I/O: Loading processed datasets")
    try:
        train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
        test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}\nVerify preprocessing pipeline output matrices exist.")

    log(f"Data: train_raw={train_raw.shape} | test_raw={test_raw.shape}")

    # 2. Feature Refinement
    log("Preprocessing: Refining structural feature matrices")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    log(f"Data: train_df features configuration={train_df.shape} | test_df={test_df.shape}")

    # 3. Handle Missing Target Scores
    before    = len(train_df)
    train_df  = train_df.dropna(subset=["score"]).reset_index(drop=True)
    dropped   = before - len(train_df)
    if dropped: 
        log(f"Data: Filtered out {dropped:,} records missing validation scores")

    # 4. Generate CV splits
    log(f"CV: Generating {N_FOLDS}-Fold Cross-Seasonal chronological matrix blocks")
    folds = build_time_seasonal_cv_folds(train_df, n_splits=N_FOLDS, season_step=13)
    log(f"CV: Chronological block partitioning sequence array complete (folds={len(folds)})")

    # 5. Iterative Execution Loop
    log("Training: Executing dual model configuration training routine")
    fold_results      = []
    fold_test_preds_a = []
    fold_test_preds_b = []

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):
        log(f"--- Fold {fold_k + 1}/{N_FOLDS} ---")
        log(f"Data: Active processing split details: train_groups={len(raw_train_groups):,} | val_groups={len(raw_val_groups):,}")
        fold_t0 = time.time()

        # Isolate target metrics cleanly to preserve validation envelope bounds
        safe_train_df_for_te = extract_training_targets_for_te(raw_train_groups)
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(safe_train_df_for_te)
        log(f"TargetEncoding: Matrix maps updated from {len(safe_train_df_for_te):,} data units")

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

            # --- Model A Optimization Execution ---
            model_a = LGBMRegressor(
                objective        = "regression_l1",
                max_depth        = 7,           
                num_leaves       = 63,          
                colsample_bytree = 0.5,         
                learning_rate    = 0.05,        
                subsample        = 0.8,
                min_child_samples= 150,         
                reg_alpha        = 0,         
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
                eval_metric = "mae",
                callbacks   = [lightgbm.early_stopping(stopping_rounds=200, verbose=False)],
            )
            val_l1_w  = model_a.predict(X_val_np)
            test_l1_w = model_a.predict(X_test_np)
            log(f"Model A: Horizon Step={week_idx+1} | best_iteration={model_a.best_iteration_} | Val MAE={float(np.mean(np.abs(val_l1_w - y_val_w))):.4f}")
            with open(ckpt_a, "wb") as fh: pickle.dump(model_a, fh)
            fold_val_preds_l1[:, week_idx] = val_l1_w
            fold_test_pred_l1[:, week_idx] = test_l1_w.astype(np.float32)

            # --- Model B Optimization Execution ---
            model_b = LGBMClassifier(
                objective        = "binary",
                max_depth        = 7,           
                num_leaves       = 85,          
                colsample_bytree = 0.5,         
                learning_rate    = 0.02,        
                subsample        = 0.8,
                min_child_samples= 500,         
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
            log(f"Model B: Horizon Step={week_idx+1} | best_iteration={model_b.best_iteration_}")
            with open(ckpt_b, "wb") as fh: pickle.dump(model_b, fh)
            fold_val_probs[:, week_idx]    = val_prob_w
            fold_test_prob[:, week_idx]    = test_prob_w.astype(np.float32)

            # Apply hard threshold to out-of-fold predictions
            oof_final_w = np.where(val_prob_w < 0.5, 0.0, val_l1_w)
            fold_val_final[:, week_idx] = oof_final_w

        fold_elapsed = time.time() - fold_t0
        week_maes_final = [float(np.mean(np.abs(fold_val_final[:, w] - y_val_np[:, w]))) for w in range(HORIZON)]
        mean_fold_mae_final = float(np.mean(week_maes_final))

        log(f"Metrics: Fold {fold_k} Mean OOF MAE = {mean_fold_mae_final:.4f} | Execution time = {fold_elapsed:.1f}s")
        print_binned_error_matrix(y_val_np.ravel(), fold_val_final.ravel(), fold_k, log)

        log(f"Diagnostics: Fold {fold_k} Model B Prob Range = [{fold_val_probs.min():.4f}, {fold_val_probs.max():.4f}]")
        log(f"Diagnostics: Fold {fold_k} Model A Score Range = [{fold_val_preds_l1.min():.4f}, {fold_val_preds_l1.max():.4f}]")

        fold_results.append((fold_k, week_maes_final, mean_fold_mae_final))
        fold_test_preds_a.append({"preds": fold_test_pred_l1, "region_ids": test_region_ids})
        fold_test_preds_b.append({"probs": fold_test_prob,    "region_ids": test_region_ids})

        del X_train_np, y_train_np, X_val_np, y_val_np, model_a, model_b
        gc.collect()

    # 6. Global Consolidation & Out-of-Sample Diagnostics
    mean_cv_maes = [r[2] for r in fold_results]
    log(f"Metrics: Overall Engine CV MAE = {np.mean(mean_cv_maes):.4f} +/- {np.std(mean_cv_maes):.4f}")

    preds_a_stack = np.stack([fp["preds"] for fp in fold_test_preds_a], axis=0)
    probs_b_stack = np.stack([fp["probs"] for fp in fold_test_preds_b], axis=0)
    test_region_ids = fold_test_preds_a[0]["region_ids"]

    sample_a_preds = preds_a_stack[:, 0, 0]
    sample_b_probs = probs_b_stack[:, 0, 0]
    log("Diagnostics: Cross-fold variance tracking (Index=0 Horizon=0)")
    log(f"  Model A outputs across folds: {np.round(sample_a_preds, 3)}")
    log(f"  Model B outputs across folds: {np.round(sample_b_probs, 3)}")

    # Export matrices to binary frames for post-processing and gating
    raw_preds_path = os.path.join(MODELS_DIR, "v52_raw_test_preds.pkl")
    with open(raw_preds_path, "wb") as f:
        pickle.dump({
            "preds_a_stack": preds_a_stack,
            "probs_b_stack": probs_b_stack,
            "region_ids": test_region_ids
        }, f)
    log(f"I/O: Raw validation frames saved to {raw_preds_path}")

    log_path = os.path.join(ROOT, "_training_log_52nd.txt")
    with open(log_path, "w") as fh: fh.write("\n".join(log_lines))
    print(f"I/O: Operational activity log structured -> {log_path}")
    print("--- [v52_train.py] Pipeline Complete ---")

    return {"overall_cv_mae": np.mean(mean_cv_maes)}

if __name__ == "__main__":
    results = main()