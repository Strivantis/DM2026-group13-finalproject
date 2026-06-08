"""
train.py – Drought Score Forecasting Pipeline (v49).

Architecture: Ordinal Expectation (Single Multiclass Classifier)
  Model – LGBMClassifier objective=multiclass, num_class=6
  Math Logic:
    1. Output probabilities P(0) to P(5) for each target integer.
    2. Compute continuous Expected Score: E[y] = sum(i * P_i).
    3. Final Prediction = np.round(E[y]).
  This eliminates the "L1 Median Under-prediction" issue for extreme droughts
  while naturally forcing non-drought areas to absolute 0.0.

Feature space: 29 features × 13 weeks + 29 deltas = 406 dimensions.
               (Using the original un-pruned dataset).

Outputs:
  submission.csv
  models/lgbm_multi_fold{k}_week{w}.pkl
  _training_log_49th.txt
"""

import os
import sys
import time
import random
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import gc

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
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_FOLDS = 5
N_FLAT_FEATURES = WINDOW_SIZE * len(FEATURE_COLS) + len(FEATURE_COLS)

# ---------------------------------------------------------------------------
# Target Encoding (TE)
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
# Custom Metric & Diagnostics
# ---------------------------------------------------------------------------
def custom_expected_mae(y_true, y_pred):
    """
    計算連續期望值 MAE，防止 Early Stopping 陷入平原期。
    安全地將 LightGBM 的 1D 多分類預測陣列重塑（Reshape）回 (N, 6)。
    """
    num_class = 6
    
    # 關鍵修復：LightGBM 在訓練期間傳入的 y_pred 是一維的 Fortran 順序陣列
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, num_class, order='F') # 'F' 代表 Fortran 順序，絕對不能打錯
        
    classes = np.arange(num_class)
    expected = np.sum(y_pred * classes, axis=1)
    mae = np.mean(np.abs(y_true - expected))
    return 'expected_mae', mae, False

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
    raise ValueError(f"Unknown interval index: {idx}")

def print_binned_error_matrix(y_true: np.ndarray, y_pred_rounded: np.ndarray, y_pred_cont: np.ndarray, fold_k: int, log_fn) -> None:
    log_fn("\n  " + "=" * 90)
    log_fn(f"  BINNED ERROR MATRIX  --  Fold {fold_k}  (Ordinal Expected -> Rounded)")
    log_fn("  " + "=" * 90)
    log_fn(f"  {'Interval':<50}  {'Count':>7}  {'AvgTrue':>8}  {'AvgExpct':>8}  {'RoundedMAE':>10}")
    log_fn("  " + "-" * 90)
    for idx, label in enumerate(INTERVAL_LABELS):
        mask = _interval_mask(y_true, idx)
        n = int(mask.sum())
        if n == 0: continue
        avg_true = float(y_true[mask].mean())
        avg_exp = float(y_pred_cont[mask].mean())
        mae_round = float(np.mean(np.abs(y_pred_rounded[mask] - y_true[mask])))
        log_fn(f"  {label:<50}  {n:>7,}  {avg_true:>8.4f}  {avg_exp:>8.4f}  {mae_round:>10.4f}")
    log_fn("  " + "=" * 90 + "\n")

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
    log("Drought Forecasting Pipeline  v49  (Ordinal Expectation)")
    log("Model: LGBMClassifier (objective=multiclass, num_class=6)")
    log(f"Feature space: {len(FEATURE_COLS)} effective × {WINDOW_SIZE} weeks + {len(FEATURE_COLS)} deltas = {N_FLAT_FEATURES} dims")
    log("Prediction: np.round( sum(P_i * i) )")
    log("=" * 90)

    # -- 1. Load processed data
    log("\nLoading processed data ...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))

    # -- 2. Feature refinement
    log("\nRefining features ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)

    before = len(train_df)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    # [Critical] Convert Target to Integer for Multiclass
    train_df["score"] = np.round(train_df["score"]).astype(int)
    if before - len(train_df): log(f"  Dropped {before - len(train_df):,} NaN-score rows.")

    # -- 3. Build CV folds
    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)

    # -- 4. Multiclass Training Loop
    fold_results = []
    fold_test_preds = []

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):
        log(f"\n{'='*90}\nFOLD {fold_k + 1} / {N_FOLDS}\n{'='*90}")
        fold_t0 = time.time()

        # TE
        train_rids = {e[0]["region_id"].iloc[0] for e in raw_train_groups}
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(train_df[train_df["region_id"].isin(train_rids)])
        
        aug_train = _augment_groups_with_te(raw_train_groups, te_map_fold, gm_fold, gzp_fold)
        aug_val = _augment_groups_with_te(raw_val_groups, te_map_fold, gm_fold, gzp_fold)

        # Tabular Matrices
        X_train_np, y_train_np, _ = build_tabular_dataset(aug_train, FEATURE_COLS)
        X_val_np,   y_val_np,   _ = build_tabular_dataset(aug_val, FEATURE_COLS)
        X_train_np = np.asfortranarray(X_train_np)
        X_val_np   = np.asfortranarray(X_val_np)

        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test_np, test_region_ids = build_tabular_test(test_df_fold, FEATURE_COLS)
        X_test_np = np.asfortranarray(X_test_np)

        fold_val_expected = np.zeros_like(y_val_np, dtype=np.float32)
        fold_test_expected = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)

        for week_idx in range(HORIZON):
            # Target must be integer
            y_train_w = y_train_np[:, week_idx].astype(int)
            y_val_w   = y_val_np[:, week_idx].astype(int)
            ckpt = os.path.join(MODELS_DIR, f"lgbm_multi_fold{fold_k}_week{week_idx}.pkl")
            log(f"\n  --- Week {week_idx + 1} ---")

            if os.path.exists(ckpt):
                with open(ckpt, "rb") as fh: model = pickle.load(fh)
                log(f"    [RESUME] {ckpt}")
            else:
                model = LGBMClassifier(
                    objective="multiclass",
                    num_class=6,
                    max_depth=7,
                    num_leaves=55,
                    colsample_bytree=0.8,
                    learning_rate=0.05,
                    subsample=0.8,
                    subsample_freq=1,
                    
                    # [關鍵修復 1] 從 250 降回 LightGBM 預設的 20
                    # 允許稀有類別 (Score 4, 5) 建立較小的葉子節點
                    min_child_samples=20,     
                    
                    # [關鍵修復 2] 加上 min_child_weight 作為防過擬合的第二道防線
                    # 避免 min_child_samples 降太低導致死背雜訊
                    min_child_weight=0.01,    
                    
                    n_estimators=15000, 
                    device="gpu",             
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
                model.fit(
                    X_train_np, y_train_w,
                    eval_set=[(X_val_np, y_val_w)],
                    eval_metric=custom_expected_mae,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=300, verbose=False),
                        lgb.log_evaluation(period=1000),
                    ],
                )
                with open(ckpt, "wb") as fh: pickle.dump(model, fh)

            # Inference: Get probabilities (N, 6)
            val_probs = model.predict_proba(X_val_np)
            test_probs = model.predict_proba(X_test_np)

            # Compute Expected Score
            classes = np.arange(6)
            fold_val_expected[:, week_idx] = np.sum(val_probs * classes, axis=1)
            fold_test_expected[:, week_idx] = np.sum(test_probs * classes, axis=1)
            
            mae_w = np.mean(np.abs(np.round(fold_val_expected[:, week_idx]) - y_val_w))
            log(f"    [OOF] Rounded MAE = {mae_w:.4f}")

            del model; gc.collect()

        fold_val_rounded = np.clip(np.round(fold_val_expected), 0, 5)
        week_maes = [float(np.mean(np.abs(fold_val_rounded[:, w] - y_val_np[:, w]))) for w in range(HORIZON)]
        mean_mae = float(np.mean(week_maes))

        log(f"\n  [Fold {fold_k}] Rounded MAE: " + "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes)))
        log(f"  [Fold {fold_k}] Mean MAE     : {mean_mae:.4f}  ({time.time() - fold_t0:.1f}s)")
        
        print_binned_error_matrix(y_val_np.ravel(), fold_val_rounded.ravel(), fold_val_expected.ravel(), fold_k, log)
        
        fold_results.append((fold_k, week_maes, mean_mae))
        fold_test_preds.append({"expected": fold_test_expected, "region_ids": test_region_ids})

        del X_train_np, X_val_np; gc.collect()

    # -- 5. CV Summary & Ensemble
    log(f"\n{'='*90}\nEnsemble & Submission\n{'='*90}")
    
    cv_maes = [res[2] for res in fold_results]
    log(f"  Overall CV Rounded MAE: {np.mean(cv_maes):.4f}  ±  {np.std(cv_maes):.4f}")

    # Ensemble: Median of continuous Expected values, then Round
    all_region_ids = fold_test_preds[0]["region_ids"]
    expected_stack = np.stack([fp["expected"] for fp in fold_test_preds], axis=0)
    
    final_expected = np.median(expected_stack, axis=0)
    final_rounded = np.clip(np.round(final_expected), 0, 5)

    log(f"  Post-Round Exact Zero: {(final_rounded == 0.0).mean():.2%}")
    log(f"  Post-Round Exact Five: {(final_rounded == 5.0).mean():.2%}")

    rows = []
    for i, rid in enumerate(all_region_ids):
        rows.append({
            "region_id":  rid,
            "pred_week1": int(final_rounded[i, 0]),
            "pred_week2": int(final_rounded[i, 1]),
            "pred_week3": int(final_rounded[i, 2]),
            "pred_week4": int(final_rounded[i, 3]),
            "pred_week5": int(final_rounded[i, 4]),
        })
    submission = pd.DataFrame(rows)
    sub_path = os.path.join(ROOT, "submission_49th.csv")
    submission.to_csv(sub_path, index=False)

    log(f"  submission.csv → {sub_path}  ({len(submission)} rows)")
    
    log_path = os.path.join(ROOT, "_training_log_49th.txt")
    with open(log_path, "w") as fh: fh.write("\n".join(log_lines))

if __name__ == "__main__":
    main()