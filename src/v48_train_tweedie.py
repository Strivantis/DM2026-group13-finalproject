"""
v48_train_tweedie.py – Drought Score Forecasting Pipeline (v48).

Architecture: Single Tweedie Tree (LightGBM)
  One LGBMRegressor per fold × forecast week → 5 folds × 5 weeks = 25 models.
  Tweedie variance_power=1.65 (compound Poisson-Gamma) natively handles zero-
  inflation and right-skewed scores; no explicit hurdle gate required.

Feature space: 23 effective features × 13 weeks + 23 deltas = 322 dimensions
               (FEATURE_COLS declares 29; DROP_COLS prunes 6 at runtime).

Inference:
  final = np.median(tweedie_fold_preds, axis=folds)
  final = np.clip(final, 0.0, 5.0)

Outputs:
  submission.csv
  models/lgbm_tweedie_fold{k}_week{w}.pkl
  _training_log_48th.txt
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
from lightgbm import LGBMRegressor
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

N_FOLDS         = 5
N_FLAT_FEATURES = WINDOW_SIZE * len(FEATURE_COLS) + len(FEATURE_COLS)  # 406 declared; 322 effective after DROP_COLS

TWEEDIE_PARAMS = dict(
    objective              = "tweedie",
    tweedie_variance_power = 1.65,   # compound Poisson-Gamma: weakens zero-mass singularity, preserves continuous range
    metric                 = "l1",   # early stopping monitors MAE, not Tweedie deviance
    learning_rate          = 0.04,
    max_depth              = 6,
    num_leaves             = 31,
    min_child_samples      = 300,    # guards against memorising extreme-score outliers
    subsample              = 0.8,
    colsample_bytree       = 0.8,
    n_estimators           = 30000,
    device                 = "gpu",
    n_jobs                 = -1,
    random_state           = 42,
    verbose                = -1,
)


# ---------------------------------------------------------------------------
# Per-fold leakage-free target encoding
# ---------------------------------------------------------------------------
def _zero_prob(x):
    return (x == 0.0).mean()


def _compute_te_stats(df: pd.DataFrame) -> tuple:
    """
    Compute per-region score statistics on training-fold rows only.

    Returns
    -------
    te_map           : dict  region_id → (mean_score, zero_prob)
    global_mean      : float
    global_zero_prob : float
    """
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
    """Inject region_mean_score and region_zero_prob into each group DataFrame."""
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
    """Broadcast TE stats onto a flat DataFrame (used for test inference)."""
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
    raise ValueError(f"Unknown interval index: {idx}")


def print_binned_error_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, fold_k: int, log_fn
) -> None:
    """Print per-drought-interval MAE breakdown for one CV fold."""
    log_fn("")
    log_fn("  " + "=" * 80)
    log_fn(f"  BINNED ERROR MATRIX  --  Fold {fold_k}  (OOF Tweedie)")
    log_fn("  " + "=" * 80)
    log_fn(
        f"  {'Interval':<50}  {'Count':>7}  {'AvgTrue':>8}  "
        f"{'AvgPred':>8}  {'MAE':>8}"
    )
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
        log_fn(
            f"  {label:<50}  {n:>7,}  {avg_true:>8.4f}  "
            f"{avg_pred:>8.4f}  {mae:>8.4f}"
        )
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
    log("Drought Forecasting Pipeline  v48  (Single Tweedie Tree)")
    log(f"Tweedie variance_power={TWEEDIE_PARAMS['tweedie_variance_power']}  "
        f"early_stopping on l1  |  25 models (5 folds × 5 weeks)")
    log(f"Feature space: 23 effective × {WINDOW_SIZE} weeks + 23 deltas = 322 dims "
        f"({len(FEATURE_COLS)} declared, DROP_COLS prunes 6)")
    log("Inference: np.median(fold preds, axis=0)  →  clip(0, 5)")
    log(f"CV: {N_FOLDS}-Fold StratifiedGroupKFold  (strata: climate cluster_id)")
    log("=" * 90)

    # -- 1. Load processed data -----------------------------------------------
    log("\nLoading processed data ...")
    try:
        train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
        test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n  --> Run `python src/preprocess.py` first."
        )

    log(f"  train: {train_raw.shape}  |  test: {test_raw.shape}")

    n_train_regions = train_raw["region_id"].nunique()
    n_test_regions  = test_raw["region_id"].nunique()
    assert n_train_regions == 2248
    assert n_test_regions  == 2248

    # -- 2. Validate V29 sentinel + feature columns ---------------------------
    assert "week_sin" in train_raw.columns, "week_sin missing – run preprocess.py"
    assert "week_cos" in train_raw.columns, "week_cos missing – run preprocess.py"
    bad = [c for c in train_raw.columns if "8w" in c or "13w" in c]
    if bad:
        log(f"  WARNING: 8w/13w drift columns found: {bad}")
    log(f"  FEATURE_COLS ({len(FEATURE_COLS)}): sentinel _v29_normalized="
        f"{'present' if '_v29_normalized' in train_raw.columns else 'ABSENT'}")

    # -- 3. Feature refinement ------------------------------------------------
    log("\nRefining features ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    log(f"  train: {train_df.shape}  |  test: {test_df.shape}")

    # -- 4. Drop NaN score rows -----------------------------------------------
    before   = len(train_df)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    dropped  = before - len(train_df)
    if dropped:
        log(f"  Dropped {dropped:,} NaN-score rows.")

    # -- 5. Target distribution -----------------------------------------------
    all_scores = train_df["score"].values
    zero_frac  = (all_scores == 0.0).mean()
    log(f"\n[Target] mean={all_scores.mean():.4f}  std={all_scores.std():.4f}  "
        f"zero={zero_frac:.2%}  positive={1-zero_frac:.2%}")
    log(f"  flat input dim: 322 effective  "
        f"(23×{WINDOW_SIZE}+23; DROP_COLS prunes 6 of {len(FEATURE_COLS)} declared)")

    # -- 6. Build CV folds ----------------------------------------------------
    log(f"\n{'='*90}")
    log(f"{N_FOLDS}-Fold StratifiedGroupKFold  (group=region_id)")
    log(f"{'='*90}")

    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    log(f"  Folds built: {len(folds)}")
    for fi, (tg, vg) in enumerate(folds):
        log(f"  Fold {fi}: train_groups={len(tg):,}  val_groups={len(vg):,}")

    # -- 7. Tweedie Training Loop ---------------------------------------------
    log(f"\n{'='*90}")
    log("Tweedie Training  (single model per fold × week)")
    log(f"{'='*90}")

    fold_results         = []
    fold_test_preds_list = []

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):

        log(f"\n{'='*90}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}")
        log(f"  train_groups: {len(raw_train_groups):,}  |  "
            f"val_groups: {len(raw_val_groups):,}")
        log(f"{'='*90}")

        fold_t0 = time.time()

        # ---- Fold-local target encoding (leakage-free) ----------------------
        train_region_ids_fold = {
            entry[0]["region_id"].iloc[0] for entry in raw_train_groups
        }
        val_region_ids_fold = {
            entry[0]["region_id"].iloc[0] for entry in raw_val_groups
        }
        train_df_fold_regions = train_df[
            train_df["region_id"].isin(train_region_ids_fold)
        ]
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(train_df_fold_regions)
        log(f"  [TE] train regions: {len(train_region_ids_fold):,}  "
            f"val regions: {len(val_region_ids_fold):,}")

        aug_train_groups = _augment_groups_with_te(
            raw_train_groups, te_map_fold, gm_fold, gzp_fold
        )
        aug_val_groups = _augment_groups_with_te(
            raw_val_groups, te_map_fold, gm_fold, gzp_fold
        )

        # ---- Resolve available feature columns ------------------------------
        feat_cols = [c for c in FEATURE_COLS if c in aug_train_groups[0][0].columns]
        log(f"  feat_cols: {len(feat_cols)} / {len(FEATURE_COLS)}")

        # ---- Build tabular matrices -----------------------------------------
        X_train_np, y_train_np, _ = build_tabular_dataset(aug_train_groups, feat_cols)
        X_val_np,   y_val_np,   _ = build_tabular_dataset(aug_val_groups,   feat_cols)
        X_train_np = np.asfortranarray(X_train_np)
        X_val_np   = np.asfortranarray(X_val_np)

        log(f"  X_train: {X_train_np.shape}  |  y_train: {y_train_np.shape}")
        log(f"  X_val  : {X_val_np.shape}    |  y_val  : {y_val_np.shape}")

        expected_dim = WINDOW_SIZE * len(feat_cols) + len(feat_cols)
        assert X_train_np.shape[1] == expected_dim, (
            f"Dim mismatch: got {X_train_np.shape[1]}, expected {expected_dim}"
        )

        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test_np, test_region_ids = build_tabular_test(test_df_fold, feat_cols)
        X_test_np = np.asfortranarray(X_test_np)
        log(f"  X_test : {X_test_np.shape}")

        # ---- Per-week Tweedie model -----------------------------------------
        fold_val_preds  = np.zeros_like(y_val_np)
        fold_test_preds = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)

        for week_idx in range(HORIZON):
            y_train_w = y_train_np[:, week_idx]
            y_val_w   = y_val_np[:,   week_idx]

            ckpt = os.path.join(
                MODELS_DIR, f"lgbm_tweedie_fold{fold_k}_week{week_idx}.pkl"
            )

            log(f"\n  --- Week {week_idx + 1} ---")

            if os.path.exists(ckpt):
                log(f"    [RESUME] {ckpt}")
                try:
                    with open(ckpt, "rb") as fh:
                        model = pickle.load(fh)
                    val_pred_w  = model.predict(X_val_np)
                    test_pred_w = model.predict(X_test_np)
                    mae = float(np.mean(np.abs(val_pred_w - y_val_w)))
                    log(f"    [Tweedie] best_iter={model.best_iteration_}  "
                        f"val_MAE={mae:.4f}  [LOADED]")
                except Exception as e:
                    log(f"    [WARN] Checkpoint corrupted ({e}), retraining.")
                    os.remove(ckpt)
                    model = None
            else:
                model = None

            if model is None:
                model = LGBMRegressor(**TWEEDIE_PARAMS)
                model.fit(
                    X_train_np, y_train_w,
                    eval_set    = [(X_val_np, y_val_w)],
                    eval_metric = "l1",
                    callbacks   = [
                        lgb.early_stopping(stopping_rounds=400, verbose=False),
                        lgb.log_evaluation(period=1000),
                    ],
                )
                val_pred_w  = model.predict(X_val_np)
                test_pred_w = model.predict(X_test_np)
                mae = float(np.mean(np.abs(val_pred_w - y_val_w)))
                log(f"    [Tweedie] best_iter={model.best_iteration_}  val_MAE={mae:.4f}")
                with open(ckpt, "wb") as fh:
                    pickle.dump(model, fh)
                log(f"    [SAVED] {ckpt}")

            fold_val_preds[:, week_idx]  = val_pred_w
            fold_test_preds[:, week_idx] = test_pred_w.astype(np.float32)
            del model
            gc.collect()

            log(f"    [OOF] MAE={float(np.mean(np.abs(fold_val_preds[:, week_idx] - y_val_w))):.4f}")

        fold_elapsed = time.time() - fold_t0

        # ---- Fold OOF summary -----------------------------------------------
        week_maes = [
            float(np.mean(np.abs(fold_val_preds[:, w] - y_val_np[:, w])))
            for w in range(HORIZON)
        ]
        mean_fold_mae = float(np.mean(week_maes))

        log(f"\n  [Fold {fold_k}] Tweedie MAE: " +
            "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes)))
        log(f"  [Fold {fold_k}] Mean MAE  : {mean_fold_mae:.4f}  ({fold_elapsed:.1f}s)")

        print_binned_error_matrix(
            y_val_np.ravel(), fold_val_preds.ravel(), fold_k, log
        )

        fold_results.append((fold_k, week_maes, mean_fold_mae))
        fold_test_preds_list.append({"preds": fold_test_preds, "region_ids": test_region_ids})

        del X_train_np, y_train_np, X_val_np, y_val_np, fold_val_preds
        gc.collect()

    # -- 8. CV summary --------------------------------------------------------
    log(f"\n{'='*90}")
    log("5-Fold CV Summary")
    log(f"{'='*90}")
    mean_cv_maes = []
    for fold_k, week_maes, mean_mae in fold_results:
        log("  Fold {:d}: {:s}  -> Mean={:.4f}".format(
            fold_k,
            "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes)),
            mean_mae,
        ))
        mean_cv_maes.append(mean_mae)

    overall_mean = float(np.mean(mean_cv_maes))
    overall_std  = float(np.std(mean_cv_maes))
    log(f"\n  Overall CV MAE: {overall_mean:.4f}  ±  {overall_std:.4f}")
    log(f"  Best  Fold: {int(np.argmin(mean_cv_maes))} "
        f"(MAE={min(mean_cv_maes):.4f})")
    log(f"  Worst Fold: {int(np.argmax(mean_cv_maes))} "
        f"(MAE={max(mean_cv_maes):.4f})")
    log("\n  Per-week breakdown:")
    for w in range(HORIZON):
        wk_maes = [fold_results[k][1][w] for k in range(N_FOLDS)]
        log(f"    Week {w+1}: mean={np.mean(wk_maes):.4f}  std={np.std(wk_maes):.4f}")

    # -- 9. Fold ensemble (median) --------------------------------------------
    log(f"\n{'='*90}")
    log(f"Tweedie Ensemble  ({N_FOLDS} folds  →  np.median)")
    log(f"{'='*90}")

    all_region_ids = fold_test_preds_list[0]["region_ids"]
    assert len(all_region_ids) == 2248

    preds_stack = np.stack([fp["preds"] for fp in fold_test_preds_list], axis=0)
    final_preds = np.median(preds_stack, axis=0)
    final_preds = np.where(final_preds < 0.05, 0.0, final_preds)
    final_preds = np.clip(final_preds, 0.0, 5.0)

    log(f"  final_preds: mean={final_preds.mean():.4f}  std={final_preds.std():.4f}  "
        f"exact-zero={(final_preds == 0.0).mean():.2%}")

    # -- 10. Submission -------------------------------------------------------
    log("\nFormatting submission.csv ...")
    rows = [
        {
            "region_id":  rid,
            "pred_week1": float(final_preds[i, 0]),
            "pred_week2": float(final_preds[i, 1]),
            "pred_week3": float(final_preds[i, 2]),
            "pred_week4": float(final_preds[i, 3]),
            "pred_week5": float(final_preds[i, 4]),
        }
        for i, rid in enumerate(all_region_ids)
    ]
    submission = pd.DataFrame(rows)
    sub_path   = os.path.join(ROOT, "submission_48th.csv")
    submission.to_csv(sub_path, index=False)

    assert len(submission) == 2248
    assert list(submission.columns) == [
        "region_id", "pred_week1", "pred_week2",
        "pred_week3", "pred_week4", "pred_week5",
    ]
    assert not submission.isnull().any().any()
    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    assert submission[pred_cols].max().max() <= 5.0 + 1e-6
    assert submission[pred_cols].min().min() >= 0.0 - 1e-6

    log(f"  submission.csv → {sub_path}  ({len(submission)} rows)")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    elapsed = time.time() - t0
    log(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    log_path = os.path.join(ROOT, "_training_log_48th.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines))
    print(f"Training log → {log_path}")

    return {
        "fold_results":    fold_results,
        "overall_cv_mae":  overall_mean,
        "std_cv_mae":      overall_std,
        "input_dim":       N_FLAT_FEATURES,
        "submission":      submission,
        "zero_frac_final": float((final_preds == 0.0).mean()),
    }


if __name__ == "__main__":
    results = main()
