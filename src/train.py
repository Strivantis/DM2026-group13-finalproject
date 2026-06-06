"""
train.py – Drought Score Forecasting Pipeline (v45 / 45th).

Architecture: Decoupled Dual-Tree Hurdle (LightGBM)
  Model A – LGBMRegressor  objective=regression_l1  (conditional median)
  Model B – LGBMClassifier objective=binary          (drought probability)
  Per CV fold × per forecast week → 5 folds × 5 weeks = 25 model pairs.

Feature space: 29 features × 13 weeks + 29 deltas = 406 dimensions.

Inference:
  l1_median  = np.median(Model-A preds, axis=folds)   robust to outlier folds
  prob_mean  = np.mean(Model-B probs,  axis=folds)    probability averaging
  final      = np.where(prob_mean < 0.5, 0.0, l1_median)
  final      = np.clip(final, 0.0, 5.0)

Outputs:
  submission.csv
  models/lgbm_a_fold{k}_week{w}.pkl
  models/lgbm_b_fold{k}_week{w}.pkl
  _training_log_45th.txt
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
N_FLAT_FEATURES = WINDOW_SIZE * len(FEATURE_COLS) + len(FEATURE_COLS)  # 406


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
    te_map       : dict  region_id → (mean_score, zero_prob)
    global_mean  : float
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
    log_fn(f"  BINNED ERROR MATRIX  --  Fold {fold_k}  (OOF Hurdle-Gated)")
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
    log("Drought Forecasting Pipeline  v45  (Decoupled Dual-Tree Hurdle)")
    log("Model A: LGBMRegressor regression_l1  |  Model B: LGBMClassifier binary")
    log(f"Feature space: {len(FEATURE_COLS)} features × {WINDOW_SIZE} weeks + "
        f"{len(FEATURE_COLS)} deltas = {N_FLAT_FEATURES} dims")
    log("OOF gate:  np.where(prob < 0.5, 0.0, l1_pred)")
    log("Test gate: np.median(A, axis=folds)  |  np.mean(B, axis=folds)")
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
    before    = len(train_df)
    train_df  = train_df.dropna(subset=["score"]).reset_index(drop=True)
    dropped   = before - len(train_df)
    if dropped:
        log(f"  Dropped {dropped:,} NaN-score rows.")

    # -- 5. Target distribution -----------------------------------------------
    all_scores = train_df["score"].values
    zero_frac  = (all_scores == 0.0).mean()
    log(f"\n[Target] mean={all_scores.mean():.4f}  std={all_scores.std():.4f}  "
        f"zero={zero_frac:.2%}  positive={1-zero_frac:.2%}")
    log(f"  flat input dim: {N_FLAT_FEATURES}  "
        f"({WINDOW_SIZE}×{len(FEATURE_COLS)}+{len(FEATURE_COLS)})")

    # -- 6. Build CV folds ----------------------------------------------------
    log(f"\n{'='*90}")
    log(f"{N_FOLDS}-Fold StratifiedGroupKFold  (group=region_id)")
    log(f"{'='*90}")

    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    log(f"  Folds built: {len(folds)}")
    for fi, (tg, vg) in enumerate(folds):
        log(f"  Fold {fi}: train_groups={len(tg):,}  val_groups={len(vg):,}")

    # -- 7. Dual-Tree Hurdle Training Loop ------------------------------------
    log(f"\n{'='*90}")
    log("Dual-Tree Hurdle Training  (Model A: regression_l1 | Model B: binary)")
    log(f"{'='*90}")

    fold_results      = []
    fold_test_preds_a = []
    fold_test_preds_b = []

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

        # ---- Per-week model pair training -----------------------------------
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

            log(f"\n  --- Week {week_idx + 1} ---")

            # Model A --------------------------------------------------------
            if os.path.exists(ckpt_a):
                log(f"    [RESUME-A] {ckpt_a}")
                try:
                    with open(ckpt_a, "rb") as fh:
                        model_a = pickle.load(fh)
                    val_l1_w  = model_a.predict(X_val_np)
                    test_l1_w = model_a.predict(X_test_np)
                    mae_a = float(np.mean(np.abs(val_l1_w - y_val_w)))
                    log(f"    [Model A] best_iter={model_a.best_iteration_}  "
                        f"val_MAE={mae_a:.4f}  [LOADED]")
                except Exception as e:
                    log(f"    [WARN-A] Checkpoint corrupted ({e}), retraining.")
                    os.remove(ckpt_a)
                    model_a = None
            else:
                model_a = None

            if model_a is None:
                # model_a = LGBMRegressor(                  #v45
                #     objective        = "regression_l1",
                #     max_depth        = 5,
                #     num_leaves       = 31,
                #     colsample_bytree = 0.5,
                #     learning_rate    = 0.02,
                #     n_estimators     = 10000, 
                #     device           = "gpu",
                #     random_state     = 42,
                #     n_jobs           = -1,
                #     verbose          = -1,
                # )
                model_a = LGBMRegressor(                    #v46
                    objective        = "regression_l1",
                    max_depth        = 6,
                    num_leaves       = 45,
                    colsample_bytree = 0.6,
                    learning_rate    = 0.04,
                    subsample        = 0.8,
                    subsample_freq   = 1,
                    min_child_samples= 300,
                    n_estimators     = 30000, 
                    device           = "gpu",
                    random_state     = 42,
                    n_jobs           = -1,
                    verbose          = -1,
                )
                model_a.fit(
                    X_train_np, y_train_w,
                    eval_set    = [(X_val_np, y_val_w)],
                    eval_metric = "mae",
                    callbacks   = [
                        lightgbm.early_stopping(stopping_rounds=400, verbose=False), # v45 150
                        lightgbm.log_evaluation(period=1000),
                    ],
                )
                val_l1_w  = model_a.predict(X_val_np)
                test_l1_w = model_a.predict(X_test_np)
                mae_a     = float(np.mean(np.abs(val_l1_w - y_val_w)))
                log(f"    [Model A] best_iter={model_a.best_iteration_}  "
                    f"val_MAE={mae_a:.4f}")
                with open(ckpt_a, "wb") as fh:
                    pickle.dump(model_a, fh)
                log(f"    [SAVED-A] {ckpt_a}")

            fold_val_preds_l1[:, week_idx] = val_l1_w
            fold_test_pred_l1[:, week_idx] = test_l1_w.astype(np.float32)
            del model_a
            gc.collect()

            # Model B --------------------------------------------------------
            if os.path.exists(ckpt_b):
                log(f"    [RESUME-B] {ckpt_b}")
                try:
                    with open(ckpt_b, "rb") as fh:
                        model_b = pickle.load(fh)
                    val_prob_w  = model_b.predict_proba(X_val_np)[:, 1]
                    test_prob_w = model_b.predict_proba(X_test_np)[:, 1]
                    log(f"    [Model B] best_iter={model_b.best_iteration_}  [LOADED]")
                except Exception as e:
                    log(f"    [WARN-B] Checkpoint corrupted ({e}), retraining.")
                    os.remove(ckpt_b)
                    model_b = None
            else:
                model_b = None

            if model_b is None:
                # model_b = LGBMClassifier(             #v45
                #     objective        = "binary",
                #     max_depth        = 5,
                #     num_leaves       = 31,
                #     colsample_bytree = 0.5,
                #     learning_rate    = 0.02,
                #     n_estimators     = 15000, # v45 5000
                #     device           = "gpu",
                #     random_state     = 42,
                #     n_jobs           = -1,
                #     verbose          = -1,
                # )
                model_b = LGBMClassifier(               #v46
                    objective        = "binary",
                    max_depth        = 6,
                    num_leaves       = 45,
                    colsample_bytree = 0.6,
                    learning_rate    = 0.04,
                    subsample        = 0.8,
                    subsample_freq   = 1,
                    min_child_samples= 300,
                    n_estimators     = 30000, 
                    device           = "gpu",
                    random_state     = 42,
                    n_jobs           = -1,
                    verbose          = -1,
                )
                model_b.fit(
                    X_train_np, y_train_b,
                    eval_set    = [(X_val_np, y_val_b)],
                    eval_metric = "binary_logloss",
                    callbacks   = [
                        lightgbm.early_stopping(stopping_rounds=400, verbose=False), # v45 150
                        lightgbm.log_evaluation(period=1000),
                    ],
                )
                val_prob_w  = model_b.predict_proba(X_val_np)[:, 1]
                test_prob_w = model_b.predict_proba(X_test_np)[:, 1]
                log(f"    [Model B] best_iter={model_b.best_iteration_}")
                with open(ckpt_b, "wb") as fh:
                    pickle.dump(model_b, fh)
                log(f"    [SAVED-B] {ckpt_b}")

            fold_val_probs[:, week_idx]    = val_prob_w
            fold_test_prob[:, week_idx]    = test_prob_w.astype(np.float32)
            del model_b
            gc.collect()

            # OOF hurdle gate -------------------------------------------------
            oof_final_w = np.where(
                fold_val_probs[:, week_idx] < 0.5, 0.0,
                fold_val_preds_l1[:, week_idx]
            )
            fold_val_final[:, week_idx] = oof_final_w
            mae_final    = float(np.mean(np.abs(oof_final_w - y_val_w)))
            zero_gated   = float((oof_final_w == 0.0).mean())
            log(f"    [OOF] HurdleMAE={mae_final:.4f}  "
                f"zero-gated={zero_gated:.2%}")

        fold_elapsed = time.time() - fold_t0

        # ---- Fold OOF summary -----------------------------------------------
        week_maes_l1    = [
            float(np.mean(np.abs(fold_val_preds_l1[:, w] - y_val_np[:, w])))
            for w in range(HORIZON)
        ]
        week_maes_final = [
            float(np.mean(np.abs(fold_val_final[:, w] - y_val_np[:, w])))
            for w in range(HORIZON)
        ]
        mean_fold_mae_final = float(np.mean(week_maes_final))

        log(f"\n  [Fold {fold_k}] L1 raw    : " +
            "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes_l1)))
        log(f"  [Fold {fold_k}] Hurdle    : " +
            "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes_final)))
        log(f"  [Fold {fold_k}] Mean MAE  : {mean_fold_mae_final:.4f}  "
            f"({fold_elapsed:.1f}s)")

        print_binned_error_matrix(
            y_val_np.ravel(), fold_val_final.ravel(), fold_k, log
        )

        fold_results.append((fold_k, week_maes_l1, week_maes_final, mean_fold_mae_final))
        fold_test_preds_a.append({"preds": fold_test_pred_l1, "region_ids": test_region_ids})
        fold_test_preds_b.append({"probs": fold_test_prob,    "region_ids": test_region_ids})

        del X_train_np, y_train_np, X_val_np, y_val_np
        del fold_val_preds_l1, fold_val_probs, fold_val_final
        gc.collect()

    # -- 8. CV summary --------------------------------------------------------
    log(f"\n{'='*90}")
    log("5-Fold CV Summary")
    log(f"{'='*90}")
    mean_cv_maes = []
    for fold_k, _, week_maes_final, mean_mae in fold_results:
        log("  Fold {:d}: {:s}  -> Mean={:.4f}".format(
            fold_k,
            "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes_final)),
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
        wk_maes = [fold_results[k][2][w] for k in range(N_FOLDS)]
        log(f"    Week {w+1}: mean={np.mean(wk_maes):.4f}  "
            f"std={np.std(wk_maes):.4f}")

    # -- 9. Asymmetric ensemble blending --------------------------------------
    log(f"\n{'='*90}")
    log(f"Asymmetric Ensemble  ({N_FOLDS} folds)")
    log("  Model A → np.median  (robust to fold outliers)")
    log("  Model B → np.mean    (probability averaging)")
    log(f"{'='*90}")

    all_region_ids = fold_test_preds_a[0]["region_ids"]
    assert len(all_region_ids) == 2248

    preds_a_stack = np.stack([fp["preds"] for fp in fold_test_preds_a], axis=0)
    probs_b_stack = np.stack([fp["probs"] for fp in fold_test_preds_b], axis=0)

    l1_median = np.median(preds_a_stack, axis=0)
    prob_mean  = np.mean(probs_b_stack,  axis=0)

    log(f"  l1_median: mean={l1_median.mean():.4f}  std={l1_median.std():.4f}")
    log(f"  prob_mean: mean={prob_mean.mean():.4f}  "
        f"fraction<0.5={( prob_mean < 0.5).mean():.2%}")

    final_preds = np.where(prob_mean < 0.5, 0.0, l1_median)
    final_preds = np.clip(final_preds, 0.0, 5.0)

    log(f"  Post-gate: mean={final_preds.mean():.4f}  "
        f"std={final_preds.std():.4f}  "
        f"exact-zero={( final_preds == 0.0).mean():.2%}")

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
    sub_path   = os.path.join(ROOT, "submission.csv")
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

    log_path = os.path.join(ROOT, "_training_log_46th.txt")
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
