"""
train.py -- Drought Score Forecasting Pipeline (v27 -- The Tweedie-Hurdle Paradigm)
====================================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv                  -- Kaggle submission (2248 rows x 6 cols)
    models/lgbm_fold{k}_week{w}.pkl -- LightGBM checkpoint per fold x week (25 total)
    _training_log_27th.txt          -- Full console log

v27 Changes (The Tweedie-Hurdle Paradigm)
------------------------------------------
  EXPAND (dataset.py): 27 explicit trend differential features (w13 - w1)
    appended to the 351-dimensional flat matrix -> 378 dimensions total.
    feature_delta = feature_w13 - feature_w1  for all 27 meteorological features.

  TRANSITION: LightGBM objective switched from regression_l1 to Compound
    Poisson-Gamma Tweedie loss (objective='tweedie', tweedie_variance_power=1.5).
    This natively models zero-inflated continuous targets by construction,
    collapsing low-mass fractional noise into absolute zeroes mathematically.

  EXPAND: Tree capacity budget increased from 10,000 to 20,000 estimators.
    Learning rate scaled down from 0.03 to 0.015 to protect against gradient
    step explosion across the wider estimator budget.
    num_leaves expanded from 31 to 63 to capture high-order interactions.
    min_child_samples=100 for deep leaf regularization.

  EARLY STOPPING: stopping_rounds=250 (adjusted for slow lr=0.015 convergence).
    eval_metric='mae' -- competition native metric prevents premature halting
    on mathematical deviance divergence.

  ABOLISH: Manual Snap-to-Zero micro-wiper gate (< 0.15) completely removed.
    Trust the integrated Tweedie exponential dispersion model to natively
    collapse low-mass fractions without hand-crafted clamping constraints.

  RETAIN: Pure np.median blending across 5 independent CV folds.
    Final np.clip(0.0, 5.0) physical safety boundary restriction.

Architecture / Optimization
-----------------------------
    LGBMRegressor x 5 per fold (one per future week)
    objective              : tweedie  (Compound Poisson-Gamma)
    tweedie_variance_power : 1.5      (equal Poisson-Gamma balance)
    device                 : gpu
    n_estimators           : 15000
    learning_rate          : 0.015
    num_leaves             : 63
    min_child_samples      : 100
    colsample_bytree       : 0.75
    subsample              : 0.75
    Early stop             : stopping_rounds=250, eval_metric='mae'
    CV                     : 5-Fold StratifiedGroupKFold, group=region_id

Feature Space
--------------
    v27: 27 features x 13 weeks (351) + 27 explicit trend deltas = 378 dimensions

Post-Processing
---------------
    Ensemble : np.median across 5 folds
    Clip     : np.clip(0.0, 5.0) physical safety guard
    (Snap-to-Zero gate abolished -- Tweedie handles this natively)
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
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import gc

# -- project root on sys.path --------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_stratified_group_cv_folds,
    build_tabular_dataset,
    build_tabular_test,
    make_flat_col_names,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
    WF_NUM_FOLDS,
    GAP_WEEKS,
    N_TS_FOLDS,
    TS_SHIFT_WEEKS,
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

# Cross-validation
N_FOLDS = 5

# v27: 27 features x 13 weeks (351) + 27 explicit trend deltas = 378 flat columns
N_FLAT_FEATURES = WINDOW_SIZE * len(FEATURE_COLS) + len(FEATURE_COLS)   # 378


# ---------------------------------------------------------------------------
# Target Encoding helpers (leakage-free, per-fold)
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
        if len(entry) == 4:
            group, i_min, i_max, _gap = entry
        else:
            group, i_min, i_max = entry
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
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    # -- 0. Pipeline description -----------------------------------------------
    log("=" * 90)
    log("Drought Forecasting Pipeline  v27  (The Tweedie-Hurdle Paradigm)")
    log("GPU LightGBM Mega-Tree  |  objective=tweedie (p=1.5)  |  n_estimators=15000")
    log("Feature Space: 27 features x 13 weeks (351) + 27 explicit deltas = 378 dims")
    log("Post-Ensemble: np.median across 5 folds  |  Snap-to-Zero gate ABOLISHED")
    log("5-Fold StratifiedGroupKFold  |  5 LGBMRegressors per fold (one per week)")
    log("=" * 90)
    log("")
    log("v27 Changes over v26:")
    log("  [EXPAND] 27 explicit trend deltas (feature_w13 - feature_w1) appended.")
    log("           Flat input dimension: 351 -> 378  (27 x 13 + 27 deltas).")
    log("  [TRANSITION] objective: regression_l1 -> tweedie (variance_power=1.5).")
    log("               Compound Poisson-Gamma formulation natively handles zero-")
    log("               inflated continuous targets without manual gating.")
    log("  [EXPAND] n_estimators: 10000 -> 15000  (prevent underfitting with lr=0.015).")
    log("  [EXPAND] num_leaves: 31 -> 63  (capture high-order feature interactions).")
    log("           min_child_samples: 100  (deep leaf regularization).")
    log("  [ADJUST] learning_rate: 0.03 -> 0.015  (protect against gradient explosion).")
    log("  [ADJUST] early_stopping: 150 -> 250 rounds  (slow lr=0.015 convergence).")
    log("  [ABOLISH] Snap-to-Zero gate (< 0.15) completely removed.")
    log("            Tweedie exponential dispersion model handles this natively.")
    log("")
    log(f"Training Config:")
    log(f"  LGBM objective         = tweedie  (Compound Poisson-Gamma)")
    log(f"  tweedie_variance_power = 1.5      (equal Poisson-Gamma balance)")
    log(f"  LGBM device            = gpu")
    log(f"  n_estimators           = 15000")
    log(f"  learning_rate          = 0.015")
    log(f"  num_leaves             = 63")
    log(f"  min_child_samples      = 100")
    log(f"  colsample_bytree       = 0.75")
    log(f"  subsample              = 0.75")
    log(f"  early_stopping         = 250 rounds (eval_metric=mae)")
    log(f"  N_FOLDS                = {N_FOLDS}")
    log(f"  N_FLAT_FEATURES        = {N_FLAT_FEATURES}  "
        f"(WINDOW_SIZE={WINDOW_SIZE} x FEAT={len(FEATURE_COLS)} + {len(FEATURE_COLS)} deltas)")

    # -- 1. Load data ----------------------------------------------------------
    log("\nLoading processed data ...")
    try:
        train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
        test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n  --> Run `python src/preprocess.py` to generate processed CSVs."
        )

    log(f"  train raw: {train_raw.shape}  |  test raw: {test_raw.shape}")

    n_train_regions = train_raw["region_id"].nunique()
    n_test_regions  = test_raw["region_id"].nunique()
    log(f"  train regions: {n_train_regions}  |  test regions: {n_test_regions}")
    assert n_train_regions == 2248, f"Expected 2248 train regions, got {n_train_regions}"
    assert n_test_regions  == 2248, f"Expected 2248 test regions,  got {n_test_regions}"

    # -- 1b. Feature Validation ------------------------------------------------
    log("\n[v27 Feature Validation]")
    assert "week_sin" in train_raw.columns, "week_sin missing -- run preprocess.py first"
    assert "week_cos" in train_raw.columns, "week_cos missing -- run preprocess.py first"
    log("  v week_sin, week_cos present.")

    bad_cols_8w_13w = [c for c in train_raw.columns if ("8w" in c or "13w" in c)]
    if bad_cols_8w_13w:
        log(f"  *** WARNING: 8w/13w features found: {bad_cols_8w_13w}")
    else:
        log("  v No 8w/13w rolling features (domain-shift columns absent).")

    lag_cols_check = [c for c in train_raw.columns if "lag" in c]
    if lag_cols_check:
        log(f"  [INFO] Lag columns present in raw CSV (will be excluded from FEATURE_COLS): "
            f"{lag_cols_check[:6]}{'...' if len(lag_cols_check) > 6 else ''}")
    log(f"  v27 FEATURE_COLS: {len(FEATURE_COLS)} base features (lag/rolling purged).")
    log(f"  v27 flat input dim: {N_FLAT_FEATURES}  "
        f"(27 x 13 = 351 + 27 explicit deltas = 378)")

    # -- 2. Feature refinement -------------------------------------------------
    log("\nRefining features (drought proxy index + log1p prec + v22/v26 pruning) ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    log(f"  train after refinement: {train_df.shape}  |  test: {test_df.shape}")

    # -- 3. Drop rows with NaN score -------------------------------------------
    before = len(train_df)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    dropped_nan = before - len(train_df)
    if dropped_nan:
        log(f"  [NaN drop] Removed {dropped_nan:,} rows with NaN score.")

    # -- 4. Target score distribution summary ---------------------------------
    log("\n[Training Target Distribution]")
    all_scores = train_df["score"].values
    zero_frac  = (all_scores == 0.0).mean()
    log(f"  mean={all_scores.mean():.4f}  std={all_scores.std():.4f}  "
        f"min={all_scores.min():.2f}  max={all_scores.max():.2f}")
    log(f"  Zero-inflation: {zero_frac:.2%} of training scores == 0.0")

    # -- 5. Feature columns ----------------------------------------------------
    log(f"\n[v27] Feature columns and flat input shape ...")
    log(f"  FEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    log(f"  flat input_size = {N_FLAT_FEATURES}  "
        f"(WINDOW_SIZE={WINDOW_SIZE} x FEATURE_COLS={len(FEATURE_COLS)} + "
        f"{len(FEATURE_COLS)} deltas = {N_FLAT_FEATURES})")

    # -- 6. Build 5-Fold StratifiedGroupKFold CV splits -----------------------
    log(f"\n{'='*90}")
    log(f"5-Fold StratifiedGroupKFold CV  [v27 -- GPU LightGBM Tweedie Mega-Tree]")
    log(f"  Group  : region_id  |  Strata : 10-quantile bins of per-region mean score")
    log(f"{'='*90}")

    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    log(f"\n  Folds built: {len(folds)}")
    for fi, (tg, vg) in enumerate(folds):
        log(f"  Fold {fi}: train_groups={len(tg):,}  val_groups={len(vg):,}")

    # -- 7. 5-Fold LightGBM Training Loop --------------------------------------
    log(f"\n{'='*90}")
    log("5-Fold LightGBM Training  [v27 GPU Tweedie Mega-Tree]")
    log("  5 independent LGBMRegressors per fold (one per target week).")
    log("  objective=tweedie (p=1.5)  |  device=gpu  |  n_estimators=15000")
    log("  num_leaves=63  |  learning_rate=0.015  |  early_stopping=250 rounds")
    log(f"{'='*90}")

    fold_results    = []
    fold_test_preds = []

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):

        log(f"\n{'='*90}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}  [v27 LightGBM Tweedie]")
        log(f"  train_groups: {len(raw_train_groups):,}  |  "
            f"val_groups: {len(raw_val_groups):,}")
        log(f"{'='*90}")

        fold_t0 = time.time()

        # -- 7a. Compute fold-local Target Encoding (leakage-free) ------------
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
        log(f"  [OOF TE] Train regions: {len(train_region_ids_fold):,}  "
            f"| Val regions: {len(val_region_ids_fold):,}")

        # -- 7b. Augment groups with fold-local TE ----------------------------
        aug_train_groups = _augment_groups_with_te(
            raw_train_groups, te_map_fold, gm_fold, gzp_fold
        )
        aug_val_groups = _augment_groups_with_te(
            raw_val_groups, te_map_fold, gm_fold, gzp_fold
        )

        # -- 7c. Determine feature columns available --------------------------
        _sample_group = aug_train_groups[0][0]
        feat_cols = [c for c in FEATURE_COLS if c in _sample_group.columns]
        log(f"  feat_cols available: {len(feat_cols)} / {len(FEATURE_COLS)}")

        # -- 7d. Build tabular matrices (13-week full flattening + 27 deltas) -
        X_train_np, y_train_np, _ = build_tabular_dataset(aug_train_groups, feat_cols)
        X_val_np,   y_val_np,   _ = build_tabular_dataset(aug_val_groups,   feat_cols)
        X_train_np = np.asfortranarray(X_train_np)
        X_val_np   = np.asfortranarray(X_val_np)

        log(f"  X_train: {X_train_np.shape}  |  y_train: {y_train_np.shape}")
        log(f"  X_val  : {X_val_np.shape}    |  y_val  : {y_val_np.shape}")

        # Verify 378-dim feature space
        expected_dim = WINDOW_SIZE * len(feat_cols) + len(feat_cols)
        assert X_train_np.shape[1] == expected_dim, (
            f"Feature dim mismatch: got {X_train_np.shape[1]}, "
            f"expected {expected_dim} (351 flat + 27 deltas)"
        )
        log(f"  v Feature dim = {X_train_np.shape[1]}  "
            f"(351 flat + 27 explicit deltas = 378)")

        # -- 7e. Build test tabular matrix for this fold ----------------------
        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test_np, test_region_ids = build_tabular_test(test_df_fold, feat_cols)
        X_test_np = np.asfortranarray(X_test_np)
        log(f"  X_test: {X_test_np.shape}")

        # -- 7f. Train 5 LGBMRegressors (one per target week) -----------------
        fold_week_models = []
        fold_val_preds   = np.zeros_like(y_val_np)   # (N_val, 5)
        fold_test_pred   = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)

        for week_idx in range(HORIZON):
            y_train_w = y_train_np[:, week_idx]   # (N_train,)
            y_val_w   = y_val_np[:,   week_idx]   # (N_val,)

            ckpt_path = os.path.join(
                MODELS_DIR, f"lgbm_fold{fold_k}_week{week_idx}.pkl"
            )

            # -- CHECKPOINT RESUME: skip training if a valid pkl already exists
            if os.path.exists(ckpt_path):
                log(f"    [RESUME] Checkpoint found: {ckpt_path}  -- loading, skip training.")
                try:
                    with open(ckpt_path, "rb") as fh:
                        model = pickle.load(fh)
                    best_iter   = model.best_iteration_
                    val_pred_w  = model.predict(X_val_np)
                    test_pred_w = model.predict(X_test_np)

                    fold_val_preds[:, week_idx] = val_pred_w
                    fold_test_pred[:, week_idx] = test_pred_w.astype(np.float32)
                    fold_week_models.append(model)

                    w_mae = float(np.mean(np.abs(val_pred_w - y_val_w)))
                    log(f"    Week {week_idx + 1}: best_iter={best_iter}  val_MAE={w_mae:.4f}  [LOADED FROM CHECKPOINT]")
                    continue
                except Exception as e:
                    log(f"    [WARN] Checkpoint corrupted ({e}), deleting and retraining: {ckpt_path}")
                    os.remove(ckpt_path)
            # -----------------------------------------------------------------

            model = LGBMRegressor(
                objective              = "tweedie",  # Compound Poisson-Gamma for zero-inflated targets
                tweedie_variance_power = 1.5,        # Equal Poisson-Gamma balance (locked exponent)
                device                 = "gpu",      # Leverage CUDA tensor cores for deep split evals
                n_estimators           = 15000,      # Expanded budget to completely prevent underfitting
                learning_rate          = 0.015,       # Scale down to protect against gradient step explosion
                num_leaves             = 63,        # Expand to capture high-order feature interactions
                min_child_samples      = 100,        # Deep leaf regularization for generalization boundary
                colsample_bytree       = 0.75,       # Column sub-sampling vs covariate shift
                subsample              = 0.75,       # Row sub-sampling for split variance reduction
                random_state           = 42,
                n_jobs                 = 16,
                verbose                = -1,
                max_bin                = 127,
            )

            model.fit(
                X_train_np, y_train_w,
                eval_set      = [(X_val_np, y_val_w)],
                eval_metric   = "mae",
                callbacks     = [
                    lightgbm.early_stopping(stopping_rounds=250, verbose=False),
                    lightgbm.log_evaluation(period=500),
                ],
            )

            best_iter  = model.best_iteration_
            val_pred_w = model.predict(X_val_np)
            test_pred_w = model.predict(X_test_np)

            fold_val_preds[:, week_idx]  = val_pred_w
            fold_test_pred[:, week_idx]  = test_pred_w.astype(np.float32)
            fold_week_models.append(model)

            w_mae = float(np.mean(np.abs(val_pred_w - y_val_w)))
            log(f"    Week {week_idx + 1}: best_iter={best_iter}  val_MAE={w_mae:.4f}")

            # Save model checkpoint
            with open(ckpt_path, "wb") as fh:
                pickle.dump(model, fh)
            log(f"    [SAVED] {ckpt_path}")
            del model
            gc.collect()


        fold_elapsed = time.time() - fold_t0

        # -- 7g. Per-fold OOF Val MAE breakdown per week ----------------------
        week_maes = [
            float(np.mean(np.abs(fold_val_preds[:, w] - y_val_np[:, w])))
            for w in range(HORIZON)
        ]
        mean_fold_mae = float(np.mean(week_maes))
        log(f"\n  [Fold {fold_k}] Week MAEs : "
            + "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes)))
        log(f"  [Fold {fold_k}] Mean Val MAE : {mean_fold_mae:.4f}")
        log(f"  [Fold {fold_k}] Elapsed      : {fold_elapsed:.1f}s")
        fold_results.append((fold_k, week_maes, mean_fold_mae))

        fold_test_preds.append({
            "preds":      fold_test_pred,
            "region_ids": test_region_ids,
        })
        del X_train_np, y_train_np, X_val_np, y_val_np, fold_week_models
        gc.collect()

    # -- 8. Cross-fold summary -------------------------------------------------
    log(f"\n{'='*90}")
    log(f"5-Fold Cross-Validation Summary  [v27 GPU LightGBM Tweedie Mega-Tree]")
    log(f"{'='*90}")
    mean_cv_maes = []
    for fold_k, week_maes, mean_mae in fold_results:
        week_str = "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes))
        log(f"  Fold {fold_k}: {week_str}  ->  Mean={mean_mae:.4f}")
        mean_cv_maes.append(mean_mae)

    overall_mean = float(np.mean(mean_cv_maes))
    overall_std  = float(np.std(mean_cv_maes))
    log(f"\n  Overall CV MAE : {overall_mean:.4f}  +-  {overall_std:.4f}")
    log(f"  Best Fold      : Fold {int(np.argmin(mean_cv_maes))} "
        f"(MAE={min(mean_cv_maes):.4f})")

    log(f"\n  Per-week CV MAE breakdown:")
    for week_idx in range(HORIZON):
        week_maes_all = [fold_results[k][1][week_idx] for k in range(N_FOLDS)]
        log(f"    Week {week_idx + 1}: mean={np.mean(week_maes_all):.4f}  "
            f"std={np.std(week_maes_all):.4f}")

    # -- 9. Pure Median Ensemble Blending (PHASE 3) ----------------------------
    log(f"\n[v27] Pure Median Ensemble Blending ({N_FOLDS}-fold)")
    log(f"  Method: np.median across all {N_FOLDS} fold prediction matrices.")
    log(f"  Snap-to-Zero gate ABOLISHED: Tweedie natively collapses low-mass fractions.")
    log(f"  Protects ensemble against individual fold outlier fluctuations.")

    all_region_ids = fold_test_preds[0]["region_ids"]
    n_regions      = len(all_region_ids)
    assert n_regions == 2248, f"Expected 2248 test regions, got {n_regions}"

    # Stack: (N_FOLDS, n_regions, HORIZON)
    preds_stack  = np.stack([fp["preds"] for fp in fold_test_preds], axis=0)
    # Strict mathematical median reduction across fold axis
    final_preds  = np.median(preds_stack, axis=0)   # (n_regions, HORIZON)

    log(f"  preds_stack shape  : {preds_stack.shape}")
    log(f"  median_preds shape : {final_preds.shape}")
    log(f"  pre-clip stats     : mean={final_preds.mean():.4f}  "
        f"std={final_preds.std():.4f}  "
        f"min={final_preds.min():.4f}  max={final_preds.max():.4f}")

    # Physical safety boundary restriction [0, 5] -- no snap-to-zero gate
    final_preds = np.clip(final_preds, 0.0, 5.0)

    # -- 10. Submission prediction diagnostics ---------------------------------
    log("\n[Submission Prediction Diagnostics]")
    all_sub_preds   = final_preds.ravel()
    zero_frac_final = float((all_sub_preds == 0.0).mean())
    p50  = float(np.percentile(all_sub_preds, 50))
    p75  = float(np.percentile(all_sub_preds, 75))
    p90  = float(np.percentile(all_sub_preds, 90))
    p95  = float(np.percentile(all_sub_preds, 95))
    p99  = float(np.percentile(all_sub_preds, 99))
    pmax = float(np.max(all_sub_preds))
    log(f"  n={len(all_sub_preds):,}  "
        f"mean={all_sub_preds.mean():.4f}  std={all_sub_preds.std():.4f}")
    log(f"  p50={p50:.4f}  p75={p75:.4f}  p90={p90:.4f}  "
        f"p95={p95:.4f}  p99={p99:.4f}  max={pmax:.4f}")
    log(f"  zero-fraction (exact 0.0): {zero_frac_final:.4f}")
    if p99 < 2.0:
        log("  *** WARNING: p99 < 2.0 -- predictions may be under-dispersed! ***")
    else:
        log("  v p99 >= 2.0 -- prediction diversity is healthy.")

    # -- 11. Format & save submission.csv -------------------------------------
    log("\nFormatting submission.csv ...")
    rows = []
    for i, region_id in enumerate(all_region_ids):
        preds = final_preds[i]
        rows.append({
            "region_id":  region_id,
            "pred_week1": float(preds[0]),
            "pred_week2": float(preds[1]),
            "pred_week3": float(preds[2]),
            "pred_week4": float(preds[3]),
            "pred_week5": float(preds[4]),
        })

    submission = pd.DataFrame(rows)
    sub_path   = os.path.join(ROOT, "submission.csv")
    submission.to_csv(sub_path, index=False)

    # -- 12. Sanity checks -----------------------------------------------------
    assert len(submission) == 2248, f"Expected 2248 rows, got {len(submission)}"
    assert list(submission.columns) == [
        "region_id", "pred_week1", "pred_week2",
        "pred_week3", "pred_week4", "pred_week5",
    ], f"Unexpected columns: {list(submission.columns)}"
    log("  v Submission assertion passed: 2248 rows, 6 columns.")

    assert not submission.isnull().any().any(), "NaN values found in submission!"
    log("  v No NaN values in submission.")

    test_regions  = set(test_df["region_id"].unique())
    train_regions = set(train_df["region_id"].unique())
    assert test_regions == train_regions, "Train/test region sets do not match!"
    log("  v Train/test regions match (2248).")

    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    assert submission[pred_cols].max().max() <= 5.0 + 1e-6
    assert submission[pred_cols].min().min() >= 0.0 - 1e-6
    log("  v All predictions in [0, 5]  (np.clip physical guard enforced).")

    log(f"\n  submission.csv -> {sub_path}")
    log(f"  Rows (excl. header): {len(submission)}")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    # -- 13. Total elapsed time ------------------------------------------------
    elapsed = time.time() - t0
    log(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    log_path = os.path.join(ROOT, "_training_log_27th.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines))
    print(f"\nTraining log saved -> {log_path}")

    return {
        "fold_results":    fold_results,
        "overall_cv_mae":  overall_mean,
        "std_cv_mae":      overall_std,
        "input_dim":       N_FLAT_FEATURES,
        "submission":      submission,
        "sub_p99":         p99,
        "zero_frac_final": zero_frac_final,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
