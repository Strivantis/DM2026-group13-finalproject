"""
train.py -- Drought Score Forecasting Pipeline (v22 – Tabular LightGBM)
========================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv                  -- Kaggle submission (2248 rows × 6 cols)
    models/lgbm_fold{k}_week{w}.txt -- Best LGBM model per fold × week (25 total)
    models/scaler_fold_{k}.pkl      -- StandardScaler per fold (5 total)
    _training_log_22nd.txt          -- Full console log

Key improvements (v22 – Extreme Tabular Simplification)
---------------------------------------------------------
  1. PyTorch ABOLISHED:
       No BiLSTM, no TCN, no DataLoaders, no AMP, no gradient clipping,
       no CosineAnnealing.  Pure NumPy + Scikit-Learn + LightGBM pipeline.

  2. Tabular Flattening (3D → 2D):
       Input shape changes from (N, 13, 40) → (N_samples, 39).
       For each sliding window [i, i+13), only the 13th week (last row)
       is extracted as the feature vector.
       Rolling statistics (_roll_sum_4w, _week_max/min/std) inside that
       row already encode all 13-week historical momentum.

  3. Feature Pruning (v22 adversarial guard):
       DROP  wind_max  (collinearity > 0.95 with wind)
       DROP  dow_sin   (lowest permutation importance)
       DROP  dp_tmp    (collinearity > 0.9999 with tmp, v19 guard retained)
       DROP  wb_tmp    (same)
       RESULT: 39 retained features.

  4. Direct Multi-Step LightGBM Regressors:
       For each of the 5 CV folds, train 5 independent LGBMRegressors
       (one per forecast week: pred_week1 … pred_week5).
       objective='regression_l1'  →  fits Conditional Median natively.
       MAE is the Kaggle evaluation metric; L1 objective aligns perfectly.
       Early stopping patience = 50 rounds on per-fold OOF validation MAE.

  5. StratifiedGroupKFold CV (5-Fold):  UNCHANGED from v21.
       Group  = region_id  (atomic unit).
       Strata = 10-quantile bins of per-region historical mean score.
       Train  = 80% of regions; Val = 20% of regions (held-out geography).

  6. OOF Scaler Alignment:
       StandardScaler fitted ONLY on training fold rows, then applied to
       val rows and the final test matrix.  No data leakage.

  7. Ensemble Inference (No Hurdle Gate):
       For each target week, average predictions across the 5 folds.
       NO Sigmoid multiplication, NO probability hard-clamping.
       np.clip(final_pred, 0.0, 5.0) as final safety guard.

LightGBM Hyperparameters (per-week model)
-----------------------------------------
    objective    = 'regression_l1'   # MAE/median fitting; handles 60% zeroes
    learning_rate= 0.04
    colsample_bytree = 0.75          # feature sub-sampling vs. covariate shift
    subsample    = 0.75              # row sub-sampling for variance reduction
    n_estimators = 1500              # large budget with early stopping
    random_state = 42
    n_jobs       = -1                # maximise i9 CPU core utilisation
    early_stopping_rounds = 50       # patience on OOF val MAE
"""

import os
import sys
import time
import random
import pickle

import numpy as np
import pandas as pd
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

# -- project root on sys.path --------------------------------------------------
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

N_FOLDS             = 5     # StratifiedGroupKFold 5-folds
LGBM_N_ESTIMATORS   = 1500  # max trees (early stopping will reduce)
LGBM_LR             = 0.04
LGBM_COLSAMPLE      = 0.75
LGBM_SUBSAMPLE      = 0.75
LGBM_EARLY_STOP     = 50    # patience (rounds without improvement)


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
    """
    Inject leakage-free TE columns into each region group DataFrame.
    Supports both 3-tuple (group, i_min, i_max) and 4-tuple entries.
    """
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
    log("=" * 85)
    log("Drought Forecasting Pipeline  v22")
    log("Extreme Tabular Simplification  |  Direct Multi-Step LightGBM Regressors")
    log("StratifiedGroupKFold CV  |  L1 (MAE) Objective  |  OOF StandardScaler")
    log("=" * 85)
    log("")
    log("Paradigm Shift (v22):")
    log("  V21 : BiLSTM + TCN Hurdle model, PyTorch 3D tensors (B, 13, 40).")
    log("  V22 : Tabular flattening to (N_samples, 39) — only 13th week extracted.")
    log("        5 independent LGBMRegressors per fold, one per forecast week.")
    log("        Objective = regression_l1 → Conditional Median → native MAE.")
    log("        No Hurdle gate, no probability sigmoid, no BCE loss.")
    log("")
    log("Feature Pruning (v22 adversarial guard):")
    log("  DROPPED wind_max  (collinearity > 0.95 with wind)")
    log("  DROPPED dow_sin   (lowest permutation importance)")
    log("  DROPPED dp_tmp / wb_tmp  (collinearity > 0.9999 with tmp, v19 guard)")
    log("  RETAINED 39 features: _roll_sum_4w, _week_max/min/std dominant.")
    log("")
    log("LGBM Hyperparameters per-week model:")
    log(f"  objective      = 'regression_l1'")
    log(f"  learning_rate  = {LGBM_LR}")
    log(f"  colsample_bytree = {LGBM_COLSAMPLE}")
    log(f"  subsample      = {LGBM_SUBSAMPLE}")
    log(f"  n_estimators   = {LGBM_N_ESTIMATORS}  (early stopping patience={LGBM_EARLY_STOP})")
    log(f"  n_jobs         = -1  (all CPU cores)")
    log(f"  random_state   = 42")
    log("")
    log("CV Strategy  : 5-Fold StratifiedGroupKFold (unchanged from v21)")
    log(f"  N_FOLDS      : {N_FOLDS}")
    log(f"  Group        : region_id")
    log(f"  Strata       : 10-quantile bins of per-region historical mean score")
    log(f"  Train        : 80% of regions per fold (geography-unseen)")
    log(f"  Val          : 20% of regions per fold (completely held-out)")
    log("")
    log("Inference    : Average predictions across 5 folds (per week).")
    log("               np.clip(final_pred, 0.0, 5.0) safety guard applied.")
    log("               NO Sigmoid gate, NO hurdle multiplication.")

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
    log("\n[v22 Feature Validation]")
    assert "week_sin" in train_raw.columns, "week_sin missing -- run preprocess.py first"
    assert "week_cos" in train_raw.columns, "week_cos missing -- run preprocess.py first"
    log("  v week_sin, week_cos present.")

    bad_cols = [c for c in train_raw.columns if ("8w" in c or "13w" in c)]
    if bad_cols:
        log(f"  *** WARNING: 8w/13w features found: {bad_cols}")
    else:
        log("  v No 8w/13w rolling features (domain-shift columns absent).")

    adv_present = [c for c in train_raw.columns if c in ("dp_tmp", "wb_tmp", "wind_max")]
    if adv_present:
        log(f"  [INFO] Adversarial columns in raw CSV (will be pruned by refine_features): "
            f"{adv_present}")
    else:
        log("  v dp_tmp, wb_tmp, wind_max absent from processed CSV.")

    v19_enriched = ["tmp_week_std", "prec_week_max", "humidity_week_std", "wind_week_std"]
    missing_enriched = [c for c in v19_enriched if c not in train_raw.columns]
    if missing_enriched:
        log(f"  *** WARNING: v19 enriched features missing: {missing_enriched}")
        log("      --> Run `python src/preprocess.py` to regenerate processed data.")
    else:
        log("  v v19 enriched weekly statistics confirmed present.")

    # -- 1c. Data Leakage Check ------------------------------------------------
    log("\n[Data Leakage Check]")
    leaky_cols = [c for c in FEATURE_COLS if "score" in c.lower()
                  and c not in ("region_mean_score", "region_zero_prob")]
    if leaky_cols:
        log(f"  *** WARNING: Potential leaky features found: {leaky_cols} ***")
    else:
        log("  v No raw-score autoregressive features in FEATURE_COLS.")
    log(f"  FEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # -- 2. Feature refinement -------------------------------------------------
    log("\nRefining features (drought proxy index + log1p prec + v22 pruning) ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    log(f"  train after refinement: {train_df.shape}  |  test: {test_df.shape}")

    # -- 3. Drop rows with NaN score -------------------------------------------
    before = len(train_df)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    dropped_nan = before - len(train_df)
    if dropped_nan:
        log(f"  [NaN drop] Removed {dropped_nan:,} rows with NaN score from train_df.")

    # -- 4. Target score distribution summary ---------------------------------
    log("\n[Training Target Distribution]")
    all_scores = train_df["score"].values
    zero_frac  = (all_scores == 0.0).mean()
    log(f"  mean={all_scores.mean():.4f}  std={all_scores.std():.4f}  "
        f"min={all_scores.min():.2f}  max={all_scores.max():.2f}")
    log(f"  Zero-inflation: {zero_frac:.2%} of training scores == 0.0")
    log(f"  LightGBM regression_l1 (Conditional Median) is optimal for "
        f"{zero_frac:.2%} zero-inflation.")
    for thresh in [1.0, 2.0, 3.0, 4.0]:
        frac = (all_scores > thresh).mean() * 100
        log(f"  score > {thresh:.1f}: {frac:.2f}%  [{int((all_scores > thresh).sum()):,} samples]")

    # -- 5. Feature columns and input shape ------------------------------------
    log("\n[v22] Feature columns ...")
    base_feat_cols = [c for c in FEATURE_COLS
                      if c in train_df.columns
                      and c not in ("region_mean_score", "region_zero_prob")]
    log(f"  Base features (excl. TE): {len(base_feat_cols)}")
    log(f"  Total with TE: {len(FEATURE_COLS)}")
    input_size = len(FEATURE_COLS)
    log(f"  input_size = {input_size}  (tabular row width; expected 39)")

    # -- 6. Build 5-Fold StratifiedGroupKFold CV splits -----------------------
    log(f"\n{'='*85}")
    log(f"5-Fold StratifiedGroupKFold CV  [v22 – Tabular LightGBM]")
    log(f"  Group  : region_id")
    log(f"  Strata : 10-quantile bins of per-region historical mean drought score")
    log(f"  Train  : 80% of regions per fold (geographically unseen)")
    log(f"  Val    : 20% of regions per fold (completely held-out)")
    log(f"{'='*85}")

    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    log(f"\n  Folds built: {len(folds)}")
    for fi, (tg, vg) in enumerate(folds):
        log(f"  Fold {fi}: train_groups={len(tg):,}  val_groups={len(vg):,}")

    # -- 7. 5-Fold Training Loop -----------------------------------------------
    fold_results    = []           # (fold_k, [week_mae x 5], mean_mae)
    fold_test_preds = []           # list of {'preds': (n_regions, 5), 'region_ids': arr}

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):

        log(f"\n{'='*85}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}  [v22 LightGBM Tabular – Direct Multi-Step]")
        log(f"  train_groups: {len(raw_train_groups):,}  |  "
            f"val_groups: {len(raw_val_groups):,}")
        log(f"{'='*85}")

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
        log(f"  [OOF TE] Train regions in fold: {len(train_region_ids_fold)}")
        log(f"  [OOF TE]  Val regions in fold : {len(val_region_ids_fold)}")
        log(f"  [OOF TE] global_mean_score={gm_fold:.4f}  "
            f"global_zero_prob={gzp_fold:.4f}")

        # -- 7b. Augment groups with fold-local TE ----------------------------
        aug_train_groups = _augment_groups_with_te(
            raw_train_groups, te_map_fold, gm_fold, gzp_fold
        )
        aug_val_groups = _augment_groups_with_te(
            raw_val_groups, te_map_fold, gm_fold, gzp_fold
        )

        # -- 7c. Determine feature columns available (all TE-augmented groups) -
        _sample_group = aug_train_groups[0][0]
        feat_cols = [c for c in FEATURE_COLS if c in _sample_group.columns]
        log(f"  feat_cols available: {len(feat_cols)} / {len(FEATURE_COLS)}")

        # -- 7d. Build tabular matrices (V22: 13th-week row extraction) --------
        log(f"\n  [V22 Tabular Flattening] Extracting 13th-week row per window ...")
        X_train, y_train, _ = build_tabular_dataset(aug_train_groups, feat_cols)
        X_val,   y_val,   _ = build_tabular_dataset(aug_val_groups,   feat_cols)

        log(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
        log(f"  X_val   : {X_val.shape}    |  y_val   : {y_val.shape}")
        assert X_train.shape[1] == len(feat_cols), \
            f"Feature column mismatch: X_train has {X_train.shape[1]} cols, " \
            f"expected {len(feat_cols)}"
        assert y_train.shape[1] == HORIZON, \
            f"Target shape mismatch: expected {HORIZON} weeks, got {y_train.shape[1]}"

        # -- 7e. Fit fold-specific StandardScaler (on train rows ONLY) --------
        log(f"\n  [OOF Scaler] Fitting StandardScaler on {X_train.shape[0]:,} "
            f"training rows ({X_train.shape[1]} features) ...")
        fold_scaler   = StandardScaler()
        X_train_sc    = fold_scaler.fit_transform(X_train)
        X_val_sc      = fold_scaler.transform(X_val)

        # -- 7f. Build test tabular matrix for this fold ----------------------
        test_df_fold  = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test, test_region_ids = build_tabular_test(test_df_fold, feat_cols)
        X_test_sc     = fold_scaler.transform(X_test)
        log(f"  X_test  : {X_test_sc.shape}  (2248 test regions)")

        # -- 7g. Train 5 independent LGBMRegressors (one per forecast week) ---
        log(f"\n  Training 5 LGBMRegressor models (one per forecast week) ...")
        fold_val_maes        = []
        fold_test_week_preds = []      # list of (n_test_regions,) arrays, length 5

        # Convert to pandas DataFrame so LightGBM carries feature names through
        # predict() and eliminates the "X does not have valid feature names" warning.
        X_train_df = pd.DataFrame(X_train_sc, columns=feat_cols)
        X_val_df   = pd.DataFrame(X_val_sc,   columns=feat_cols)
        X_test_df  = pd.DataFrame(X_test_sc,  columns=feat_cols)

        for week_idx in range(HORIZON):
            # .copy() avoids LightGBM "np.ndarray subset" peak-memory warning
            y_tr_w  = y_train[:, week_idx].copy()   # (N_train,) contiguous
            y_val_w = y_val[:,   week_idx].copy()   # (N_val,)   contiguous

            lgbm_model = LGBMRegressor(
                objective      = "regression_l1",
                learning_rate  = LGBM_LR,
                colsample_bytree = LGBM_COLSAMPLE,
                subsample      = LGBM_SUBSAMPLE,
                n_estimators   = LGBM_N_ESTIMATORS,
                random_state   = 42,
                n_jobs         = -1,
                verbose        = -1,
            )

            lgbm_model.fit(
                X_train_df, y_tr_w,
                eval_set    = [(X_val_df, y_val_w)],
                eval_metric = "mae",
                callbacks   = [
                    lightgbm.early_stopping(stopping_rounds=LGBM_EARLY_STOP,
                                            verbose=False),
                    lightgbm.log_evaluation(period=200),
                ],
            )

            best_iter  = lgbm_model.best_iteration_
            val_pred   = lgbm_model.predict(X_val_df)
            val_mae_w  = float(np.mean(np.abs(val_pred - y_val_w)))
            fold_val_maes.append(val_mae_w)

            # Test prediction for this week
            test_pred_w = lgbm_model.predict(X_test_df)
            fold_test_week_preds.append(test_pred_w)

            log(f"    Week {week_idx + 1}: best_iter={best_iter:4d}  "
                f"val_MAE={val_mae_w:.4f}")

            # Save LGBM model
            model_path = os.path.join(
                MODELS_DIR, f"lgbm_fold{fold_k}_week{week_idx}.txt"
            )
            lgbm_model.booster_.save_model(model_path)

        mean_fold_mae = float(np.mean(fold_val_maes))
        log(f"\n  [Fold {fold_k}] Week MAEs : "
            + "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(fold_val_maes)))
        log(f"  [Fold {fold_k}] Mean Val MAE : {mean_fold_mae:.4f}")
        fold_results.append((fold_k, fold_val_maes, mean_fold_mae))

        # -- 7h. Stack test predictions: (n_test_regions, 5) -----------------
        fold_test_pred_matrix = np.stack(fold_test_week_preds, axis=1)
        # fold_test_week_preds: list of 5 arrays shape (n_test_regions,)
        # → stack along axis=1 → (n_test_regions, 5)
        fold_test_preds.append({
            "preds":      fold_test_pred_matrix,
            "region_ids": test_region_ids,
        })

        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, f"scaler_fold_{fold_k}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(fold_scaler, f)
        log(f"  Fold scaler saved -> {scaler_path}")

    # -- 8. Cross-fold summary -------------------------------------------------
    log(f"\n{'='*85}")
    log(f"5-Fold Cross-Validation Summary  [v22 Direct Multi-Step LightGBM]")
    log(f"{'='*85}")
    mean_cv_maes = []
    for fold_k, week_maes, mean_mae in fold_results:
        week_str = "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes))
        log(f"  Fold {fold_k}: {week_str}  →  Mean={mean_mae:.4f}")
        mean_cv_maes.append(mean_mae)

    overall_mean = float(np.mean(mean_cv_maes))
    overall_std  = float(np.std(mean_cv_maes))
    log(f"\n  Overall CV MAE : {overall_mean:.4f}  ±  {overall_std:.4f}")
    log(f"  Best Fold      : Fold {int(np.argmin(mean_cv_maes))} "
        f"(MAE={min(mean_cv_maes):.4f})")

    # Per-week CV MAE summary
    log(f"\n  Per-week CV MAE breakdown:")
    for week_idx in range(HORIZON):
        week_maes_all = [fold_results[k][1][week_idx] for k in range(N_FOLDS)]
        log(f"    Week {week_idx + 1}: mean={np.mean(week_maes_all):.4f}  "
            f"std={np.std(week_maes_all):.4f}")

    # -- 9. Ensemble Inference (average across 5 folds) -----------------------
    log(f"\n[v22] Ensemble Blending ({N_FOLDS}-fold simple average) ...")
    log(f"  No Hurdle gate. No Sigmoid. Direct MAE regressor output averaged.")

    # All folds predict the same 2248 regions in the same order (groupby sorts)
    all_region_ids = fold_test_preds[0]["region_ids"]
    n_regions      = len(all_region_ids)
    assert n_regions == 2248, f"Expected 2248 test regions, got {n_regions}"

    # Stack: (N_FOLDS, n_regions, HORIZON)
    preds_stack = np.stack(
        [fp["preds"] for fp in fold_test_preds], axis=0
    )

    # Simple average across folds
    mean_preds = np.mean(preds_stack, axis=0)   # (n_regions, HORIZON)

    log(f"  preds_stack shape : {preds_stack.shape}")
    log(f"  mean_preds shape  : {mean_preds.shape}")
    log(f"  mean_preds stats  : mean={mean_preds.mean():.4f}  "
        f"std={mean_preds.std():.4f}  "
        f"min={mean_preds.min():.4f}  max={mean_preds.max():.4f}")

    # Final safety clip: [0.0, 5.0]
    final_preds = np.clip(mean_preds, 0.0, 5.0)

    # -- 10. Submission prediction diagnostics ---------------------------------
    log("\n[Submission Prediction Diagnostics]")
    all_sub_preds = final_preds.ravel()
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
    log(f"  Train zero-inflation was {(all_scores == 0.0).mean():.4f}")

    # -- 11. Format & save submission.csv -------------------------------------
    log("\nFormatting submission.csv ...")
    rows = []
    for i, region_id in enumerate(all_region_ids):
        preds = final_preds[i]   # (5,)
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
    assert test_regions == train_regions, \
        "Train/test region sets do not match!"
    log("  v Train/test regions match (2248).")

    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    assert submission[pred_cols].max().max() <= 5.0 + 1e-6
    assert submission[pred_cols].min().min() >= 0.0 - 1e-6
    log("  v All predictions in [0, 5]  (np.clip guard enforced).")

    log(f"\n  submission.csv -> {sub_path}")
    log(f"  Rows (excl. header): {len(submission)}")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    # -- 13. Total elapsed time ------------------------------------------------
    elapsed = time.time() - t0
    log(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    log_path = os.path.join(ROOT, "_training_log_22nd.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nTraining log saved -> {log_path}")

    return {
        "fold_results":     fold_results,
        "overall_cv_mae":   overall_mean,
        "std_cv_mae":       overall_std,
        "input_size":       input_size,
        "submission":       submission,
        "sub_p99":          p99,
        "zero_frac_final":  zero_frac_final,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
