"""
train.py -- Drought Score Forecasting Pipeline (v28 -- The Calibrated Tabular Hurdle)
======================================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv                        -- Kaggle submission (2248 rows x 6 cols)
    models/lgbm_a_fold{k}_week{w}.pkl     -- Model A (L1 Regressor) checkpoint (25 total)
    models/lgbm_b_fold{k}_week{w}.pkl     -- Model B (Binary Classifier) checkpoint (25 total)
    _training_log_28th.txt                -- Full console log

v28 Changes (The Calibrated Tabular Hurdle)
-------------------------------------------
  ABOLISH: Single-head Tweedie regressor (objective='tweedie') completely removed.
           Tweedie exponential link cannot produce exact zeroes; zero-fraction in
           submission was 0.0000 despite 59.63% zeros in training ground-truth.

  IMPLEMENT: Decoupled Dual-Tree Hurdle Pipeline -- two distinct LightGBM models
             trained in parallel inside each CV fold x week:

    Model A  (The Median Predictor):
      objective='regression_l1'  -- Natively optimizes the Conditional Median (L1 Loss)
      max_depth=5                 -- Strict depth ceiling to prevent memorizing deltas
      num_leaves=31               -- Capacity matched to depth constraint
      colsample_bytree=0.5        -- High feature regularization, alternative split paths
      learning_rate=0.02          -- Stable, low-oscillation convergence step
      n_estimators=10000          -- Wide budget controlled by early stopping
      early_stop=150 rounds, eval_metric='mae'

    Model B  (The Drought Probability Classifier):
      objective='binary'          -- Pure binary cross-entropy probability tracking
      max_depth=5                 -- Structural regularization synchronized with Model A
      num_leaves=31               -- Capacity matched to depth constraint
      colsample_bytree=0.5        -- Feature sub-sampling identical at 50%
      learning_rate=0.02
      n_estimators=5000           -- Focused tree budget for classification boundaries
      early_stop=150 rounds, eval_metric='binary_logloss'
      Target: (y > 0.0).astype(int)

  OOF COMBINATION:
      oof_final = np.where(oof_prob < 0.5, 0.0, oof_l1)
      Zero-gate derived from calibrated classifier probability -- no hand-crafted threshold.

  BINNED ERROR MATRIX:
      6 physical drought intervals audited per fold:
        Interval 0: Absolute Zero     (y == 0.0)
        Interval 1: Mild Drought      (0.0 < y <= 1.0)
        Interval 2: Moderate Drought  (1.0 < y <= 2.0)
        Interval 3: Severe Drought    (2.0 < y <= 3.0)
        Interval 4: Extreme Drought   (3.0 < y <= 4.0)
        Interval 5: Exceptional Drought (4.0 < y <= 5.0)
      Reports: count, avg_true, avg_pred, interval_MAE per bin.

  TEST INFERENCE (Asymmetric Ensemble Compression):
      Model A channel: strict np.median across 5 folds (robust to outlier folds)
      Model B channel: np.mean across 5 folds (probability averaging for calibration)
      Final gate: final_preds = np.where(prob_mean < 0.5, 0.0, l1_median)
      Safety clip: np.clip(0.0, 5.0)

  RETAIN: 378-dimensional flat tabular layout from v27 exactly.
          (27 features x 13 weeks = 351 + 27 explicit trend deltas = 378)
          5-Fold StratifiedGroupKFold, group=region_id.

Architecture
-------------
    Model A: LGBMRegressor  x 5 per fold (one per future week)
    Model B: LGBMClassifier x 5 per fold (one per future week)
    objective_A        : regression_l1
    objective_B        : binary
    device             : gpu
    max_depth          : 5
    num_leaves         : 31
    colsample_bytree   : 0.5
    learning_rate      : 0.02
    n_estimators_A     : 10000
    n_estimators_B     : 5000
    early_stop_rounds  : 150
    CV                 : 5-Fold StratifiedGroupKFold, group=region_id

Feature Space
--------------
    v28 retains v27: 27 features x 13 weeks (351) + 27 explicit deltas = 378 dimensions

Post-Processing
---------------
    Model A ensemble : np.median across 5 folds
    Model B ensemble : np.mean  across 5 folds
    Zero-gate        : np.where(prob_mean < 0.5, 0.0, l1_median)
    Clip             : np.clip(0.0, 5.0) physical safety guard
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

# v28 retains v27 378-dim feature space exactly
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
# Binned Error Matrix diagnostic
# ---------------------------------------------------------------------------
INTERVAL_LABELS = [
    "Interval 0  [Absolute Zero     y == 0.0       ]",
    "Interval 1  [Mild Drought      0.0 < y <= 1.0 ]",
    "Interval 2  [Moderate Drought  1.0 < y <= 2.0 ]",
    "Interval 3  [Severe Drought    2.0 < y <= 3.0 ]",
    "Interval 4  [Extreme Drought   3.0 < y <= 4.0 ]",
    "Interval 5  [Exceptional       4.0 < y <= 5.0 ]",
]

def _interval_mask(y_true: np.ndarray, interval_idx: int) -> np.ndarray:
    """Return boolean mask for samples belonging to drought interval idx."""
    if interval_idx == 0:
        return y_true == 0.0
    elif interval_idx == 1:
        return (y_true > 0.0) & (y_true <= 1.0)
    elif interval_idx == 2:
        return (y_true > 1.0) & (y_true <= 2.0)
    elif interval_idx == 3:
        return (y_true > 2.0) & (y_true <= 3.0)
    elif interval_idx == 4:
        return (y_true > 3.0) & (y_true <= 4.0)
    elif interval_idx == 5:
        return (y_true > 4.0) & (y_true <= 5.0)
    else:
        raise ValueError(f"Unknown interval index: {interval_idx}")


def print_binned_error_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                               fold_k: int, log_fn) -> None:
    """
    Print a Binned Error Matrix auditing 6 physical drought intervals.

    Parameters
    ----------
    y_true  : 1-D array of true ground-truth targets (all weeks flattened or per-week)
    y_pred  : 1-D array of OOF final predictions after the hurdle gate
    fold_k  : fold index for display labeling
    log_fn  : callable (e.g. the local `log` function) for dual print+capture
    """
    log_fn("")
    log_fn("  " + "=" * 80)
    log_fn(f"  BINNED ERROR MATRIX  --  Fold {fold_k}  (OOF Hurdle-Gated Predictions)")
    log_fn("  " + "=" * 80)
    header = (
        f"  {'Interval':<50}  {'Count':>7}  {'AvgTrue':>8}  "
        f"{'AvgPred':>8}  {'MAE':>8}"
    )
    log_fn(header)
    log_fn("  " + "-" * 78)

    for idx, label in enumerate(INTERVAL_LABELS):
        mask = _interval_mask(y_true, idx)
        n    = int(mask.sum())
        if n == 0:
            log_fn(
                f"  {label:<50}  {n:>7}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}"
            )
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

    # -- 0. Pipeline description -----------------------------------------------
    log("=" * 90)
    log("Drought Forecasting Pipeline  v28  (The Calibrated Tabular Hurdle)")
    log("Decoupled Dual-Tree Hurdle  |  Model A: regression_l1  |  Model B: binary")
    log("Feature Space: 27 features x 13 weeks (351) + 27 explicit deltas = 378 dims")
    log("OOF Gate : np.where(oof_prob < 0.5, 0.0, oof_l1)")
    log("Test Blend: Model A -> np.median (5 folds)  |  Model B -> np.mean (5 folds)")
    log("Final Gate: np.where(prob_mean < 0.5, 0.0, l1_median)  |  clip [0, 5]")
    log("5-Fold StratifiedGroupKFold  |  5 model pairs per fold (one pair per week)")
    log("=" * 90)
    log("")
    log("v28 Architectural Changes over v27:")
    log("  [ABOLISH] Tweedie single-head regressor completely removed.")
    log("            Tweedie exp-link cannot produce exact zeroes -- zero-fraction")
    log("            in v27 submission was 0.0000 vs 59.63% training zeros.")
    log("  [IMPLEMENT] Decoupled Dual-Tree Hurdle Pipeline:")
    log("    Model A -- LGBMRegressor(objective='regression_l1')")
    log("               max_depth=5  num_leaves=31  colsample_bytree=0.5")
    log("               lr=0.02  n_estimators=10000  early_stop=150 (eval=mae)")
    log("    Model B -- LGBMClassifier(objective='binary')")
    log("               max_depth=5  num_leaves=31  colsample_bytree=0.5")
    log("               lr=0.02  n_estimators=5000   early_stop=150 (eval=binary_logloss)")
    log("               target = (y > 0.0).astype(int)")
    log("  [IMPLEMENT] Binned Error Matrix: 6 drought-interval calibration audit per fold.")
    log("  [IMPLEMENT] Asymmetric Ensemble Compression:")
    log("              Model A  -> np.median (robust to fold outliers)")
    log("              Model B  -> np.mean   (probability averaging = calibration)")
    log("  [RETAIN] 378-dim flat tabular layout from v27.")
    log("")
    log(f"Training Config:")
    log(f"  Model A (Regressor)  objective       = regression_l1")
    log(f"  Model B (Classifier) objective       = binary")
    log(f"  LGBM device                          = gpu")
    log(f"  max_depth                            = 5")
    log(f"  num_leaves                           = 31")
    log(f"  colsample_bytree                     = 0.5")
    log(f"  learning_rate                        = 0.02")
    log(f"  n_estimators (Model A)               = 10000")
    log(f"  n_estimators (Model B)               = 5000")
    log(f"  early_stopping_rounds                = 150")
    log(f"  eval_metric (Model A)                = mae")
    log(f"  eval_metric (Model B)                = binary_logloss")
    log(f"  N_FOLDS                              = {N_FOLDS}")
    log(f"  N_FLAT_FEATURES                      = {N_FLAT_FEATURES}  "
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
    log("\n[v28 Feature Validation]")
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
        log(f"  [INFO] Lag columns present in raw CSV (excluded from FEATURE_COLS): "
            f"{lag_cols_check[:6]}{'...' if len(lag_cols_check) > 6 else ''}")
    log(f"  v28 FEATURE_COLS: {len(FEATURE_COLS)} base features (lag/rolling purged).")
    log(f"  v28 flat input dim: {N_FLAT_FEATURES}  "
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
    log(f"  --> Hurdle classifier target: {(all_scores > 0.0).sum():,} positives "
        f"({(all_scores > 0.0).mean():.2%}) vs "
        f"{(all_scores == 0.0).sum():,} zeroes ({zero_frac:.2%})")

    # -- 5. Feature columns ----------------------------------------------------
    log(f"\n[v28] Feature columns and flat input shape ...")
    log(f"  FEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    log(f"  flat input_size = {N_FLAT_FEATURES}  "
        f"(WINDOW_SIZE={WINDOW_SIZE} x FEATURE_COLS={len(FEATURE_COLS)} + "
        f"{len(FEATURE_COLS)} deltas = {N_FLAT_FEATURES})")

    # -- 6. Build 5-Fold StratifiedGroupKFold CV splits -----------------------
    log(f"\n{'='*90}")
    log(f"5-Fold StratifiedGroupKFold CV  [v28 -- Decoupled Dual-Tree Hurdle]")
    log(f"  Group  : region_id  |  Strata : 10-quantile bins of per-region mean score")
    log(f"{'='*90}")

    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    log(f"\n  Folds built: {len(folds)}")
    for fi, (tg, vg) in enumerate(folds):
        log(f"  Fold {fi}: train_groups={len(tg):,}  val_groups={len(vg):,}")

    # -- 7. 5-Fold Dual-Tree Hurdle Training Loop ------------------------------
    log(f"\n{'='*90}")
    log("5-Fold Dual-Tree Hurdle Training  [v28]")
    log("  Per fold x week: Model A (L1 Regressor) + Model B (Binary Classifier)")
    log("  Both use max_depth=5 for strict temporal regularization.")
    log("  OOF output gated: oof_final = np.where(oof_prob < 0.5, 0.0, oof_l1)")
    log(f"{'='*90}")

    fold_results      = []   # (fold_k, week_maes_l1, week_maes_final, mean_mae_final)
    fold_test_preds_a = []   # list of {"preds": (n_regions, 5), "region_ids": [...]}
    fold_test_preds_b = []   # list of {"probs": (n_regions, 5), "region_ids": [...]}

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):

        log(f"\n{'='*90}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}  [v28 Decoupled Dual-Tree Hurdle]")
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

        # Verify 378-dim feature space (retained from v27)
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

        # -- 7f. Train MODEL A + MODEL B per target week ----------------------
        fold_val_preds_l1   = np.zeros_like(y_val_np)    # raw L1 regression outputs
        fold_val_probs      = np.zeros_like(y_val_np)    # raw classifier probabilities
        fold_val_final      = np.zeros_like(y_val_np)    # hurdle-gated OOF predictions
        fold_test_pred_l1   = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)
        fold_test_prob      = np.zeros((X_test_np.shape[0], HORIZON), dtype=np.float32)

        for week_idx in range(HORIZON):
            y_train_w   = y_train_np[:, week_idx]              # (N_train,) continuous
            y_val_w     = y_val_np[:,   week_idx]              # (N_val,)   continuous
            y_train_b   = (y_train_w > 0.0).astype(int)        # (N_train,) binary
            y_val_b     = (y_val_w   > 0.0).astype(int)        # (N_val,)   binary

            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            ckpt_b = os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl")

            log(f"\n  --- Week {week_idx + 1} ---")

            # ----------------------------------------------------------------
            # MODEL A  (L1 Regressor -- Conditional Median)
            # ----------------------------------------------------------------
            if os.path.exists(ckpt_a):
                log(f"    [RESUME-A] Checkpoint found: {ckpt_a}  -- loading.")
                try:
                    with open(ckpt_a, "rb") as fh:
                        model_a = pickle.load(fh)
                    val_l1_w  = model_a.predict(X_val_np)
                    test_l1_w = model_a.predict(X_test_np)
                    best_a = model_a.best_iteration_
                    mae_a  = float(np.mean(np.abs(val_l1_w - y_val_w)))
                    log(f"    [Model A] best_iter={best_a}  val_MAE={mae_a:.4f}  [LOADED]")
                except Exception as e:
                    log(f"    [WARN-A] Checkpoint corrupted ({e}), deleting and retraining.")
                    os.remove(ckpt_a)
                    model_a = None
            else:
                model_a = None

            if model_a is None:
                model_a = LGBMRegressor(
                    objective        = "regression_l1",  # Natively optimizes Conditional Median (L1)
                    max_depth        = 5,                # Strict depth ceiling, prevents delta noise memorization
                    num_leaves       = 31,               # Capacity matched with depth constraint
                    colsample_bytree = 0.5,              # High feature regularization, alternative split paths
                    learning_rate    = 0.02,             # Stable low-oscillation convergence step
                    n_estimators     = 10000,            # Wide budget controlled by early stopping
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
                        lightgbm.early_stopping(stopping_rounds=150, verbose=False),
                        lightgbm.log_evaluation(period=500),
                    ],
                )
                val_l1_w  = model_a.predict(X_val_np)
                test_l1_w = model_a.predict(X_test_np)
                best_a    = model_a.best_iteration_
                mae_a     = float(np.mean(np.abs(val_l1_w - y_val_w)))
                log(f"    [Model A] best_iter={best_a}  val_MAE={mae_a:.4f}")
                with open(ckpt_a, "wb") as fh:
                    pickle.dump(model_a, fh)
                log(f"    [SAVED-A] {ckpt_a}")

            fold_val_preds_l1[:, week_idx]  = val_l1_w
            fold_test_pred_l1[:, week_idx]  = test_l1_w.astype(np.float32)
            del model_a
            gc.collect()

            # ----------------------------------------------------------------
            # MODEL B  (Binary Classifier -- Drought Probability)
            # ----------------------------------------------------------------
            if os.path.exists(ckpt_b):
                log(f"    [RESUME-B] Checkpoint found: {ckpt_b}  -- loading.")
                try:
                    with open(ckpt_b, "rb") as fh:
                        model_b = pickle.load(fh)
                    val_prob_w  = model_b.predict_proba(X_val_np)[:, 1]
                    test_prob_w = model_b.predict_proba(X_test_np)[:, 1]
                    best_b = model_b.best_iteration_
                    log(f"    [Model B] best_iter={best_b}  [LOADED]")
                except Exception as e:
                    log(f"    [WARN-B] Checkpoint corrupted ({e}), deleting and retraining.")
                    os.remove(ckpt_b)
                    model_b = None
            else:
                model_b = None

            if model_b is None:
                model_b = LGBMClassifier(
                    objective        = "binary",          # Pure binary cross-entropy probability tracking
                    max_depth        = 5,                # Structural regularization synchronized with Model A
                    num_leaves       = 31,               # Capacity matched with depth constraint
                    colsample_bytree = 0.5,              # Feature sub-sampling identical at 50%
                    learning_rate    = 0.02,
                    n_estimators     = 5000,             # Focused tree budget for classification boundaries
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
                        lightgbm.early_stopping(stopping_rounds=150, verbose=False),
                        lightgbm.log_evaluation(period=500),
                    ],
                )
                val_prob_w  = model_b.predict_proba(X_val_np)[:, 1]
                test_prob_w = model_b.predict_proba(X_test_np)[:, 1]
                best_b      = model_b.best_iteration_
                log(f"    [Model B] best_iter={best_b}")
                with open(ckpt_b, "wb") as fh:
                    pickle.dump(model_b, fh)
                log(f"    [SAVED-B] {ckpt_b}")

            fold_val_probs[:, week_idx]     = val_prob_w
            fold_test_prob[:, week_idx]     = test_prob_w.astype(np.float32)
            del model_b
            gc.collect()

            # ----------------------------------------------------------------
            # Apply local hurdle gate for OOF diagnostics
            # ----------------------------------------------------------------
            oof_l1_w   = fold_val_preds_l1[:, week_idx]
            oof_prob_w = fold_val_probs[:,   week_idx]
            oof_final_w = np.where(oof_prob_w < 0.5, 0.0, oof_l1_w)
            fold_val_final[:, week_idx] = oof_final_w

            mae_final = float(np.mean(np.abs(oof_final_w - y_val_w)))
            zero_gate_frac = float((oof_final_w == 0.0).mean())
            log(f"    [OOF Gate] HurdleMAE={mae_final:.4f}  "
                f"zero-gated={zero_gate_frac:.2%}  "
                f"(prob<0.5 threshold applied)")

        fold_elapsed = time.time() - fold_t0

        # -- 7g. Per-fold OOF Val MAE breakdown (L1 raw + Hurdle-gated) ------
        week_maes_l1    = [
            float(np.mean(np.abs(fold_val_preds_l1[:, w] - y_val_np[:, w])))
            for w in range(HORIZON)
        ]
        week_maes_final = [
            float(np.mean(np.abs(fold_val_final[:, w] - y_val_np[:, w])))
            for w in range(HORIZON)
        ]
        mean_fold_mae_l1    = float(np.mean(week_maes_l1))
        mean_fold_mae_final = float(np.mean(week_maes_final))

        log(f"\n  [Fold {fold_k}] Week MAEs (Model A raw L1) : "
            + "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes_l1)))
        log(f"  [Fold {fold_k}] Week MAEs (Hurdle-Gated)   : "
            + "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes_final)))
        log(f"  [Fold {fold_k}] Mean Val MAE (raw L1)      : {mean_fold_mae_l1:.4f}")
        log(f"  [Fold {fold_k}] Mean Val MAE (hurdle gate) : {mean_fold_mae_final:.4f}")
        log(f"  [Fold {fold_k}] Elapsed                    : {fold_elapsed:.1f}s")

        # -- 7h. Binned Error Matrix (flatten all weeks for fold-level audit) -
        y_true_all = y_val_np.ravel()
        y_pred_all = fold_val_final.ravel()
        print_binned_error_matrix(y_true_all, y_pred_all, fold_k, log)

        fold_results.append(
            (fold_k, week_maes_l1, week_maes_final, mean_fold_mae_final)
        )

        fold_test_preds_a.append({
            "preds":      fold_test_pred_l1,
            "region_ids": test_region_ids,
        })
        fold_test_preds_b.append({
            "probs":      fold_test_prob,
            "region_ids": test_region_ids,
        })

        del X_train_np, y_train_np, X_val_np, y_val_np
        del fold_val_preds_l1, fold_val_probs, fold_val_final
        gc.collect()

    # -- 8. Cross-fold summary -------------------------------------------------
    log(f"\n{'='*90}")
    log(f"5-Fold Cross-Validation Summary  [v28 Decoupled Dual-Tree Hurdle]")
    log(f"{'='*90}")
    mean_cv_maes_final = []
    for fold_k, week_maes_l1, week_maes_final, mean_mae_final in fold_results:
        final_str = "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes_final))
        log(f"  Fold {fold_k} [Hurdle-Gated]: {final_str}  ->  Mean={mean_mae_final:.4f}")
        mean_cv_maes_final.append(mean_mae_final)

    overall_mean = float(np.mean(mean_cv_maes_final))
    overall_std  = float(np.std(mean_cv_maes_final))
    log(f"\n  Overall CV MAE (Hurdle-Gated) : {overall_mean:.4f}  +-  {overall_std:.4f}")
    log(f"  Best Fold                      : Fold {int(np.argmin(mean_cv_maes_final))} "
        f"(MAE={min(mean_cv_maes_final):.4f})")
    log(f"  Worst Fold                     : Fold {int(np.argmax(mean_cv_maes_final))} "
        f"(MAE={max(mean_cv_maes_final):.4f})")

    log(f"\n  Per-week CV MAE breakdown (Hurdle-Gated):")
    for week_idx in range(HORIZON):
        wk_maes = [fold_results[k][2][week_idx] for k in range(N_FOLDS)]
        log(f"    Week {week_idx + 1}: mean={np.mean(wk_maes):.4f}  "
            f"std={np.std(wk_maes):.4f}")

    # -- 9. Asymmetric Ensemble Blending (Phase 3) -----------------------------
    log(f"\n{'='*90}")
    log(f"[v28] Asymmetric Ensemble Compression  ({N_FOLDS} folds)")
    log(f"  Model A (L1 Regressor)  -> strict np.MEDIAN across {N_FOLDS} folds")
    log(f"       Rationale: median is robust to individual fold outlier fluctuations.")
    log(f"  Model B (Classifier)    -> np.MEAN  across {N_FOLDS} folds")
    log(f"       Rationale: probability averaging stabilizes calibrated thresholds.")
    log(f"  Zero-Gate: final_preds = np.where(prob_mean < 0.5, 0.0, l1_median)")
    log(f"{'='*90}")

    all_region_ids = fold_test_preds_a[0]["region_ids"]
    n_regions      = len(all_region_ids)
    assert n_regions == 2248, f"Expected 2248 test regions, got {n_regions}"

    # Stack Model A predictions: (N_FOLDS, n_regions, HORIZON)
    preds_a_stack = np.stack(
        [fp["preds"] for fp in fold_test_preds_a], axis=0
    )  # (5, 2248, 5)

    # Stack Model B probabilities: (N_FOLDS, n_regions, HORIZON)
    probs_b_stack = np.stack(
        [fp["probs"] for fp in fold_test_preds_b], axis=0
    )  # (5, 2248, 5)

    # Asymmetric compression
    l1_median = np.median(preds_a_stack, axis=0)   # (2248, 5)  -- robust median
    prob_mean  = np.mean(probs_b_stack,  axis=0)   # (2248, 5)  -- calibrated mean prob

    log(f"  preds_a_stack shape : {preds_a_stack.shape}")
    log(f"  probs_b_stack shape : {probs_b_stack.shape}")
    log(f"  l1_median  stats    : mean={l1_median.mean():.4f}  "
        f"std={l1_median.std():.4f}  "
        f"min={l1_median.min():.4f}  max={l1_median.max():.4f}")
    log(f"  prob_mean  stats    : mean={prob_mean.mean():.4f}  "
        f"std={prob_mean.std():.4f}  "
        f"min={prob_mean.min():.4f}  max={prob_mean.max():.4f}")
    log(f"  prob_mean < 0.5 fraction: {(prob_mean < 0.5).mean():.2%}  "
        f"(will be zero-gated)")

    # Evaluate thresholding strictly after the cross-fold blend has stabilized probabilities
    final_preds = np.where(prob_mean < 0.5, 0.0, l1_median)

    log(f"\n  Post-gate pre-clip stats:")
    log(f"    mean={final_preds.mean():.4f}  std={final_preds.std():.4f}  "
        f"min={final_preds.min():.4f}  max={final_preds.max():.4f}")
    log(f"    exact-zero fraction: {(final_preds == 0.0).mean():.2%}")

    # Physical safety boundary restriction [0, 5]
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
    log(f"  zero-fraction (exact 0.0): {zero_frac_final:.4f}  "
        f"(training zero-baseline was {zero_frac:.2%})")
    if p99 < 2.0:
        log("  *** WARNING: p99 < 2.0 -- predictions may be under-dispersed! ***")
    else:
        log("  v p99 >= 2.0 -- prediction diversity is healthy.")
    if abs(zero_frac_final - zero_frac) > 0.15:
        log(f"  *** WARNING: zero-fraction deviation > 15%  "
            f"(pred={zero_frac_final:.2%} vs train={zero_frac:.2%})")
    else:
        log(f"  v Zero-fraction within 15% tolerance of training baseline.")

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

    log_path = os.path.join(ROOT, "_training_log_28th.txt")
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
