"""
v30_soft_inference.py -- Training-Free Soft Hurdle Inference (v30)
===================================================================
Usage:
    python src/v30_soft_inference.py

Outputs:
    submission_v30.csv  -- Kaggle submission (2248 rows x 6 cols)

v30 Post-Processing Overhaul: Soft Hurdle (Expected Value Blending)
--------------------------------------------------------------------
  This script is TRAINING-FREE. It loads the 50 pre-trained v29 LightGBM
  checkpoint files (5 folds x 5 weeks x 2 heads = 50 .pkl files) from the
  models/ directory and runs inference exclusively on the test feature matrix.

  KEY CHANGES vs v29:
  ─────────────────────────────────────────────────────────────────────────
  [ABOLISH HARD THRESHOLD]
    v29 used a per-week OOF-optimized hard gate:
        final = np.where(prob < best_threshold[w], 0.0, l1_pred)
    v30 COMPLETELY REMOVES this zero-gate logic.

  [SOFT HURDLE  (Expected Value Blending)]
    Instead of a hard binary cutoff, we compute the expected severity via
    continuous multiplication of the probability and severity channels:
        soft_pred = prob_mean * l1_median
    Rationale: This preserves gradient information near the drought boundary
    and smoothly penalizes low-confidence severity estimates without forcing
    a discontinuous jump to zero.

  [MILD DROUGHT CALIBRATION CORRECTION  (< 1.5 → x0.85)]
    Observations in the mild drought region (0 < severity < 1.5) are
    systematically over-predicted by the v29 dual-head model.  A 0.85
    deflation anchor is applied to pull these fractional noise predictions
    smoothly back toward the zero baseline:
        calibrated = np.where(soft_pred < 1.5, soft_pred * 0.85, soft_pred)

  [FINAL CLIP]
    Physical boundary guard: np.clip(calibrated, 0.0, 5.0)

  ENSEMBLE RULE (retained from v29):
    Model A (L1 Regressor)  → np.median across 5 folds  (robust compression)
    Model B (Binary Classifier) → np.mean  across 5 folds  (probability calibration)

  OUTPUT FORMAT:
    submission_v30.csv: 2248 rows × 6 cols
    [region_id, pred_week1, pred_week2, pred_week3, pred_week4, pred_week5]

Model Checkpoint Naming Convention (v29):
    models/lgbm_a_fold{k}_week{w}.pkl   -- Model A (L1 Regressor), k=0-4, w=0-4
    models/lgbm_b_fold{k}_week{w}.pkl   -- Model B (Binary Classifier), k=0-4, w=0-4
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path (mirrors train.py convention)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_tabular_test,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")
OUTPUT_PATH   = os.path.join(ROOT, "submission_v30.csv")

N_FOLDS         = 5
N_FLAT_FEATURES = WINDOW_SIZE * len(FEATURE_COLS) + len(FEATURE_COLS)  # 378

# Soft Hurdle calibration constants
MILD_DROUGHT_CEILING    = 1.5   # Predictions below this threshold are dampened
MILD_DROUGHT_SCALE      = 0.85  # Deflation anchor for mild drought region
CLIP_MIN                = 0.0
CLIP_MAX                = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    """Simple dual print + flush helper."""
    print(msg, flush=True)


def load_model(path: str):
    """Deserialize a pickled LightGBM model from disk."""
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    return model


def _zero_prob(x):
    return (x == 0.0).mean()


def _compute_global_te_stats(train_df: pd.DataFrame) -> tuple:
    """
    Compute global Target Encoding statistics from the full training set.
    Mirrors the _compute_te_stats / _merge_te_to_df logic in train.py, but
    uses ALL training regions (no fold split) to produce a single stable
    global TE map for test-time inference.

    Returns
    -------
    te_map          : dict  {region_id -> (mean_score, zero_prob)}
    global_mean     : float  fallback mean for unseen regions
    global_zero_prob: float  fallback zero-prob for unseen regions
    """
    te_stats = (
        train_df.groupby("region_id")["score"]
        .agg(region_mean_score="mean", region_zero_prob=_zero_prob)
        .reset_index()
    )
    global_mean       = float(te_stats["region_mean_score"].mean())
    global_zero_prob  = float(te_stats["region_zero_prob"].mean())
    te_map = {
        row["region_id"]: (float(row["region_mean_score"]),
                           float(row["region_zero_prob"]))
        for _, row in te_stats.iterrows()
    }
    return te_map, global_mean, global_zero_prob


def _merge_te_to_df(df: pd.DataFrame, te_map: dict,
                    global_mean: float, global_zero_prob: float) -> pd.DataFrame:
    """
    Inject region_mean_score and region_zero_prob columns into df.
    Mirrors the identically named function in train.py.
    """
    df = df.copy()
    df["region_mean_score"] = df["region_id"].map(
        lambda rid: te_map.get(rid, (global_mean, global_zero_prob))[0]
    ).astype(np.float32)
    df["region_zero_prob"] = df["region_id"].map(
        lambda rid: te_map.get(rid, (global_mean, global_zero_prob))[1]
    ).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    # ── 0. Pipeline banner ────────────────────────────────────────────────
    log("=" * 90)
    log("Drought Forecasting Inference  v30  (Training-Free Soft Hurdle)")
    log("NO TRAINING -- Loads 50 pre-trained v29 LightGBM checkpoints from models/")
    log("=" * 90)
    log("")
    log("v30 Post-Processing Changes over v29:")
    log("  [ABOLISH HARD THRESHOLD]  The per-week OOF dynamic zero-gate is REMOVED.")
    log("  [SOFT HURDLE]             soft_pred = prob_mean * l1_median")
    log("                            (Expected severity = P(drought) x Conditional severity)")
    log("  [MILD DROUGHT CALIBRATION] calibrated = np.where(soft_pred < 1.5,")
    log("                                               soft_pred * 0.85, soft_pred)")
    log("  [CLIP]                    np.clip(calibrated, 0.0, 5.0)")
    log("")
    log(f"  Ensemble rule (retained from v29):")
    log(f"    Model A (L1 Regressor)      -> np.MEDIAN across {N_FOLDS} folds")
    log(f"    Model B (Binary Classifier) -> np.MEAN  across {N_FOLDS} folds")
    log(f"  N_FOLDS           : {N_FOLDS}")
    log(f"  HORIZON (weeks)   : {HORIZON}")
    log(f"  N_FLAT_FEATURES   : {N_FLAT_FEATURES}  (378 = 27×13 flat + 27 deltas)")
    log(f"  CHECKPOINT DIR    : {MODELS_DIR}")
    log(f"  OUTPUT PATH       : {OUTPUT_PATH}")
    log("")

    # ── 1. Verify checkpoint availability ────────────────────────────────
    log("[Step 1] Verifying model checkpoint availability ...")
    missing = []
    for fold_k in range(N_FOLDS):
        for week_idx in range(HORIZON):
            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            ckpt_b = os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl")
            if not os.path.exists(ckpt_a):
                missing.append(ckpt_a)
            if not os.path.exists(ckpt_b):
                missing.append(ckpt_b)

    if missing:
        log(f"  *** ERROR: {len(missing)} checkpoint(s) not found:")
        for m in missing:
            log(f"      - {m}")
        raise FileNotFoundError(
            f"{len(missing)} required v29 model checkpoints are missing from {MODELS_DIR}.\n"
            "  --> Ensure the v29 training run completed successfully (train.py)."
        )
    log(f"  ✓ All {N_FOLDS * HORIZON * 2} checkpoints present "
        f"({N_FOLDS} folds × {HORIZON} weeks × 2 heads).")

    # ── 2. Load & preprocess test data ───────────────────────────────────
    log("\n[Step 2] Loading test feature matrix ...")
    test_csv  = os.path.join(PROCESSED_DIR, "test_processed.csv")
    train_csv = os.path.join(PROCESSED_DIR, "train_processed.csv")

    for csv_path in (test_csv, train_csv):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Required CSV not found: {csv_path}\n"
                "  --> Run `python src/preprocess.py` to generate processed CSVs."
            )

    test_raw  = pd.read_csv(test_csv)
    train_raw = pd.read_csv(train_csv)
    log(f"  Raw test shape      : {test_raw.shape}")
    log(f"  Raw train shape     : {train_raw.shape}  (used for TE stats only)")

    n_test_regions = test_raw["region_id"].nunique()
    log(f"  Unique test regions : {n_test_regions}")
    assert n_test_regions == 2248, (
        f"Expected 2248 test regions, got {n_test_regions}"
    )

    # Feature refinement (mirrors train.py: drought proxy index + log1p prec + pruning)
    log("\n  Applying feature refinement (drought proxy index, log1p prec, v22/v26 pruning) ...")
    test_df  = refine_features(test_raw,  is_train=False)
    train_df = refine_features(train_raw, is_train=True)
    log(f"  Refined test shape  : {test_df.shape}")
    log(f"  Refined train shape : {train_df.shape}")

    # Drop NaN-score rows from train (mirrors train.py Step 3) before computing TE
    before = len(train_df)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    dropped = before - len(train_df)
    if dropped:
        log(f"  [NaN drop] Removed {dropped:,} rows with NaN score from train before TE.")

    # ── 2a. Compute global Target Encoding stats from full training set ──
    log("\n  Computing global Target Encoding stats from training set ...")
    log("  (region_mean_score & region_zero_prob -- required by v29 models as 2 of 27 features)")
    te_map, global_mean, global_zero_prob = _compute_global_te_stats(train_df)
    log(f"  TE map size          : {len(te_map):,} regions")
    log(f"  global_mean_score    : {global_mean:.4f}")
    log(f"  global_zero_prob     : {global_zero_prob:.4f}")

    # ── 2b. Inject TE columns into test_df ───────────────────────────────
    log("\n  Injecting TE columns (region_mean_score, region_zero_prob) into test_df ...")
    test_df_te = _merge_te_to_df(test_df, te_map, global_mean, global_zero_prob)
    log(f"  test_df with TE shape: {test_df_te.shape}")

    # Determine which FEATURE_COLS are actually present after TE injection
    available_feat_cols = [c for c in FEATURE_COLS if c in test_df_te.columns]
    log(f"  Feature cols available : {len(available_feat_cols)} / {len(FEATURE_COLS)}")

    if len(available_feat_cols) != len(FEATURE_COLS):
        missing_feats = [c for c in FEATURE_COLS if c not in test_df_te.columns]
        log(f"  *** WARNING: {len(missing_feats)} FEATURE_COLS still absent after TE injection:")
        for mf in missing_feats:
            log(f"      - {mf}")

    expected_dim = WINDOW_SIZE * len(available_feat_cols) + len(available_feat_cols)
    log(f"  Expected tabular dim   : {expected_dim}  "
        f"(WINDOW_SIZE={WINDOW_SIZE} × {len(available_feat_cols)} + "
        f"{len(available_feat_cols)} deltas)")

    # Build flat tabular test matrix with TE-augmented test_df
    X_test_np, test_region_ids = build_tabular_test(test_df_te, available_feat_cols)
    X_test_np = np.asfortranarray(X_test_np)

    log(f"  X_test shape           : {X_test_np.shape}")
    assert X_test_np.shape[0] == 2248, (
        f"Expected 2248 test rows, got {X_test_np.shape[0]}"
    )
    assert X_test_np.shape[1] == expected_dim, (
        f"Unexpected X_test feature dim: {X_test_np.shape[1]}, "
        f"expected {expected_dim}  (check FEATURE_COLS vs available columns)"
    )
    log(f"  ✓ Test matrix validated: 2248 rows × {expected_dim} features")

    # ── 3. Run inference across all 50 checkpoints ───────────────────────
    log(f"\n[Step 3] Generating fold predictions (5 folds × {HORIZON} weeks × 2 heads) ...")
    log(f"  Ensemble strategy:")
    log(f"    Model A (L1 Regressor)      : strict np.MEDIAN  (robust to outlier folds)")
    log(f"    Model B (Binary Classifier) : arithmetic np.MEAN (probability calibration)")

    # Accumulators: shape (N_FOLDS, n_regions, HORIZON)
    all_model_a_preds = np.zeros((N_FOLDS, 2248, HORIZON), dtype=np.float32)
    all_model_b_probs = np.zeros((N_FOLDS, 2248, HORIZON), dtype=np.float32)

    for fold_k in range(N_FOLDS):
        log(f"\n  Loading fold {fold_k} checkpoints ...")
        for week_idx in range(HORIZON):
            ckpt_a = os.path.join(MODELS_DIR, f"lgbm_a_fold{fold_k}_week{week_idx}.pkl")
            ckpt_b = os.path.join(MODELS_DIR, f"lgbm_b_fold{fold_k}_week{week_idx}.pkl")

            # ── Model A: L1 Regressor → raw continuous severity ──────────
            model_a = load_model(ckpt_a)
            pred_a  = model_a.predict(X_test_np).astype(np.float32)  # (2248,)
            all_model_a_preds[fold_k, :, week_idx] = pred_a
            del model_a

            # ── Model B: Binary Classifier → drought probability ─────────
            model_b = load_model(ckpt_b)
            prob_b  = model_b.predict_proba(X_test_np)[:, 1].astype(np.float32)  # (2248,)
            all_model_b_probs[fold_k, :, week_idx] = prob_b
            del model_b

            log(f"    fold={fold_k}  week={week_idx + 1}  "
                f"ModelA_mean={pred_a.mean():.4f}  "
                f"ModelB_prob_mean={prob_b.mean():.4f}  [LOADED]")

    log(f"\n  ✓ All {N_FOLDS * HORIZON * 2} models loaded and predictions generated.")

    # ── 4. Asymmetric ensemble compression ───────────────────────────────
    log(f"\n[Step 4] Asymmetric ensemble compression ...")

    # Model B → mean probability (calibration-stable)
    prob_mean = np.mean(all_model_b_probs, axis=0)   # (2248, 5)

    # Model A → median severity prediction (robust L1 compression)
    l1_median = np.median(all_model_a_preds, axis=0)  # (2248, 5)

    log(f"  prob_mean  stats  : mean={prob_mean.mean():.4f}  "
        f"std={prob_mean.std():.4f}  "
        f"min={prob_mean.min():.4f}  max={prob_mean.max():.4f}")
    log(f"  l1_median  stats  : mean={l1_median.mean():.4f}  "
        f"std={l1_median.std():.4f}  "
        f"min={l1_median.min():.4f}  max={l1_median.max():.4f}")

    # ── 5. Soft Hurdle: Expected Value = Probability × Severity ──────────
    log(f"\n[Step 5] Applying Soft Hurdle (Expected Value Blending) ...")
    log(f"  Formula: soft_pred = prob_mean * l1_median")
    log(f"  (NO hard threshold gate -- continuous probability-weighted severity)")

    # Soft Hurdle: Expected Value = Probability x Continuous Severity Scale
    soft_pred = prob_mean * l1_median  # (2248, 5)

    log(f"  soft_pred  stats  : mean={soft_pred.mean():.4f}  "
        f"std={soft_pred.std():.4f}  "
        f"min={soft_pred.min():.4f}  max={soft_pred.max():.4f}")

    for week_idx in range(HORIZON):
        log(f"    Week {week_idx + 1}: mean={soft_pred[:, week_idx].mean():.4f}  "
            f"std={soft_pred[:, week_idx].std():.4f}  "
            f"max={soft_pred[:, week_idx].max():.4f}")

    # ── 6. Mild Drought Calibration Correction ────────────────────────────
    log(f"\n[Step 6] Applying Mild Drought Calibration Correction ...")
    log(f"  Rule: soft_pred < {MILD_DROUGHT_CEILING} → ×{MILD_DROUGHT_SCALE}  "
        f"(deflation anchor for over-predicted fractional drought region)")
    log(f"  Goal: Dampen fractional noise in the low-drought region (< {MILD_DROUGHT_CEILING}) "
        f"and pull false-positive scores smoothly toward zero baseline.")

    # If the soft expected value falls under 1.5, apply a 0.85 scaling deflation anchor
    calibrated_pred = np.where(soft_pred < MILD_DROUGHT_CEILING,
                                soft_pred * MILD_DROUGHT_SCALE,
                                soft_pred)  # (2248, 5)

    mild_frac = float((soft_pred < MILD_DROUGHT_CEILING).mean())
    log(f"  Fraction of predictions scaled (soft_pred < {MILD_DROUGHT_CEILING}): "
        f"{mild_frac:.2%}")
    log(f"  calibrated stats  : mean={calibrated_pred.mean():.4f}  "
        f"std={calibrated_pred.std():.4f}  "
        f"min={calibrated_pred.min():.4f}  max={calibrated_pred.max():.4f}")
    log(f"  Mean shift (soft → calibrated): "
        f"{(calibrated_pred.mean() - soft_pred.mean()):.6f}")

    # ── 7. Final boundary clip ────────────────────────────────────────────
    log(f"\n[Step 7] Applying final boundary clip [{CLIP_MIN}, {CLIP_MAX}] ...")

    final_submission = np.clip(calibrated_pred, CLIP_MIN, CLIP_MAX)  # (2248, 5)

    log(f"  final_submission stats : mean={final_submission.mean():.4f}  "
        f"std={final_submission.std():.4f}  "
        f"min={final_submission.min():.4f}  max={final_submission.max():.4f}")

    all_preds_flat  = final_submission.ravel()
    zero_frac_final = float((all_preds_flat == 0.0).mean())
    p50  = float(np.percentile(all_preds_flat, 50))
    p75  = float(np.percentile(all_preds_flat, 75))
    p90  = float(np.percentile(all_preds_flat, 90))
    p95  = float(np.percentile(all_preds_flat, 95))
    p99  = float(np.percentile(all_preds_flat, 99))
    pmax = float(np.max(all_preds_flat))

    log(f"\n[Submission Prediction Diagnostics]")
    log(f"  n={len(all_preds_flat):,}  "
        f"mean={all_preds_flat.mean():.4f}  std={all_preds_flat.std():.4f}")
    log(f"  p50={p50:.4f}  p75={p75:.4f}  p90={p90:.4f}  "
        f"p95={p95:.4f}  p99={p99:.4f}  max={pmax:.4f}")
    log(f"  zero-fraction (exact 0.0): {zero_frac_final:.4f}")

    if p99 < 2.0:
        log("  *** WARNING: p99 < 2.0 -- predictions may be under-dispersed! ***")
    else:
        log(f"  ✓ p99 >= 2.0 -- prediction diversity is healthy.")

    for week_idx in range(HORIZON):
        wk = final_submission[:, week_idx]
        log(f"  Week {week_idx + 1} : mean={wk.mean():.4f}  std={wk.std():.4f}  "
            f"max={wk.max():.4f}  zero_frac={float((wk == 0.0).mean()):.2%}")

    # ── 8. Format & export submission_v30.csv ────────────────────────────
    log(f"\n[Step 8] Formatting and exporting {OUTPUT_PATH} ...")

    rows = []
    for i, region_id in enumerate(test_region_ids):
        preds = final_submission[i]
        rows.append({
            "region_id":  region_id,
            "pred_week1": float(preds[0]),
            "pred_week2": float(preds[1]),
            "pred_week3": float(preds[2]),
            "pred_week4": float(preds[3]),
            "pred_week5": float(preds[4]),
        })

    submission = pd.DataFrame(rows)
    submission.to_csv(OUTPUT_PATH, index=False)

    # ── 9. Sanity checks ─────────────────────────────────────────────────
    log("\n[Step 9] Sanity checks ...")

    assert len(submission) == 2248, (
        f"Expected 2248 rows, got {len(submission)}"
    )
    log("  ✓ Row count: 2248")

    expected_cols = ["region_id", "pred_week1", "pred_week2",
                     "pred_week3", "pred_week4", "pred_week5"]
    assert list(submission.columns) == expected_cols, (
        f"Unexpected columns: {list(submission.columns)}"
    )
    log(f"  ✓ Columns: {list(submission.columns)}")

    assert not submission.isnull().any().any(), (
        "NaN values found in submission!"
    )
    log("  ✓ No NaN values in submission.")

    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    assert submission[pred_cols].max().max() <= CLIP_MAX + 1e-6, (
        f"Values exceed {CLIP_MAX}!"
    )
    assert submission[pred_cols].min().min() >= CLIP_MIN - 1e-6, (
        f"Values below {CLIP_MIN}!"
    )
    log(f"  ✓ All predictions in [{CLIP_MIN}, {CLIP_MAX}] (np.clip physical guard enforced).")

    # ── 10. Final summary ────────────────────────────────────────────────
    elapsed = time.time() - t0

    log("")
    log("=" * 90)
    log("v30 Soft Hurdle Inference  --  COMPLETE")
    log("=" * 90)
    log(f"  Output             : {OUTPUT_PATH}")
    log(f"  Rows               : {len(submission)}  (Kaggle eval: 2248 regions)")
    log(f"  Columns            : {list(submission.columns)}")
    log(f"  Total elapsed      : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    log("")
    log("  Post-Processing Summary:")
    log(f"    [1] Loaded 50 v29 checkpoints (5 folds × {HORIZON} weeks × 2 heads)")
    log(f"    [2] Model A ensemble : np.MEDIAN across {N_FOLDS} folds  (L1 severity)")
    log(f"    [3] Model B ensemble : np.MEAN  across {N_FOLDS} folds  (drought prob)")
    log(f"    [4] Soft Hurdle      : soft_pred = prob_mean × l1_median")
    log(f"    [5] Mild Calibration : np.where(soft_pred < {MILD_DROUGHT_CEILING}, "
        f"soft_pred × {MILD_DROUGHT_SCALE}, soft_pred)")
    log(f"    [6] Clip             : np.clip(calibrated, {CLIP_MIN}, {CLIP_MAX})")
    log("")
    log(f"  Preview:")
    log(submission.head(5).to_string(index=False))
    log("")

    return {
        "submission_path":   OUTPUT_PATH,
        "n_rows":            len(submission),
        "mean_pred":         float(all_preds_flat.mean()),
        "std_pred":          float(all_preds_flat.std()),
        "zero_frac":         zero_frac_final,
        "p99":               p99,
        "elapsed_s":         elapsed,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
