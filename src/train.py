"""
train.py -- Drought Score Forecasting Pipeline (v36 -- Temporal Fusion Transformer)
=====================================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv              -- Kaggle submission (2248 rows × 6 cols)
    _training_log_36th.txt      -- Full console log

v36 Architecture (TFT — pytorch_forecasting + PyTorch Lightning)
-----------------------------------------------------------------
  INPUT PIPELINE:
    Raw, un-flattened 13-week sequential meteorological data per region.
    NO manual Z-score normalisation, NO tabular flattening, NO horizontal deltas.
    TimeSeriesDataSet handles windowing + per-encoder-window target normalisation.
    time_idx : week_idx (train 0–781; test encoder 782–794; future decoder 795–799)

  MODEL:
    TemporalFusionTransformer.from_dataset()
      hidden_size             = 64   (LSTM / Gating hidden state capacity)
      attention_head_size     = 4    (multi-head self-attention heads)
      dropout                 = 0.2
      hidden_continuous_size  = 32   (continuous variable embedding size)
      output_size             = 7    (quantiles: 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
      loss                    = QuantileLoss()
      reduce_on_plateau_patience = 4

  TRAINING:
    5-Fold Rolling Time-Based CV  (no group-overlap leakage between folds)
      Fold 0 : train [0, 756]  val [757, 761]
      Fold 1 : train [0, 761]  val [762, 766]
      Fold 2 : train [0, 766]  val [767, 771]
      Fold 3 : train [0, 771]  val [772, 776]
      Fold 4 : train [0, 776]  val [777, 781]
    PyTorch Lightning Trainer
      max_epochs=50, accelerator="gpu", devices=1
      gradient_clip_val=0.1
      EarlyStopping(val_loss, patience=5, min_delta=1e-4)
      LearningRateMonitor(epoch)

  INFERENCE:
    For each fold: tft.predict(test_dataloader, mode="raw")
    p50 channel (index 3 of 7 quantiles) extracted: pred[:, :, 3]
    5-fold p50 arrays stacked and median-compressed:
      final_preds = np.median(all_fold_p50_matrices, axis=0)

  POST-INFERENCE:
    Conservative < 0.10 zero-floor:
      final_preds = np.where(final_preds < 0.10, 0.0, final_preds)
    Physical clip  : np.clip(final_preds, 0.0, 5.0)
    Test Set 6-tier Binned Distribution printed before submission write.

  SUBMISSION:
    2248 rows × 6 columns  |  region_id, pred_week1 … pred_week5

v36 Changes over v35
---------------------
  [INTRODUCE] TemporalFusionTransformer (pytorch_forecasting) — full TFT architecture.
  [INTRODUCE] TimeSeriesDataSet — raw sequential 13-week encoder / 5-week decoder.
  [INTRODUCE] QuantileLoss() (7 quantiles) — pinball objective optimisation.
  [INTRODUCE] PyTorch Lightning Trainer — GPU-accelerated training w/ callbacks.
  [INTRODUCE] 5-Fold Rolling Time-Based CV (no group leak; clean temporal windows).
  [INTRODUCE] p50 (q=0.5, index 3) median channel extraction during inference.
  [INTRODUCE] Asymmetrical 5-fold ensemble: np.median(all_fold_p50, axis=0).
  [INTRODUCE] Conservative < 0.10 zero-floor (replaces probability hurdle gate).

  [ABOLISH]   LGBMRegressor / LGBMClassifier Hurdle architecture.
  [ABOLISH]   354-dimensional tabular flattening matrix.
  [ABOLISH]   Per-region Z-score anomaly normalisation.
  [ABOLISH]   Manual Hurdle gating + per-week threshold calibration.
  [ABOLISH]   StratifiedGroupKFold (replaced by rolling temporal CV).
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# -- Project root on sys.path -------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    WINDOW_SIZE,
    HORIZON,
    N_FOLDS,
    load_data,
    get_fold_boundaries,
    build_training_dataset,
    build_val_dataset,
    build_test_inference_dataset,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE  : int   = 256
NUM_WORKERS : int   = 4
MAX_EPOCHS  : int   = 50
LR          : float = 1e-3

# TFT architecture hyper-parameters
TFT_HIDDEN_SIZE            : int   = 64
TFT_ATTENTION_HEAD_SIZE    : int   = 4
TFT_DROPOUT                : float = 0.2
TFT_HIDDEN_CONTINUOUS_SIZE : int   = 32
TFT_OUTPUT_SIZE            : int   = 7   # 7 quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
TFT_PLATEAU_PATIENCE       : int   = 4

# p50 is index 3 in [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
P50_IDX : int = 3

# Conservative zero-floor
ZERO_FLOOR_THRESHOLD : float = 0.10

# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------
TEST_BIN_LABELS = [
    "Bin 0  [Absolute Zero       preds == 0.0        ]",
    "Bin 1  [Mild Drought        0.0 < preds <= 1.0  ]",
    "Bin 2  [Moderate Drought    1.0 < preds <= 2.0  ]",
    "Bin 3  [Severe Drought      2.0 < preds <= 3.0  ]",
    "Bin 4  [Extreme Drought     3.0 < preds <= 4.0  ]",
    "Bin 5  [Exceptional Drought 4.0 < preds <= 5.0  ]",
]


def _test_bin_mask(preds: np.ndarray, bin_idx: int) -> np.ndarray:
    if bin_idx == 0:
        return preds == 0.0
    elif bin_idx == 1:
        return (preds > 0.0) & (preds <= 1.0)
    elif bin_idx == 2:
        return (preds > 1.0) & (preds <= 2.0)
    elif bin_idx == 3:
        return (preds > 2.0) & (preds <= 3.0)
    elif bin_idx == 4:
        return (preds > 3.0) & (preds <= 4.0)
    elif bin_idx == 5:
        return (preds > 4.0) & (preds <= 5.0)
    else:
        raise ValueError(f"Unknown bin index: {bin_idx}")


def print_test_binned_distribution(final_preds: np.ndarray, log_fn) -> None:
    """
    Print the 6-tier drought intensity bin distribution for the final
    test predictions.  Called immediately before writing submission.csv.
    """
    flat = final_preds.ravel()
    total = len(flat)

    log_fn("")
    log_fn("  " + "=" * 88)
    log_fn("  TEST SET BINNED PREDICTION MATRIX  [v36 TFT p50 Inference Diagnostic]")
    log_fn("  Empirical distribution of median-ensembled, zero-floored, clipped preds")
    log_fn("  " + "=" * 88)
    header = (
        f"  {'Bin / Drought Category':<52}  {'Count':>8}  "
        f"{'% Share':>9}  {'AvgPred':>9}  {'MinPred':>9}  {'MaxPred':>9}"
    )
    log_fn(header)
    log_fn("  " + "-" * 86)

    for idx, label in enumerate(TEST_BIN_LABELS):
        mask  = _test_bin_mask(flat, idx)
        count = int(mask.sum())
        pct   = count / total * 100.0
        if count == 0:
            log_fn(
                f"  {label:<52}  {count:>8,}  {pct:>8.2f}%  "
                f"{'N/A':>9}  {'N/A':>9}  {'N/A':>9}"
            )
        else:
            avg_p = float(flat[mask].mean())
            min_p = float(flat[mask].min())
            max_p = float(flat[mask].max())
            log_fn(
                f"  {label:<52}  {count:>8,}  {pct:>8.2f}%  "
                f"{avg_p:>9.4f}  {min_p:>9.4f}  {max_p:>9.4f}"
            )

    log_fn("  " + "-" * 86)
    log_fn(f"  Total predictions : {total:,}")
    log_fn(f"  Exact-zero fraction: {(flat == 0.0).mean():.4f}  "
           f"({(flat == 0.0).sum():,} of {total:,})")
    log_fn(f"  Non-zero fraction  : {(flat > 0.0).mean():.4f}  "
           f"({(flat > 0.0).sum():,} of {total:,})")
    log_fn(f"  Mean  : {flat.mean():.4f}  Std : {flat.std():.4f}")
    log_fn(f"  p50   : {np.percentile(flat, 50):.4f}  "
           f"p90  : {np.percentile(flat, 90):.4f}  "
           f"p99  : {np.percentile(flat, 99):.4f}  Max : {flat.max():.4f}")
    log_fn("  " + "=" * 88)
    log_fn("")


# ---------------------------------------------------------------------------
# p50 extraction helper
# ---------------------------------------------------------------------------

def extract_p50(raw_predictions, p50_idx: int = P50_IDX) -> np.ndarray:
    """
    Extract the p50 (median) quantile channel from TFT raw predictions.

    pytorch_forecasting 1.x returns a NamedTuple-like object. The quantile
    prediction tensor is accessible at .prediction with shape
    (n_samples, max_prediction_length, output_size).

    Returns
    -------
    p50 : np.ndarray  shape (n_samples, max_prediction_length)
    """
    # Robust extraction — handle both attribute names across PF versions
    if hasattr(raw_predictions, "prediction"):
        pred_tensor = raw_predictions.prediction
    elif hasattr(raw_predictions, "output"):
        pred_tensor = raw_predictions.output
    else:
        # Assume raw tensor was returned directly
        pred_tensor = raw_predictions

    # Move to CPU and convert
    if hasattr(pred_tensor, "cpu"):
        pred_tensor = pred_tensor.cpu()
    return pred_tensor[:, :, p50_idx].numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    t0        = time.time()
    log_lines : list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(str(msg))

    # -- 0. Banner -------------------------------------------------------------
    log("=" * 95)
    log("Drought Forecasting Pipeline  v36  (Temporal Fusion Transformer)")
    log("Model   : TemporalFusionTransformer  (pytorch_forecasting)  + QuantileLoss()")
    log("Input   : Raw 13-week sequential meteorological windows per region")
    log("          NO tabular flattening  |  NO manual Z-score normalisation")
    log("Output  : 7 quantiles [0.02–0.98]  →  p50 (index 3) extracted for submission")
    log("Ensemble: 5-Fold Rolling CV  →  np.median(p50 across folds, axis=0)")
    log("ZeroFloor : preds < 0.10  →  0.0  (conservative micro-noise suppression)")
    log("Clip    : np.clip(0.0, 5.0)  physical constraint")
    log("=" * 95)
    log("")
    log("TFT Architecture:")
    log(f"  hidden_size            = {TFT_HIDDEN_SIZE}")
    log(f"  attention_head_size    = {TFT_ATTENTION_HEAD_SIZE}")
    log(f"  dropout                = {TFT_DROPOUT}")
    log(f"  hidden_continuous_size = {TFT_HIDDEN_CONTINUOUS_SIZE}")
    log(f"  output_size            = {TFT_OUTPUT_SIZE}  "
        f"(quantiles: 0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)")
    log(f"  loss                   = QuantileLoss()")
    log(f"  learning_rate          = {LR}")
    log(f"  max_epochs             = {MAX_EPOCHS}")
    log(f"  batch_size             = {BATCH_SIZE}")
    log("")
    log("5-Fold Rolling Time-Based CV Boundaries:")
    for k in range(N_FOLDS):
        te, vs, ve = get_fold_boundaries(k)
        log(f"  Fold {k}: train time_idx [0, {te}]  →  val time_idx [{vs}, {ve}]")
    log("")

    # -- 1. Load data ----------------------------------------------------------
    log("Loading processed data ...")
    train_df, test_df = load_data()
    log(f"  train_df : {train_df.shape}  |  test_df : {test_df.shape}")
    log(f"  train regions : {train_df['region_id'].nunique():,}")
    log(f"  test  regions : {test_df['region_id'].nunique():,}")
    assert train_df["region_id"].nunique() == 2248, "Expected 2248 train regions"
    assert test_df["region_id"].nunique()  == 2248, "Expected 2248 test regions"

    # Target distribution summary
    scores = train_df["score"].values
    log(f"\n[Training Target Distribution]")
    log(f"  mean={scores.mean():.4f}  std={scores.std():.4f}  "
        f"min={scores.min():.4f}  max={scores.max():.4f}")
    log(f"  Zero-fraction : {(scores == 0.0).mean():.4f}  "
        f"({(scores == 0.0).sum():,} rows)")
    log(f"  Non-zero      : {(scores > 0.0).mean():.4f}  "
        f"({(scores > 0.0).sum():,} rows)")

    # -- 2. 5-Fold Rolling TFT Training Loop -----------------------------------
    log(f"\n{'='*95}")
    log("5-Fold Rolling Time-Based TFT Training")
    log(f"  Each fold trains a separate TFT and predicts the test set.")
    log(f"  p50 predictions are collected and median-ensembled post-training.")
    log(f"{'='*95}")

    all_fold_p50_matrices : list[np.ndarray] = []   # each (2248, 5)
    fold_val_losses       : list[float]      = []
    test_dataset          = None    # built once in fold 0 and reused
    test_ordered_regions  : list[str] = []

    for fold_k in range(N_FOLDS):

        fold_t0 = time.time()
        train_end, val_start, val_end = get_fold_boundaries(fold_k)

        log(f"\n{'='*95}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}  [TFT Rolling CV]")
        log(f"  train time_idx : [0, {train_end}]  "
            f"(= {train_end + 1} weeks × 2248 regions)")
        log(f"  val   time_idx : [{val_start}, {val_end}]  "
            f"(= {val_end - val_start + 1} weeks × 2248 regions)")
        log(f"{'='*95}")

        # -- 2a. Build datasets for this fold ----------------------------------
        fold_train_df = train_df[train_df["time_idx"] <= train_end].copy()
        fold_ctx_df   = train_df[train_df["time_idx"] <= val_end  ].copy()

        log(f"\n  Building TimeSeriesDataSet (training fold {fold_k}) ...")
        training_dataset = build_training_dataset(fold_train_df)

        log(f"  Building TimeSeriesDataSet (validation fold {fold_k}) ...")
        val_dataset = build_val_dataset(training_dataset, fold_ctx_df, val_start)

        log(f"  training dataset size      : {len(training_dataset):,} samples")
        log(f"  validation dataset size    : {len(val_dataset):,} samples")

        # -- 2b. Build test inference dataset once (uses fold 0's encoders) ---
        if fold_k == 0:
            log(f"\n  Building test inference TimeSeriesDataSet (fold 0 encoders) ...")
            test_dataset, test_ordered_regions = build_test_inference_dataset(
                training_dataset, test_df
            )
            log(f"  test inference dataset size: {len(test_dataset):,} samples "
                f"(= 2248 regions × 1 window)")

        # -- 2c. DataLoaders ---------------------------------------------------
        train_dataloader = training_dataset.to_dataloader(
            train      = True,
            batch_size = BATCH_SIZE,
            num_workers= NUM_WORKERS,
        )
        val_dataloader = val_dataset.to_dataloader(
            train       = False,
            batch_size  = BATCH_SIZE,
            num_workers = NUM_WORKERS,
        )
        test_dataloader = test_dataset.to_dataloader(
            train       = False,
            batch_size  = BATCH_SIZE,
            num_workers = NUM_WORKERS,
        )

        # -- 2d. Initialise TFT model ------------------------------------------
        log(f"\n  Initialising TemporalFusionTransformer (fold {fold_k}) ...")
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate             = LR,
            hidden_size               = TFT_HIDDEN_SIZE,
            attention_head_size       = TFT_ATTENTION_HEAD_SIZE,
            dropout                   = TFT_DROPOUT,
            hidden_continuous_size    = TFT_HIDDEN_CONTINUOUS_SIZE,
            output_size               = TFT_OUTPUT_SIZE,
            loss                      = QuantileLoss(),
            reduce_on_plateau_patience= TFT_PLATEAU_PATIENCE,
            log_interval              = 10,
            log_val_interval          = 1,
        )
        log(f"  Parameter count: {sum(p.numel() for p in tft.parameters()):,}")

        # -- 2e. Lightning Trainer ---------------------------------------------
        callbacks = [
            EarlyStopping(
                monitor   = "val_loss",
                patience  = 5,
                min_delta = 1e-4,
                mode      = "min",
                verbose   = True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        trainer = pl.Trainer(
            max_epochs       = MAX_EPOCHS,
            accelerator      = "gpu",
            devices          = 1,
            gradient_clip_val= 0.1,
            callbacks        = callbacks,
            enable_progress_bar = True,
            logger           = True,    # default TensorBoard logger
        )

        log(f"\n  Fitting TFT (fold {fold_k})  ...  "
            f"[max_epochs={MAX_EPOCHS}, early_stop patience=5]")
        trainer.fit(
            tft,
            train_dataloaders = train_dataloader,
            val_dataloaders   = val_dataloader,
        )

        # Capture best validation loss
        best_val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
        if hasattr(best_val_loss, "item"):
            best_val_loss = float(best_val_loss.item())
        else:
            best_val_loss = float(best_val_loss)
        fold_val_losses.append(best_val_loss)

        stopped_epoch = trainer.current_epoch
        log(f"\n  [Fold {fold_k}] Training complete.")
        log(f"  Stopped at epoch : {stopped_epoch}")
        log(f"  Best val_loss    : {best_val_loss:.6f}")

        # -- 2f. Inference: extract p50 ----------------------------------------
        log(f"\n  Running test inference (fold {fold_k}) ...")
        tft.eval()

        # return_x=False: we use test_ordered_regions for region mapping —
        # no need to unpack group index tensors.
        # In pytorch_forecasting 1.x, predict() returns a NamedTuple-like
        # structure; capture the full result and let extract_p50() handle it.
        pred_result = tft.predict(
            test_dataloader,
            mode     = "raw",
            return_x = False,
        )

        # Extract p50 quantile channel  (index 3 in 7-quantile output)
        p50_fold = extract_p50(pred_result, p50_idx=P50_IDX)
        # p50_fold shape: (n_samples, HORIZON) = (2248, 5)

        log(f"  p50_fold shape         : {p50_fold.shape}")
        log(f"  p50_fold stats  mean={p50_fold.mean():.4f}  "
            f"std={p50_fold.std():.4f}  "
            f"min={p50_fold.min():.4f}  max={p50_fold.max():.4f}")

        all_fold_p50_matrices.append(p50_fold)

        fold_elapsed = time.time() - fold_t0
        log(f"  Fold {fold_k} elapsed : {fold_elapsed:.1f}s  ({fold_elapsed/60:.1f} min)")

    # -- 3. Cross-fold summary -------------------------------------------------
    log(f"\n{'='*95}")
    log(f"5-Fold Rolling CV Summary  [v36 TFT QuantileLoss]")
    log(f"{'='*95}")
    for k, vl in enumerate(fold_val_losses):
        log(f"  Fold {k}: best val_loss = {vl:.6f}")
    finite_losses = [v for v in fold_val_losses if not np.isnan(v)]
    if finite_losses:
        log(f"\n  Mean val_loss : {np.mean(finite_losses):.6f}  "
            f"±  {np.std(finite_losses):.6f}")
        log(f"  Best fold     : {int(np.argmin(fold_val_losses))} "
            f"(val_loss = {min(finite_losses):.6f})")

    # -- 4. Asymmetrical ensemble compression ----------------------------------
    log(f"\n{'='*95}")
    log(f"[v36] 5-Fold p50 Ensemble Compression")
    log(f"  Strategy: np.median(all_fold_p50_matrices, axis=0)")
    log(f"{'='*95}")

    # Stack to (N_FOLDS, 2248, 5) then collapse via strict median
    p50_stack   = np.stack(all_fold_p50_matrices, axis=0)   # (5, 2248, 5)
    final_preds = np.median(p50_stack, axis=0)               # (2248, 5)

    log(f"  p50_stack shape        : {p50_stack.shape}")
    log(f"  final_preds shape      : {final_preds.shape}")
    log(f"  Pre-floor stats: mean={final_preds.mean():.4f}  "
        f"std={final_preds.std():.4f}  "
        f"min={final_preds.min():.4f}  max={final_preds.max():.4f}")

    # -- 5. Conservative zero-floor (<0.10) ------------------------------------
    log(f"\n[v36] Conservative Zero-Floor  (threshold < {ZERO_FLOOR_THRESHOLD})")
    before_zero_frac = float((final_preds == 0.0).mean())
    final_preds = np.where(final_preds < ZERO_FLOOR_THRESHOLD, 0.0, final_preds)
    after_zero_frac  = float((final_preds == 0.0).mean())

    log(f"  Zero fraction before floor : {before_zero_frac:.4f}")
    log(f"  Zero fraction after  floor : {after_zero_frac:.4f}")
    log(f"  Micro-noise eliminated     : {after_zero_frac - before_zero_frac:.4f}")

    # -- 6. Physical clip [0, 5] -----------------------------------------------
    final_submission = np.clip(final_preds, 0.0, 5.0).astype(np.float32)

    log(f"\n  Post-clip stats [0, 5]:")
    log(f"    mean={final_submission.mean():.4f}  "
        f"std={final_submission.std():.4f}  "
        f"min={final_submission.min():.4f}  "
        f"max={final_submission.max():.4f}")

    # -- 7. Test Set Binned Distribution Audit ---------------------------------
    log(f"\n{'='*95}")
    log(f"[v36] Test Set 6-Tier Binned Distribution Audit")
    log(f"{'='*95}")
    print_test_binned_distribution(final_submission, log)

    # -- 8. Submission diagnostics  -------------------------------------------
    log("[v36 Submission Prediction Diagnostics]")
    flat    = final_submission.ravel()
    p50_v   = float(np.percentile(flat, 50))
    p75_v   = float(np.percentile(flat, 75))
    p90_v   = float(np.percentile(flat, 90))
    p95_v   = float(np.percentile(flat, 95))
    p99_v   = float(np.percentile(flat, 99))
    log(f"  n={len(flat):,}  mean={flat.mean():.4f}  std={flat.std():.4f}")
    log(f"  p50={p50_v:.4f}  p75={p75_v:.4f}  p90={p90_v:.4f}  "
        f"p95={p95_v:.4f}  p99={p99_v:.4f}  max={flat.max():.4f}")
    log(f"  Exact zero fraction: {(flat == 0.0).mean():.4f}")

    if p99_v < 1.5:
        log("  *** WARNING: p99 < 1.5 -- predictions may be under-dispersed! ***")
    else:
        log(f"  ✓ p99 >= 1.5 -- prediction dynamic range is healthy.")

    # -- 9. Build submission DataFrame -----------------------------------------
    log("\nFormatting submission.csv ...")

    # Map prediction rows back to region_ids.
    # test_ordered_regions was built by sorted(unique region_ids in inference_df).
    # The test_dataloader iterates through samples in the order they were created
    # by TimeSeriesDataSet, which follows the sorted region + time_idx order of
    # the inference dataframe (one sample per region).
    assert len(test_ordered_regions) == final_submission.shape[0], (
        f"Region count mismatch: {len(test_ordered_regions)} regions vs "
        f"{final_submission.shape[0]} prediction rows."
    )
    assert final_submission.shape == (2248, HORIZON), (
        f"Expected final_submission shape (2248, {HORIZON}), "
        f"got {final_submission.shape}"
    )

    rows = []
    for i, region_id in enumerate(test_ordered_regions):
        p = final_submission[i]
        rows.append({
            "region_id" : region_id,
            "pred_week1": float(p[0]),
            "pred_week2": float(p[1]),
            "pred_week3": float(p[2]),
            "pred_week4": float(p[3]),
            "pred_week5": float(p[4]),
        })

    submission = pd.DataFrame(rows)
    sub_path   = os.path.join(ROOT, "submission.csv")
    submission.to_csv(sub_path, index=False)

    # -- 10. Sanity checks -----------------------------------------------------
    assert len(submission) == 2248, f"Expected 2248 rows, got {len(submission)}"
    assert list(submission.columns) == [
        "region_id", "pred_week1", "pred_week2",
        "pred_week3", "pred_week4", "pred_week5",
    ], f"Unexpected columns: {list(submission.columns)}"
    log("  ✓ Submission assertion passed: 2248 rows, 6 columns.")

    assert not submission.isnull().any().any(), "NaN values found in submission!"
    log("  ✓ No NaN values in submission.")

    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    assert submission[pred_cols].max().max() <= 5.0 + 1e-6
    assert submission[pred_cols].min().min() >= 0.0 - 1e-6
    log("  ✓ All predictions in [0, 5]  (np.clip physical guard enforced).")

    log(f"\n  submission.csv  →  {sub_path}")
    log(f"  Rows (excl. header): {len(submission)}")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    # -- 11. Total elapsed time -----------------------------------------------
    elapsed = time.time() - t0
    log(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    log_path = os.path.join(ROOT, "_training_log_36th.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines))
    print(f"\nTraining log saved  →  {log_path}")

    return {
        "fold_val_losses"   : fold_val_losses,
        "p50_stack_shape"   : p50_stack.shape,
        "final_preds_shape" : final_submission.shape,
        "submission"        : submission,
        "sub_max"           : float(flat.max()),
        "sub_p99"           : p99_v,
        "zero_frac_final"   : float((flat == 0.0).mean()),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
