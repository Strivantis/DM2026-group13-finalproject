# ---------------------------------------------------------------------------
# [v37.1] Hardware Acceleration — Tensor Core float32 matmul precision
# Must be set BEFORE any other torch operations or module imports.
# ---------------------------------------------------------------------------
import torch
torch.set_float32_matmul_precision('high')

"""
train.py -- Drought Score Forecasting Pipeline (v37 -- CORAL-TFT Ordinal CDF)
==============================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv              -- Kaggle submission (2248 rows × 6 cols)
    _training_log_37th.txt      -- Full console log

v37 Architecture (CORAL-TFT — Ordinal CDF Paradigm)
-----------------------------------------------------
  INPUT PIPELINE:
    TimeSeriesDataSet with 33 meteorological / lag / hydrology reals
    (expanded from 8 in v36).
    target_normalizer = TorchNormalizer(method="identity"):
      NO normalization → raw [0, 5] drought scores for CORAL threshold
      comparisons.
    time_idx : week_idx (train 0–781; test encoder 782–794; decoder 795–799)

  BACKBONE:
    TemporalFusionTransformer.from_dataset()
      hidden_size             = 64
      attention_head_size     = 4
      dropout                 = 0.2
      hidden_continuous_size  = 32
      output_size             = 1  (scalar logit: g(H_{i,t}))
      loss placeholder        = QuantileLoss(quantiles=[0.5])
        → used ONLY to size the TFT output layer to 1 neuron;
          never computed during training (CORALTFTWrapper owns all loss).

  CORAL HEAD (CORALTFTWrapper):
    coral_bias_raw (K=50,) unconstrained → monotonic biases via
      b = -cumsum(softplus(raw))   [b_1 > b_2 > ... > b_50]
    p̂_{ik} = σ( g(H_{i,t}) + b_k )   →  (B, 5, 50) CDF probabilities

  LOSS (Masked Ordinal BCE):
    z_{ik} = 1 if y_i ≥ t_k (Δt=0.1, K=50)  else 0
    L = -1/K ∑_k [ z_k log(p̂_k) + (1-z_k) log(1-p̂_k) ]
    Anti-anchor mechanism: zero-score samples produce near-zero gradient
    on high-drought bins as p̂ → 0, freeing tail thresholds.

  TRAINING:
    5-Fold Rolling Time-Based CV (no group-overlap leakage):
      Fold 0 : train [0, 756]  val [757, 761]
      ...     (see get_fold_boundaries for full layout)
    PyTorch Lightning Trainer
      max_epochs=50, accelerator="gpu", devices=1
      gradient_clip_val=0.1
      EarlyStopping(val_loss, patience=5, min_delta=1e-4)
      LearningRateMonitor(epoch)
    Optimizer: AdamW(lr=1e-3, weight_decay=1e-3)
    Scheduler: CosineAnnealingLR(T_max=50, eta_min=1e-6)

  INFERENCE:
    trainer.predict(coral_model, test_dataloader)
    → predict_step() calls decode_median(coral_probs)
      ŷ_i = Σ_k I(p̂_{ik} ≥ 0.5) × 0.1  (Conditional Median)
    5-fold decoded predictions stacked → np.median(axis=0) ensemble

  VALIDATION MONITOR (per epoch, printed by CORALTFTWrapper):
    1. CDF Saturation Entropy per drought bin
    2. Conditional Median MAE / Bias per bin
    3. Tail Recovery Delta: true 4.0–5.0 labels → ŷ > 3.5?

  POST-INFERENCE:
    np.clip(final_preds, 0, 5)   physical constraint
    Zero-floor suppressed (ordinal BCE naturally learns y=0 signal).
    Test Set 6-tier Binned Distribution printed before submission write.

  SUBMISSION:
    2248 rows × 6 cols | region_id, pred_week1 … pred_week5

v37 Changes over v36
--------------------
  [INTRODUCE] CORALTFTWrapper — full CORAL ordinal CDF pipeline.
  [INTRODUCE] QuantileLoss(quantiles=[0.5]) as TFT output_size=1 placeholder.
  [INTRODUCE] Masked Ordinal BCE loss (inside CORALTFTWrapper).
  [INTRODUCE] Conditional Median Decoder (Σ I(p≥0.5)·Δt).
  [INTRODUCE] CDF Saturation Entropy + Tail Recovery monitor (per epoch).
  [INTRODUCE] trainer.predict() + torch.cat() ensemble collection.
  [INTRODUCE] Expanded 33-feature unknown-real matrix.
  [INTRODUCE] TorchNormalizer(method="identity") — raw score preservation.

  [ABOLISH]   QuantileLoss() as training objective.
  [ABOLISH]   p50 extraction (index 3 of 7 quantiles).
  [ABOLISH]   extract_p50() function.
  [ABOLISH]   Conservative < 0.10 zero-floor.
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
    ORDINAL_K,
    ORDINAL_DELTA_T,
    ORDINAL_THRESHOLDS,
    load_data,
    get_fold_boundaries,
    build_training_dataset,
    build_val_dataset,
    build_test_inference_dataset,
)
from src.model import CORALTFTWrapper, CORAL_K, CORAL_DELTA_T

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE  : int   = 1024
NUM_WORKERS : int   = 8    # [v37.1] PCIe bandwidth maximisation
MAX_EPOCHS  : int   = 50
LR          : float = 2e-3  # [v37.1] peak LR for warmup-cosine schedule

# TFT backbone hyper-parameters
# output_size = 1 → scalar logit for the CORAL head
# QuantileLoss(quantiles=[0.5]) is a placeholder to size output_layer to 1 neuron;
# the loss is never computed in training (CORALTFTWrapper overrides training_step).
TFT_HIDDEN_SIZE            : int   = 64
TFT_ATTENTION_HEAD_SIZE    : int   = 4
TFT_DROPOUT                : float = 0.2
TFT_HIDDEN_CONTINUOUS_SIZE : int   = 32
TFT_PLATEAU_PATIENCE       : int   = 4

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
    flat  = final_preds.ravel()
    total = len(flat)

    log_fn("")
    log_fn("  " + "=" * 90)
    log_fn("  TEST SET BINNED PREDICTION MATRIX  [v37 CORAL-TFT Conditional Median Inference]")
    log_fn("  Empirical distribution of median-ensembled, clipped decoded predictions")
    log_fn("  " + "=" * 90)
    header = (
        f"  {'Bin / Drought Category':<52}  {'Count':>8}  "
        f"{'% Share':>9}  {'AvgPred':>9}  {'MinPred':>9}  {'MaxPred':>9}"
    )
    log_fn(header)
    log_fn("  " + "-" * 88)

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

    log_fn("  " + "-" * 88)
    log_fn(f"  Total predictions : {total:,}")
    log_fn(f"  Exact-zero fraction: {(flat == 0.0).mean():.4f}  "
           f"({(flat == 0.0).sum():,} of {total:,})")
    log_fn(f"  Non-zero fraction  : {(flat > 0.0).mean():.4f}  "
           f"({(flat > 0.0).sum():,} of {total:,})")
    log_fn(f"  Mean  : {flat.mean():.4f}  Std : {flat.std():.4f}")
    log_fn(f"  p50   : {np.percentile(flat, 50):.4f}  "
           f"p90  : {np.percentile(flat, 90):.4f}  "
           f"p99  : {np.percentile(flat, 99):.4f}  Max : {flat.max():.4f}")
    log_fn("  " + "=" * 90)
    log_fn("")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    t0        = time.time()
    log_lines : list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(str(msg))

    # -- 0. Banner --------------------------------------------------------------
    log("=" * 97)
    log("Drought Forecasting Pipeline  v38  (CORAL-TFT — Parametric tau-Decoder + OOF Calibration)")
    log("Model   : CORALTFTWrapper  (TemporalFusionTransformer + CORAL Consistent Bias Head)")
    log("Input   : Raw 13-week sequential meteorological windows per region")
    log("          37 unknown reals | 2 known reals (sin/cos) | NO Z-score normalisation")
    log("Backbone: TFT (output_size=1) -> scalar logit per step")
    log(f"CORAL   : K={ORDINAL_K} thresholds dt={ORDINAL_DELTA_T} -> [{ORDINAL_THRESHOLDS[0]}...{ORDINAL_THRESHOLDS[-1]}]")
    log("Loss    : Masked Ordinal BCE  z_{ik}=1 iff y>=t_k | Anti-anchor zero-mass gradient")
    log("[v38] Decode  : Parametric tau-Decoder  y_hat = sum_k I(p_k >= best_tau) * 0.1")
    log("[v38] Calibrate: OOF 1D Grid Search  tau in [0.40, 0.75] (36 candidates) per fold")
    log("Ensemble: 5-Fold Rolling CV  ->  np.median(decoded, axis=0)")
    log("Clip    : np.clip(0.0, 5.0)  physical constraint")
    log("=" * 97)
    log("")
    log("TFT Backbone Architecture:")
    log(f"  hidden_size            = {TFT_HIDDEN_SIZE}")
    log(f"  attention_head_size    = {TFT_ATTENTION_HEAD_SIZE}")
    log(f"  dropout                = {TFT_DROPOUT}")
    log(f"  hidden_continuous_size = {TFT_HIDDEN_CONTINUOUS_SIZE}")
    log(f"  output_size            = 1  (scalar logit; QuantileLoss([0.5]) sizing placeholder)")
    log(f"  learning_rate          = {LR}  (AdamW; CosineAnnealingLR in CORALTFTWrapper)")
    log(f"  max_epochs             = {MAX_EPOCHS}")
    log(f"  batch_size             = {BATCH_SIZE}")
    log("")
    log("5-Fold Rolling Time-Based CV Boundaries:")
    for k in range(N_FOLDS):
        te, vs, ve = get_fold_boundaries(k)
        log(f"  Fold {k}: train time_idx [0, {te}]  →  val time_idx [{vs}, {ve}]")
    log("")

    # -- 1. Load data -----------------------------------------------------------
    log("Loading processed data ...")
    train_df, test_df = load_data()
    log(f"  train_df : {train_df.shape}  |  test_df : {test_df.shape}")
    log(f"  train regions : {train_df['region_id'].nunique():,}")
    log(f"  test  regions : {test_df['region_id'].nunique():,}")
    assert train_df["region_id"].nunique() == 2248, "Expected 2248 train regions"
    assert test_df["region_id"].nunique()  == 2248, "Expected 2248 test regions"

    # -- [v38.3] Load regional historical drought structural priors ─────────────
    # 1. Read regional historical drought structural priors
    priors_path = os.path.join(ROOT, "data", "processed", "region_priors.csv")
    if not os.path.exists(priors_path):
        raise FileNotFoundError(f"找不到先驗字典: {priors_path}")
    priors_df = pd.read_csv(priors_path)

    # 2. Build explicit map correlation dictionary: region_id -> region_zero_prob
    region_zero_prob_dict = dict(zip(priors_df["region_id"], priors_df["region_zero_prob"]))

    log(f"\n[v38.3] Regional Prior Dictionary Loaded:")
    log(f"  Source   : {priors_path}")
    log(f"  Regions  : {len(region_zero_prob_dict):,} entries")
    log(f"  zero_prob range : [{min(region_zero_prob_dict.values()):.4f}, "
        f"{max(region_zero_prob_dict.values()):.4f}]  "
        f"mean={sum(region_zero_prob_dict.values())/len(region_zero_prob_dict):.4f}")
    log(f"  Fallback  : 0.5963 (macro mean) for any missing region_id")

    # Target distribution summary
    scores = train_df["score"].values
    log(f"\n[Training Target Distribution]")
    log(f"  mean={scores.mean():.4f}  std={scores.std():.4f}  "
        f"min={scores.min():.4f}  max={scores.max():.4f}")
    log(f"  Zero-fraction : {(scores == 0.0).mean():.4f}  "
        f"({(scores == 0.0).sum():,} rows)  — the 1.0 Anchor Degradation source")
    log(f"  Non-zero      : {(scores > 0.0).mean():.4f}  "
        f"({(scores > 0.0).sum():,} rows)")
    log(f"  p90={np.percentile(scores, 90):.4f}  "
        f"p99={np.percentile(scores, 99):.4f}  "
        f"tail(≥4.0)={(scores >= 4.0).sum():,} rows")

    # Ordinal encoding preview
    log(f"\n[v37] Ordinal CDF Encoding Preview:")
    log(f"  K={ORDINAL_K} thresholds step Δt={ORDINAL_DELTA_T}")
    log(f"  y=0.0 → z=[0,0,...,0]  (all-zero, 59.63% of training samples)")
    log(f"  y=2.3 → z=[1 (×23), 0 (×27)]  (moderate drought)")
    log(f"  y=4.5 → z=[1 (×45), 0 (×5)]   (extreme drought)")
    log(f"  y=5.0 → z=[1,1,...,1]           (maximum drought)")

    # -- 2. 5-Fold Rolling CORAL-TFT Training Loop ─────────────────────────────
    log(f"\n{'='*97}")
    log("5-Fold Rolling Time-Based CORAL-TFT Training")
    log(f"  Each fold trains a CORALTFTWrapper using Masked Ordinal BCE.")
    log(f"  Conditional Median decoded predictions are collected per fold.")
    log(f"  5-fold decoded arrays are median-ensembled post-training.")
    log(f"{'='*97}")

    all_fold_decoded     : list[np.ndarray] = []   # each (2248, 5)
    fold_val_bce_losses  : list[float]      = []
    test_dataset         = None
    test_ordered_regions : list[str]        = []

    for fold_k in range(N_FOLDS):

        fold_t0 = time.time()
        train_end, val_start, val_end = get_fold_boundaries(fold_k)

        log(f"\n{'='*97}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}  [CORAL-TFT Rolling CV]")
        log(f"  train time_idx : [0, {train_end}]  "
            f"(= {train_end + 1} weeks × 2248 regions)")
        log(f"  val   time_idx : [{val_start}, {val_end}]  "
            f"(= {val_end - val_start + 1} weeks × 2248 regions)")
        log(f"{'='*97}")

        # -- 2a. Build datasets for this fold -----------------------------------
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

        # -- 2c. DataLoaders ----------------------------------------------------
        # [v37.1] pin_memory=True → page-locked PCIe transfers for GPU throughput
        train_dataloader = training_dataset.to_dataloader(
            train       = True,
            batch_size  = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory  = True,
        )
        val_dataloader = val_dataset.to_dataloader(
            train       = False,
            batch_size  = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory  = True,
        )
        test_dataloader = test_dataset.to_dataloader(
            train       = False,
            batch_size  = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory  = True,
        )

        # -- 2d. Build TFT backbone with output_size=1 --------------------------
        # QuantileLoss(quantiles=[0.5]) is used ONLY to set output_size=1 in the
        # TFT's output_layer.  The loss is never computed — CORALTFTWrapper owns
        # the full training / validation step with Masked Ordinal BCE.
        log(f"\n  Initialising TFT backbone (output_size=1, fold {fold_k}) ...")
        tft_backbone = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate             = LR,
            hidden_size               = TFT_HIDDEN_SIZE,
            attention_head_size       = TFT_ATTENTION_HEAD_SIZE,
            dropout                   = TFT_DROPOUT,
            hidden_continuous_size    = TFT_HIDDEN_CONTINUOUS_SIZE,
            loss                      = QuantileLoss(quantiles=[0.5]),
            reduce_on_plateau_patience= TFT_PLATEAU_PATIENCE,
            log_interval              = -1,   # suppress TFT's internal logging
            log_val_interval          = -1,
        )
        log(f"  TFT backbone parameter count: "
            f"{sum(p.numel() for p in tft_backbone.parameters()):,}")

        # -- 2e. Wrap in CORAL head ─────────────────────────────────────────────
        log(f"\n  Wrapping TFT in CORALTFTWrapper (fold {fold_k}) ...")
        coral_model = CORALTFTWrapper(
            tft_backbone = tft_backbone,
            n_ordinal    = CORAL_K,
            lr           = LR,
            max_epochs   = MAX_EPOCHS,
        )
        log(coral_model.architecture_summary())
        log(f"  Total CORAL-TFT parameter count: {coral_model.count_parameters():,}")
        log(f"  CORAL bias_raw initial shape    : {tuple(coral_model.coral_bias_raw.shape)}")

        # -- 2f. Lightning Trainer ─────────────────────────────────────────────
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

        # [v37.1] gradient_clip_val=1.0 + norm algorithm shields attention heads
        # from divergent gradient spikes across 50 ordinal dimensions.
        trainer = pl.Trainer(
            max_epochs              = MAX_EPOCHS,
            accelerator             = "gpu",
            devices                 = 1,
            gradient_clip_val       = 1.0,
            gradient_clip_algorithm = "norm",
            callbacks               = callbacks,
            enable_progress_bar     = True,
            logger                  = True,
        )

        log(f"\n  Fitting CORALTFTWrapper (fold {fold_k})  ...  "
            f"[max_epochs={MAX_EPOCHS}, early_stop patience=5]")
        log(f"  Loss: Masked Ordinal BCE  |  Decode: Conditional Median  ŷ=Σ I(p≥0.5)·0.1")
        trainer.fit(
            coral_model,
            train_dataloaders = train_dataloader,
            val_dataloaders   = val_dataloader,
        )

        # Capture best validation BCE loss
        best_val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
        if hasattr(best_val_loss, "item"):
            best_val_loss = float(best_val_loss.item())
        else:
            best_val_loss = float(best_val_loss)
        fold_val_bce_losses.append(best_val_loss)

        stopped_epoch = trainer.current_epoch
        log(f"\n  [Fold {fold_k}] Training complete.")
        log(f"  Stopped at epoch      : {stopped_epoch}")
        log(f"  Best val_loss (BCE)   : {best_val_loss:.6f}")

        # Log final CORAL bias statistics
        biases_np = coral_model.get_monotonic_biases().detach().cpu().numpy()
        log(f"  CORAL biases [b_1..b_50]:")
        log(f"    b_1 (lowest threshold)  = {biases_np[0]:.4f}")
        log(f"    b_25 (mid threshold)    = {biases_np[24]:.4f}")
        log(f"    b_50 (highest threshold)= {biases_np[-1]:.4f}")
        log(f"    range = [{biases_np.min():.4f}, {biases_np.max():.4f}]  "
            f"(monotone: {(np.diff(biases_np) <= 0).all()})")

        # ======================================================================
        # [v38] Phase 3: OOF 1D Grid Search Calibrator
        # After trainer.fit(), use the validation set to find the best tau
        # that minimises out-of-fold MAE before touching the test horizon.
        # ======================================================================
        log(f"\n  [v38] Running OOF val predict for tau calibration (fold {fold_k}) ...")
        # 1. Isolate the out-of-fold validation set's prediction probabilities and ground-truth values
        val_predictions = trainer.predict(coral_model, val_dataloader)
        # val_predictions: list of dicts with keys 'coral_probs' and 'target'
        val_coral_probs = torch.cat([x["coral_probs"] for x in val_predictions], dim=0).numpy()
        val_y_true      = torch.cat([x["target"]      for x in val_predictions], dim=0).numpy()
        # val_coral_probs shape : (N_val, HORIZON, K=50)
        # val_y_true      shape : (N_val, HORIZON)

        # 2. Execute a 1D grid search over 36 evenly spaced candidates to locate the optimal splitting frontier
        tau_grid    = np.linspace(0.40, 0.75, 36)
        best_tau    = 0.5
        min_fold_mae = float("inf")

        for tau in tau_grid:
            # Utilise the newly refactored parametric decoding logic
            y_val_pred   = np.sum(val_coral_probs >= tau, axis=-1) * 0.1
            current_mae  = np.mean(np.abs(val_y_true - y_val_pred))

            if current_mae < min_fold_mae:
                min_fold_mae = current_mae
                best_tau     = tau

        log(f"  ==> [Fold {fold_k} Calibrated] Best Tau = {best_tau:.4f} | "
            f"Val OOF MAE = {min_fold_mae:.4f}")

        # Log tau grid scan summary
        log(f"  tau grid search range: [{tau_grid[0]:.2f}, {tau_grid[-1]:.2f}]  "
            f"({len(tau_grid)} candidates)")

        # Free val prediction buffers before test inference
        del val_predictions, val_coral_probs, val_y_true

        # ======================================================================
        # [v38] Phase 3 (cont.): Deploy fold-specific best_tau to infer the unseen test horizon
        # ======================================================================
        # ======================================================================
        # [v38.3] Per-Region Percentile Calibration (tau_i) Decoder
        # Replaces global best_tau with a region-specific threshold derived from
        # each zone's historical zero-drought density in region_priors.csv.
        # ======================================================================
        log(f"\n  [v38.3] Running test inference (fold {fold_k}) — Per-Region tau_i Calibration ...")
        test_predictions = trainer.predict(coral_model, test_dataloader)
        # test_predictions: list of dicts with keys 'coral_probs' and 'target'
        test_coral_probs = torch.cat(
            [x["coral_probs"] for x in test_predictions], dim=0
        ).numpy()
        # test_coral_probs shape: (2248, 5, 50) -> representing σ(logit + b_k)
        # Ensure 'test_ordered_regions' aligns exactly with the 2248 matrix array sorting order

        decoded_fold_list = []
        tau_i_values = []   # collect for diagnostics
        for i, region_id in enumerate(test_ordered_regions):
            # 1. Retrieve historical zero-score empirical frequency baseline
            # Fallback to macro expectation 0.5963 if mapping is missing
            p_zero_prior = region_zero_prob_dict.get(region_id, 0.5963)

            # 2. Extract the continuous 5-week sequence onset probabilities (Bin index 0 mapping >= 0.1)
            # base_probs shape: (5,)
            base_probs = test_coral_probs[i, :, 0]

            # 3. Formulate the dynamic target percentile threshold tau_i scaled to 0-100 index range
            calibrated_tau_i = np.percentile(base_probs, p_zero_prior * 100)

            # 4. Apply physical defense guardrails to insulate local volatility spikes
            tau_i = np.clip(calibrated_tau_i, 0.45, 0.75)

            # 5. Execute discrete conditional summation using the micro-calibrated tau_i bounds
            # Step weights of 0.1 apply strictly when survival confidence clears regional threshold
            region_pred = np.sum(test_coral_probs[i] >= tau_i, axis=-1) * 0.1
            decoded_fold_list.append(region_pred)
            tau_i_values.append(float(tau_i))

        # Re-compress historical sequence stack back into matrix architecture of shape (2248, 5)
        decoded_fold = np.stack(decoded_fold_list, axis=0).astype(np.float32)

        tau_arr = np.array(tau_i_values)
        log(f"  decoded_fold shape      : {decoded_fold.shape}")
        log(f"  [v38.3] Per-region tau_i stats:")
        log(f"    mean={tau_arr.mean():.4f}  std={tau_arr.std():.4f}  "
            f"min={tau_arr.min():.4f}  max={tau_arr.max():.4f}")
        log(f"    at_guardrail_low(0.45)  : {(tau_arr == 0.45).sum():,} regions")
        log(f"    at_guardrail_high(0.75) : {(tau_arr == 0.75).sum():,} regions")
        log(f"  decoded_fold stats [v38.3 per-region tau_i]:")
        log(f"    mean={decoded_fold.mean():.4f}  std={decoded_fold.std():.4f}  "
            f"min={decoded_fold.min():.4f}  max={decoded_fold.max():.4f}")
        log(f"    zero-frac = {(decoded_fold == 0.0).mean():.4f}  "
            f"p90={np.percentile(decoded_fold, 90):.4f}  "
            f"p99={np.percentile(decoded_fold, 99):.4f}")

        # Free test prediction buffer and working lists
        del test_predictions, test_coral_probs, decoded_fold_list, tau_i_values, tau_arr

        all_fold_decoded.append(decoded_fold)

        fold_elapsed = time.time() - fold_t0
        log(f"  Fold {fold_k} elapsed : {fold_elapsed:.1f}s  ({fold_elapsed/60:.1f} min)")

        # ======================================================================
        # [v38] Phase 4: Hard RAM/VRAM Memory Purge Defense
        # Eliminate PyTorch Lightning background process cache build-ups and
        # prevent segmentation faults during multiple fold execution handshakes.
        # ======================================================================
        import gc
        del trainer, coral_model, train_dataloader, val_dataloader, training_dataset, val_dataset
        del fold_train_df, fold_ctx_df
        gc.collect()
        torch.cuda.empty_cache()
        # ======================================================================


    # -- 3. Cross-fold summary --------------------------------------------------
    log(f"\n{'='*97}")
    log(f"5-Fold Rolling CV Summary  [v37 CORAL-TFT Masked Ordinal BCE]")
    log(f"{'='*97}")
    for k, vl in enumerate(fold_val_bce_losses):
        log(f"  Fold {k}: best val_loss (Ordinal BCE) = {vl:.6f}")
    finite_losses = [v for v in fold_val_bce_losses if not np.isnan(v)]
    if finite_losses:
        log(f"\n  Mean val_loss : {np.mean(finite_losses):.6f}  "
            f"±  {np.std(finite_losses):.6f}")
        log(f"  Best fold     : {int(np.argmin(fold_val_bce_losses))} "
            f"(val_loss = {min(finite_losses):.6f})")

    # -- 4. 5-Fold Conditional Median Ensemble ─────────────────────────────────
    log(f"\n{'='*97}")
    log(f"[v37] 5-Fold Conditional Median Ensemble Compression")
    log(f"  Strategy: np.median(all_fold_decoded, axis=0)")
    log(f"  Each fold produces ŷ = Σ_k I(p̂_k≥0.5)·0.1 (step 0.1, K=50)")
    log(f"{'='*97}")

    # Stack to (N_FOLDS, 2248, 5) then collapse via strict median
    decoded_stack = np.stack(all_fold_decoded, axis=0)    # (5, 2248, 5)
    final_preds   = np.median(decoded_stack, axis=0)      # (2248, 5)

    log(f"  decoded_stack shape : {decoded_stack.shape}")
    log(f"  final_preds shape   : {final_preds.shape}")
    log(f"  Pre-clip stats: mean={final_preds.mean():.4f}  "
        f"std={final_preds.std():.4f}  "
        f"min={final_preds.min():.4f}  max={final_preds.max():.4f}")

    # Per-fold zero-fractions (ordinal BCE tail escape audit)
    log(f"\n  Per-fold zero-fraction of decoded predictions:")
    for k, d in enumerate(all_fold_decoded):
        log(f"    Fold {k}: zero-frac={( d==0.0).mean():.4f}  "
            f"p90={np.percentile(d, 90):.4f}  max={d.max():.4f}")

    # -- 5. Physical clip [0, 5] ───────────────────────────────────────────────
    # Note: No zero-floor applied in v37.
    # Ordinal BCE with CORAL naturally learns to predict p̂ → 0 for all thresholds
    # when drought score = 0.0, producing ŷ = 0.0 via conditional median.
    # An artificial floor would incorrectly suppress genuine near-zero predictions.
    final_submission = np.clip(final_preds, 0.0, 5.0).astype(np.float32)

    log(f"\n  Post-clip stats [0, 5] (no zero-floor in v37 — ordinal BCE handles zeros):")
    log(f"    mean={final_submission.mean():.4f}  "
        f"std={final_submission.std():.4f}  "
        f"min={final_submission.min():.4f}  "
        f"max={final_submission.max():.4f}")
    log(f"    zero-frac = {(final_submission == 0.0).mean():.4f}")

    # -- 6. Test Set Binned Distribution Audit ─────────────────────────────────
    log(f"\n{'='*97}")
    log(f"[v37] Test Set 6-Tier Binned Distribution Audit")
    log(f"{'='*97}")
    print_test_binned_distribution(final_submission, log)

    # -- 7. Submission diagnostics ─────────────────────────────────────────────
    log("[v37 Submission Prediction Diagnostics  — Conditional Median Decoded]")
    flat  = final_submission.ravel()
    p50_v = float(np.percentile(flat, 50))
    p75_v = float(np.percentile(flat, 75))
    p90_v = float(np.percentile(flat, 90))
    p95_v = float(np.percentile(flat, 95))
    p99_v = float(np.percentile(flat, 99))
    log(f"  n={len(flat):,}  mean={flat.mean():.4f}  std={flat.std():.4f}")
    log(f"  p50={p50_v:.4f}  p75={p75_v:.4f}  p90={p90_v:.4f}  "
        f"p95={p95_v:.4f}  p99={p99_v:.4f}  max={flat.max():.4f}")
    log(f"  Exact zero fraction: {(flat == 0.0).mean():.4f}")

    if p99_v < 1.5:
        log("  *** WARNING: p99 < 1.5 — predictions may be under-dispersed! "
            "CORAL tail recovery may need more epochs. ***")
    else:
        log(f"  ✓ p99 >= 1.5 — prediction dynamic range is healthy.")

    # Tail recovery audit on final ensemble
    tail_mask = (flat >= 3.5)
    tail_frac = tail_mask.mean()
    log(f"  Predictions ≥ 3.5 (tail recovery signal) : "
        f"{tail_frac:.4f}  ({tail_mask.sum():,} / {len(flat):,})")
    if tail_frac > 0.001:
        log("  ✓ Non-trivial tail recovery detected — escaped 1.0 anchor sink!")
    else:
        log("  ✗ Very few predictions ≥ 3.5 — ensemble may still be anchor-biased")

    # -- 8. Build submission DataFrame ─────────────────────────────────────────
    log("\nFormatting submission.csv ...")

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

    # -- 9. Sanity checks ───────────────────────────────────────────────────────
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

    # -- 10. Total elapsed time ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    log(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    log_path = os.path.join(ROOT, "_training_log_38th.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines))
    print(f"\nTraining log saved  →  {log_path}")

    return {
        "fold_val_bce_losses"  : fold_val_bce_losses,
        "decoded_stack_shape"  : decoded_stack.shape,
        "final_preds_shape"    : final_submission.shape,
        "submission"           : submission,
        "sub_max"              : float(flat.max()),
        "sub_p99"              : p99_v,
        "zero_frac_final"      : float((flat == 0.0).mean()),
        "tail_recovery_frac"   : float(tail_frac),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
