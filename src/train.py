"""
train.py -- Drought Score Forecasting Pipeline (v11 – BiLSTM + Learnable Horizon + Gap-Aware CV + OOF Scaling)
=============================================================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv               -- Kaggle submission (2248 rows + header)
    models/fold_0_best.pt        -- Best weights for fold 0
    models/fold_1_best.pt        -- Best weights for fold 1
    models/fold_2_best.pt        -- Best weights for fold 2
    models/scaler.pkl            -- Final StandardScaler (fit on full train set; for test inference)
    _training_log_11th.txt       -- Full console log

Key improvements (v11 – BiLSTM + Learnable Horizon + Gap-Aware CV + OOF Scaling)
----------------------------------------------------------------------------------
  1. Massive Architecture Scale-Up (model.py)
       - BiLSTM: bidirectional=True.  LSTM output width = hidden_size * 2 = 512.
       - HIDDEN_SIZE=256 (was 64), NUM_LAYERS=3 (was 2).
       - Learnable Horizon Embedding: nn.Embedding(5, 32) replaces linspace scalar.
         Each of the 5 horizon steps now has a 32-dim learned representation.

  2. Gap-Aware Walk-Forward CV (dataset.py)
       - WINDOW_SIZE increased from 13 to 26 (half a year of context).
       - GAP_WEEKS=4: training fold strictly ends at val_start - 4 weeks,
         simulating the real 4-week deployment gap present in the test set.

  3. Strict Out-Of-Fold (OOF) StandardScaler (train.py)
       - Global scaler.fit() REMOVED from before the fold loop.
       - Each fold instantiates a NEW StandardScaler, fitted EXCLUSIVELY on
         that fold's training feature matrix.  Both train and val datasets for
         that fold are transformed with this fold-local scaler.
       - For Test Set inference: a final StandardScaler is fitted on the ENTIRE
         1.7M training set and used to transform the test set.  Saved as scaler.pkl.

  4. Hardware OOM Protection (train.py)
       - try-except RuntimeError around the fold training loop.
       - Default BATCH_SIZE=512.  If CUDA OOM is detected, cache is flushed,
         BATCH_SIZE falls back to 256, and training restarts for that fold.
       - pin_memory=True retained throughout.

  Retained from v10:
  5. Dynamic Loss Weighting (Burn-in) – epochs 1-20: Loss_B only; 21+: +0.1*Loss_A
  6. Manual LR Warm-up – epochs 1-5: 1e-5 → 1e-3 linearly
  7. Two-Stage Multi-Task Architecture (BCE + Smooth L1)
  8. Leakage-Free Target Encoding (fold-specific)
  9. Cyclical Time Features (week_sin, week_cos)
  10. Fold Ensembling (3 fold checkpoints averaged for test prediction)
  11. Temporal Attention (v8), LayerNorm (v7)
  12. NUM_EPOCHS=200, PATIENCE=35, np.clip(pred, 0, 5)

Hardware note
-------------
  RTX 4070 Laptop has 8 GB VRAM.  Default BATCH_SIZE=512 with OOM fallback to 256.
"""

import os
import sys
import time
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# -- project root on sys.path -------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_walk_forward_folds,
    DroughtDataset,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
    WF_NUM_FOLDS,
    WF_FOLD_WEEKS,
    GAP_WEEKS,
)
from src.model import DroughtLSTM

# ---------------------------------------------------------------------------
# Reproducibility  (v7+: must be called before ANY torch/numpy/random usage)
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Seed all RNG sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# v11: Scale-up
HIDDEN_SIZE   = 256           # v11: was 64
NUM_LAYERS    = 3             # v11: was 2
DROPOUT       = 0.4
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-3
BATCH_SIZE    = 512           # v11: OOM fallback to 256 inside fold loop
NUM_EPOCHS    = 200           # extended budget; early stopping governs each fold
PATIENCE      = 35            # deeper convergence patience on pure val MAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# AMP: only enabled for CUDA
USE_AMP = DEVICE.type == "cuda"

# GradScaler for AMP (no-op on CPU)
_scaler = GradScaler(device="cuda", enabled=USE_AMP)

# ---------------------------------------------------------------------------
# Loss Criteria (module-level for reuse)
# ---------------------------------------------------------------------------
_l1_criterion  = nn.L1Loss()
_bce_criterion = nn.BCEWithLogitsLoss()


# ---------------------------------------------------------------------------
# Continuous Smooth Loss  (v7 – retained in v9/v10/v11, applied to final_output)
# ---------------------------------------------------------------------------
# Mathematical definition:
#   For each (pred_i, target_i) pair:
#     huber_i  = SmoothL1(pred_i, target_i)               [element-wise, reduction='none']
#     weight_i = 1.0 + (target_i / 5.0)^2 * 3.0          [continuous, smooth curve]
#     loss     = mean(weight_i * huber_i)
#
# Properties:
#   - weight at target=0 : 1.0  (no amplification for no-drought samples)
#   - weight at target=3 : 1.0 + (3/5)^2 * 3.0 = 2.08   (moderate amplification)
#   - weight at target=5 : 1.0 + (5/5)^2 * 3.0 = 4.0    (maximum amplification)
#   - No step function -> smooth gradients and stable training.
# ---------------------------------------------------------------------------
def continuous_smooth_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss_B (v9/v10/v11): Continuous quadratically-weighted Smooth L1 / Huber loss.
    Applied to final_output (= P x Severity).

    Parameters
    ----------
    pred   : (B, H) model final_output predictions
    target : (B, H) ground-truth scores in [0, 5]

    Returns
    -------
    scalar loss
    """
    element_loss = F.smooth_l1_loss(pred, target, reduction="none")  # (B, H)
    weight = 1.0 + (target / 5.0) ** 2 * 3.0                        # (B, H)
    return (weight * element_loss).mean()


def joint_loss(
    final_output: torch.Tensor,
    logits_output: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
) -> torch.Tensor:
    """
    v10/v11 Dynamic Joint Loss (Burn-in Schedule)

    Burn-in Phase  (epoch <= 20): Loss = Loss_B ONLY
      Rationale: Let the regression head (Severity) learn magnitude freely
      before the BCE loss locks the Probability head onto binary predictions.

    Post Burn-in   (epoch > 20) : Loss = Loss_B + 0.1 * Loss_A
      Rationale: Introduce BCE with reduced weight (0.1) so it guides the
      Probability head without dominating the regression signal.

    Loss_A = BCEWithLogitsLoss(logits_output, binary_target)
             where binary_target = (target > 0.0).float()
    Loss_B = Continuous Smooth L1 Loss(final_output, target)

    Parameters
    ----------
    final_output : (B, H)  Branch_A x Branch_B  (Expected Severity)
    logits_output: (B, H)  Branch_A raw logits (pre-sigmoid)
    target       : (B, H)  ground-truth scores in [0, 5]
    epoch        : int     current epoch number (1-indexed)

    Returns
    -------
    scalar total loss
    """
    binary_target = (target > 0.0).float()  # (B, H); 1 = drought exists, 0 = no drought
    loss_b = continuous_smooth_loss(final_output, target)

    if epoch <= 20:
        # Burn-in phase: regression ONLY -- let Severity head calibrate freely
        return loss_b
    else:
        # Post burn-in: introduce BCE with light weight (0.1)
        loss_a = _bce_criterion(logits_output, binary_target)
        return loss_b + 0.1 * loss_a


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, epoch: int):
    """
    One training epoch -- v10/v11 Dynamic Joint Loss with AMP.

    v10/v11: epoch number passed to joint_loss for burn-in schedule.
         Epochs 1-20: Loss_B only.  Epoch 21+: Loss_B + 0.1 * Loss_A.
    """
    model.train()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=USE_AMP):
            final_output, logits_output = model(X)
            loss = joint_loss(final_output, logits_output, y, epoch)

        _scaler.scale(loss).backward()
        _scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        _scaler.step(optimizer)
        _scaler.update()

        bs = X.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n


@torch.no_grad()
def eval_mae(model, loader, device) -> float:
    """
    Evaluate pure (unweighted) MAE on final_output -- used for early stopping.
    Only uses final_output (Branch_A x Branch_B); ignores logits_output.
    This strictly aligns with the Kaggle metric.
    """
    model.eval()
    total_mae, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            final_output, _ = model(X)
        total_mae += _l1_criterion(final_output, y).item() * X.size(0)
        n += X.size(0)
    return total_mae / n if n > 0 else float("inf")


@torch.no_grad()
def eval_prediction_percentiles(model, loader, device, log) -> dict:
    """
    Diagnostic hook: collect all final_output predictions and log percentile stats.
    Self-Correction Check: if p99 < 2.0, the model is still evading extremes.
    """
    model.eval()
    all_preds = []
    for X, y in loader:
        X = X.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            final_output, _ = model(X)
        all_preds.append(final_output.cpu().float().numpy())

    if not all_preds:
        return {}

    preds_flat = np.concatenate(all_preds, axis=0).ravel()
    p50  = float(np.percentile(preds_flat, 50))
    p90  = float(np.percentile(preds_flat, 90))
    p95  = float(np.percentile(preds_flat, 95))
    p99  = float(np.percentile(preds_flat, 99))
    pmax = float(np.max(preds_flat))

    log(f"  [Prediction Diagnostics] n={len(preds_flat):,}")
    log(f"    p50={p50:.4f}  p90={p90:.4f}  p95={p95:.4f}  p99={p99:.4f}  max={pmax:.4f}")
    if p99 < 2.0:
        log("    *** WARNING: p99 < 2.0 -- model is still evading extremes (collapse not fixed)! ***")
    else:
        log("    v p99 >= 2.0 -- model is predicting away from the mean.")

    return {"p50": p50, "p90": p90, "p95": p95, "p99": p99, "max": pmax}


def make_model(input_size: int) -> DroughtLSTM:
    return DroughtLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
    ).to(DEVICE)


def make_optimiser(model):
    return torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )


def make_scheduler(optimiser):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10
    )


def _make_loader(dataset, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Build DataLoader with hardware-optimised settings for i9 + RTX 4070."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,            # i9-13980HX has 32 threads -> 8 workers for IO
        pin_memory=True,          # zero-copy transfer to GPU
        persistent_workers=True,
        prefetch_factor=2,
    )


# ---------------------------------------------------------------------------
# Single training run with early stopping (used for CV folds)
# ---------------------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int,
    patience: int,
    ckpt_path: str,
    log,
    show_header: bool = True,
):
    """
    Train model with early stopping on pure (unweighted) MAE.

    v10/v11 Training:
      - Manual LR warm-up epochs 1-5: linear ramp 1e-5 -> 1e-3.
      - ReduceLROnPlateau only active from epoch 6+.
      - Dynamic joint loss: regression-only burn-in epochs 1-20,
        then Loss_B + 0.1*Loss_A from epoch 21+.
    Early stopping: pure L1Loss(final_output, y) -- strictly Kaggle-metric-aligned.

    Returns best_val_mae, best_epoch, history.
    """
    optimiser = make_optimiser(model)
    scheduler = make_scheduler(optimiser)

    if show_header:
        log("=" * 65)
        log(f"{'Epoch':>6} {'TrainLoss':>11} {'ValMAE':>10} {'LR':>10}")
        log("=" * 65)

    best_val_mae = float("inf")
    best_epoch   = 1
    no_improve   = 0
    history      = []

    for epoch in range(1, num_epochs + 1):
        # --- v10/v11: Manual LR Warm-up (Epochs 1-5) ---
        # Linear ramp from 1e-5 to 1e-3 over the first 5 epochs.
        # Prevents ReduceLROnPlateau from firing during the volatile
        # feature-alignment phase before the model has learned anything.
        if epoch <= 5:
            lr = 1e-5 + (1e-3 - 1e-5) * ((epoch - 1) / 4.0)
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr

        train_loss = train_epoch(model, train_loader, optimiser, DEVICE, epoch)
        val_mae    = eval_mae(model, val_loader, DEVICE)

        # --- v10/v11: Only step ReduceLROnPlateau after warm-up is complete ---
        if epoch > 5:
            scheduler.step(val_mae)

        curr_lr = optimiser.param_groups[0]["lr"]

        history.append((epoch, train_loss, val_mae))

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            best_epoch   = epoch
            no_improve   = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        marker = " *" if improved else ""
        log(
            f"{epoch:>6}  {train_loss:>11.4f}"
            f"  {val_mae:>10.4f}  {curr_lr:>10.2e}{marker}"
        )

        if no_improve >= patience:
            log(
                f"\n  Early stopping at epoch {epoch} "
                f"(no MAE improvement for {patience} epochs)."
            )
            break

    log("=" * 65)
    log(f"  Best Val MAE: {best_val_mae:.4f}  (epoch {best_epoch})")
    return best_val_mae, best_epoch, history


# ---------------------------------------------------------------------------
# Target Encoding helpers (v9 – leakage-free)
# ---------------------------------------------------------------------------
def _zero_prob(x):
    """Per-region fraction of exactly-zero scores. Used in agg()."""
    return (x == 0.0).mean()


def _compute_te_stats(df: pd.DataFrame) -> tuple:
    """
    Compute target encoding statistics from a DataFrame with a 'score' column.

    Returns
    -------
    te_map : dict  region_id -> (region_mean_score, region_zero_prob)
    global_mean  : float  (fallback for unseen regions)
    global_zero_prob : float  (fallback for unseen regions)
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


def _augment_groups_with_te(
    groups: list,
    te_map: dict,
    global_mean: float,
    global_zero_prob: float,
) -> list:
    """
    Inject target encoding features into each (group_df, i_min, i_max) tuple.
    TE values are CONSTANT within a region (static features).

    Parameters
    ----------
    groups           : list of (group_df, i_min, i_max)
    te_map           : dict  region_id -> (mean_score, zero_prob)
    global_mean      : fallback region_mean_score for unseen regions
    global_zero_prob : fallback region_zero_prob for unseen regions

    Returns
    -------
    augmented list of (group_df_with_te, i_min, i_max)
    """
    result = []
    for group, i_min, i_max in groups:
        g = group.copy()
        rid = g["region_id"].iloc[0]
        mean_s, zero_p = te_map.get(rid, (global_mean, global_zero_prob))
        g["region_mean_score"] = np.float32(mean_s)
        g["region_zero_prob"]  = np.float32(zero_p)
        result.append((g, i_min, i_max))
    return result


def _merge_te_to_df(
    df: pd.DataFrame,
    te_map: dict,
    global_mean: float,
    global_zero_prob: float,
) -> pd.DataFrame:
    """
    Add region_mean_score and region_zero_prob columns to a DataFrame.
    Used for train_df (final scaler fitting) and test_df (inference).
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
# Fold Ensembling inference helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_test_set(model, test_df, feat_cols, scaler, log) -> dict:
    """
    Run inference for every region in test_df using the given model.
    Model returns (final_output, logits_output) -- only final_output is used.

    Returns
    -------
    predictions : dict  region_id -> np.array shape (5,)  [clipped to [0,5]]
    """
    model.eval()
    predictions = {}

    for region_id, group in test_df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < WINDOW_SIZE:
            pad_rows = pd.concat(
                [group.iloc[[0]]] * (WINDOW_SIZE - n) + [group],
                ignore_index=True,
            )
            group = pad_rows

        window_df = group.iloc[-WINDOW_SIZE:]
        X = window_df[feat_cols].values.astype(np.float32)
        X = scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with autocast(device_type=DEVICE.type, enabled=USE_AMP):
            final_output, _ = model(X_tensor)

        # Safety clip -- ensures Kaggle-valid range
        pred = final_output.squeeze(0).cpu().float().numpy()
        pred = np.clip(pred, 0.0, 5.0)
        predictions[region_id] = pred

    return predictions


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    # -- 0. Architecture & loss description ------------------------------------
    log("=" * 75)
    log("Drought Forecasting Pipeline  v11")
    log("BiLSTM + Scale-Up + Learnable Horizon Embedding + Gap-Aware CV + OOF Scaling")
    log("=" * 75)
    log("Architecture : Two-Stage BiLSTM MTL + Learnable Horizon Embedding  [v11]")
    log(f"  BiLSTM     : hidden_size={HIDDEN_SIZE}/dir -> {HIDDEN_SIZE*2} effective  [v11: was 64]")
    log(f"  Layers     : {NUM_LAYERS}  [v11: was 2]")
    log(f"  context_vector (B,{HIDDEN_SIZE*2}) -> expand (B,5,{HIDDEN_SIZE*2})")
    log(f"  horizon_ids [0-4] -> Embedding(5,32) -> (B,5,32)  [v11: learnable]")
    log(f"  encoded_state (B,5,{HIDDEN_SIZE*2+32})  [v11: replaces linspace scalar]")
    log(f"  Branch A   : Linear({HIDDEN_SIZE*2+32}->32) -> GELU -> Linear(32->1) -> (B,5)")
    log( "               P(drought) logits -- sigmoid() inline in forward()")
    log(f"  Branch B   : Linear({HIDDEN_SIZE*2+32}->32) -> GELU -> Linear(32->1) -> (B,5)")
    log( "               Severity >= 0  via Softplus()")
    log( "  Output     : final = sigmoid(logits_A) x Branch_B  (Expected Severity)")
    log( "Loss         : [v10/v11] Dynamic Burn-in Schedule")
    log( "  Epoch 1-20 : Loss = Loss_B ONLY  (regression burn-in)")
    log( "  Epoch 21+  : Loss = Loss_B + 0.1 * Loss_A  (BCE introduced lightly)")
    log( "  Loss_B     : Continuous Smooth L1  W_i = 1.0 + (y_i/5)^2 * 3.0")
    log( "  Loss_A     : BCEWithLogitsLoss(logits_output, binary_target)")
    log( "LR Schedule  : [v10/v11] Manual Warm-up Epochs 1-5: 1e-5 -> 1e-3 (linear)")
    log( "               Epoch 6+: ReduceLROnPlateau (factor=0.5, patience=10)")
    log( "Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]")
    log( "Pooling      : Temporal Attention  [v8: retained; v11: updated for hidden*2]")
    log( "LayerNorm    : LayerNorm(input_size) before LSTM  [v7: retained]")
    log( "Features     : 37  (11 weather + 2 cyclic + 9 rolling + 8 lag + 5 drought + 2 TE)")
    log( "  Cyclic     : week_sin, week_cos  [v9: retained]")
    log( "  TE         : region_mean_score, region_zero_prob  [v9: retained]")
    log( "CV Strategy  : Gap-Aware Walk-Forward  [v11: GAP_WEEKS=4]")
    log(f"  WINDOW_SIZE: {WINDOW_SIZE} weeks  [v11: was 13]")
    log(f"  GAP_WEEKS  : {GAP_WEEKS} weeks  [v11: train cutoff = val_start - {GAP_WEEKS}]")
    log( "OOF Scaling  : Fold-local StandardScaler (NO global pre-fit)  [v11]")
    log( "Test Scaler  : Final StandardScaler fit on full 1.7M train set  [v11]")
    log( "OOM Guard    : try-except RuntimeError -> fallback BATCH_SIZE=256  [v11]")
    log( "Strategy     : Fold Ensembling  (3 folds x 5 weeks, avg test predictions)")
    log(f"Epochs       : {NUM_EPOCHS}  |  Patience: {PATIENCE}")
    log(f"Seed         : 42  (cuDNN deterministic={torch.backends.cudnn.deterministic})")
    log(f"BatchSize    : {BATCH_SIZE}  (OOM fallback: 256)")

    # -- 1. Load data ----------------------------------------------------------
    log("\nLoading processed data ...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    log(f"  train raw: {train_raw.shape}  |  test raw: {test_raw.shape}")

    n_train_regions = train_raw["region_id"].nunique()
    n_test_regions  = test_raw["region_id"].nunique()
    log(f"  train regions: {n_train_regions}  |  test regions: {n_test_regions}")
    assert n_train_regions == 2248, f"Expected 2248 train regions, got {n_train_regions}"
    assert n_test_regions  == 2248, f"Expected 2248 test regions,  got {n_test_regions}"

    # -- 1b. Validate v9 features in processed CSV ----------------------------
    log("\n[v9 Feature Validation]")
    assert "week_sin" in train_raw.columns, "week_sin missing -- run eda.py first"
    assert "week_cos" in train_raw.columns, "week_cos missing -- run eda.py first"
    assert "month" not in train_raw.columns, "month still present -- should be dropped by preprocess"
    assert "week_of_year" not in train_raw.columns, "week_of_year still present -- should be dropped"
    log("  v week_sin, week_cos present  |  month, week_of_year correctly absent.")

    # -- 1c. Data Leakage Check ------------------------------------------------
    log("\n[Data Leakage Check]")
    leaky_cols = [c for c in FEATURE_COLS if "score" in c.lower()
                  and c not in ("region_mean_score", "region_zero_prob")]
    if leaky_cols:
        log(f"  *** WARNING: Potential leaky features found: {leaky_cols} ***")
    else:
        log("  v No raw-score autoregressive features in FEATURE_COLS.")
    log(f"  FEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # -- 2. Feature refinement (incl. Drought Index + log1p precip) -----------
    log("\nRefining features (drought proxy index + log1p precipitation) ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    log(f"  train after refinement: {train_df.shape}  |  test: {test_df.shape}")

    # -- 3. Drop rows with NaN score (prevents NaN loss) ----------------------
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
    log(f"  [v9] Zero-inflation: {zero_frac:.2%} of training scores == 0.0")
    for thresh in [1.0, 2.0, 3.0, 4.0]:
        frac = (all_scores > thresh).mean() * 100
        log(f"  score > {thresh:.1f}: {frac:.2f}%  [{int((all_scores > thresh).sum()):,} samples]")

    # -- 5. Full-train Target Encoding (for scaler fitting + test inference) --
    log("\n[v9] Computing full-train Target Encoding statistics ...")
    te_map_full, global_mean_te, global_zero_prob_te = _compute_te_stats(train_df)
    log(f"  Regions with TE stats : {len(te_map_full)}")
    log(f"  Global region_mean_score : {global_mean_te:.4f}")
    log(f"  Global region_zero_prob  : {global_zero_prob_te:.4f}")

    # Add full-train TE to train_df (used later for test scaler fitting)
    train_df = _merge_te_to_df(train_df, te_map_full, global_mean_te, global_zero_prob_te)
    log("  v region_mean_score, region_zero_prob added to train_df (full-train stats)")

    # -- 6. Determine feature columns and input_size (v11: NO global scaler fit) --
    log("\n[v11] Determining feature columns (OOF Scaling: NO global pre-fit) ...")
    feat_cols  = [c for c in FEATURE_COLS if c in train_df.columns]
    input_size = len(feat_cols)
    log(f"  Input features ({input_size}): {feat_cols}")

    assert input_size == 37, (
        f"Expected 37 features (35 base + 2 TE), got {input_size}. "
        f"Check that preprocess.py was run (eda.py) and TE cols are present."
    )

    log("  v OOF Scaling: each fold will instantiate its OWN StandardScaler,")
    log("                 fitted EXCLUSIVELY on that fold's training features.")
    log("  v Test Scaler: a final StandardScaler will be fit on the ENTIRE train set.")

    # -- 7. Walk-Forward Cross-Validation with Fold Checkpointing -------------
    log(f"\n{'='*75}")
    log(f"Walk-Forward Cross-Validation  ({WF_NUM_FOLDS} folds x {WF_FOLD_WEEKS} weeks)")
    log(f"[v11] Gap-Aware: train cutoff = val_start - GAP_WEEKS ({GAP_WEEKS} weeks)")
    log(f"[v11] Window Size: {WINDOW_SIZE} weeks (was 13)")
    log(f"Train loss  : v10/v11 Burn-in: epochs 1-20 = Smooth L1 only; 21+ = Smooth L1 + 0.1*BCE")
    log(f"Val metric  : pure MAE  (unweighted, Kaggle-aligned)")
    log(f"TE strategy : fold-specific (train rows only) -> leakage-free")
    log(f"Scaler      : OOF (new StandardScaler per fold, fit on fold train features)")
    log(f"Checkpoints : fold_0_best.pt / fold_1_best.pt / fold_2_best.pt")
    log(f"Strategy    : Fold Ensembling  (v8 retained)")
    log(f"AMP : {USE_AMP}  |  batch_size={BATCH_SIZE} (OOM fallback: 256)  |  num_workers=8")
    log(f"{'='*75}")

    folds = build_walk_forward_folds(train_df)
    fold_maes         = []
    fold_best_epochs  = []
    fold_percentiles  = []
    fold_ckpt_paths   = []
    fold_scalers      = []   # v11: retain fold scalers (not used for test, but logged)

    for fold_k, (fold_train_groups, fold_val_groups) in enumerate(folds):
        log(f"\n-- Fold {fold_k + 1}/{WF_NUM_FOLDS} --")

        # ---- v9: Compute FOLD-SPECIFIC Target Encoding (leakage-free) ------
        # Use rows 0..val_start from training groups exclusively.
        # val_start = train_i_max + WINDOW_SIZE + HORIZON + GAP_WEEKS (v11)
        fold_te_rows = []
        for group, i_min, i_max in fold_train_groups:
            val_start = min(i_max + WINDOW_SIZE + HORIZON + GAP_WEEKS, len(group))
            fold_te_rows.append(group.iloc[:val_start])

        if fold_te_rows:
            fold_te_df = pd.concat(fold_te_rows, ignore_index=True)
            # Only use rows where score is not NaN for TE computation
            fold_te_df = fold_te_df.dropna(subset=["score"])
            fold_te_map, fold_global_mean, fold_global_zero_prob = _compute_te_stats(fold_te_df)
        else:
            fold_te_map = te_map_full
            fold_global_mean      = global_mean_te
            fold_global_zero_prob = global_zero_prob_te

        log(f"  [v9 TE] Regions covered: {len(fold_te_map)}"
            f"  |  fold_mean_score={fold_global_mean:.4f}"
            f"  |  fold_zero_prob={fold_global_zero_prob:.4f}")

        # Augment group dfs with fold-specific TE values
        aug_train_groups = _augment_groups_with_te(
            fold_train_groups, fold_te_map, fold_global_mean, fold_global_zero_prob
        )
        aug_val_groups = _augment_groups_with_te(
            fold_val_groups, fold_te_map, fold_global_mean, fold_global_zero_prob
        )

        # ---- v11: Strict OOF Scaling ----------------------------------------
        # Instantiate a NEW StandardScaler per fold.
        # Fit EXCLUSIVELY on the fold's training feature matrix (never on val).
        # Then transform both train and val datasets with this fold-local scaler.
        log(f"  [v11 OOF Scaler] Fitting fold-{fold_k+1} StandardScaler on fold train features ...")
        fold_scaler = StandardScaler()

        # Collect raw feature matrix from all fold train groups
        fold_train_feat_parts = []
        for group, i_min, i_max in aug_train_groups:
            local_feat_cols = [c for c in FEATURE_COLS if c in group.columns]
            fold_train_feat_parts.append(group[local_feat_cols].values.astype(np.float32))

        if fold_train_feat_parts:
            fold_train_feat_matrix = np.concatenate(fold_train_feat_parts, axis=0)
            fold_scaler.fit(fold_train_feat_matrix)
            log(f"  [v11 OOF Scaler] Fit on {len(fold_train_feat_matrix):,} rows "
                f"({fold_train_feat_matrix.shape[1]} features)")
        else:
            log(f"  [v11 OOF Scaler] WARNING: No train rows for fold {fold_k+1}, using identity scaler.")

        fold_scalers.append(fold_scaler)

        # Create datasets with fold-local scaler
        fold_train_ds = DroughtDataset(aug_train_groups, scaler=fold_scaler)
        fold_val_ds   = DroughtDataset(aug_val_groups,   scaler=fold_scaler)
        log(f"  Train seqs: {len(fold_train_ds):,}  |  Val seqs: {len(fold_val_ds):,}")

        # ---- Tensor Shape Verification (fold 0 only) ------------------------
        fold_model = make_model(input_size)
        fold_ckpt  = os.path.join(MODELS_DIR, f"fold_{fold_k}_best.pt")
        fold_ckpt_paths.append(fold_ckpt)

        # ---- v11: OOM-Protected Training ------------------------------------
        # Start with BATCH_SIZE=512; if CUDA OOM is caught, flush cache,
        # fall back to BATCH_SIZE=256, and restart training for this fold.
        batch_size_to_use = BATCH_SIZE

        def _run_fold_training(bs):
            """Helper: build loaders + optionally verify shapes + train."""
            try:
                loader_tr  = _make_loader(fold_train_ds, shuffle=True,  batch_size=bs)
                loader_val = _make_loader(fold_val_ds,   shuffle=False, batch_size=bs)
            except Exception:
                loader_tr  = DataLoader(fold_train_ds, batch_size=bs,
                                        shuffle=True,  num_workers=0, pin_memory=USE_AMP)
                loader_val = DataLoader(fold_val_ds,   batch_size=bs,
                                        shuffle=False, num_workers=0, pin_memory=USE_AMP)

            # Tensor shape verification (fold 0 only)
            if fold_k == 0:
                first_X, first_y = next(iter(loader_tr))
                log(f"\n  [Tensor Shape Verification] First Batch (Fold 1):")
                log(f"    X shape : {tuple(first_X.shape)}"
                    f"  ->  (Batch={first_X.shape[0]}, Seq={first_X.shape[1]}, Features={first_X.shape[2]})")
                log(f"    y shape : {tuple(first_y.shape)}"
                    f"  ->  (Batch={first_y.shape[0]}, Horizon={first_y.shape[1]})")
                log(f"    Expected: Features={input_size}  "
                    f"(11 weather + 2 cyclic + 9 rolling + 8 lag + 5 drought + 2 TE = 37)")
                log(f"    Window : {WINDOW_SIZE} weeks  [v11: was 13]")
                assert first_X.shape[1] == WINDOW_SIZE, \
                    f"Seq mismatch: got {first_X.shape[1]}, expected {WINDOW_SIZE}"
                assert first_X.shape[2] == input_size, \
                    f"Feature mismatch: got {first_X.shape[2]}, expected {input_size}"
                assert first_y.shape[1] == HORIZON, \
                    f"Horizon mismatch: got {first_y.shape[1]}, expected {HORIZON}"
                log(f"    v Shape assertion PASSED.\n")
                del first_X, first_y

            mae, epoch, hist = train_model(
                fold_model, loader_tr, loader_val,
                num_epochs=NUM_EPOCHS,
                patience=PATIENCE,
                ckpt_path=fold_ckpt,
                log=log,
                show_header=True,
            )
            return mae, epoch, hist, loader_tr, loader_val

        try:
            best_fold_mae, best_fold_epoch, _, fold_loader_tr, fold_loader_val = \
                _run_fold_training(batch_size_to_use)

        except RuntimeError as oom_err:
            if "out of memory" in str(oom_err).lower():
                log(f"\n  [v11 OOM] CUDA out-of-memory at batch_size={batch_size_to_use}.")
                log(f"  [v11 OOM] Flushing CUDA cache and retrying with batch_size=256 ...")
                torch.cuda.empty_cache()

                # Re-instantiate model to clear any partially-allocated weights
                fold_model = make_model(input_size)
                batch_size_to_use = 256

                best_fold_mae, best_fold_epoch, _, fold_loader_tr, fold_loader_val = \
                    _run_fold_training(batch_size_to_use)
            else:
                raise  # Re-raise non-OOM RuntimeErrors immediately

        fold_maes.append(best_fold_mae)
        fold_best_epochs.append(best_fold_epoch)
        log(f"  Fold {fold_k+1} Best Val MAE: {best_fold_mae:.4f}  |  Best Epoch: {best_fold_epoch}")
        log(f"  Checkpoint saved -> {fold_ckpt}")

        # --- Diagnostic Hook: load best checkpoint and check prediction distribution ---
        log(f"\n  [Fold {fold_k+1} Prediction Percentiles -- best checkpoint]")
        if os.path.exists(fold_ckpt) and best_fold_mae < float("inf"):
            fold_model.load_state_dict(torch.load(fold_ckpt, map_location=DEVICE))
            pct = eval_prediction_percentiles(fold_model, fold_loader_val, DEVICE, log)
        else:
            log("  [Skip] No valid checkpoint saved for this fold (training produced NaN).")
            pct = {}
        fold_percentiles.append(pct)

        # Free GPU memory after each fold
        del fold_model, fold_loader_tr, fold_loader_val, fold_train_ds, fold_val_ds
        torch.cuda.empty_cache()

    avg_val_mae     = float(np.mean(fold_maes))
    mean_best_epoch = float(np.mean(fold_best_epochs))

    log(f"\n{'='*75}")
    log(f"Fold MAEs          : {[f'{m:.4f}' for m in fold_maes]}")
    log(f"Average_Val_MAE    : {avg_val_mae:.4f}")
    log(f"Fold Best Epochs   : {fold_best_epochs}")
    log(f"Mean Best Epoch    : {mean_best_epoch:.1f}")
    log(f"Strategy (v8/v9)   : Fold Ensembling (NO blind retraining)")
    log(f"\nFold Prediction Percentiles Summary:")
    for i, p in enumerate(fold_percentiles):
        if p:
            log(f"  Fold {i+1}: p50={p.get('p50',float('nan')):.3f}  "
                f"p95={p.get('p95',float('nan')):.3f}  "
                f"p99={p.get('p99',float('nan')):.3f}  "
                f"max={p.get('max',float('nan')):.3f}")
    log(f"{'='*75}")

    # -- 8. Fit FINAL Test Scaler on ENTIRE training set ----------------------
    # v11: The test scaler is separate from any fold scaler.
    # It is fitted on the full 1.7M training set to maximize coverage of the
    # feature distribution before transforming the test set.
    log(f"\n[v11] Fitting final Test StandardScaler on FULL training set ...")
    test_scaler = StandardScaler()
    full_train_feat_matrix = train_df[feat_cols].values.astype(np.float32)
    test_scaler.fit(full_train_feat_matrix)
    log(f"  Test scaler fit on {len(full_train_feat_matrix):,} rows  "
        f"({full_train_feat_matrix.shape[1]} features)")

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(test_scaler, f)
    log(f"  Test scaler saved -> {os.path.join(MODELS_DIR, 'scaler.pkl')}")

    # -- 8b. Prepare test_df with full-train TE for inference -----------------
    log(f"\n[v9] Preparing test_df with full-train Target Encoding ...")
    test_df = _merge_te_to_df(test_df, te_map_full, global_mean_te, global_zero_prob_te)
    log(f"  test_df after TE injection: {test_df.shape}")
    log(f"  v region_mean_score, region_zero_prob added to test_df (full-train stats)")

    # -- 9. Fold Ensemble Inference on Test Set -------------------------------
    log(f"\n{'='*75}")
    log(f"Fold Ensemble Inference  (v8/v9 -- Average Blending)")
    log(f"  Blending {len(fold_ckpt_paths)} fold checkpoints:")
    for p in fold_ckpt_paths:
        log(f"    {p}")
    log(f"  Scaler used for inference: FINAL TEST SCALER (fit on full train set)  [v11]")
    log(f"  final_pred = mean(pred_0, pred_1, pred_2)")
    log(f"  Safety clip: np.clip(final_pred, 0.0, 5.0)")
    log(f"{'='*75}")

    # Print architecture summary using a temporary model
    _tmp_model = make_model(input_size)
    log("\n" + _tmp_model.architecture_summary(input_size))
    del _tmp_model

    all_fold_pred_dicts = []

    for fold_k, ckpt_path in enumerate(fold_ckpt_paths):
        log(f"\n  [Fold {fold_k+1}] Loading checkpoint: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            log(f"  *** WARNING: Checkpoint not found: {ckpt_path}. Skipping this fold. ***")
            continue

        fold_model = make_model(input_size)
        fold_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        fold_model.eval()

        # v11: Use test_scaler (fit on full train) for test inference
        fold_pred_dict = predict_test_set(fold_model, test_df, feat_cols, test_scaler, log)
        all_fold_pred_dicts.append(fold_pred_dict)
        log(f"  [Fold {fold_k+1}] Inference complete. Regions predicted: {len(fold_pred_dict)}")

        del fold_model
        torch.cuda.empty_cache()

    if not all_fold_pred_dicts:
        raise RuntimeError("No valid fold checkpoints found. Cannot build ensemble.")

    log(f"\n  Blending {len(all_fold_pred_dicts)} fold predictions via simple average ...")

    all_region_ids = sorted(all_fold_pred_dicts[0].keys())

    predictions = {}
    for region_id in all_region_ids:
        fold_preds = np.array([d[region_id] for d in all_fold_pred_dicts])  # (K, 5)
        final_pred = fold_preds.mean(axis=0)                                  # (5,)
        final_pred = np.clip(final_pred, 0.0, 5.0)                           # Kaggle format
        predictions[region_id] = final_pred

    # -- 10. Submission-level prediction diagnostics --------------------------
    log("\n[Submission Prediction Diagnostics]")
    all_sub_preds = np.array(list(predictions.values())).ravel()
    p50  = float(np.percentile(all_sub_preds, 50))
    p90  = float(np.percentile(all_sub_preds, 90))
    p95  = float(np.percentile(all_sub_preds, 95))
    p99  = float(np.percentile(all_sub_preds, 99))
    pmax = float(np.max(all_sub_preds))
    log(f"  n={len(all_sub_preds):,}  mean={all_sub_preds.mean():.4f}  std={all_sub_preds.std():.4f}")
    log(f"  p50={p50:.4f}  p90={p90:.4f}  p95={p95:.4f}  p99={p99:.4f}  max={pmax:.4f}")
    if p99 < 2.0:
        log("  *** WARNING: Submission p99 < 2.0 -- model collapse still present! ***")
    else:
        log("  v Submission p99 >= 2.0 -- model collapse is BROKEN.")

    # Zero-inflation: did the model predict many exact zeros?
    zero_pred_frac = (all_sub_preds < 0.05).mean()
    log(f"  Fraction of near-zero predictions (<0.05): {zero_pred_frac:.2%}")

    # -- 11. Format & save submission.csv -------------------------------------
    log("\nFormatting submission.csv ...")
    rows = []
    for region_id, preds in sorted(predictions.items()):
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

    # -- 12. Sanity checks ----------------------------------------------------
    assert len(submission) == 2248, (
        f"Expected 2248 rows, got {len(submission)}"
    )
    assert list(submission.columns) == [
        "region_id", "pred_week1", "pred_week2",
        "pred_week3", "pred_week4", "pred_week5",
    ], f"Unexpected columns: {list(submission.columns)}"
    log("  v Submission assertion passed: 2248 rows, 6 columns.")

    test_regions  = set(test_df["region_id"].unique())
    train_regions = set(train_df["region_id"].unique())
    assert test_regions == train_regions, (
        f"Region mismatch: train={len(train_regions)}, test={len(test_regions)}"
    )
    log("  v Train/test regions match (2248).")

    log(f"  submission.csv -> {sub_path}")
    log(f"  Rows (excl. header): {len(submission)}")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    # Save full training log
    log(f"\nTotal elapsed: {(time.time() - t0):.1f}s")
    log_path = os.path.join(ROOT, "_training_log_11th.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nTraining log saved -> {log_path}")

    return {
        "fold_maes":              fold_maes,
        "fold_best_epochs":       fold_best_epochs,
        "avg_val_mae":            avg_val_mae,
        "mean_best_epoch":        mean_best_epoch,
        "input_size":             input_size,
        "submission":             submission,
        "fold_percentiles":       fold_percentiles,
        "sub_p99":                p99,
        "num_folds_ensembled":    len(all_fold_pred_dicts),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
