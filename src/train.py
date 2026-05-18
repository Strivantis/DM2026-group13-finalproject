"""
train.py -- Drought Score Forecasting Pipeline (v7 – Architecture Refinement & Smooth Loss)
============================================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv               -- Kaggle submission (2248 rows + header)
    models/final_model_v7.pt     -- Blind full-retrained model (TARGET_EPOCHS, 100% data)
    models/scaler.pkl            -- Fitted StandardScaler
    models/training_history.csv  -- Per-epoch metrics for final training run
    _training_log_7th.txt        -- Full console log

Key improvements (v7 – Architecture Refinement & Smooth Loss)
--------------------------------------------------------------
  1. Continuous Smooth Loss      -- Replaces hard-threshold weighted loss.
                                    W_i = 1.0 + (y_i / 5.0)^2 * 3.0
                                    Smooth gradient transitions; no sudden jumps.
  2. LayerNorm on Inputs         -- nn.LayerNorm(input_size) before LSTM to
                                    stabilise training under concept drift.
  3. Global Average Pooling      -- Replaces naive last-step extraction.
                                    All 13 weekly hidden states are averaged so
                                    the full temporal profile informs predictions.
  4. No Clamp in Model           -- torch.clamp removed from forward pass;
                                    Softplus output is unbounded to keep gradients
                                    active across the full score range.
  5. Inference Safety Clip       -- np.clip(predictions, 0.0, 5.0) applied to
                                    NumPy arrays before writing submission.csv.
  6. Strict Reproducibility      -- set_seed(42) called at startup; deterministic
                                    cuDNN; no dynamic OOM batch-size fallback.
  7. Hardcoded BATCH_SIZE=512    -- Removes all OOM-retry complexity.
  8. Diagnostic Logging          -- 95th and 99th prediction percentiles logged
                                    after every CV fold to detect collapse.

Hardware note
-------------
  RTX 4070 Laptop has 8 GB VRAM.  BATCH_SIZE fixed to 512 for stability.
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
    build_full_train_groups,
    DroughtDataset,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
    WF_NUM_FOLDS,
    WF_FOLD_WEEKS,
)
from src.model import DroughtLSTM

# ---------------------------------------------------------------------------
# Reproducibility  (v7 – must be called before ANY torch/numpy/random usage)
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

HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-3
BATCH_SIZE    = 512          # hardcoded; no OOM fallback (v7)
NUM_EPOCHS    = 150          # extended; early stopping governs CV folds
PATIENCE      = 25           # patience on pure val MAE (CV only)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# AMP: only enabled for CUDA
USE_AMP = DEVICE.type == "cuda"

# GradScaler for AMP (no-op on CPU)
_scaler = GradScaler(device="cuda", enabled=USE_AMP)


# ---------------------------------------------------------------------------
# Continuous Smooth Loss  (v7 – replaces hard-threshold weighted loss)
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
#   - No step function → smooth gradients and stable training.
# ---------------------------------------------------------------------------
def continuous_smooth_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Continuous quadratically-weighted Smooth L1 / Huber loss.

    Parameters
    ----------
    pred   : (B, H) model predictions
    target : (B, H) ground-truth scores in [0, 5]

    Returns
    -------
    scalar loss
    """
    element_loss = F.smooth_l1_loss(pred, target, reduction="none")  # (B, H)
    # Continuous weight: quadratic ramp from 1.0 (at target=0) to 4.0 (at target=5)
    weight = 1.0 + (target / 5.0) ** 2 * 3.0                        # (B, H)
    return (weight * element_loss).mean()


# Plain MAE for evaluation/diagnostics (no weighting – fair metric)
_l1_criterion = nn.L1Loss()


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    """One training epoch – Continuous Smooth Loss with AMP."""
    model.train()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=USE_AMP):
            pred = model(X)
            loss = continuous_smooth_loss(pred, y)

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
    """Evaluate pure (unweighted) MAE – used for early stopping & fold scoring."""
    model.eval()
    total_mae, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            pred = model(X)
        total_mae += _l1_criterion(pred, y).item() * X.size(0)
        n += X.size(0)
    return total_mae / n if n > 0 else float("inf")


@torch.no_grad()
def eval_prediction_percentiles(model, loader, device, log) -> dict:
    """
    Diagnostic hook: collect all predictions and log percentile statistics.
    Self-Correction Check: if p99 < 2.0, the model is still evading extremes.
    """
    model.eval()
    all_preds = []
    for X, y in loader:
        X = X.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            pred = model(X)
        all_preds.append(pred.cpu().float().numpy())

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
        log("    *** WARNING: p99 < 2.0 — model is still evading extremes (collapse not fixed)! ***")
    else:
        log("    ✓ p99 >= 2.0 — model is predicting away from the mean.")

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
        num_workers=8,            # i9-13980HX has 32 threads → 8 workers for IO
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
    Loss during training: Continuous Smooth Loss (v7).

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
        train_loss = train_epoch(model, train_loader, optimiser, DEVICE)
        val_mae    = eval_mae(model, val_loader, DEVICE)
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
# Blind full retraining (v7) -- no validation, no early stopping
# ---------------------------------------------------------------------------
def train_model_blind(
    model,
    train_loader,
    num_epochs: int,
    ckpt_path: str,
    log,
):
    """
    Blind full retraining on 100% of training data for exactly num_epochs.
    Uses CosineAnnealingLR.  Continuous Smooth Loss (v7).  AMP enabled.
    """
    optimiser = make_optimiser(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=num_epochs
    )

    log("=" * 55)
    log(f"{'Epoch':>6} {'TrainLoss':>11} {'LR':>10}")
    log("=" * 55)

    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, DEVICE)
        scheduler.step()
        curr_lr = optimiser.param_groups[0]["lr"]

        history.append((epoch, train_loss))
        log(f"{epoch:>6}  {train_loss:>11.4f}  {curr_lr:>10.2e}")

    log("=" * 55)
    torch.save(model.state_dict(), ckpt_path)
    log(f"  Final model saved -> {ckpt_path}")
    return history


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
    log("=" * 65)
    log("Drought Forecasting Pipeline  v7  (Architecture Refinement & Smooth Loss)")
    log("=" * 65)
    log("Loss      : Continuous Smooth Loss  W_i = 1.0 + (y_i/5)^2 * 3.0")
    log("Activation: Softplus  [NO clamp – unbounded; np.clip at inference]")
    log("Pooling   : Global Average Pooling (dim=1)  [all 13 steps averaged]")
    log("LayerNorm : LayerNorm(input_size) applied before LSTM")
    log("Head      : Linear(64→32)→GELU→Dropout(0.3)→Linear(32→5)")
    log(f"Seed      : 42  (cuDNN deterministic={torch.backends.cudnn.deterministic})")
    log(f"BatchSize : {BATCH_SIZE}  (hardcoded, no OOM fallback)")

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

    # -- 1b. Data Leakage Check -------------------------------------------------
    log("\n[Data Leakage Check]")
    leaky_cols = [c for c in FEATURE_COLS if "score" in c.lower()]
    if leaky_cols:
        log(f"  *** WARNING: Potential leaky features found: {leaky_cols} ***")
    else:
        log("  ✓ No score-based autoregressive features in FEATURE_COLS.")
    log(f"  FEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # -- 2. Feature refinement (incl. Drought Index + log1p precip) ------------
    log("\nRefining features (drought proxy index + log1p precipitation) ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)

    feat_cols  = [c for c in FEATURE_COLS if c in train_df.columns]
    input_size = len(feat_cols)
    log(f"  Input features ({input_size}): {feat_cols}")
    log(f"  train after refinement: {train_df.shape}  |  test: {test_df.shape}")

    # -- 3. Fit scaler on training features ------------------------------------
    log("\nFitting StandardScaler on training feature matrix ...")
    scaler = StandardScaler()
    train_feat_matrix = train_df[feat_cols].values.astype(np.float32)
    scaler.fit(train_feat_matrix)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # -- 3b. Drop rows with NaN score (prevents NaN loss) ----------------------
    before = len(train_df)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    dropped_nan = before - len(train_df)
    if dropped_nan:
        log(f"  [NaN drop] Removed {dropped_nan:,} rows with NaN score from train_df.")

    # -- 4. Target score distribution summary ----------------------------------
    log("\n[Training Target Distribution]")
    all_scores = train_df["score"].values
    log(f"  mean={all_scores.mean():.4f}  std={all_scores.std():.4f}  "
        f"min={all_scores.min():.2f}  max={all_scores.max():.2f}")
    for thresh in [1.0, 2.0, 3.0, 4.0]:
        frac = (all_scores > thresh).mean() * 100
        log(f"  score > {thresh:.1f}: {frac:.2f}%  [{int((all_scores > thresh).sum()):,} samples]")

    # -- 5. Walk-Forward Cross-Validation --------------------------------------
    log(f"\n{'='*65}")
    log(f"Walk-Forward Cross-Validation  ({WF_NUM_FOLDS} folds × {WF_FOLD_WEEKS} weeks)")
    log(f"Train loss  : Continuous Smooth Loss  W_i = 1.0 + (y_i/5)^2 * 3.0")
    log(f"Val metric  : pure MAE  (unweighted, fair comparison)")
    log(f"AMP : {USE_AMP}  |  batch_size={BATCH_SIZE}  |  num_workers=8")
    log(f"{'='*65}")

    folds = build_walk_forward_folds(train_df)
    fold_maes         = []
    fold_best_epochs  = []
    fold_percentiles  = []

    for fold_k, (fold_train_groups, fold_val_groups) in enumerate(folds):
        log(f"\n-- Fold {fold_k + 1}/{WF_NUM_FOLDS} --")
        fold_train_ds = DroughtDataset(fold_train_groups, scaler=scaler)
        fold_val_ds   = DroughtDataset(fold_val_groups,   scaler=scaler)
        log(f"  Train seqs: {len(fold_train_ds):,}  |  Val seqs: {len(fold_val_ds):,}")

        try:
            fold_loader_tr  = _make_loader(fold_train_ds, shuffle=True)
            fold_loader_val = _make_loader(fold_val_ds,   shuffle=False)
        except Exception:
            fold_loader_tr  = DataLoader(fold_train_ds, batch_size=BATCH_SIZE,
                                         shuffle=True,  num_workers=0, pin_memory=USE_AMP)
            fold_loader_val = DataLoader(fold_val_ds,   batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0, pin_memory=USE_AMP)

        fold_model = make_model(input_size)
        fold_ckpt  = os.path.join(MODELS_DIR, f"fold_{fold_k}_best.pt")

        best_fold_mae, best_fold_epoch, _ = train_model(
            fold_model,
            fold_loader_tr,
            fold_loader_val,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            ckpt_path=fold_ckpt,
            log=log,
            show_header=True,
        )

        fold_maes.append(best_fold_mae)
        fold_best_epochs.append(best_fold_epoch)
        log(f"  Fold {fold_k+1} Best Val MAE: {best_fold_mae:.4f}  |  Best Epoch: {best_fold_epoch}")

        # --- Diagnostic Hook: load best checkpoint and check prediction distribution ---
        log(f"\n  [Fold {fold_k+1} Prediction Percentiles – best checkpoint]")
        if os.path.exists(fold_ckpt) and best_fold_mae < float("inf"):
            fold_model.load_state_dict(torch.load(fold_ckpt, map_location=DEVICE))
            pct = eval_prediction_percentiles(fold_model, fold_loader_val, DEVICE, log)
        else:
            log("  [Skip] No valid checkpoint saved for this fold (training produced NaN).")
            pct = {}
        fold_percentiles.append(pct)

    avg_val_mae     = float(np.mean(fold_maes))
    mean_best_epoch = float(np.mean(fold_best_epochs))
    TARGET_EPOCHS   = max(1, int(round(mean_best_epoch * 1.05)))

    log(f"\n{'='*65}")
    log(f"Fold MAEs         : {[f'{m:.4f}' for m in fold_maes]}")
    log(f"Average_Val_MAE   : {avg_val_mae:.4f}")
    log(f"Fold Best Epochs  : {fold_best_epochs}")
    log(f"Mean Best Epoch   : {mean_best_epoch:.1f}")
    log(f"TARGET_EPOCHS (v7): {TARGET_EPOCHS}  (mean * 1.05, rounded)")
    log(f"\nFold Prediction Percentiles Summary:")
    for i, p in enumerate(fold_percentiles):
        if p:
            log(f"  Fold {i+1}: p50={p.get('p50',float('nan')):.3f}  "
                f"p95={p.get('p95',float('nan')):.3f}  "
                f"p99={p.get('p99',float('nan')):.3f}  "
                f"max={p.get('max',float('nan')):.3f}")
    log(f"{'='*65}")

    # -- 6. Blind full retraining on 100% training data (v7) -------------------
    log(f"\n{'='*65}")
    log(f"Blind Full Retraining  (TARGET_EPOCHS={TARGET_EPOCHS}, 100% data, v7)")
    log(f"Loss      : Continuous Smooth Loss  W_i = 1.0 + (y_i/5)^2 * 3.0")
    log(f"Scheduler : CosineAnnealingLR(T_max={TARGET_EPOCHS})")
    log(f"AMP       : {USE_AMP}  |  batch_size={BATCH_SIZE}")
    log("No validation split. No early stopping.")
    log(f"{'='*65}")

    # Free GPU & CPU memory from fold models/datasets before building full dataset
    torch.cuda.empty_cache()

    full_train_groups = build_full_train_groups(train_df)
    full_train_ds     = DroughtDataset(full_train_groups, scaler=scaler)
    log(f"  Full train sequences: {len(full_train_ds):,}")

    try:
        full_train_loader = DataLoader(
            full_train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
        )
    except Exception:
        full_train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE,
                                       shuffle=True, num_workers=0, pin_memory=USE_AMP)

    final_model = make_model(input_size)
    log("\n" + final_model.architecture_summary(input_size))

    v7_ckpt = os.path.join(MODELS_DIR, "final_model_v7.pt")

    final_history = train_model_blind(
        final_model, full_train_loader,
        num_epochs=TARGET_EPOCHS,
        ckpt_path=v7_ckpt,
        log=log,
    )

    log(f"Blind training complete. Epochs trained : {TARGET_EPOCHS}")
    log(f"Training time so far: {(time.time() - t0):.1f}s")

    # Save training history CSV
    hist_df = pd.DataFrame(final_history, columns=["epoch", "train_loss"])
    hist_df.to_csv(os.path.join(MODELS_DIR, "training_history.csv"), index=False)

    # -- 7. Inference on test set ----------------------------------------------
    log("\nRunning inference on test set ...")
    final_model.load_state_dict(torch.load(v7_ckpt, map_location=DEVICE))
    final_model.eval()

    predictions = {}   # region_id -> np.array of shape (5,)

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

        with torch.no_grad():
            with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                pred = final_model(X_tensor).squeeze(0).cpu().float().numpy()

        # v7: Safety clip – model is now unbounded; clip to Kaggle-valid range
        pred = np.clip(pred, 0.0, 5.0)

        predictions[region_id] = pred

    # -- 8. Submission-level prediction diagnostics ----------------------------
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
        log("  *** WARNING: Submission p99 < 2.0 — model collapse still present! ***")
    else:
        log("  ✓ Submission p99 >= 2.0 — model collapse is BROKEN.")

    # -- 9. Format & save submission.csv ---------------------------------------
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

    # -- 10. Sanity checks ------------------------------------------------------
    assert len(submission) == 2248, (
        f"Expected 2248 rows, got {len(submission)}"
    )
    assert list(submission.columns) == [
        "region_id", "pred_week1", "pred_week2",
        "pred_week3", "pred_week4", "pred_week5",
    ], f"Unexpected columns: {list(submission.columns)}"
    log("  ✓ Submission assertion passed: 2248 rows, 6 columns.")

    test_regions  = set(test_df["region_id"].unique())
    train_regions = set(train_df["region_id"].unique())
    assert test_regions == train_regions, (
        f"Region mismatch: train={len(train_regions)}, test={len(test_regions)}"
    )
    log("  ✓ Train/test regions match (2248).")

    log(f"  submission.csv → {sub_path}")
    log(f"  Rows (excl. header): {len(submission)}")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    # Save full training log
    log(f"\nTotal elapsed: {(time.time() - t0):.1f}s")
    log_path = os.path.join(ROOT, "_training_log_7th.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nTraining log saved -> {log_path}")

    return {
        "fold_maes":         fold_maes,
        "fold_best_epochs":  fold_best_epochs,
        "avg_val_mae":       avg_val_mae,
        "target_epochs":     TARGET_EPOCHS,
        "input_size":        input_size,
        "submission":        submission,
        "final_history":     final_history,
        "fold_percentiles":  fold_percentiles,
        "sub_p99":           p99,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
