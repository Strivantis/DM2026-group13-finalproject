"""
train.py -- Drought Score Forecasting Pipeline (v16 – Dilated TCN + Gap-Gating + Zero-Suppression)
===================================================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv                  -- Kaggle submission (2248 rows + header)
    models/single_fold_best.pt      -- Best weights for the single fold
    models/scaler.pkl               -- StandardScaler (fit on full single-fold train set)
    _training_log_16th.txt          -- Full console log

Key improvements (v16)
-----------------------
  1. Dilated TCN Stream + Confidence Gap-Gating (model.py):
       - CNN replaced by 1D Dilated TCN (d=1,2,4; receptive field=29 weeks).
       - Confidence Gap-Gate: G = Sigmoid(Linear(1,1)(gap)) applied to
         lstm_context and tcn_context, suppressing both when gap is large.
       - BiLSTM hidden downsized 256 → 128 (effective output 512 → 256).
       - branch_in: 690 → 434 (256+128+32+2+16).

  2. Focal Loss (Branch A):
       - BCEWithLogitsLoss replaced by custom Focal Loss (γ=2.0).
       - Forces the model to focus on hard active-drought boundary samples
         rather than trivially classifying the zero-majority class.

  3. Masked Regression Burn-in (Branch B, epochs 1–20):
       - During the initial 20-epoch burn-in, Loss_B is computed ONLY on
         samples where target > 0.0  (active drought timesteps).
       - Zero-target samples contribute 0 gradient to the regression head,
         preventing the model from optimising towards zero as an easy attractor.
       - After epoch 20, full unmasked regression loss is restored.

  4. Maintained V15 Training Dynamics:
       - BATCH_SIZE=512  (OOM fallback to 256)
       - Peak LR=1e-3, Manual Warm-up Epochs 1-5
       - ReduceLROnPlateau after Epoch 5
       - NUM_EPOCHS=200, PATIENCE=35
       - SingleFold Data-Maximizing CV

  5. Feature count updated: 37 → 29
       - All 8w and 13w rolling features removed (domain-shift mitigation).
       - Input assertion updated to expect 29 features.

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
    build_single_fold,
    build_gap_replay_folds,       # kept for backward compat; not used directly
    build_walk_forward_folds,     # kept for backward compat; not used directly
    compute_actual_gaps,
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
# Reproducibility
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

HIDDEN_SIZE   = 128           # v16: BiLSTM hidden per direction (was 256 in v15)
                               # Effective output = HIDDEN_SIZE * 2 = 256
NUM_LAYERS    = 3
DROPOUT       = 0.4
LEARNING_RATE = 1e-3          # peak LR
WEIGHT_DECAY  = 1e-2
BATCH_SIZE    = 512           # OOM fallback=256
NUM_EPOCHS    = 200
PATIENCE      = 35            # deep convergence on maximized single fold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# AMP: only enabled for CUDA
USE_AMP = DEVICE.type == "cuda"

# GradScaler for AMP (no-op on CPU)
_scaler = GradScaler(device="cuda", enabled=USE_AMP)

# ---------------------------------------------------------------------------
# Loss Criteria
# ---------------------------------------------------------------------------
_l1_criterion = nn.L1Loss()


# ---------------------------------------------------------------------------
# Focal Loss  (v16 – replaces BCEWithLogitsLoss for Branch A)
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Binary Focal Loss with logits.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logit) when target=1, and p_t = 1 - sigmoid(logit) when
    target=0.

    γ=2.0 down-weights easy negatives (zero-drought weeks) and concentrates
    gradient mass on hard active-drought boundary samples.

    References
    ----------
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    # Numerically stable via F.binary_cross_entropy_with_logits
    bce_loss  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t       = torch.exp(-bce_loss)                   # exp(-BCE) = p_t
    focal_wgt = (1.0 - p_t) ** gamma                   # (B, H)
    return (focal_wgt * bce_loss).mean()


# ---------------------------------------------------------------------------
# Masked Regression Loss  (v16 Burn-in – Branch B, epochs 1-20)
# ---------------------------------------------------------------------------
def masked_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Zero-Inflation Masked Regression Loss (v16 burn-in).

    Computes the weighted Smooth L1 loss ONLY on samples where target > 0.
    Zero-target samples have their loss masked to 0.0.

    During the burn-in phase (epochs 1-20), this forces the regression head to
    learn droughts exclusively from active-drought samples, avoiding the
    trivial zero-attractor solution.

    Returns 0-tensor (requires_grad=True to allow backward()) when no active
    samples exist in the batch.
    """
    active_mask = target > 0.0    # (B, H) boolean
    n_active    = active_mask.sum().item()

    if n_active == 0:
        # Degenerate batch: all zeros — return a zero loss that still backprops
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_active   = pred[active_mask]      # (n_active,)
    target_active = target[active_mask]    # (n_active,)

    element_loss = F.smooth_l1_loss(pred_active, target_active, reduction="none")
    weight = 1.0 + (target_active / 5.0) ** 2 * 3.0
    return (weight * element_loss).mean()


# ---------------------------------------------------------------------------
# Full Continuous Smooth Loss  (v7 – retained, used post burn-in)
# ---------------------------------------------------------------------------
def continuous_smooth_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss_B (post burn-in): Continuous quadratically-weighted Smooth L1 / Huber loss.
    Applied to the full target tensor (including zeros).
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
    v16 Zero-Inflation Suppression Loss Schedule.

    Burn-in Phase  (epoch <= 20):
        Loss = Masked Regression Loss  (Branch B, active samples ONLY)
        Regression is computed ONLY where target > 0.0.
        Zero-target samples are masked out (loss contribution = 0).
        Branch A (Focal Loss) is NOT included during burn-in.

    Post Burn-in   (epoch > 20):
        Loss = continuous_smooth_loss(full) + 0.1 * focal_loss(γ=2.0)
        Full unmasked regression is restored.
        Focal Loss replaces BCEWithLogitsLoss for sharper boundary learning.
    """
    if epoch <= 20:
        # Burn-in: masked regression on active drought samples only
        return masked_regression_loss(final_output, target)
    else:
        binary_target = (target > 0.0).float()
        loss_b = continuous_smooth_loss(final_output, target)
        loss_a = focal_loss(logits_output, binary_target, gamma=2.0)
        return loss_b + 0.1 * loss_a


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, epoch: int):
    """
    One training epoch -- v16: unpack (X, y, target_time, gap_size) from loader.
    """
    model.train()
    total_loss, n = 0.0, 0
    for X, y, target_time, gap_size in loader:
        X, y = X.to(device), y.to(device)
        target_time = target_time.to(device)
        gap_size    = gap_size.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=USE_AMP):
            final_output, logits_output = model(X, target_time, gap_size)
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
    v16: unpack (X, y, target_time, gap_size) from loader.
    """
    model.eval()
    total_mae, n = 0.0, 0
    for X, y, target_time, gap_size in loader:
        X, y = X.to(device), y.to(device)
        target_time = target_time.to(device)
        gap_size    = gap_size.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            final_output, _ = model(X, target_time, gap_size)
        total_mae += _l1_criterion(final_output, y).item() * X.size(0)
        n += X.size(0)
    return total_mae / n if n > 0 else float("inf")


@torch.no_grad()
def eval_prediction_percentiles(model, loader, device, log) -> dict:
    """
    Diagnostic hook: collect all final_output predictions and log percentile stats.
    v16: unpack (X, y, target_time, gap_size) from loader.
    """
    model.eval()
    all_preds = []
    for X, y, target_time, gap_size in loader:
        X           = X.to(device)
        target_time = target_time.to(device)
        gap_size    = gap_size.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            final_output, _ = model(X, target_time, gap_size)
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
        log("    *** WARNING: p99 < 2.0 -- model is still evading extremes (zero-collapse)! ***")
    else:
        log("    v p99 >= 2.0 -- model is predicting away from zero-collapse.")

    return {"p50": p50, "p90": p90, "p95": p95, "p99": p99, "max": pmax}


def make_model(input_size: int) -> DroughtLSTM:
    return DroughtLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,    # v16: 128 (was 256)
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
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


# ---------------------------------------------------------------------------
# Single training run with early stopping
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

    v16 Training:
      - Manual LR warm-up epochs 1-5: linear ramp 1e-5 -> 1e-3.
      - ReduceLROnPlateau only active from epoch 6+.
      - v16 Zero-Inflation Loss Schedule:
          Epochs 1-20 (burn-in): Masked Regression (active samples only).
          Epochs 21+:  continuous_smooth_loss(full) + 0.1 * FocalLoss(γ=2.0).
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
        # --- v15/v16: Manual LR Warm-up (Epochs 1-5) ---
        # Linear ramp from 1e-5 to 1e-3 over the first 5 epochs.
        if epoch <= 5:
            lr = 1e-5 + (1e-3 - 1e-5) * ((epoch - 1) / 4.0)
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr

        train_loss = train_epoch(model, train_loader, optimiser, DEVICE, epoch)
        val_mae    = eval_mae(model, val_loader, DEVICE)

        # Only step ReduceLROnPlateau after warm-up is complete
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

        # Loss mode annotation for logging
        loss_mode = "MaskedReg" if epoch <= 20 else "Full+Focal"
        marker    = " *" if improved else ""
        log(
            f"{epoch:>6}  {train_loss:>11.4f}"
            f"  {val_mae:>10.4f}  {curr_lr:>10.2e}  [{loss_mode}]{marker}"
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
    Inject target encoding features into each (group_df, i_min, i_max, gap) tuple.
    TE values are CONSTANT within a region (static features).

    v14/v15/v16: handles 4-tuple (group, i_min, i_max, actual_gap).
    """
    result = []
    for entry in groups:
        if len(entry) == 4:
            group, i_min, i_max, actual_gap = entry
        else:
            group, i_min, i_max = entry
            actual_gap = GAP_WEEKS
        g = group.copy()
        rid = g["region_id"].iloc[0]
        mean_s, zero_p = te_map.get(rid, (global_mean, global_zero_prob))
        g["region_mean_score"] = np.float32(mean_s)
        g["region_zero_prob"]  = np.float32(zero_p)
        result.append((g, i_min, i_max, actual_gap))
    return result


def _merge_te_to_df(
    df: pd.DataFrame,
    te_map: dict,
    global_mean: float,
    global_zero_prob: float,
) -> pd.DataFrame:
    """
    Add region_mean_score and region_zero_prob columns to a DataFrame.
    Used for train_df (scaler fitting) and test_df (inference).
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
# Inference helper  (v15/v16: single model, no fold ensembling)
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_test_set(model, test_df, feat_cols, scaler, actual_gaps, log) -> dict:
    """
    Run inference for every region in test_df using the given model.

    v16: single model (single_fold_best.pt), no ensembling.
    Retained v14 target_time + gap_size construction.
    Submission clipped to [0, 5].

    Returns
    -------
    predictions : dict  region_id -> np.array shape (5,)  [clipped to [0,5]]
    """
    model.eval()
    predictions = {}

    for region_id, group in test_df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)

        # Pad if fewer than WINDOW_SIZE rows exist
        if n < WINDOW_SIZE:
            pad_rows = pd.concat(
                [group.iloc[[0]]] * (WINDOW_SIZE - n) + [group],
                ignore_index=True,
            )
            group = pad_rows
            n = len(group)

        # The input window is the last WINDOW_SIZE rows BEFORE the test target.
        window_df = group.iloc[:WINDOW_SIZE]
        X = window_df[feat_cols].values.astype(np.float32)
        X = scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        # (1, WINDOW_SIZE, F)

        # target_time: week_sin/cos of the 5 test target weeks
        target_rows = group.iloc[-HORIZON:]
        if "week_sin" in group.columns and "week_cos" in group.columns:
            tt = np.stack([
                target_rows["week_sin"].values.astype(np.float32),
                target_rows["week_cos"].values.astype(np.float32),
            ], axis=-1)   # (5, 2)
        else:
            tt = np.zeros((HORIZON, 2), dtype=np.float32)
        target_time_tensor = torch.tensor(tt, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        # (1, 5, 2)

        # gap_size: actual_gap for this region
        actual_gap = actual_gaps.get(region_id, GAP_WEEKS)
        gap_tensor = torch.tensor(
            [[actual_gap / 100.0]], dtype=torch.float32
        ).to(DEVICE)
        # (1, 1)

        with autocast(device_type=DEVICE.type, enabled=USE_AMP):
            final_output, _ = model(X_tensor, target_time_tensor, gap_tensor)

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
    log("=" * 80)
    log("Drought Forecasting Pipeline  v16")
    log("Dilated TCN + Confidence Gap-Gating + Zero-Inflation Loss Suppression")
    log("=" * 80)
    log("Architecture : Dual-Stream Dilated TCN + BiLSTM MTL  [v16]")
    log(f"  == LSTM Stream ==")
    log(f"  BiLSTM     : hidden_size={HIDDEN_SIZE}/dir -> {HIDDEN_SIZE*2} effective  "
        f"[v16: hidden 256→128]")
    log(f"  Layers     : {NUM_LAYERS}")
    log(f"  lstm_context (B,{HIDDEN_SIZE*2})  ×  G  ->  gated (B,{HIDDEN_SIZE*2}) "
        f"-> expand (B,5,{HIDDEN_SIZE*2})")
    log(f"  == Dilated TCN Stream (v16 NEW) ==")
    log(f"  Input      : (B, 29, 13)  [features × window]")
    log(f"  TCN Layer 1: Conv1d(29→128, k=3, d=1, pad=1) + GELU -> (B,128,13)")
    log(f"  TCN Layer 2: Conv1d(128→128, k=3, d=2, pad=2) + GELU -> (B,128,13)")
    log(f"  TCN Layer 3: Conv1d(128→128, k=3, d=4, pad=4) + GELU -> (B,128,13)")
    log(f"  Receptive field: 29 weeks (full window coverage)")
    log(f"  GlobalPool : AdaptiveAvgPool1d(1) -> squeeze -> (B,128)")
    log(f"  tcn_context (B,128)  ×  G  ->  gated (B,128) -> expand (B,5,128)")
    log(f"  == Confidence Gap-Gating (v16 NEW) ==")
    log(f"  G = Sigmoid(Linear(1,1)(gap_size))  ->  scalar ∈ [0,1]  (B,1)")
    log(f"  Suppresses both streams proportionally to deployment gap uncertainty")
    log(f"  == Fusion ==")
    log(f"  horizon_ids [0-4] -> Embedding(5,32) -> (B,5,32)  [v11: learnable]")
    log(f"  target_time (B,5,2) -- week_sin/cos of 5 future target weeks  [v14]")
    log(f"  gap_size (B,1) -> Linear(1,16) -> (B,16) -> expand (B,5,16)  [v14]")
    log(f"  encoded_state (B,5,{HIDDEN_SIZE*2+128+32+2+16})  "
        f"[v16: 256+128+32+2+16=434]")
    log(f"  Branch A   : Linear({HIDDEN_SIZE*2+128+32+2+16}->128) -> GELU -> "
        f"Dropout(0.2) -> Linear(128->1)  [v16]")
    log( "               P(drought) logits -- sigmoid() inline in forward()")
    log(f"  Branch B   : Linear({HIDDEN_SIZE*2+128+32+2+16}->128) -> GELU -> "
        f"Dropout(0.2) -> Linear(128->1)  [v16]")
    log( "               Severity >= 0  via Softplus()")
    log( "  Output     : final = sigmoid(logits_A) x Branch_B  (Expected Severity)")
    log( "CV Strategy  : [v15] Single-Fold Data-Maximizing (100% regions, last 5 wks as Val_Y)")
    log(f"  WINDOW_SIZE: {WINDOW_SIZE} weeks  [v16: 13w cap at test-set horizon]")
    log( "  actual_gap : computed from real calendar distance (train last -> test first)")
    log( "  Val_Y      : last 5 rows per region (fixed, no rotation)")
    log( "  Train      : ALL historical sliding windows ending before Val_Y (data max)")
    log( "  Scaler     : StandardScaler fit on entire maximized training split  [v15]")
    log( "  Fallback   : ZeroDataWaste (shrink gap if history too short)")
    log( "Loss         : [v16] Zero-Inflation Suppression Schedule")
    log( "  Epoch 1-20 : Loss = Masked Regression ONLY  (active samples, target > 0)")
    log( "               Zero-target samples receive 0 gradient in Branch B")
    log( "  Epoch 21+  : Loss = Full Smooth L1 + 0.1 * FocalLoss(γ=2.0)")
    log( "  Loss_B     : Continuous Smooth L1  W_i = 1.0 + (y_i/5)^2 * 3.0")
    log( "  Loss_A     : FocalLoss(γ=2.0)  [v16: replaces BCEWithLogitsLoss]")
    log(f"LR Schedule  : [v15] Manual Warm-up Epochs 1-5: 1e-5 -> 1e-3 (linear)")
    log( "               Epoch 6+: ReduceLROnPlateau (factor=0.5, patience=10)")
    log( "Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]")
    log( "Pooling      : Temporal Attention [v8: retained; v16: hidden=128]")
    log( "LayerNorm    : LayerNorm(input_size) before LSTM  [v7: retained]")
    log( "Features     : 29  (11 weather + 2 cyclic + 3 rolling-4w + 4 lag1 "
         "+ 4 lag2 + 3 drought-4w + 2 TE)  [v16: removed 8w/13w]")
    log( "  Cyclic     : week_sin, week_cos  [v9: retained]")
    log( "  TE         : region_mean_score, region_zero_prob  [v9: retained]")
    log( "Scaler       : Single StandardScaler fit on maximized train split  [v15]")
    log( "Checkpoint   : single_fold_best.pt  [v15: replaces fold_k_best.pt]")
    log( "Inference    : Single model (no fold ensembling)  [v15]")
    log( "OOM Guard    : try-except RuntimeError -> fallback BATCH_SIZE=256  [v12]")
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

    # -- 1b. Validate v16 features in processed CSV ----------------------------
    log("\n[v16 Feature Validation]")
    assert "week_sin" in train_raw.columns, "week_sin missing -- run preprocess.py first"
    assert "week_cos" in train_raw.columns, "week_cos missing -- run preprocess.py first"
    assert "month" not in train_raw.columns, "month still present -- should be dropped by preprocess"
    assert "week_of_year" not in train_raw.columns, "week_of_year still present -- should be dropped"
    log("  v week_sin, week_cos present  |  month, week_of_year correctly absent.")

    # Verify no 8w/13w domain-shift features
    bad_cols = [c for c in train_raw.columns if ("8w" in c or "13w" in c)]
    if bad_cols:
        log(f"  *** WARNING: 8w/13w features found: {bad_cols}")
    else:
        log("  v No 8w/13w rolling features (domain-shift columns absent).")

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
    log(f"  [v16] Zero-inflation: {zero_frac:.2%} of training scores == 0.0")
    log(f"  [v16] Masked Regression (burn-in) targets {1 - zero_frac:.2%} active samples")
    for thresh in [1.0, 2.0, 3.0, 4.0]:
        frac = (all_scores > thresh).mean() * 100
        log(f"  score > {thresh:.1f}: {frac:.2f}%  [{int((all_scores > thresh).sum()):,} samples]")

    # -- 4b. [v14/v15/v16] Compute per-region actual deployment gap ---------------
    log("\n[v16] Computing per-region actual deployment gaps ...")
    actual_gaps = compute_actual_gaps(train_raw, test_raw)
    gap_values  = np.array(list(actual_gaps.values()))
    log(f"  Regions with gap computed : {len(actual_gaps)}")
    log(f"  Gap stats: min={gap_values.min():.0f}  "
        f"p25={np.percentile(gap_values,25):.0f}  "
        f"median={np.median(gap_values):.0f}  "
        f"p75={np.percentile(gap_values,75):.0f}  "
        f"max={gap_values.max():.0f}  "
        f"mean={gap_values.mean():.1f}")
    from collections import Counter
    gap_counter = Counter(gap_values.astype(int).tolist())
    top5 = gap_counter.most_common(5)
    log(f"  Top-5 most common gaps: {top5}")
    log(f"  [v16] Confidence Gap-Gate will learn G=Sigmoid(Linear(normalized_gap))")

    # -- 5. Full-train Target Encoding (for scaler fitting + test inference) --
    log("\n[v9] Computing full-train Target Encoding statistics ...")
    te_map_full, global_mean_te, global_zero_prob_te = _compute_te_stats(train_df)
    log(f"  Regions with TE stats : {len(te_map_full)}")
    log(f"  Global region_mean_score : {global_mean_te:.4f}")
    log(f"  Global region_zero_prob  : {global_zero_prob_te:.4f}")

    # Add full-train TE to train_df (used for scaler fitting and single-fold splits)
    train_df = _merge_te_to_df(train_df, te_map_full, global_mean_te, global_zero_prob_te)
    log("  v region_mean_score, region_zero_prob added to train_df (full-train stats)")

    # -- 6. Determine feature columns and input_size ---------------------------
    log("\n[v16] Determining feature columns ...")
    feat_cols  = [c for c in FEATURE_COLS if c in train_df.columns]
    input_size = len(feat_cols)
    log(f"  Input features ({input_size}): {feat_cols}")

    assert input_size == 29, (
        f"Expected 29 features (27 base + 2 TE), got {input_size}. "
        f"v16 removed all 8w/13w rolling features — check that preprocess.py "
        f"was run and TE cols are present."
    )

    # -- 7. [v15/v16] Single-Fold Data-Maximizing Split -----------------------
    log(f"\n{'='*80}")
    log(f"Single-Fold Data-Maximizing Training  [v15/v16]")
    log(f"[v16] Strategy: 100% of regions  →  Val_Y = last 5 weeks (fixed)")
    log(f"[v16] Train    : ALL historical sliding windows before Val_Y (data-maximizing)")
    log(f"[v16] Gap-Replay: actual_gap replaces fixed GAP_WEEKS={GAP_WEEKS}")
    log(f"[v16] Window Size: {WINDOW_SIZE} weeks")
    log(f"Train loss  : v16 Zero-Suppression: Masked Reg (ep 1-20) | Full + Focal (ep 21+)")
    log(f"Val metric  : pure MAE  (unweighted, Kaggle-aligned)")
    log(f"TE strategy : full-train stats (leakage-free TE inserted before split)")
    log(f"Scaler      : Single StandardScaler fit on maximized train features  [v15]")
    log(f"Checkpoint  : single_fold_best.pt")
    log(f"AMP : {USE_AMP}  |  batch_size={BATCH_SIZE} (OOM fallback: 256)  |  num_workers=8")
    log(f"{'='*80}")

    # Build single fold
    raw_train_groups, raw_val_groups = build_single_fold(train_df, actual_gaps)
    log(f"\n  [v16 Single-Fold] Raw train groups: {len(raw_train_groups)}  "
        f"|  Raw val groups: {len(raw_val_groups)}")

    # Augment both with full-train TE
    aug_train_groups = _augment_groups_with_te(
        raw_train_groups, te_map_full, global_mean_te, global_zero_prob_te
    )
    aug_val_groups = _augment_groups_with_te(
        raw_val_groups, te_map_full, global_mean_te, global_zero_prob_te
    )

    # -- 7b. [v15/v16] Fit single StandardScaler on maximized train split ------
    log(f"\n  [v16 Scaler] Fitting StandardScaler on maximized single-fold train features ...")
    single_scaler = StandardScaler()

    train_feat_parts = []
    for entry in aug_train_groups:
        if len(entry) == 4:
            group, i_min, i_max, eff_gap = entry
        else:
            group, i_min, i_max = entry
        local_feat_cols = [c for c in FEATURE_COLS if c in group.columns]
        train_feat_parts.append(group[local_feat_cols].values.astype(np.float32))

    if train_feat_parts:
        train_feat_matrix = np.concatenate(train_feat_parts, axis=0)
        single_scaler.fit(train_feat_matrix)
        log(f"  [v16 Scaler] Fit on {len(train_feat_matrix):,} rows "
            f"({train_feat_matrix.shape[1]} features)")
    else:
        log(f"  [v16 Scaler] WARNING: No train rows found — using identity scaler.")

    # Create datasets with the single scaler
    train_ds = DroughtDataset(aug_train_groups, scaler=single_scaler)
    val_ds   = DroughtDataset(aug_val_groups,   scaler=single_scaler)
    log(f"  Train sequences: {len(train_ds):,}  |  Val sequences: {len(val_ds):,}")

    # -- 7c. Build model and checkpoint path ----------------------------------
    model     = make_model(input_size)
    ckpt_path = os.path.join(MODELS_DIR, "single_fold_best.pt")

    # -- 7d. OOM-Protected Training -------------------------------------------
    batch_size_to_use = BATCH_SIZE

    def _run_training(bs):
        """Helper: build loaders + optionally verify shapes + train."""
        try:
            loader_tr  = _make_loader(train_ds, shuffle=True,  batch_size=bs)
            loader_val = _make_loader(val_ds,   shuffle=False, batch_size=bs)
        except Exception:
            loader_tr  = DataLoader(train_ds, batch_size=bs,
                                    shuffle=True,  num_workers=0, pin_memory=USE_AMP)
            loader_val = DataLoader(val_ds,   batch_size=bs,
                                    shuffle=False, num_workers=0, pin_memory=USE_AMP)

        # Tensor shape verification
        first_X, first_y, first_tt, first_gs = next(iter(loader_tr))
        log(f"\n  [v16 Tensor Shape Verification] First Batch:")
        log(f"    X shape          : {tuple(first_X.shape)}"
            f"  ->  (Batch, Seq={first_X.shape[1]}, Features={first_X.shape[2]})")
        log(f"    y shape          : {tuple(first_y.shape)}"
            f"  ->  (Batch, Horizon={first_y.shape[1]})")
        log(f"    target_time shape: {tuple(first_tt.shape)}"
            f"  ->  (Batch, Horizon=5, 2=week_sin/cos)  [v14]")
        log(f"    gap_size shape   : {tuple(first_gs.shape)}"
            f"  ->  (Batch, 1=normalised_gap)  [v14]")
        log(f"    Expected: Features={input_size}, Window={WINDOW_SIZE}, Horizon={HORIZON}")
        assert first_X.shape[1] == WINDOW_SIZE, \
            f"Seq mismatch: got {first_X.shape[1]}, expected {WINDOW_SIZE}"
        assert first_X.shape[2] == input_size, \
            f"Feature mismatch: got {first_X.shape[2]}, expected {input_size}"
        assert first_y.shape[1] == HORIZON, \
            f"Horizon mismatch: got {first_y.shape[1]}, expected {HORIZON}"
        assert first_tt.shape[1] == HORIZON and first_tt.shape[2] == 2, \
            f"target_time mismatch: got {tuple(first_tt.shape)}, expected (B,5,2)"
        assert first_gs.shape[1] == 1, \
            f"gap_size mismatch: got {tuple(first_gs.shape)}, expected (B,1)"
        log(f"    v Shape assertion PASSED (v16: target_time and gap_size verified).\n")
        del first_X, first_y, first_tt, first_gs

        mae, epoch, hist = train_model(
            model, loader_tr, loader_val,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            ckpt_path=ckpt_path,
            log=log,
            show_header=True,
        )
        return mae, epoch, hist, loader_tr, loader_val

    try:
        best_mae, best_epoch_num, _, loader_tr, loader_val = \
            _run_training(batch_size_to_use)

    except RuntimeError as oom_err:
        if "out of memory" in str(oom_err).lower():
            log(f"\n  [v16 OOM] CUDA out-of-memory at batch_size={batch_size_to_use}.")
            log(f"  [v16 OOM] Flushing CUDA cache and retrying with batch_size=256 ...")
            torch.cuda.empty_cache()

            # Re-instantiate model to clear partially-allocated weights
            model = make_model(input_size)
            batch_size_to_use = 256

            best_mae, best_epoch_num, _, loader_tr, loader_val = \
                _run_training(batch_size_to_use)
        else:
            raise

    log(f"\n  [v16 Single-Fold] Best Val MAE : {best_mae:.4f}")
    log(f"  [v16 Single-Fold] Best Epoch  : {best_epoch_num}")
    log(f"  Checkpoint saved -> {ckpt_path}")

    # --- Diagnostic Hook: prediction distribution ---
    log(f"\n  [v16 Single-Fold Prediction Percentiles -- best checkpoint]")
    if os.path.exists(ckpt_path) and best_mae < float("inf"):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        single_fold_pct = eval_prediction_percentiles(model, loader_val, DEVICE, log)
    else:
        log("  [Skip] No valid checkpoint saved (training produced NaN).")
        single_fold_pct = {}

    # Free GPU memory
    del model, loader_tr, loader_val, train_ds, val_ds
    torch.cuda.empty_cache()

    log(f"\n{'='*80}")
    log(f"Single-Fold Best Val MAE   : {best_mae:.4f}")
    log(f"Single-Fold Best Epoch     : {best_epoch_num}")
    if single_fold_pct:
        log(f"Val Prediction Percentiles :"
            f"  p50={single_fold_pct.get('p50', float('nan')):.3f}"
            f"  p95={single_fold_pct.get('p95', float('nan')):.3f}"
            f"  p99={single_fold_pct.get('p99', float('nan')):.3f}"
            f"  max={single_fold_pct.get('max', float('nan')):.3f}")
    log(f"{'='*80}")

    # -- 8. Save single scaler ------------------------------------------------
    log(f"\n[v16] Saving single StandardScaler (fit on maximized train split) ...")
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(single_scaler, f)
    log(f"  Scaler saved -> {os.path.join(MODELS_DIR, 'scaler.pkl')}")

    # -- 8b. Prepare test_df with full-train TE for inference -----------------
    log(f"\n[v9] Preparing test_df with full-train Target Encoding ...")
    test_df = _merge_te_to_df(test_df, te_map_full, global_mean_te, global_zero_prob_te)
    log(f"  test_df after TE injection: {test_df.shape}")
    log(f"  v region_mean_score, region_zero_prob added to test_df (full-train stats)")

    # -- 9. Single-Model Inference on Test Set --------------------------------
    log(f"\n{'='*80}")
    log(f"Single-Model Inference  [v16 – single_fold_best.pt]")
    log(f"  Checkpoint: {ckpt_path}")
    log(f"  Scaler: single_fold_best scaler (fit on maximized train split)  [v15]")
    log(f"  [v14] target_time: week_sin/cos of 5 test target weeks")
    log(f"  [v14] gap_size: per-region actual_gap / 100.0")
    log(f"  [v16] Gap-Gating: G = Sigmoid(Linear(gap_size)) applied to both streams")
    log(f"  Safety clip: np.clip(pred, 0.0, 5.0)")
    log(f"{'='*80}")

    # Print architecture summary
    _tmp_model = make_model(input_size)
    log("\n" + _tmp_model.architecture_summary(input_size))
    del _tmp_model

    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}. Training may have failed.")

    inference_model = make_model(input_size)
    inference_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    inference_model.eval()

    predictions = predict_test_set(
        inference_model, test_df, feat_cols, single_scaler, actual_gaps, log
    )
    log(f"\n  Inference complete. Regions predicted: {len(predictions)}")

    del inference_model
    torch.cuda.empty_cache()

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
        log("  *** WARNING: Submission p99 < 2.0 -- model zero-collapse still present! ***")
    else:
        log("  v Submission p99 >= 2.0 -- zero-collapse suppression is working.")

    zero_pred_frac = (all_sub_preds < 0.05).mean()
    log(f"  Fraction of near-zero predictions (<0.05): {zero_pred_frac:.2%}")

    # -- 11. Format & save submission.csv -------------------------------------
    log("\nFormatting submission.csv ...")
    rows = []
    for region_id, preds in sorted(predictions.items()):
        rows.append({
            "region_id":  region_id,
            "pred_week1": float(np.clip(preds[0], 0, 5)),
            "pred_week2": float(np.clip(preds[1], 0, 5)),
            "pred_week3": float(np.clip(preds[2], 0, 5)),
            "pred_week4": float(np.clip(preds[3], 0, 5)),
            "pred_week5": float(np.clip(preds[4], 0, 5)),
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

    # Values in [0, 5] guaranteed by np.clip in predict_test_set + rows above
    assert submission[["pred_week1","pred_week2","pred_week3",
                        "pred_week4","pred_week5"]].max().max() <= 5.0 + 1e-6
    assert submission[["pred_week1","pred_week2","pred_week3",
                        "pred_week4","pred_week5"]].min().min() >= 0.0 - 1e-6
    log("  v All predictions in [0, 5]  (np.clip enforced).")

    log(f"  submission.csv -> {sub_path}")
    log(f"  Rows (excl. header): {len(submission)}")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    # Save full training log
    log(f"\nTotal elapsed: {(time.time() - t0):.1f}s")
    log_path = os.path.join(ROOT, "_training_log_16th.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nTraining log saved -> {log_path}")

    return {
        "best_mae":            best_mae,
        "best_epoch":          best_epoch_num,
        "input_size":          input_size,
        "submission":          submission,
        "single_fold_pct":     single_fold_pct,
        "sub_p99":             p99,
        "actual_gaps_summary": {
            "min":    int(gap_values.min()),
            "median": int(np.median(gap_values)),
            "max":    int(gap_values.max()),
            "mean":   float(gap_values.mean()),
        },
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
