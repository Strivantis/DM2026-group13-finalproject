"""
train.py -- Drought Score Forecasting Pipeline (v18 – Hard Thresholding + StratifiedGroupKFold + Loss Re-weighting)
====================================================================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv                  -- Kaggle submission (2248 rows + header, 5-fold ensemble average)
    models/fold_{k}_best.pt         -- Best weights per fold (k=0..4)
    models/scaler_fold_{k}.pkl      -- StandardScaler per fold (fit on fold's train regions only)
    _training_log_18th.txt          -- Full console log

Key improvements (v18)
-----------------------
  1. StratifiedGroupKFold CV (dataset.py build_region_group_cv_folds):
       - REPLACES GroupKFold with StratifiedGroupKFold(n_splits=5).
       - Stratification: 10 quantile bins of per-region historical mean score.
         pd.qcut(region_means, q=10, labels=False, duplicates='drop')
       - Every fold now contains an identical proportion of "perpetual desert"
         vs "frequent rainy" regions, eliminating the Fold 2 outlier variance.
       - groups=region_id unchanged (no cross-region leakage).
       - OOF Scaler/TE alignment fully preserved from v17.

  2. Inference Hard Thresholding Gate (model.py forward):
       - final_output now uses: torch.where(prob < 0.5, 0.0, prob * severity)
       - Completely terminates fractional noise (e.g., 0.3) for regions that
         should predict absolute zero, matching the 59.64% zero-inflation ceiling.
       - Raw loss branches (logits_output, severity_output) are UNAFFECTED;
         backward pass uses only hurdle_loss(logits, severity, y).
       - eval_mae() and early stopping now directly optimise for the
         hard-thresholded final_output -- fully Kaggle-metric aligned.

  3. Focal Loss Weight Re-calibration (hurdle_loss):
       - Total = 2.0 * Loss_A (FocalLoss) + 1.0 * Loss_B (Smooth L1)
       - Increased from 1.0 to 2.0 because the inference layer now relies on
         the classification head as a hard switch; minimising classification
         errors is paramount.

  4. Preserved Hyperparameters (from v17):
       - BATCH_SIZE=512 (OOM fallback to 256)
       - Peak LR=1e-3, Manual Warm-up Epochs 1-5
       - ReduceLROnPlateau after Epoch 5
       - NUM_EPOCHS=200, PATIENCE=35
       - WEIGHT_DECAY=1e-2
       - BiLSTM + Dilated TCN architecture unchanged

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
    build_region_group_cv_folds,
    build_single_fold,             # retained for backward compat; not called
    build_gap_replay_folds,        # retained for backward compat; not called
    build_walk_forward_folds,      # retained for backward compat; not called
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

N_FOLDS       = 5             # v18: 5-Fold Stratified Region Group CV
HIDDEN_SIZE   = 128           # v16: BiLSTM hidden per direction (was 256 in v15)
                               # Effective output = HIDDEN_SIZE * 2 = 256
NUM_LAYERS    = 3
DROPOUT       = 0.4
LEARNING_RATE = 1e-3          # peak LR
WEIGHT_DECAY  = 1e-2
BATCH_SIZE    = 512           # OOM fallback=256
NUM_EPOCHS    = 200
PATIENCE      = 35

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
# Focal Loss  (v16/v17/v18 – Branch A classification loss)
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

    gamma=2.0 down-weights easy negatives (zero-drought weeks) and concentrates
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
# Decoupled Hurdle Model Loss  (v18 – Loss_A weight increased to 2.0)
# ---------------------------------------------------------------------------
def hurdle_loss(
    logits_output: torch.Tensor,
    severity_output: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    v18 Decoupled Hurdle Model Loss.

    Branch A (Classification Head):
        FocalLoss(gamma=2.0) computed across ALL samples in the batch.
        binary_target = (target > 0.0).float()
        Loss weight: 2.0  [v18: increased from 1.0 -- classification is now a
                           hard switch in inference; errors are paramount]

    Branch B (Regression Head):
        Continuous weighted Smooth L1 Loss computed STRICTLY on samples
        where the true target > 0.0.
        mask = target > 0.0
        If mask.sum() > 0:  loss_b = mean(smooth_l1(sev[mask], tgt[mask]) * w[mask])
                             w_i = 1.0 + (tgt_i / 5.0)^2 * 3.0
        If all targets == 0: loss_b = 0.0
        Loss weight: 1.0

    Total Loss = 2.0 * Loss_A + 1.0 * Loss_B  [v18: was 1.0+1.0 in v17]

    This allows Branch B to specialise strictly in drought severity scales,
    while Branch A masters the zero-inflation boundary classification.
    The two branches receive mathematically isolated gradient flows.
    """
    # Branch A: FocalLoss across ALL samples
    binary_target = (target > 0.0).float()
    loss_a = focal_loss(logits_output, binary_target, gamma=2.0)

    # Branch B: Weighted Smooth L1 strictly on target > 0
    mask = target > 0.0
    if mask.sum() > 0:
        sev_masked = severity_output[mask]
        tgt_masked = target[mask]
        element_loss = F.smooth_l1_loss(sev_masked, tgt_masked, reduction="none")
        weight       = 1.0 + (tgt_masked / 5.0) ** 2 * 3.0
        loss_b       = (weight * element_loss).mean()
    else:
        # Degenerate batch: all zeros -- return zero loss that still allows backward
        loss_b = torch.tensor(0.0, device=target.device, requires_grad=True)

    # v18: classification loss weighted 2x to stabilise the hard-threshold gate
    return 2.0 * loss_a + 1.0 * loss_b


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    """
    One training epoch -- v17/v18: unpack (X, y, target_time, gap_size) from loader.
    Uses decoupled hurdle_loss(logits_output, severity_output, y).
    """
    model.train()
    total_loss, n = 0.0, 0
    for X, y, target_time, gap_size in loader:
        X, y = X.to(device), y.to(device)
        target_time = target_time.to(device)
        gap_size    = gap_size.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=USE_AMP):
            final_output, logits_output, severity_output = model(X, target_time, gap_size)
            loss = hurdle_loss(logits_output, severity_output, y)

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
    v18: final_output uses Hard Thresholding Gate (prob < 0.5 => 0.0), so
    early stopping now directly optimises for the thresholded output.
    """
    model.eval()
    total_mae, n = 0.0, 0
    for X, y, target_time, gap_size in loader:
        X, y = X.to(device), y.to(device)
        target_time = target_time.to(device)
        gap_size    = gap_size.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            final_output, _, _ = model(X, target_time, gap_size)
        total_mae += _l1_criterion(final_output, y).item() * X.size(0)
        n += X.size(0)
    return total_mae / n if n > 0 else float("inf")


@torch.no_grad()
def eval_prediction_percentiles(model, loader, device, log) -> dict:
    """
    Diagnostic hook: collect all final_output predictions and log percentile stats.
    v17/v18: model returns (final_output, logits_output, severity_output).
    """
    model.eval()
    all_preds = []
    for X, y, target_time, gap_size in loader:
        X           = X.to(device)
        target_time = target_time.to(device)
        gap_size    = gap_size.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            final_output, _, _ = model(X, target_time, gap_size)
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
        log("    *** WARNING: p99 < 2.0 -- model may still be evading extremes! ***")
    else:
        log("    v p99 >= 2.0 -- model is predicting away from zero-collapse.")

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

    v18 Training:
      - Manual LR warm-up epochs 1-5: linear ramp 1e-5 -> 1e-3.
      - ReduceLROnPlateau only active from epoch 6+.
      - v18 Decoupled Hurdle Loss (always active, no burn-in schedule):
          Loss = 2.0 * FocalLoss(all) + 1.0 * Weighted_Smooth_L1(target>0 only)
    Early stopping: pure L1Loss(final_output, y) -- strictly Kaggle-metric-aligned.
    Note: final_output uses Hard Thresholding Gate, so MAE is thresholded-aligned.

    Returns best_val_mae, best_epoch, history.
    """
    optimiser = make_optimiser(model)
    scheduler = make_scheduler(optimiser)

    if show_header:
        log("=" * 70)
        log(f"{'Epoch':>6} {'TrainLoss':>11} {'ValMAE':>10} {'LR':>10}  Loss")
        log("=" * 70)

    best_val_mae = float("inf")
    best_epoch   = 1
    no_improve   = 0
    history      = []

    for epoch in range(1, num_epochs + 1):
        # --- v18: Manual LR Warm-up (Epochs 1-5) ---
        # Linear ramp from 1e-5 to 1e-3 over the first 5 epochs.
        if epoch <= 5:
            lr = 1e-5 + (1e-3 - 1e-5) * ((epoch - 1) / 4.0)
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr

        train_loss = train_epoch(model, train_loader, optimiser, DEVICE)
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

        marker = " *" if improved else ""
        log(
            f"{epoch:>6}  {train_loss:>11.4f}"
            f"  {val_mae:>10.4f}  {curr_lr:>10.2e}  [HurdleDecoupled_v18]{marker}"
        )

        if no_improve >= patience:
            log(
                f"\n  Early stopping at epoch {epoch} "
                f"(no MAE improvement for {patience} epochs)."
            )
            break

    log("=" * 70)
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

    v17/v18: called per-fold on fold's train-region rows only (OOF TE alignment).

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

    v14/v15/v16/v17/v18: handles 4-tuple (group, i_min, i_max, actual_gap).
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
    Used for test_df (inference) with fold-specific TE.
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
# Inference helper  (v18: per-fold model + per-fold scaler + per-fold TE)
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_test_set(model, test_df, feat_cols, scaler, actual_gaps, log) -> dict:
    """
    Run inference for every region in test_df using the given model.

    v18: called once per fold. Predictions are averaged across all 5 folds
    in main() to produce the final submission.
    Hard Thresholding Gate is applied inside model.forward() -- predictions
    where P(drought) < 0.5 are already set to 0.0 before reaching here.
    Submission additionally clipped to [0, 5].

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
            # v18: model returns (final_output, logits_output, severity_output)
            # final_output already has Hard Thresholding applied inside forward()
            final_output, _, _ = model(X_tensor, target_time_tensor, gap_tensor)

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
    log("=" * 85)
    log("Drought Forecasting Pipeline  v18")
    log("Dilated TCN + Exponential Decay Gap-Gating + Hard Thresholding + Decoupled Hurdle Model")
    log("5-Fold Stratified Region Group CV  (StratifiedGroupKFold by region_id)")
    log("=" * 85)
    log("Architecture : Dual-Stream Dilated TCN + BiLSTM MTL  [v16, retained]")
    log(f"  == LSTM Stream ==")
    log(f"  BiLSTM     : hidden_size={HIDDEN_SIZE}/dir -> {HIDDEN_SIZE*2} effective")
    log(f"  Layers     : {NUM_LAYERS}")
    log(f"  lstm_context (B,{HIDDEN_SIZE*2})  x  G  ->  gated (B,{HIDDEN_SIZE*2}) "
        f"-> expand (B,5,{HIDDEN_SIZE*2})")
    log(f"  == Dilated TCN Stream (v16 - retained) ==")
    log(f"  Input      : (B, 29, 13)  [features x window]")
    log(f"  TCN Layer 1: Conv1d(29->128, k=3, d=1, pad=1) + GELU -> (B,128,13)")
    log(f"  TCN Layer 2: Conv1d(128->128, k=3, d=2, pad=2) + GELU -> (B,128,13)")
    log(f"  TCN Layer 3: Conv1d(128->128, k=3, d=4, pad=4) + GELU -> (B,128,13)")
    log(f"  Receptive field: 29 weeks (full window coverage)")
    log(f"  GlobalPool : AdaptiveAvgPool1d(1) -> squeeze -> (B,128)")
    log(f"  tcn_context (B,128)  x  G  ->  gated (B,128) -> expand (B,5,128)")
    log(f"  == Exponential Decay Gap-Gating (v17, retained) ==")
    log(f"  gap_lambda : nn.Parameter (learnable scalar, init=5.0)")
    log(f"  G = exp(-max(eps,|gap_lambda|)*gap_size)  (B,1)  in (0, 1]")
    log(f"  Physics constraint: G->1 at gap=0, G->0 as gap->inf (strictly monotone)")
    log(f"  == Fusion ==")
    log(f"  horizon_ids [0-4] -> Embedding(5,32) -> (B,5,32)")
    log(f"  target_time (B,5,2) -- week_sin/cos of 5 future target weeks")
    log(f"  gap_size (B,1) -> Linear(1,16) -> (B,16) -> expand (B,5,16)")
    log(f"  encoded_state (B,5,{HIDDEN_SIZE*2+128+32+2+16})  [256+128+32+2+16=434]")
    log(f"  Branch A   : Linear(434->128) -> GELU -> Dropout(0.2) -> Linear(128->1)")
    log( "               P(drought) logits -- sigmoid() inline in forward()")
    log(f"  Branch B   : Linear(434->128) -> GELU -> Dropout(0.2) -> Linear(128->1)")
    log( "               Severity >= 0  via Softplus()")
    log( "  Output     : [v18 NEW] Hard Thresholding Gate:")
    log( "               prob = sigmoid(logits_A)")
    log( "               final = where(prob < 0.5, 0.0, prob * Branch_B)")
    log( "  Returns    : (final_output, logits_output, severity_output)  [v17: 3 tensors]")
    log( "CV Strategy  : [v18] 5-Fold Stratified Region Group CV  (StratifiedGroupKFold)")
    log(f"  N_FOLDS    : {N_FOLDS}")
    log(f"  WINDOW_SIZE: {WINDOW_SIZE} weeks")
    log( "  Stratify   : 10-quantile bins of per-region mean score (pd.qcut q=10)")
    log( "  Val regions: 20% of geographical regions entirely withheld per fold")
    log( "  Train      : ALL sliding windows for 80% of regions")
    log( "  Val        : last 5 weeks for 20% of regions (gap-replay windowing)")
    log( "  OOF Scaler : StandardScaler fit ONLY on 4-fold train regions per fold")
    log( "  OOF TE     : region_mean/zero_prob from 4-fold train regions ONLY")
    log( "  Submission : average predictions across all 5 fold models")
    log( "Loss         : [v18] Decoupled Hurdle Model Loss + Loss Re-weighting")
    log( "  Loss_A     : FocalLoss(gamma=2.0) across ALL samples, weight=2.0  [v18: 1.0->2.0]")
    log( "  Loss_B     : Weighted Smooth L1 on target>0 ONLY, weight=1.0")
    log( "               w_i = 1.0 + (y_i/5)^2 * 3.0")
    log( "  Total      : 2.0*Loss_A + 1.0*Loss_B  (classification head prioritised)")
    log(f"LR Schedule  : Manual Warm-up Epochs 1-5: 1e-5 -> 1e-3 (linear)")
    log( "               Epoch 6+: ReduceLROnPlateau (factor=0.5, patience=10)")
    log( "Early Stop   : pure L1Loss(final_output, y)  [Kaggle MAE aligned]")
    log( "               v18: final_output is Hard-Thresholded -- MAE is gate-aligned")
    log( "Pooling      : Temporal Attention [v8: retained]")
    log( "LayerNorm    : LayerNorm(input_size) before LSTM  [v7: retained]")
    log( "Features     : 29  (11 weather + 2 cyclic + 3 rolling-4w + 4 lag1 "
         "+ 4 lag2 + 3 drought-4w + 2 TE)  [v16: removed 8w/13w]")
    log( "  Cyclic     : week_sin, week_cos")
    log( "  TE         : region_mean_score, region_zero_prob  [OOF per fold in v17/v18]")
    log( "Scaler       : OOF StandardScaler per fold  [v17: replaces single scaler]")
    log( "Checkpoint   : fold_{k}_best.pt  [one per fold, 5 total]")
    log( "Inference    : 5-fold ensemble average")
    log( "OOM Guard    : try-except RuntimeError -> fallback BATCH_SIZE=256")
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

    # -- 1b. Validate v16/v17/v18 features in processed CSV -------------------
    log("\n[v18 Feature Validation]")
    assert "week_sin" in train_raw.columns, "week_sin missing -- run preprocess.py first"
    assert "week_cos" in train_raw.columns, "week_cos missing -- run preprocess.py first"
    assert "month" not in train_raw.columns, "month still present -- should be dropped by preprocess"
    assert "week_of_year" not in train_raw.columns, "week_of_year still present"
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
    log(f"  [v18] Zero-inflation: {zero_frac:.2%} of training scores == 0.0")
    log(f"  [v18] Hurdle Loss_B targets {1 - zero_frac:.2%} active samples (target>0)")
    for thresh in [1.0, 2.0, 3.0, 4.0]:
        frac = (all_scores > thresh).mean() * 100
        log(f"  score > {thresh:.1f}: {frac:.2f}%  [{int((all_scores > thresh).sum()):,} samples]")

    # -- 4b. Compute per-region actual deployment gap --------------------------
    log("\n[v18] Computing per-region actual deployment gaps ...")
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
    log(f"  [v18] Exponential Decay Gate: G = exp(-|gap_lambda| * gap_size)")

    # -- 5. Determine feature columns and input_size ---------------------------
    log("\n[v18] Determining feature columns ...")
    # Need a temp TE-augmented df just to resolve feat_cols
    # (TE will be added per-fold; here just identify which base cols are present)
    feat_cols  = [c for c in FEATURE_COLS if c in train_df.columns
                  or c in ("region_mean_score", "region_zero_prob")]
    # Final resolution happens after TE injection; for now just check base count
    base_feat_cols = [c for c in FEATURE_COLS
                      if c in train_df.columns
                      and c not in ("region_mean_score", "region_zero_prob")]
    log(f"  Base features (excluding TE): {len(base_feat_cols)}")
    log(f"  Expected total with TE: {len(FEATURE_COLS)}")
    input_size = len(FEATURE_COLS)   # 29 (TE added per-fold)

    assert input_size == 29, (
        f"Expected 29 features (27 base + 2 TE), got {input_size}. "
        f"v16/v17/v18: removed all 8w/13w rolling features."
    )
    log(f"  input_size = {input_size}  (29 confirmed)")

    # -- 6. Build 5-Fold Stratified Region Group CV splits ---------------------
    log(f"\n{'='*85}")
    log(f"5-Fold Stratified Region Group CV  [v18]")
    log(f"  StratifiedGroupKFold(n_splits={N_FOLDS}) grouped by region_id, stratified by 10 score bins")
    log(f"  Each fold: 20% of regions withheld entirely for validation")
    log(f"  Train groups: ALL sliding windows for 80% train-fold regions")
    log(f"  Val groups  : Last {HORIZON} weeks for 20% val-fold regions (gap-replay)")
    log(f"  OOF Scaler  : fit on 4-fold train features only (no val/test leakage)")
    log(f"  OOF TE      : region stats from 4-fold train regions only")
    log(f"  Submission  : ensemble average of all {N_FOLDS} fold predictions")
    log(f"{'='*85}")

    folds = build_region_group_cv_folds(train_df, actual_gaps, n_splits=N_FOLDS)
    log(f"\n  Folds built: {len(folds)}")
    for fi, (tg, vg) in enumerate(folds):
        log(f"  Fold {fi}: train_groups={len(tg):,}  val_groups={len(vg):,}")

    # -- 7. 5-Fold Training Loop -----------------------------------------------
    fold_results     = []   # list of (fold_k, best_mae, best_epoch)
    fold_test_preds  = []   # list of dict {region_id -> (5,) array}
    fold_val_pcts    = []   # list of percentile dicts

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):

        log(f"\n{'='*85}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}  [v18 Stratified Region Group CV]")
        log(f"  train_groups: {len(raw_train_groups):,}  |  "
            f"val_groups: {len(raw_val_groups):,}")
        log(f"{'='*85}")

        # -- 7a. Compute fold-local TE (from train-fold regions ONLY) ----------
        train_region_ids_fold = {
            entry[0]["region_id"].iloc[0] for entry in raw_train_groups
        }
        train_df_fold_regions = train_df[
            train_df["region_id"].isin(train_region_ids_fold)
        ]
        te_map_fold, gm_fold, gzp_fold = _compute_te_stats(train_df_fold_regions)
        log(f"  [OOF TE] Regions in train-fold: {len(train_region_ids_fold)}")
        log(f"  [OOF TE] global_mean_score={gm_fold:.4f}  "
            f"global_zero_prob={gzp_fold:.4f}")

        # -- 7b. Augment groups with fold-local TE ----------------------------
        aug_train_groups = _augment_groups_with_te(
            raw_train_groups, te_map_fold, gm_fold, gzp_fold
        )
        aug_val_groups = _augment_groups_with_te(
            raw_val_groups, te_map_fold, gm_fold, gzp_fold
        )

        # -- 7c. Fit fold-specific StandardScaler on train features only ------
        log(f"\n  [OOF Scaler] Fitting StandardScaler on fold {fold_k} train features ...")
        fold_scaler = StandardScaler()

        train_feat_parts = []
        for entry in aug_train_groups:
            group, i_min, i_max, eff_gap = entry
            local_feat_cols = [c for c in FEATURE_COLS if c in group.columns]
            train_feat_parts.append(group[local_feat_cols].values.astype(np.float32))

        if train_feat_parts:
            train_feat_matrix = np.concatenate(train_feat_parts, axis=0)
            fold_scaler.fit(train_feat_matrix)
            log(f"  [OOF Scaler] Fit on {len(train_feat_matrix):,} rows "
                f"({train_feat_matrix.shape[1]} features)")
        else:
            log(f"  [OOF Scaler] WARNING: No train rows found -- using identity scaler.")

        # Determine final feat_cols from augmented group
        _sample_group = aug_train_groups[0][0] if aug_train_groups else None
        if _sample_group is not None:
            feat_cols = [c for c in FEATURE_COLS if c in _sample_group.columns]
        log(f"  [OOF Scaler] feat_cols ({len(feat_cols)}): confirmed {len(feat_cols)} features")

        # -- 7d. Create datasets -----------------------------------------------
        train_ds = DroughtDataset(aug_train_groups, scaler=fold_scaler)
        val_ds   = DroughtDataset(aug_val_groups,   scaler=fold_scaler)
        log(f"  Train sequences: {len(train_ds):,}  |  Val sequences: {len(val_ds):,}")

        # -- 7e. Build model and checkpoint path ------------------------------
        model     = make_model(input_size)
        ckpt_path = os.path.join(MODELS_DIR, f"fold_{fold_k}_best.pt")
        log(f"  Checkpoint path: {ckpt_path}")
        log(f"  Model params: {model.count_parameters():,}")

        # -- 7f. OOM-Protected Training ----------------------------------------
        batch_size_to_use = BATCH_SIZE

        def _run_fold_training(bs):
            try:
                loader_tr  = _make_loader(train_ds, shuffle=True,  batch_size=bs)
                loader_val = _make_loader(val_ds,   shuffle=False, batch_size=bs)
            except Exception:
                loader_tr  = DataLoader(train_ds, batch_size=bs,
                                        shuffle=True,  num_workers=0, pin_memory=USE_AMP)
                loader_val = DataLoader(val_ds,   batch_size=bs,
                                        shuffle=False, num_workers=0, pin_memory=USE_AMP)

            # Tensor shape verification (first fold only)
            if fold_k == 0:
                first_X, first_y, first_tt, first_gs = next(iter(loader_tr))
                log(f"\n  [v18 Tensor Shape Verification] First Batch (Fold 0):")
                log(f"    X shape          : {tuple(first_X.shape)}"
                    f"  ->  (Batch, Seq={first_X.shape[1]}, Features={first_X.shape[2]})")
                log(f"    y shape          : {tuple(first_y.shape)}"
                    f"  ->  (Batch, Horizon={first_y.shape[1]})")
                log(f"    target_time shape: {tuple(first_tt.shape)}"
                    f"  ->  (Batch, Horizon=5, 2=week_sin/cos)")
                log(f"    gap_size shape   : {tuple(first_gs.shape)}"
                    f"  ->  (Batch, 1=normalised_gap)")
                assert first_X.shape[1] == WINDOW_SIZE, \
                    f"Seq mismatch: got {first_X.shape[1]}, expected {WINDOW_SIZE}"
                assert first_X.shape[2] == input_size, \
                    f"Feature mismatch: got {first_X.shape[2]}, expected {input_size}"
                assert first_y.shape[1] == HORIZON, \
                    f"Horizon mismatch: got {first_y.shape[1]}, expected {HORIZON}"
                assert first_tt.shape[1] == HORIZON and first_tt.shape[2] == 2
                assert first_gs.shape[1] == 1
                log(f"    v Shape assertion PASSED.\n")
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
                _run_fold_training(batch_size_to_use)

        except RuntimeError as oom_err:
            if "out of memory" in str(oom_err).lower():
                log(f"\n  [OOM] CUDA OOM at batch_size={batch_size_to_use}.")
                log(f"  [OOM] Flushing cache, retry with batch_size=256 ...")
                torch.cuda.empty_cache()
                model = make_model(input_size)
                batch_size_to_use = 256
                best_mae, best_epoch_num, _, loader_tr, loader_val = \
                    _run_fold_training(batch_size_to_use)
            else:
                raise

        log(f"\n  [Fold {fold_k}] Best Val MAE : {best_mae:.4f}  (epoch {best_epoch_num})")
        log(f"  Checkpoint saved -> {ckpt_path}")
        fold_results.append((fold_k, best_mae, best_epoch_num))

        # Prediction percentile diagnostics on val set
        log(f"\n  [Fold {fold_k} Prediction Percentiles -- best checkpoint]")
        if os.path.exists(ckpt_path) and best_mae < float("inf"):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            fold_pct = eval_prediction_percentiles(model, loader_val, DEVICE, log)
        else:
            log("  [Skip] No valid checkpoint (training produced NaN).")
            fold_pct = {}
        fold_val_pcts.append(fold_pct)

        # -- 7g. Save fold scaler --------------------------------------------
        scaler_path = os.path.join(MODELS_DIR, f"scaler_fold_{fold_k}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(fold_scaler, f)
        log(f"  Fold scaler saved -> {scaler_path}")

        # -- 7h. Run test inference with this fold's model + scaler + TE ------
        log(f"\n  [Fold {fold_k}] Running test set inference ...")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}.")

        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        fold_preds   = predict_test_set(
            model, test_df_fold, feat_cols, fold_scaler, actual_gaps, log
        )
        fold_test_preds.append(fold_preds)
        log(f"  [Fold {fold_k}] Regions predicted: {len(fold_preds)}")

        # Free GPU memory before next fold
        del model, loader_tr, loader_val, train_ds, val_ds
        torch.cuda.empty_cache()

    # -- 8. Cross-fold summary -------------------------------------------------
    log(f"\n{'='*85}")
    log(f"5-Fold Cross-Validation Summary  [v18]")
    log(f"{'='*85}")
    mae_values = [r[1] for r in fold_results]
    for fold_k, best_mae, best_epoch in fold_results:
        log(f"  Fold {fold_k}: Val MAE={best_mae:.4f}  (epoch {best_epoch})")
    log(f"  Mean Val MAE : {np.mean(mae_values):.4f}  +-  {np.std(mae_values):.4f}")
    log(f"  Best Fold    : Fold {np.argmin(mae_values)} "
        f"(MAE={min(mae_values):.4f})")

    # -- 9. Ensemble: average predictions across all 5 folds -------------------
    log(f"\n[v18] Ensembling {N_FOLDS}-fold test predictions (simple average) ...")
    # All folds predict for all 2248 regions
    all_region_ids = sorted(fold_test_preds[0].keys())
    final_predictions = {}
    for rid in all_region_ids:
        stack = np.stack([fp[rid] for fp in fold_test_preds], axis=0)  # (5, 5)
        final_predictions[rid] = stack.mean(axis=0)                     # (5,)

    log(f"  Ensemble complete. Total regions: {len(final_predictions)}")

    # -- 10. Submission-level prediction diagnostics --------------------------
    log("\n[Submission Prediction Diagnostics]")
    all_sub_preds = np.array(list(final_predictions.values())).ravel()
    p50  = float(np.percentile(all_sub_preds, 50))
    p90  = float(np.percentile(all_sub_preds, 90))
    p95  = float(np.percentile(all_sub_preds, 95))
    p99  = float(np.percentile(all_sub_preds, 99))
    pmax = float(np.max(all_sub_preds))
    log(f"  n={len(all_sub_preds):,}  mean={all_sub_preds.mean():.4f}  std={all_sub_preds.std():.4f}")
    log(f"  p50={p50:.4f}  p90={p90:.4f}  p95={p95:.4f}  p99={p99:.4f}  max={pmax:.4f}")
    if p99 < 2.0:
        log("  *** WARNING: Submission p99 < 2.0 -- model zero-collapse may still present! ***")
    else:
        log("  v Submission p99 >= 2.0 -- zero-collapse suppression is working.")

    zero_pred_frac = (all_sub_preds < 0.05).mean()
    log(f"  Fraction of near-zero predictions (<0.05): {zero_pred_frac:.2%}")
    log(f"  [v18] Hard Thresholding active: predictions where P(drought)<0.5 forced to 0.0")

    # -- 11. Format & save submission.csv -------------------------------------
    log("\nFormatting submission.csv ...")
    rows = []
    for region_id, preds in sorted(final_predictions.items()):
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

    # -- 13. Print architecture summary (using last fold's checkpoint) --------
    log(f"\n[v18 Architecture Summary]")
    _tmp_model = make_model(input_size)
    log("\n" + _tmp_model.architecture_summary(input_size))
    del _tmp_model

    # Save full training log
    log(f"\nTotal elapsed: {(time.time() - t0):.1f}s")
    log_path = os.path.join(ROOT, "_training_log_18th.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nTraining log saved -> {log_path}")

    return {
        "fold_results":     fold_results,
        "mean_val_mae":     float(np.mean(mae_values)),
        "std_val_mae":      float(np.std(mae_values)),
        "input_size":       input_size,
        "submission":       submission,
        "sub_p99":          p99,
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
