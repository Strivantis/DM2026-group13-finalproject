"""
train.py -- Drought Score Forecasting Pipeline (v21 – Pure Continuous-Time Prediction)
=======================================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv                  -- Kaggle submission (2248 rows × 6 cols)
    models/fold_{k}_best.pt         -- Best weights per fold (k=0..4)
    models/scaler_fold_{k}.pkl      -- StandardScaler per fold
    _training_log_21st.txt          -- Full console log

Key improvements (v21 – Paradigm Shift)
-----------------------------------------
  1. Pure Continuous-Time Prediction:
       Given 13 consecutive weeks X, predict the immediately following 5 weeks Y.
       Gap = 0.  No GapGate.  No gap_size tensor anywhere in the pipeline.

  2. StratifiedGroupKFold CV (5-Fold):
       Group  = region_id (each region is one atomic unit, never split).
       Strata = 10-quantile bins of per-region historical mean drought score.
       Train  = 80% of regions (geography unseen during validation).
       Val    = 20% of regions (completely held-out geography).
       Forces the model to generalise climate physics, not memorise regions.

  3. DataLoader tuple changes:
       V20: (X, y, target_time, gap_size, group_id)  [5-tuple]
       V21: (X, y, target_time, group_id)            [4-tuple]

  4. Model forward signature:
       V20: model(X, target_time, gap_size) -> (logits, severity)
       V21: model(X, target_time)           -> (logits, severity)

  5. Loss (unchanged from v20):
       Loss A: BCEWithLogitsLoss(logits, (target>0).float())  [all samples]
       Loss B: SmoothL1Loss(severity[mask], target[mask])      [mask=(target>0)]
       Total  = 1.0 * Loss_A + 1.0 * Loss_B

  6. Scheduler (unchanged from v20):
       CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-5).
       Manual warm-up epochs 1-5: linear ramp 1e-5 → 1e-3.

  7. Post-Ensemble Median Rule (unchanged from v20):
       mean_prob = mean(probs, axis=0)
       mean_sev  = mean(sevs,  axis=0)
       final     = where(mean_prob < 0.5, 0.0, mean_sev)
       Hard threshold ONLY after ensemble averaging.

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
    build_stratified_group_cv_folds,
    build_temporal_shift_cv_folds,   # backward compat alias
    build_region_group_cv_folds,     # backward compat alias
    build_single_fold,
    build_gap_replay_folds,
    build_walk_forward_folds,
    compute_actual_gaps,
    DroughtDataset,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
    WF_NUM_FOLDS,
    WF_FOLD_WEEKS,
    GAP_WEEKS,
    N_TS_FOLDS,
    TS_SHIFT_WEEKS,
)
from src.model import DroughtLSTM

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
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

N_FOLDS       = 5             # StratifiedGroupKFold 5-folds
HIDDEN_SIZE   = 128
NUM_LAYERS    = 3
DROPOUT       = 0.4
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-2
BATCH_SIZE    = 512
NUM_EPOCHS    = 200
PATIENCE      = 35

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

USE_AMP = DEVICE.type == "cuda"
_scaler = GradScaler(device="cuda", enabled=USE_AMP)

# ---------------------------------------------------------------------------
# Loss criteria
# ---------------------------------------------------------------------------
# Loss A: BCE with logits — probability calibration (all samples)
_bce_criterion = nn.BCEWithLogitsLoss()

# Loss B: Huber / SmoothL1 — severity regression (positive targets only)
_huber_criterion = nn.SmoothL1Loss()

# L1 for MAE evaluation (early stopping monitor)
_l1_criterion = nn.L1Loss()


# ---------------------------------------------------------------------------
# Decoupled Hurdle Loss  (v20/v21)
# ---------------------------------------------------------------------------
def hurdle_loss(
    logits: torch.Tensor,
    severity: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    v21 Decoupled BCE + Masked SmoothL1 Loss (1:1 ratio).

    Loss A (Probability Calibration):
        binary_target = (target > 0.0).float()   [all samples]
        loss_a = BCEWithLogitsLoss(logits, binary_target)

    Loss B (Severity Regression, Masked):
        mask = (target > 0.0)
        If mask.sum() > 0:
            loss_b = SmoothL1Loss(severity[mask], target[mask])
        Else:
            loss_b = 0.0   [no positive targets in batch — skip regression]

    Total Loss = 1.0 * loss_a + 1.0 * loss_b

    Parameters
    ----------
    logits   : (B, H)  raw probability logits from Branch A
    severity : (B, H)  non-negative severity from Branch B (Softplus)
    target   : (B, H)  ground truth drought scores in [0, 5]

    Returns
    -------
    total_loss : scalar tensor
    loss_a     : scalar tensor (BCE)
    loss_b     : float (SmoothL1 or 0.0)
    """
    # Loss A: all-sample binary calibration
    binary_target = (target > 0.0).float()
    loss_a = _bce_criterion(logits, binary_target)

    # Loss B: masked severity regression on positive targets only
    mask = target > 0.0
    if mask.sum() > 0:
        loss_b = _huber_criterion(severity[mask], target[mask])
    else:
        loss_b = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    total_loss = 1.0 * loss_a + 1.0 * loss_b
    return total_loss, loss_a, loss_b


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    """
    One training epoch with decoupled BCE + Masked SmoothL1 loss.

    v21: unpack (X, y, target_time, group_id)  [4-tuple; gap_size removed]
    model.forward(X, target_time) -> (logits_output, severity_output)
    """
    model.train()
    total_loss, n = 0.0, 0

    for X, y, target_time, group_id in loader:
        X, y        = X.to(device), y.to(device)
        target_time = target_time.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=USE_AMP):
            logits_output, severity_output = model(X, target_time)

            total, loss_a, loss_b = hurdle_loss(logits_output, severity_output, y)

        _scaler.scale(total).backward()
        _scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        _scaler.step(optimizer)
        _scaler.update()

        bs = X.size(0)
        total_loss += total.item() * bs
        n += bs
    return total_loss / n if n > 0 else float("inf")


@torch.no_grad()
def eval_mae(model, loader, device) -> float:
    """
    Evaluate pure (unweighted) MAE on hurdle combined predictions.

    Combined prediction: sigmoid(logits) >= 0.5 ? severity : 0.0

    v21: 4-tuple unpacking; model.forward(X, target_time) only.
    """
    model.eval()
    total_mae, n = 0.0, 0
    for X, y, target_time, group_id in loader:
        X, y        = X.to(device), y.to(device)
        target_time = target_time.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            logits_output, severity_output = model(X, target_time)

        prob = torch.sigmoid(logits_output)       # (B, 5) in [0, 1]
        # Hurdle combined prediction: 0 if prob < 0.5 else severity
        combined = torch.where(prob < 0.5,
                               torch.zeros_like(severity_output),
                               severity_output)   # (B, 5)
        total_mae += _l1_criterion(combined, y).item() * X.size(0)
        n += X.size(0)
    return total_mae / n if n > 0 else float("inf")


@torch.no_grad()
def eval_prediction_percentiles(model, loader, device, log) -> dict:
    """
    Diagnostic hook: collect all combined predictions and log percentile stats.
    v21: 4-tuple unpacking; model.forward(X, target_time) only.
    """
    model.eval()
    all_probs = []
    all_sevs  = []
    for X, y, target_time, group_id in loader:
        X           = X.to(device)
        target_time = target_time.to(device)
        with autocast(device_type=device.type, enabled=USE_AMP):
            logits_output, severity_output = model(X, target_time)

        prob = torch.sigmoid(logits_output)
        combined = torch.where(prob < 0.5,
                               torch.zeros_like(severity_output),
                               severity_output)
        all_probs.append(prob.cpu().float().numpy())
        all_sevs.append(combined.cpu().float().numpy())

    if not all_probs:
        return {}

    probs_flat = np.concatenate(all_probs, axis=0).ravel()
    preds_flat = np.concatenate(all_sevs, axis=0).ravel()

    p50  = float(np.percentile(preds_flat, 50))
    p90  = float(np.percentile(preds_flat, 90))
    p95  = float(np.percentile(preds_flat, 95))
    p99  = float(np.percentile(preds_flat, 99))
    pmax = float(np.max(preds_flat))
    mean_prob = float(probs_flat.mean())
    frac_above_05 = float((probs_flat >= 0.5).mean())

    log(f"  [Prediction Diagnostics] n={len(preds_flat):,}")
    log(f"    p50={p50:.4f}  p90={p90:.4f}  p95={p95:.4f}  p99={p99:.4f}  max={pmax:.4f}")
    log(f"    mean_prob={mean_prob:.4f}  frac_prob>=0.5={frac_above_05:.4f}")
    if p99 < 2.0:
        log("    *** WARNING: p99 < 2.0 -- model may still be evading extremes! ***")
    else:
        log("    v p99 >= 2.0 -- model is predicting away from zero-collapse.")

    return {"p50": p50, "p90": p90, "p95": p95, "p99": p99, "max": pmax,
            "mean_prob": mean_prob, "frac_prob_ge_05": frac_above_05}


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
    """
    v21: CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-5).
    T_0=15: first restart after 15 epochs.
    T_mult=2: each restart doubles the interval (15 → 30 → 60 ...).
    eta_min=1e-5: minimum LR floor.
    """
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=15, T_mult=2, eta_min=1e-5
    )


def _make_loader(dataset, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
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

    v21 Training:
      - Manual LR warm-up epochs 1-5: linear ramp 1e-5 → 1e-3.
      - CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-5) from epoch 6+.
      - Decoupled BCE + Masked SmoothL1 loss (1:1 ratio).
      - No time-decay weighting (strict unweighted baseline).
    Early stopping: pure MAE of hurdle combined prediction — Kaggle aligned.

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
        # --- Manual LR Warm-up (Epochs 1-5) ---
        if epoch <= 5:
            lr = 1e-5 + (1e-3 - 1e-5) * ((epoch - 1) / 4.0)
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr

        train_loss = train_epoch(model, train_loader, optimiser, DEVICE)
        val_mae    = eval_mae(model, val_loader, DEVICE)

        # CosineAnnealingWarmRestarts steps per epoch after warm-up
        if epoch > 5:
            scheduler.step()

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
            f"  {val_mae:>10.4f}  {curr_lr:>10.2e}  [Hurdle_v21]{marker}"
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
    Supports both 3-tuple (group, i_min, i_max) and 4-tuple (group, i_min,
    i_max, _gap) entries for backward compatibility.
    """
    result = []
    for entry in groups:
        if len(entry) == 4:
            group, i_min, i_max, _gap = entry
        else:
            group, i_min, i_max = entry
        g = group.copy()
        rid = g["region_id"].iloc[0]
        mean_s, zero_p = te_map.get(rid, (global_mean, global_zero_prob))
        g["region_mean_score"] = np.float32(mean_s)
        g["region_zero_prob"]  = np.float32(zero_p)
        # Return 3-tuple (V21: no gap)
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
# Inference helper  (v21 – dual-head; per-fold collect prob & severity)
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_test_set(model, test_df, feat_cols, scaler, log) -> tuple:
    """
    Run inference for every region in test_df using the given model.

    v21: model.forward(X, target_time) -> (logits, severity) tuple.
         No gap_size tensor anywhere.
    Returns raw (prob, severity) dicts — NO hard threshold here.
    Hard threshold applied post-ensemble in main().

    Returns
    -------
    probs_dict     : dict  region_id -> np.array shape (5,)  [sigmoid(logits)]
    severity_dict  : dict  region_id -> np.array shape (5,)  [Softplus severity]
    """
    model.eval()
    probs_dict    = {}
    severity_dict = {}

    for region_id, group in test_df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)

        if n < WINDOW_SIZE:
            pad_rows = pd.concat(
                [group.iloc[[0]]] * (WINDOW_SIZE - n) + [group],
                ignore_index=True,
            )
            group = pad_rows
            n = len(group)

        window_df = group.iloc[:WINDOW_SIZE]
        X = window_df[feat_cols].values.astype(np.float32)
        X = scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # target_time: use the HORIZON rows immediately following the window
        # For test set there is no ground-truth future, so use cyclic features
        # from test_df rows (columns week_sin/week_cos of the test rows).
        target_rows = group.iloc[-HORIZON:]
        if "week_sin" in group.columns and "week_cos" in group.columns:
            tt = np.stack([
                target_rows["week_sin"].values.astype(np.float32),
                target_rows["week_cos"].values.astype(np.float32),
            ], axis=-1)
        else:
            tt = np.zeros((HORIZON, 2), dtype=np.float32)
        target_time_tensor = torch.tensor(tt, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with autocast(device_type=DEVICE.type, enabled=USE_AMP):
            # v21: no gap_size
            logits_output, severity_output = model(X_tensor, target_time_tensor)

        prob     = torch.sigmoid(logits_output).squeeze(0).cpu().float().numpy()  # (5,)
        severity = severity_output.squeeze(0).cpu().float().numpy()               # (5,)

        probs_dict[region_id]    = prob
        severity_dict[region_id] = severity

    return probs_dict, severity_dict


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
    log("Drought Forecasting Pipeline  v21")
    log("Pure Continuous-Time Prediction  |  StratifiedGroupKFold CV  |  Post-Ensemble Median Rule")
    log("CosineAnnealingWarmRestarts(T_0=15, T_mult=2)  |  5-Fold StratifiedGroupKFold")
    log("=" * 85)
    log("Paradigm Shift (v21):")
    log("  V20 ERROR: Treating Train/Test Dataset Gap as Input→Prediction Gap.")
    log("  V21 FIX : Gap = 0.  Given 13 consecutive weeks X, predict next 5 weeks Y.")
    log("  Gap-Gate (gap_lambda, G) ABOLISHED.  Gap Embedding ABOLISHED.")
    log("")
    log("Architecture : Dual-Stream Dilated TCN + BiLSTM  (v16 trunk retained)")
    log(f"  == LSTM Stream ==")
    log(f"  BiLSTM     : hidden_size={HIDDEN_SIZE}/dir -> {HIDDEN_SIZE*2} effective")
    log(f"  Layers     : {NUM_LAYERS}")
    log(f"  lstm_context (B,{HIDDEN_SIZE*2}) -> expand (B,5,{HIDDEN_SIZE*2})")
    log(f"  == Dilated TCN Stream (v16 - retained) ==")
    log(f"  Input      : (B, 40, 13)  [v19/v21: input_size=40 enriched features]")
    log(f"  TCN Layer 1: Conv1d(40->128, k=3, d=1, pad=1) + GELU -> (B,128,13)")
    log(f"  TCN Layer 2: Conv1d(128->128, k=3, d=2, pad=2) + GELU -> (B,128,13)")
    log(f"  TCN Layer 3: Conv1d(128->128, k=3, d=4, pad=4) + GELU -> (B,128,13)")
    log(f"  Receptive field: 29 weeks (full window coverage)")
    log(f"  GlobalPool : AdaptiveAvgPool1d(1) -> squeeze -> (B,128)")
    log(f"  tcn_context (B,128) -> expand (B,5,128)")
    log(f"  [v21] Gap-Gate ABOLISHED: no gap_lambda, no G multiplication")
    log(f"  == Fusion ==")
    log(f"  horizon_ids [0-4] -> Embedding(5,32) -> (B,5,32)")
    log(f"  target_time (B,5,2) -- week_sin/cos of 5 future target weeks")
    log(f"  [v21] gap_embed ABOLISHED: no gap_size input, no Linear(1,16)")
    log(f"  encoded_state (B,5,{HIDDEN_SIZE*2+128+32+2})  [256+128+32+2=418]")
    log(f"  == Branch A: Probability Logits Head ==")
    log(f"  Linear(418->128) -> GELU -> Dropout(0.2) -> Linear(128->1)")
    log(f"  squeeze(-1) -> (B,5) raw logits  (NO Sigmoid in model)")
    log(f"  == Branch B: Severity Regressor Head ==")
    log(f"  Linear(418->128) -> GELU -> Dropout(0.2) -> Linear(128->1)")
    log( "  squeeze(-1) -> (B,5) | Softplus() -> non-negative severity")
    log( "  Returns: (logits_output, severity_output) -- dual-head tuple")
    log( "CV Strategy  : [v21] 5-Fold StratifiedGroupKFold")
    log(f"  N_FOLDS       : {N_FOLDS}")
    log(f"  WINDOW_SIZE   : {WINDOW_SIZE} weeks")
    log( "  Group         : region_id  (atomic unit, never split across folds)")
    log( "  Strata        : 10-quantile bins of per-region historical mean score")
    log( "  Train         : 80% of regions per fold (geography-unseen)")
    log( "  Val           : 20% of regions per fold (completely held-out)")
    log( "  Submission    : Post-Ensemble Median Rule (after 5-fold averaging)")
    log( "Loss         : [v21] Decoupled BCE + Masked SmoothL1 (1:1 ratio)")
    log( "  Loss A: BCEWithLogitsLoss(logits, (target>0).float())  [all samples]")
    log( "  Loss B: SmoothL1Loss(severity[mask], target[mask])  [mask=(target>0)]")
    log( "  Total = 1.0 * Loss_A + 1.0 * Loss_B")
    log(f"LR Schedule  : Manual Warm-up Epochs 1-5: 1e-5 -> 1e-3 (linear)")
    log( "               Epoch 6+: CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-5)")
    log( "Early Stop   : pure MAE of hurdle combined prediction  [Kaggle aligned]")
    log( "Inference    : Per fold: prob=sigmoid(logits), sev=severity  [no gap_size]")
    log( "               mean_prob = mean(probs, axis=0)")
    log( "               mean_sev  = mean(sevs, axis=0)")
    log( "               final     = where(mean_prob < 0.5, 0.0, mean_sev)")
    log( "Features     : 40  (11 base weather + 11 enriched weekly stats + 2 cyclic")
    log( "               + 3 rolling-4w + 4 lag1 + 4 lag2 + 3 drought-4w + 2 TE)")
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

    # -- 1b. Feature Validation -----------------------------------------------
    log("\n[v21 Feature Validation]")
    assert "week_sin" in train_raw.columns, "week_sin missing -- run preprocess.py first"
    assert "week_cos" in train_raw.columns, "week_cos missing -- run preprocess.py first"
    assert "month" not in train_raw.columns, "month still present -- preprocess issue"
    assert "week_of_year" not in train_raw.columns, "week_of_year still present"
    log("  v week_sin, week_cos present  |  month, week_of_year correctly absent.")

    # Verify no 8w/13w domain-shift features
    bad_cols = [c for c in train_raw.columns if ("8w" in c or "13w" in c)]
    if bad_cols:
        log(f"  *** WARNING: 8w/13w features found: {bad_cols}")
    else:
        log("  v No 8w/13w rolling features (domain-shift columns absent).")

    # Verify adversarial features pruned
    adv_present = [c for c in train_raw.columns if c in ("dp_tmp", "wb_tmp")]
    if adv_present:
        log(f"  *** WARNING: Adversarial collinear features still in CSV: {adv_present}")
    else:
        log("  v dp_tmp, wb_tmp correctly pruned from processed CSV.")

    # Verify v19 enriched features present
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
    log("\nRefining features (drought proxy index + log1p precipitation) ...")
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
    log(f"  [v21] Zero-inflation baseline: {zero_frac:.2%} of training scores == 0.0")
    log(f"  Post-Ensemble Median Rule target: preserve >= {zero_frac:.2%} zero fraction")
    for thresh in [1.0, 2.0, 3.0, 4.0]:
        frac = (all_scores > thresh).mean() * 100
        log(f"  score > {thresh:.1f}: {frac:.2f}%  [{int((all_scores > thresh).sum()):,} samples]")

    # -- 5. Determine feature columns and input_size ---------------------------
    log("\n[v21] Determining feature columns ...")
    base_feat_cols = [c for c in FEATURE_COLS
                      if c in train_df.columns
                      and c not in ("region_mean_score", "region_zero_prob")]
    log(f"  Base features (excluding TE): {len(base_feat_cols)}")
    log(f"  Expected total with TE: {len(FEATURE_COLS)}")
    input_size = len(FEATURE_COLS)   # 40 (TE added per-fold)

    assert input_size == 40, (
        f"Expected 40 features (38 base + 2 TE), got {input_size}. "
        f"v21: 11 base weather + 11 enriched weekly stats + 2 cyclic "
        f"+ 3 rolling-4w + 4 lag1 + 4 lag2 + 3 drought-4w + 2 TE = 40."
    )
    log(f"  input_size = {input_size}  (40 confirmed)")

    # -- 6. Build 5-Fold StratifiedGroupKFold CV splits -----------------------
    log(f"\n{'='*85}")
    log(f"5-Fold StratifiedGroupKFold CV  [v21]")
    log(f"  Group  : region_id  (atomic unit; each region stays in one fold side)")
    log(f"  Strata : 10-quantile bins of per-region historical mean drought score")
    log(f"  Train  : 80% of regions per fold (geographically unseen)")
    log(f"  Val    : 20% of regions per fold (completely held-out geography)")
    log(f"  Gap    : 0  (V21 pure continuous-time prediction)")
    log(f"{'='*85}")

    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    log(f"\n  Folds built: {len(folds)}")
    for fi, (tg, vg) in enumerate(folds):
        log(f"  Fold {fi}: train_groups={len(tg):,}  val_groups={len(vg):,}")

    # -- 7. 5-Fold Training Loop -----------------------------------------------
    fold_results     = []
    fold_probs_preds = []   # List of per-fold probs dicts  (region_id -> (5,))
    fold_sevs_preds  = []   # List of per-fold sevs dicts   (region_id -> (5,))
    fold_val_pcts    = []

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):

        log(f"\n{'='*85}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}  [v21 StratifiedGroupKFold – Pure Hurdle Model]")
        log(f"  train_groups: {len(raw_train_groups):,}  |  "
            f"val_groups: {len(raw_val_groups):,}")
        log(f"{'='*85}")

        # -- 7a. Compute fold-local TE -----------------------------------------
        # TE is computed over training-fold regions only (leakage-free).
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

        # -- 7c. Fit fold-specific StandardScaler -----------------------------
        log(f"\n  [OOF Scaler] Fitting StandardScaler on fold {fold_k} train features ...")
        fold_scaler = StandardScaler()

        train_feat_parts = []
        for entry in aug_train_groups:
            group, i_min, i_max = entry[:3]
            local_feat_cols = [c for c in FEATURE_COLS if c in group.columns]
            train_feat_parts.append(group[local_feat_cols].values.astype(np.float32))

        if train_feat_parts:
            train_feat_matrix = np.concatenate(train_feat_parts, axis=0)
            fold_scaler.fit(train_feat_matrix)
            log(f"  [OOF Scaler] Fit on {len(train_feat_matrix):,} rows "
                f"({train_feat_matrix.shape[1]} features)")
        else:
            log(f"  [OOF Scaler] WARNING: No train rows found -- using identity scaler.")

        _sample_group = aug_train_groups[0][0] if aug_train_groups else None
        if _sample_group is not None:
            feat_cols = [c for c in FEATURE_COLS if c in _sample_group.columns]
        log(f"  [OOF Scaler] feat_cols ({len(feat_cols)}): confirmed {len(feat_cols)} features")

        # -- 7d. Create datasets -----------------------------------------------
        train_ds = DroughtDataset(aug_train_groups, scaler=fold_scaler)
        val_ds   = DroughtDataset(aug_val_groups,   scaler=fold_scaler)
        log(f"  Train sequences: {len(train_ds):,}  |  Val sequences: {len(val_ds):,}")

        # -- 7e. Build model and checkpoint -----------------------------------
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
                first_X, first_y, first_tt, first_gid = next(iter(loader_tr))
                log(f"\n  [v21 Tensor Shape Verification] First Batch (Fold 0):")
                log(f"    X shape          : {tuple(first_X.shape)}"
                    f"  ->  (Batch, Seq={first_X.shape[1]}, Features={first_X.shape[2]})")
                log(f"    y shape          : {tuple(first_y.shape)}"
                    f"  ->  (Batch, Horizon={first_y.shape[1]})")
                log(f"    target_time shape: {tuple(first_tt.shape)}"
                    f"  ->  (Batch, Horizon=5, 2=week_sin/cos)")
                log(f"    group_id shape   : {tuple(first_gid.shape)}"
                    f"  ->  (Batch, 1=chronological_position)")
                log(f"    [v21] gap_size   : ABOLISHED (not in tuple)")
                assert first_X.shape[1] == WINDOW_SIZE, \
                    f"Seq mismatch: got {first_X.shape[1]}, expected {WINDOW_SIZE}"
                assert first_X.shape[2] == input_size, \
                    f"Feature mismatch: got {first_X.shape[2]}, expected {input_size}"
                assert first_y.shape[1] == HORIZON
                assert first_tt.shape[1] == HORIZON and first_tt.shape[2] == 2
                assert first_gid.shape[1] == 1

                # Dual-head forward shape check (v21: no gap_size)
                _test_model = model
                _test_model.eval()
                with torch.no_grad():
                    _lgt, _sev = _test_model(
                        first_X[:2].to(DEVICE),
                        first_tt[:2].to(DEVICE),
                    )
                assert _lgt.shape == (2, HORIZON), \
                    f"logits shape mismatch: {tuple(_lgt.shape)}, expected (2, {HORIZON})"
                assert _sev.shape == (2, HORIZON), \
                    f"severity shape mismatch: {tuple(_sev.shape)}, expected (2, {HORIZON})"
                log(f"    logits shape     : {tuple(_lgt.shape)}  ->  (Batch, Horizon)")
                log(f"    severity shape   : {tuple(_sev.shape)}  ->  (Batch, Horizon)")
                log(f"    v Shape assertion PASSED (input_size={input_size}).\n")
                del first_X, first_y, first_tt, first_gid, _lgt, _sev
                _test_model.train()

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

        # Prediction percentile diagnostics
        log(f"\n  [Fold {fold_k} Prediction Percentiles -- best checkpoint]")
        if os.path.exists(ckpt_path) and best_mae < float("inf"):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            fold_pct = eval_prediction_percentiles(model, loader_val, DEVICE, log)
        else:
            log("  [Skip] No valid checkpoint (training produced NaN).")
            fold_pct = {}
        fold_val_pcts.append(fold_pct)

        # Save fold scaler
        scaler_path = os.path.join(MODELS_DIR, f"scaler_fold_{fold_k}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(fold_scaler, f)
        log(f"  Fold scaler saved -> {scaler_path}")

        # Test inference -- collect raw (prob, severity) per fold
        log(f"\n  [Fold {fold_k}] Running test set inference (raw prob + severity) ...")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}.")

        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        test_df_fold = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        fold_probs, fold_sevs = predict_test_set(
            model, test_df_fold, feat_cols, fold_scaler, log
        )
        fold_probs_preds.append(fold_probs)
        fold_sevs_preds.append(fold_sevs)

        # Log per-fold raw prediction stats (before ensemble)
        sample_probs = np.concatenate(list(fold_probs.values()), axis=0)
        sample_sevs  = np.concatenate(list(fold_sevs.values()),  axis=0)
        log(f"  [Fold {fold_k}] Raw prob  -- mean={sample_probs.mean():.4f}  "
            f"frac>=0.5={float((sample_probs>=0.5).mean()):.4f}")
        log(f"  [Fold {fold_k}] Raw sev   -- mean={sample_sevs.mean():.4f}  "
            f"p99={float(np.percentile(sample_sevs,99)):.4f}")
        log(f"  [Fold {fold_k}] Regions predicted: {len(fold_probs)}")

        del model, loader_tr, loader_val, train_ds, val_ds
        torch.cuda.empty_cache()

    # -- 8. Cross-fold summary -------------------------------------------------
    log(f"\n{'='*85}")
    log(f"5-Fold Cross-Validation Summary  [v21 StratifiedGroupKFold Hurdle Model]")
    log(f"{'='*85}")
    mae_values = [r[1] for r in fold_results]
    for fold_k, best_mae, best_epoch in fold_results:
        log(f"  Fold {fold_k}: Val MAE={best_mae:.4f}  (epoch {best_epoch})")
    log(f"  Mean Val MAE : {np.mean(mae_values):.4f}  +-  {np.std(mae_values):.4f}")
    log(f"  Best Fold    : Fold {np.argmin(mae_values)} "
        f"(MAE={min(mae_values):.4f})")

    # -- 9. Post-Ensemble Median Rule ------------------------------------------
    log(f"\n[v21] Post-Ensemble Median Rule ({N_FOLDS}-fold) ...")
    log(f"  Step 1: Compute mean_prob = mean(probs, axis=0) across {N_FOLDS} folds")
    log(f"  Step 2: Compute mean_sev  = mean(sevs,  axis=0) across {N_FOLDS} folds")
    log(f"  Step 3: final = where(mean_prob < 0.5, 0.0, mean_sev)")
    log(f"  Hard threshold ONLY applied after ensemble averaging.")

    all_region_ids = sorted(fold_probs_preds[0].keys())

    # Build stacked arrays for vectorised ensemble
    # probs_stack: (N_FOLDS, n_regions, HORIZON)
    probs_stack = np.stack(
        [np.stack([fp[rid] for rid in all_region_ids], axis=0)
         for fp in fold_probs_preds],
        axis=0,
    )
    sevs_stack = np.stack(
        [np.stack([fs[rid] for rid in all_region_ids], axis=0)
         for fs in fold_sevs_preds],
        axis=0,
    )

    # Ensemble averaging (BEFORE hard threshold)
    mean_prob     = np.mean(probs_stack, axis=0)   # (n_regions, HORIZON)
    mean_severity = np.mean(sevs_stack,  axis=0)   # (n_regions, HORIZON)

    log(f"  mean_prob  -- min={mean_prob.min():.4f}  mean={mean_prob.mean():.4f}  "
        f"frac>=0.5={float((mean_prob>=0.5).mean()):.4f}")
    log(f"  mean_sev   -- mean={mean_severity.mean():.4f}  "
        f"p99={float(np.percentile(mean_severity,99)):.4f}")

    # Post-Ensemble Hard Thresholding
    final_preds = np.where(mean_prob < 0.5, 0.0, mean_severity)
    # (n_regions, HORIZON)

    zero_frac_final = float((final_preds == 0.0).mean())
    log(f"  final_preds zero-fraction: {zero_frac_final:.4f}  "
        f"(train zero-inflation was {(all_scores==0.0).mean():.4f})")

    final_predictions = {rid: final_preds[i] for i, rid in enumerate(all_region_ids)}
    log(f"  Ensemble + thresholding complete. Total regions: {len(final_predictions)}")

    # -- 10. Submission prediction diagnostics --------------------------------
    log("\n[Submission Prediction Diagnostics]")
    all_sub_preds = final_preds.ravel()
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
        log("  v Submission p99 >= 2.0 -- prediction diversity is healthy.")
    log(f"  [v21] Post-Ensemble Median Rule applied -- zero-fraction={zero_frac_final:.4f}")

    # -- 11. Format & save submission.csv -------------------------------------
    log("\nFormatting submission.csv ...")
    rows = []
    for i, region_id in enumerate(all_region_ids):
        preds = final_preds[i]
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
    assert len(submission) == 2248, f"Expected 2248 rows, got {len(submission)}"
    assert list(submission.columns) == [
        "region_id", "pred_week1", "pred_week2",
        "pred_week3", "pred_week4", "pred_week5",
    ], f"Unexpected columns: {list(submission.columns)}"
    log("  v Submission assertion passed: 2248 rows, 6 columns.")

    # NaN check (strict)
    assert not submission.isnull().any().any(), "NaN values found in submission!"
    log("  v No NaN values in submission.")

    test_regions  = set(test_df["region_id"].unique())
    train_regions = set(train_df["region_id"].unique())
    assert test_regions == train_regions
    log("  v Train/test regions match (2248).")

    assert submission[["pred_week1","pred_week2","pred_week3",
                        "pred_week4","pred_week5"]].max().max() <= 5.0 + 1e-6
    assert submission[["pred_week1","pred_week2","pred_week3",
                        "pred_week4","pred_week5"]].min().min() >= 0.0 - 1e-6
    log("  v All predictions in [0, 5]  (np.clip last-resort guard enforced).")

    log(f"  submission.csv -> {sub_path}")
    log(f"  Rows (excl. header): {len(submission)}")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    # -- 13. Architecture summary ---------------------------------------------
    log(f"\n[v21 Architecture Summary]")
    _tmp_model = make_model(input_size)
    log("\n" + _tmp_model.architecture_summary(input_size))
    del _tmp_model

    log(f"\nTotal elapsed: {(time.time() - t0):.1f}s")
    log_path = os.path.join(ROOT, "_training_log_21st.txt")
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
        "zero_frac_final":  zero_frac_final,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
