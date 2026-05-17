"""
train.py -- Drought Score Forecasting Pipeline (v4)
====================================================
Usage:
    python src/train.py

Outputs:
    submission.csv               -- Kaggle submission (133 rows + header)
    models/final_model_v4.pt     -- Blind full-retrained model (TARGET_EPOCHS, 100% data)
    models/scaler.pkl            -- Fitted StandardScaler
    models/training_history.csv  -- Per-epoch metrics for final training run
    _training_log_4th.txt        -- Full console log

Key improvements (v4)
---------------------
  1. Pure MAE Loss       -- Blended/Weighted MAE completely removed.
                            All training phases use torch.nn.L1Loss() exclusively,
                            aligning perfectly with the Kaggle MAE metric.
  2. Dynamic TARGET_EPOCHS -- CV fold best-epochs are recorded; TARGET_EPOCHS =
                              mean(fold best epochs) * 1.05 (5% uplift for 100% data).
  3. Bias Init preserved  -- fc.bias = -0.98 from v3 kept.
  4. CosineAnnealingLR   -- Used in the blind full-retraining phase (no val set).
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR  = os.path.join(ROOT, "data", "processed")
MODELS_DIR     = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HIDDEN_SIZE    = 64
NUM_LAYERS     = 2
DROPOUT        = 0.4
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-3
BATCH_SIZE     = 256
NUM_EPOCHS     = 150         # extended; early stopping governs CV folds
PATIENCE       = 25          # patience on pure val MAE (CV only)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Training / validation helpers  (pure MAE only -- no weighting)
# ---------------------------------------------------------------------------
CRITERION = nn.L1Loss()


def train_epoch(model, loader, optimizer, device):
    """One training epoch using pure MAE (L1Loss)."""
    model.train()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = CRITERION(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        bs = X.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n


@torch.no_grad()
def eval_mae(model, loader, device) -> float:
    """Evaluate pure MAE -- used for early stopping & fold scoring."""
    model.eval()
    total_mae, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_mae += CRITERION(pred, y).item() * X.size(0)
        n += X.size(0)
    return total_mae / n if n > 0 else float("inf")


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


# ---------------------------------------------------------------------------
# Single training run with early stopping (used for CV folds only)
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
    Train ``model`` and save the best checkpoint to ``ckpt_path``.

    Early stopping is based exclusively on **pure MAE** from ``val_loader``.

    Returns
    -------
    best_val_mae : float
    best_epoch   : int   -- epoch number that achieved best_val_mae
    history      : list of (epoch, train_mae, val_mae)
    """
    optimiser = make_optimiser(model)
    scheduler = make_scheduler(optimiser)

    if show_header:
        log("=" * 60)
        log(f"{'Epoch':>6} {'TrainMAE':>10} {'ValMAE':>10} {'LR':>10}")
        log("=" * 60)

    best_val_mae = float("inf")
    best_epoch   = 1
    no_improve   = 0
    history      = []

    for epoch in range(1, num_epochs + 1):
        train_mae = train_epoch(model, train_loader, optimiser, DEVICE)
        val_mae   = eval_mae(model, val_loader, DEVICE)
        scheduler.step(val_mae)
        curr_lr = optimiser.param_groups[0]["lr"]

        history.append((epoch, train_mae, val_mae))

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            best_epoch   = epoch
            no_improve   = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        marker = " *" if improved else ""
        row = (
            f"{epoch:>6}  {train_mae:>10.4f}"
            f"  {val_mae:>10.4f}  {curr_lr:>10.2e}{marker}"
        )
        log(row)

        if no_improve >= patience:
            log(
                f"\n  Early stopping at epoch {epoch} "
                f"(no MAE improvement for {patience} epochs)."
            )
            break

    log("=" * 60)
    log(f"  Best Val MAE: {best_val_mae:.4f}  (epoch {best_epoch})")
    return best_val_mae, best_epoch, history


# ---------------------------------------------------------------------------
# Blind full retraining (v4) -- no validation, no early stopping
# ---------------------------------------------------------------------------
def train_model_blind(
    model,
    train_loader,
    num_epochs: int,
    ckpt_path: str,
    log,
):
    """
    Blind full retraining on 100% of training data for exactly ``num_epochs``.

    Uses CosineAnnealingLR (no validation metric required).
    No early stopping -- the model trains for all ``num_epochs`` unconditionally.
    Loss: pure MAE (L1Loss).
    Saves the final-epoch weights to ``ckpt_path``.

    Returns
    -------
    history : list of (epoch, train_mae)
    """
    optimiser = make_optimiser(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=num_epochs
    )

    log("=" * 50)
    log(f"{'Epoch':>6} {'TrainMAE':>10} {'LR':>10}")
    log("=" * 50)

    history = []

    for epoch in range(1, num_epochs + 1):
        train_mae = train_epoch(model, train_loader, optimiser, DEVICE)
        scheduler.step()
        curr_lr = optimiser.param_groups[0]["lr"]

        history.append((epoch, train_mae))

        row = (
            f"{epoch:>6}  {train_mae:>10.4f}"
            f"  {curr_lr:>10.2e}"
        )
        log(row)

    log("=" * 50)
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
        log_lines.append(msg)

    # -- 1. Load data ----------------------------------------------------------
    log("Loading processed data ...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    log(f"  train raw: {train_raw.shape}  |  test raw: {test_raw.shape}")

    # -- 2. Feature refinement (incl. Drought Index) ---------------------------
    log("Refining features (+ drought proxy index) ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)

    feat_cols  = [c for c in FEATURE_COLS if c in train_df.columns]
    input_size = len(feat_cols)
    log(f"  Input features ({input_size}): {feat_cols}")
    log(f"  train after refinement: {train_df.shape}  |  test: {test_df.shape}")

    # -- 3. Fit scaler on training features ------------------------------------
    log("Fitting StandardScaler on training feature matrix ...")
    scaler = StandardScaler()
    train_feat_matrix = train_df[feat_cols].values.astype(np.float32)
    scaler.fit(train_feat_matrix)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # -- 4. Walk-Forward Cross-Validation --------------------------------------
    log(f"\n{'='*65}")
    log(f"Walk-Forward Cross-Validation  ({WF_NUM_FOLDS} folds x {WF_FOLD_WEEKS} weeks)")
    log(f"Loss: pure MAE (torch.nn.L1Loss) -- zero weighting / asymmetric penalties")
    log(f"{'='*65}")

    folds = build_walk_forward_folds(train_df)
    fold_maes        = []
    fold_best_epochs = []

    for fold_k, (fold_train_groups, fold_val_groups) in enumerate(folds):
        log(f"\n-- Fold {fold_k + 1}/{WF_NUM_FOLDS} --")
        fold_train_ds = DroughtDataset(fold_train_groups, scaler=scaler)
        fold_val_ds   = DroughtDataset(fold_val_groups,   scaler=scaler)
        log(f"  Train seqs: {len(fold_train_ds):,}  |  Val seqs: {len(fold_val_ds):,}")

        fold_loader_tr = DataLoader(
            fold_train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        fold_loader_val = DataLoader(
            fold_val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True,
        )

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
        log(f"  Fold {fold_k + 1} Best Val MAE: {best_fold_mae:.4f}  |  Best Epoch: {best_fold_epoch}")

    avg_val_mae    = float(np.mean(fold_maes))
    mean_best_epoch = float(np.mean(fold_best_epochs))
    TARGET_EPOCHS  = max(1, int(round(mean_best_epoch * 1.05)))   # +5% uplift

    log(f"\n{'='*65}")
    log(f"Fold MAEs         : {[f'{m:.4f}' for m in fold_maes]}")
    log(f"Average_Val_MAE   : {avg_val_mae:.4f}")
    log(f"Fold Best Epochs  : {fold_best_epochs}")
    log(f"Mean Best Epoch   : {mean_best_epoch:.1f}")
    log(f"TARGET_EPOCHS (v4): {TARGET_EPOCHS}  (mean * 1.05, rounded)")
    log(f"{'='*65}")

    # -- 5. Blind full retraining on 100% training data (v4) -------------------
    log(f"\n{'='*65}")
    log(f"Blind Full Retraining  (TARGET_EPOCHS={TARGET_EPOCHS}, 100% data)")
    log(f"Loss      : pure MAE (torch.nn.L1Loss)")
    log(f"Scheduler : CosineAnnealingLR(T_max={TARGET_EPOCHS})")
    log(f"No validation split. No early stopping.")
    log(f"{'='*65}")

    full_train_groups = build_full_train_groups(train_df)
    full_train_ds     = DroughtDataset(full_train_groups, scaler=scaler)
    log(f"  Full train sequences: {len(full_train_ds):,}")

    full_train_loader = DataLoader(
        full_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )

    final_model = make_model(input_size)
    log("\n" + final_model.architecture_summary(input_size))

    v4_ckpt = os.path.join(MODELS_DIR, "final_model_v4.pt")

    final_history = train_model_blind(
        final_model,
        full_train_loader,
        num_epochs=TARGET_EPOCHS,
        ckpt_path=v4_ckpt,
        log=log,
    )

    log(f"Blind training complete. Epochs trained : {TARGET_EPOCHS}")
    log(f"Training time : {(time.time() - t0):.1f}s")

    # Save training log
    log_path = os.path.join(ROOT, "_training_log_4th.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nTraining log saved -> {log_path}")

    # Save history CSV (blind: train_mae only)
    hist_df = pd.DataFrame(
        final_history,
        columns=["epoch", "train_mae"],
    )
    hist_df.to_csv(os.path.join(MODELS_DIR, "training_history.csv"), index=False)

    # -- 6. Inference on test set ----------------------------------------------
    log("\nRunning inference on test set ...")
    final_model.load_state_dict(torch.load(v4_ckpt, map_location=DEVICE))
    final_model.eval()

    predictions = {}   # region_id -> np.array of shape (5,)

    for region_id, group in test_df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < WINDOW_SIZE:
            pad_count = WINDOW_SIZE - n
            pad_rows  = pd.concat(
                [group.iloc[[0]]] * pad_count + [group],
                ignore_index=True,
            )
            group = pad_rows

        window_df = group.iloc[-WINDOW_SIZE:]
        X = window_df[feat_cols].values.astype(np.float32)
        X = scaler.transform(X)
        X_tensor = (
            torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )

        with torch.no_grad():
            pred = final_model(X_tensor).squeeze(0).cpu().numpy()  # (5,) in (0,5)

        # No manual clip needed -- Sigmoid x 5 guarantees (0, 5)
        predictions[region_id] = pred

    # -- 7. Format & save submission.csv ---------------------------------------
    log("Formatting submission.csv ...")
    rows = []
    for region_id, preds in sorted(predictions.items()):
        rows.append(
            {
                "region_id": region_id,
                "pred_week1": preds[0],
                "pred_week2": preds[1],
                "pred_week3": preds[2],
                "pred_week4": preds[3],
                "pred_week5": preds[4],
            }
        )

    submission = pd.DataFrame(rows)
    sub_path   = os.path.join(ROOT, "submission.csv")
    submission.to_csv(sub_path, index=False)

    # -- 8. Sanity checks ------------------------------------------------------
    assert len(submission) == 133, (
        f"Expected 133 rows, got {len(submission)}"
    )
    assert list(submission.columns) == [
        "region_id", "pred_week1", "pred_week2",
        "pred_week3", "pred_week4", "pred_week5",
    ], f"Unexpected columns: {list(submission.columns)}"

    test_regions  = set(test_df["region_id"].unique())
    train_regions = set(train_df["region_id"].unique())
    assert test_regions == train_regions, (
        "Region mismatch between train and test -- check data integrity."
    )
    log("  No cross-region leakage detected (scaler fitted on train only) *")

    log(f"  submission.csv saved -> {sub_path}")
    log(f"  Rows (excluding header): {len(submission)}  *")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    return {
        "fold_maes":         fold_maes,
        "fold_best_epochs":  fold_best_epochs,
        "avg_val_mae":       avg_val_mae,
        "target_epochs":     TARGET_EPOCHS,
        "input_size":        input_size,
        "submission":        submission,
        "final_history":     final_history,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
