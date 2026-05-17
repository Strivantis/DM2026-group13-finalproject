"""
train.py – Drought Score Forecasting Pipeline
=============================================
Usage:
    python src/train.py

Outputs:
    submission.csv           – Kaggle submission (133 rows + header)
    models/best_model.pt     – Best checkpoint (lowest val MAE)
    _training_log.txt        – Epoch-level train/val MAE log
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

# ── project root on sys.path ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_train_val_groups,
    DroughtDataset,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
)
from src.model import DroughtLSTM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 256
NUM_EPOCHS    = 120
PATIENCE      = 20    # early-stopping patience (epochs without val improvement)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Weighted MAE loss
# Penalise high-drought classes (score ≥ 3) more heavily to combat imbalance
# ---------------------------------------------------------------------------
def weighted_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(target)
    weights[target >= 2.0] = 1.5
    weights[target >= 3.0] = 2.0
    weights[target >= 4.0] = 3.0
    return (weights * torch.abs(pred - target)).mean()


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_mae, n = 0.0, 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        bs = X.size(0)
        total_loss += loss.item() * bs
        total_mae  += nn.L1Loss()(pred.detach(), y).item() * bs
        n += bs
    return total_loss / n, total_mae / n


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_mae, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_mae += nn.L1Loss()(pred, y).item() * X.size(0)
        n += X.size(0)
    return total_mae / n


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    # ── 1. Load data ────────────────────────────────────────────────────────
    log("Loading processed data …")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    log(f"  train raw: {train_raw.shape}  |  test raw: {test_raw.shape}")

    # ── 2. Feature refinement ───────────────────────────────────────────────
    log("Refining features …")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)

    feat_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    input_size = len(feat_cols)
    log(f"  Input features ({input_size}): {feat_cols}")
    log(f"  train after refinement: {train_df.shape}  |  test: {test_df.shape}")

    # ── 3. Fit scaler on training features ──────────────────────────────────
    log("Fitting StandardScaler on training feature matrix …")
    scaler = StandardScaler()
    # Flatten all train sequences to (N_rows, F) for fitting
    train_feat_matrix = train_df[feat_cols].values.astype(np.float32)
    scaler.fit(train_feat_matrix)

    # Persist scaler for reproducibility
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # ── 4. Build train / val datasets ───────────────────────────────────────
    log("Building train / val index groups …")
    train_groups, val_groups = build_train_val_groups(train_df)
    log(f"  Train regions: {len(train_groups)}  |  Val regions: {len(val_groups)}")

    train_ds = DroughtDataset(train_groups, scaler=scaler)
    val_ds   = DroughtDataset(val_groups,   scaler=scaler)
    log(f"  Train sequences: {len(train_ds):,}  |  Val sequences: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── 5. Model, optimiser, scheduler ─────────────────────────────────────
    model = DroughtLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
    ).to(DEVICE)

    log("\n" + model.architecture_summary(input_size))

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=8
    )
    criterion = weighted_mae

    # ── 6. Training loop ─────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log(f"{'Epoch':>6} {'Train Loss':>12} {'Train MAE':>10} {'Val MAE':>10} {'LR':>10}")
    log("=" * 65)

    best_val_mae = float("inf")
    no_improve   = 0
    best_ckpt    = os.path.join(MODELS_DIR, "best_model.pt")
    history      = []          # [(epoch, train_loss, train_mae, val_mae), …]

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_mae = train_epoch(
            model, train_loader, optimiser, criterion, DEVICE
        )
        val_mae = val_epoch(model, val_loader, DEVICE)
        scheduler.step(val_mae)
        curr_lr = optimiser.param_groups[0]["lr"]

        history.append((epoch, train_loss, train_mae, val_mae))

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            no_improve   = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1

        marker = " ✓" if improved else ""
        row = (f"{epoch:>6}  {train_loss:>12.4f}  {train_mae:>10.4f}"
               f"  {val_mae:>10.4f}  {curr_lr:>10.2e}{marker}")
        log(row)

        if no_improve >= PATIENCE:
            log(f"\nEarly stopping triggered at epoch {epoch} "
                f"(no improvement for {PATIENCE} epochs).")
            break

    log("=" * 65)
    log(f"Best Val MAE: {best_val_mae:.4f}")
    log(f"Training time: {(time.time() - t0):.1f}s")

    # Save training log
    log_path = os.path.join(ROOT, "_training_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nTraining log saved → {log_path}")

    # Save history CSV
    hist_df = pd.DataFrame(history, columns=["epoch", "train_loss", "train_mae", "val_mae"])
    hist_df.to_csv(os.path.join(MODELS_DIR, "training_history.csv"), index=False)

    # ── 7. Inference on test set ─────────────────────────────────────────
    log("\nRunning inference on test set …")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    model.eval()

    predictions = {}   # region_id → np.array of shape (5,)

    for region_id, group in test_df.groupby("region_id"):
        group = group.reset_index(drop=True)
        # Take the LAST WINDOW_SIZE rows as input
        n = len(group)
        if n < WINDOW_SIZE:
            # Edge case: pad with first row if fewer than WINDOW rows available
            pad_count = WINDOW_SIZE - n
            pad_rows  = pd.concat([group.iloc[[0]] * pad_count, group],
                                   ignore_index=True)
            group = pad_rows

        window_df = group.iloc[-WINDOW_SIZE:]
        X = window_df[feat_cols].values.astype(np.float32)   # (W, F)

        # Scale
        X = scaler.transform(X)                               # (W, F)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, W, F)

        with torch.no_grad():
            pred = model(X_tensor).squeeze(0).cpu().numpy()  # (5,)

        # Clip predictions to valid score range [0, 5]
        pred = np.clip(pred, 0.0, 5.0)
        predictions[region_id] = pred

    # ── 8. Format & save submission.csv ─────────────────────────────────
    log("Formatting submission.csv …")
    rows = []
    for region_id, preds in sorted(predictions.items()):
        rows.append({
            "region_id": region_id,
            "pred_week1": preds[0],
            "pred_week2": preds[1],
            "pred_week3": preds[2],
            "pred_week4": preds[3],
            "pred_week5": preds[4],
        })

    submission = pd.DataFrame(rows)
    sub_path   = os.path.join(ROOT, "submission.csv")
    submission.to_csv(sub_path, index=False)

    # ── Sanity checks ────────────────────────────────────────────────────
    assert len(submission) == 133, (
        f"Expected 133 rows, got {len(submission)}"
    )
    assert list(submission.columns) == [
        "region_id", "pred_week1", "pred_week2", "pred_week3",
        "pred_week4", "pred_week5"
    ], f"Unexpected columns: {list(submission.columns)}"

    log(f"  submission.csv saved → {sub_path}")
    log(f"  Rows (excluding header): {len(submission)}  ✓")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    return history, best_val_mae, input_size, submission


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
