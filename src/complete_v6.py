"""
complete_v6.py -- Fast completion: blind retrain + inference only
=================================================================
Skips cross-validation (already done). Uses saved scaler.pkl.
TARGET_EPOCHS estimated from fold results: fold1=59, fold2=9 → mean~34 → *1.05 = 36
Using 50 epochs to be safe (conservative).

Generates:
    submission.csv
    models/final_model_v6.pt
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features, build_full_train_groups,
    DroughtDataset, FEATURE_COLS, WINDOW_SIZE, HORIZON,
)
from src.model import DroughtLSTM

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")

# From CV results: fold1=59, fold2=9. fold3 unknown but likely ~30-50.
# Using conservative 50 epochs for blind retrain.
TARGET_EPOCHS = 50
BATCH_SIZE    = 512       # Reduced from 1024 to avoid OOM on blind retrain
NUM_WORKERS   = 4         # Reduced from 8
HIDDEN_SIZE, NUM_LAYERS, DROPOUT, HORIZON_ = 64, 2, 0.4, 5
LEARNING_RATE, WEIGHT_DECAY = 1e-3, 1e-3
DROUGHT_THRESHOLD, DROUGHT_PENALTY = 3.0, 4.0

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print(f"Device: {DEVICE}  |  TARGET_EPOCHS={TARGET_EPOCHS}  |  batch_size={BATCH_SIZE}")

_grad_scaler = GradScaler(device="cuda", enabled=USE_AMP)


def weighted_smooth_l1_loss(pred, target):
    element_loss = F.smooth_l1_loss(pred, target, reduction="none")
    weight = torch.where(target > DROUGHT_THRESHOLD,
                         torch.full_like(target, DROUGHT_PENALTY),
                         torch.ones_like(target))
    return (weight * element_loss).mean()


def train_epoch(model, loader, optimizer):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with autocast(device_type=DEVICE.type, enabled=USE_AMP):
            loss = weighted_smooth_l1_loss(model(X), y)
        _grad_scaler.scale(loss).backward()
        _grad_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        _grad_scaler.step(optimizer)
        _grad_scaler.update()
        total += loss.item() * X.size(0)
        n     += X.size(0)
    return total / n


def main():
    # 1. Load data
    print("Loading data ...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))

    # 2. Feature refinement
    print("Refining features ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)

    feat_cols  = [c for c in FEATURE_COLS if c in train_df.columns]
    input_size = len(feat_cols)

    # 3. Load saved scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    print(f"Loading scaler from {scaler_path} ...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # 4. Drop NaN scores
    before = len(train_df)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    print(f"  Dropped {before - len(train_df):,} NaN-score rows.")

    # 5. Build full train dataset
    print("Building full training dataset ...")
    full_groups = build_full_train_groups(train_df)
    full_ds     = DroughtDataset(full_groups, scaler=scaler)
    print(f"  Full train sequences: {len(full_ds):,}")

    loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)

    # 6. Blind retrain
    model = DroughtLSTM(input_size=input_size, hidden_size=HIDDEN_SIZE,
                        num_layers=NUM_LAYERS, dropout=DROPOUT,
                        horizon=HORIZON_).to(DEVICE)
    print(model.architecture_summary(input_size))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TARGET_EPOCHS)

    print(f"\nBlind retraining for {TARGET_EPOCHS} epochs ...")
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'LR':>10}")
    for epoch in range(1, TARGET_EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {loss:>10.4f}  {lr:>10.2e}")

    ckpt = os.path.join(MODELS_DIR, "final_model_v6.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"\nModel saved -> {ckpt}")

    # 7. Inference
    print("\nRunning inference ...")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    predictions = {}
    for region_id, group in test_df.groupby("region_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < WINDOW_SIZE:
            group = pd.concat([group.iloc[[0]] * (WINDOW_SIZE - n), group],
                              ignore_index=True)
        X = group.iloc[-WINDOW_SIZE:][feat_cols].values.astype("float32")
        X = scaler.transform(X)
        X_t = torch.tensor(X).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                pred = model(X_t).squeeze(0).cpu().float().numpy()
        predictions[region_id] = pred

    # 8. Diagnostics
    all_p = np.array(list(predictions.values())).ravel()
    print(f"\n[Submission Diagnostics]")
    print(f"  mean={all_p.mean():.4f}  std={all_p.std():.4f}")
    print(f"  p50={np.percentile(all_p,50):.4f}  p90={np.percentile(all_p,90):.4f}  "
          f"p95={np.percentile(all_p,95):.4f}  p99={np.percentile(all_p,99):.4f}  "
          f"max={all_p.max():.4f}")
    if np.percentile(all_p, 99) < 2.0:
        print("  *** WARNING: p99 < 2.0 — collapse still present! ***")
    else:
        print("  ✓ p99 >= 2.0 — model collapse is BROKEN.")

    # 9. Save submission.csv
    rows = [{"region_id": rid,
             "pred_week1": float(p[0]), "pred_week2": float(p[1]),
             "pred_week3": float(p[2]), "pred_week4": float(p[3]),
             "pred_week5": float(p[4])}
            for rid, p in sorted(predictions.items())]
    sub = pd.DataFrame(rows)
    sub_path = os.path.join(ROOT, "submission.csv")
    sub.to_csv(sub_path, index=False)
    assert len(sub) == 2248
    assert list(sub.columns) == ["region_id","pred_week1","pred_week2",
                                  "pred_week3","pred_week4","pred_week5"]
    print(f"\n✓ submission.csv saved: {len(sub)} rows")
    print(sub.head(5).to_string(index=False))
    return sub


if __name__ == "__main__":
    main()
