"""
src/train.py  — v40 PatchTST + CORAL  Training & Inference Pipeline
=====================================================================
Workflow:
  1. Load train / test processed CSVs
  2. Build native DroughtDataset (sliding windows, NO manual scaling)
  3. Train CORALPatchTSTWrapper (PatchTST + CORAL head) via PyTorch Lightning
  4. Run test inference → collect CORAL exceedance probabilities (2248, 5, 50)
  5. Uncapped Per-Region Prior Decoder  τ ∈ [0.40, 0.999]
  6. Write submission.csv  (2248 rows × 6 columns)

⚠️  DO NOT run this file via Cline automation.
    The user executes:  python src/train.py
"""

from __future__ import annotations

import gc
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

# Tensor Cores precision — trades negligible accuracy for significant throughput gain
torch.set_float32_matmul_precision("high")
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader

# ── Project imports ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    DroughtDataset,
    TestDataset,
    build_region_index,
    FEATURE_COLS,
    N_FEATURES,
    CONTEXT_LEN,
    HORIZON_LEN,
)
from src.model import CORALPatchTSTWrapper


# ── Hyper-parameters ─────────────────────────────────────────────────────────
BATCH_SIZE   = 1024
NUM_WORKERS  = 8
PIN_MEMORY   = True
MAX_EPOCHS   = 30
VAL_WINDOWS  = 75      # trailing time-windows reserved per region for validation
LR           = 1e-3
HIDDEN_SIZE  = 64
N_HEADS      = 4
PATCH_LEN    = 4
STRIDE       = 2
SEED         = 42

pl.seed_everything(SEED, workers=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(ROOT, "data", "processed")
TRAIN_CSV  = os.path.join(DATA_DIR, "train_processed.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test_processed.csv")
PRIORS_CSV = os.path.join(DATA_DIR, "region_priors.csv")
SUB_PATH   = os.path.join(ROOT, "submission.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("[v40] Loading data …")
train_df  = pd.read_csv(TRAIN_CSV)
test_df   = pd.read_csv(TEST_CSV)
priors_df = pd.read_csv(PRIORS_CSV)

print(f"  Train: {train_df.shape}  |  Test: {test_df.shape}")
print(f"  Regions: {train_df['region_id'].nunique()} train  "
      f"/ {test_df['region_id'].nunique()} test")

# Build per-region zero-inflation prior lookup
region_zero_prob_dict: dict[str, float] = dict(
    zip(priors_df["region_id"], priors_df["region_zero_prob"])
)
# Global fallback prior (train-set mean zero-inflation rate)
GLOBAL_ZERO_PRIOR: float = float(priors_df["region_zero_prob"].mean())
print(f"  Global zero-prior fallback: {GLOBAL_ZERO_PRIOR:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build Datasets / DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
print("\n[v40] Building datasets …")
# Shared region index: must cover both train and test regions
all_region_ids = sorted(
    set(train_df["region_id"].unique()) | set(test_df["region_id"].unique())
)
region_index: dict[str, int] = {rid: i for i, rid in enumerate(all_region_ids)}
print(f"  Total unique regions: {len(region_index)}")

train_ds = DroughtDataset(
    train_df, region_index, val_split=False, val_windows=VAL_WINDOWS
)
val_ds = DroughtDataset(
    train_df, region_index, val_split=True,  val_windows=VAL_WINDOWS
)
test_ds = TestDataset(test_df, region_index)

print(f"  Train windows: {len(train_ds):,}")
print(f"  Val   windows: {len(val_ds):,}")
print(f"  Test  regions: {len(test_ds)}")

train_loader = DataLoader(
    train_ds,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = NUM_WORKERS,
    pin_memory  = PIN_MEMORY,
    drop_last   = True,
    persistent_workers = (NUM_WORKERS > 0),
)
val_loader = DataLoader(
    val_ds,
    batch_size  = BATCH_SIZE * 2,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = PIN_MEMORY,
    persistent_workers = (NUM_WORKERS > 0),
)
test_loader = DataLoader(
    test_ds,
    batch_size  = BATCH_SIZE * 2,
    shuffle     = False,   # preserve order for region mapping
    num_workers = NUM_WORKERS,
    pin_memory  = PIN_MEMORY,
    persistent_workers = (NUM_WORKERS > 0),
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model instantiation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[v40] Instantiating CORALPatchTSTWrapper …")
model = CORALPatchTSTWrapper(
    n_features  = N_FEATURES,
    hidden_size = HIDDEN_SIZE,
    n_heads     = N_HEADS,
    patch_len   = PATCH_LEN,
    stride      = STRIDE,
    lr          = LR,
)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {n_params:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lightning Trainer + Training
# ─────────────────────────────────────────────────────────────────────────────
ckpt_dir = os.path.join(ROOT, "models", "v40")
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath   = ckpt_dir,
        filename  = "coral_patchtst_v40_{epoch:02d}_{val_mae:.4f}",
        monitor   = "val_mae",
        mode      = "min",
        save_top_k = 1,
        verbose   = True,
    ),
    EarlyStopping(
        monitor  = "val_mae",
        patience = 8,
        mode     = "min",
        verbose  = True,
    ),
    LearningRateMonitor(logging_interval="epoch"),
]

trainer = pl.Trainer(
    max_epochs        = MAX_EPOCHS,
    accelerator       = "gpu" if torch.cuda.is_available() else "cpu",
    devices           = 1,
    callbacks         = callbacks,
    log_every_n_steps = 50,
    enable_progress_bar = True,
    deterministic     = False,   # allow cuDNN optimizations
    precision         = "32-true",   # Full FP32 — guards RevIN ε from AMP underflow
)

import glob

print("\n[v40] Preparing to train …")

# 尋找已存在的檢查點 (副檔名為 .ckpt)
ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
last_ckpt = None

if ckpt_files:
    # 根據檔案修改時間排序，找出最新儲存的檢查點
    last_ckpt = max(ckpt_files, key=os.path.getctime)
    print(f"  [Info] 找到之前的檢查點，將從此處繼續訓練: {last_ckpt}")
else:
    print("  [Info] 未找到任何檢查點，將從 Epoch 0 開始全新訓練。")

print("\n[v40] Training …")
# 將 last_ckpt 傳遞給 ckpt_path 參數
trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Load best checkpoint → Inference on test set
# ─────────────────────────────────────────────────────────────────────────────
best_ckpt_path = trainer.checkpoint_callback.best_model_path
print(f"\n[v40] Best checkpoint: {best_ckpt_path}")
best_model = CORALPatchTSTWrapper.load_from_checkpoint(best_ckpt_path)
best_model.eval()
best_model.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(device)

print("[v40] Running test inference …")
all_probs: list[np.ndarray] = []   # each: (batch, 5, 50)

with torch.no_grad():
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        # predict_proba() = sigmoid(forward()) — returns [0,1] probs for decoder
        probs = best_model.predict_proba(x_batch)  # (B, 5, 50)
        all_probs.append(probs.cpu().numpy())

test_coral_probs = np.concatenate(all_probs, axis=0)  # (2248, 5, 50)
print(f"  CORAL probability tensor shape: {test_coral_probs.shape}")

# Ordered region IDs matching inference batch order
test_ordered_regions: list[str] = test_ds.ordered_region_ids
assert len(test_ordered_regions) == test_coral_probs.shape[0], (
    f"Region count mismatch: {len(test_ordered_regions)} vs {test_coral_probs.shape[0]}"
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. [v40.1] DUAL-STRATEGY DECODER  (Decoder Collapse Fix)
# ─────────────────────────────────────────────────────────────────────────────

# ==============================================================================
# [V40.1] DECODER COLLAPSE FIX: DUAL-STRATEGY TOGGLE
# USE_SOFT_PRIOR = False -> Strategy A: Rigid Global 0.5 Threshold (Fast Vectorized)
# USE_SOFT_PRIOR = True  -> Strategy B: Soft Prior Probability Shift (Bayesian adjustment)
# PRIOR_SHIFT_MULTIPLIER -> Tunes the strength of the prior shift (e.g., 0.1 to 0.3)
# ==============================================================================
USE_SOFT_PRIOR          = FALSE
PRIOR_SHIFT_MULTIPLIER  = 0.2

all_fold_decoded: list[np.ndarray] = []

if not USE_SOFT_PRIOR:
    # ------------------------------------------------------------------
    # Strategy A: Rigid Global 0.5 Decoder
    # Completely vectorized — no per-region loop, no prior lookup.
    # Applies a single universal threshold τ = 0.5 across all regions.
    # Best when the model's probability scale is well-calibrated (~0.5)
    # and no Bayesian regional adjustment is desired.
    # ------------------------------------------------------------------
    print("==> [v40.1] Using Strategy A: Rigid Global 0.5 Decoder")
    decoded_fold = np.sum(test_coral_probs >= 0.5, axis=-1) * 0.1  # (2248, 5)
    decoded_fold = np.clip(decoded_fold, 0.0, 5.0)
    all_fold_decoded.append(decoded_fold)

else:
    # ------------------------------------------------------------------
    # Strategy B: Soft Prior Probability Shift
    # Per-region Bayesian adjustment: shifts raw probabilities up/down
    # by (p_drought_prior - 0.5) * PRIOR_SHIFT_MULTIPLIER before
    # applying the rigid 0.5 cutoff.
    #   • Chronically dry regions (low zero_prob → high p_drought) → bump up
    #   • Wet regions (high zero_prob → low p_drought)             → slight penalty
    # PRIOR_SHIFT_MULTIPLIER controls the magnitude (0.1 = subtle, 0.3 = strong)
    # ------------------------------------------------------------------
    print(f"==> [v40.1] Using Strategy B: Soft Prior Shift "
          f"(Multiplier: {PRIOR_SHIFT_MULTIPLIER})")
    decoded_fold_list: list[np.ndarray] = []

    for i, region_id in enumerate(test_ordered_regions):
        p_zero_prior = region_zero_prob_dict.get(region_id, GLOBAL_ZERO_PRIOR)

        # Probability of experiencing drought = complement of zero-inflation rate
        p_drought_prior = 1.0 - p_zero_prior

        # Bayesian shift: positive for dry regions, negative for wet regions
        adjusted_probs = (
            test_coral_probs[i]
            + (p_drought_prior - 0.5) * PRIOR_SHIFT_MULTIPLIER
        )  # (5, 50)

        # Integrate using rigid 0.5 threshold on shifted probabilities
        region_pred = np.sum(adjusted_probs >= 0.5, axis=-1) * 0.1  # (5,)
        region_pred = np.clip(region_pred, 0.0, 5.0)

        decoded_fold_list.append(region_pred)

    decoded_fold = np.stack(decoded_fold_list, axis=0)   # (2248, 5)
    all_fold_decoded.append(decoded_fold)

# Average across folds (single fold → no change; ready for multi-fold extension)
final_preds = np.mean(all_fold_decoded, axis=0)      # (2248, 5)
print(f"  Final predictions shape: {final_preds.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VRAM Defense: purge training objects
# ─────────────────────────────────────────────────────────────────────────────
del trainer, model, best_model, all_probs, test_coral_probs
del train_loader, val_loader
gc.collect()
torch.cuda.empty_cache()
print("[v40] VRAM purged.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Build & save submission
# ─────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame(
    {
        "region_id": test_ordered_regions,
        "pred_week1": final_preds[:, 0],
        "pred_week2": final_preds[:, 1],
        "pred_week3": final_preds[:, 2],
        "pred_week4": final_preds[:, 3],
        "pred_week5": final_preds[:, 4],
    }
).sort_values("region_id").reset_index(drop=True)

assert submission.shape == (2248, 6), (
    f"Submission shape error: expected (2248, 6), got {submission.shape}"
)

submission.to_csv(SUB_PATH, index=False)
print(f"\n[v40] Submission saved → {SUB_PATH}")
print(f"  Shape: {submission.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. 6-Tier binned share percentages
# ─────────────────────────────────────────────────────────────────────────────
all_values = final_preds.flatten()
total = len(all_values)

tier_edges = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.001]
tier_labels = [
    "Tier-0  [0.0]      ",
    "Tier-1  (0.0–0.5]  ",
    "Tier-2  (0.5–1.0]  ",
    "Tier-3  (1.0–2.0]  ",
    "Tier-4  (2.0–3.0]  ",
    "Tier-5  (3.0–5.0]  ",
]

print("\n[v40] 6-Tier Binned Share Percentages:")
print("-" * 45)
for j in range(len(tier_labels)):
    lo, hi = tier_edges[j], tier_edges[j + 1]
    count = int(np.sum((all_values >= lo) & (all_values < hi)))
    pct = count / total * 100
    print(f"  {tier_labels[j]}  {count:5d} / {total}  ({pct:5.2f}%)")
print("-" * 45)
print(f"  Overall mean score: {all_values.mean():.4f}")
print(f"  Overall  std score: {all_values.std():.4f}")
print(f"  Fraction == 0.0   : {(all_values == 0.0).mean() * 100:.2f}%")

print("\n[v40] Pipeline complete. ✓")
