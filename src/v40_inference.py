"""
src/inference.py — v40 CORAL Inference & Decoder Only Pipeline
=====================================================================
Workflow:
  1. Load test data and minimal train data (for region_index alignment)
  2. Load the best .ckpt model from models/v40/
  3. Run test inference → collect CORAL probabilities
  4. Dual-Strategy Decoder to generate final predictions
  5. Save submission.csv and display 6-Tier stats
"""

from __future__ import annotations

import gc
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Tensor Cores precision
torch.set_float32_matmul_precision("high")

# ── Project imports ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import TestDataset
from src.model import CORALPatchTSTWrapper

# ── 調整這些參數來測試不同的 Submission ───────────────────────────────────────
# ==============================================================================
# [V40.1] DECODER COLLAPSE FIX: DUAL-STRATEGY TOGGLE
# USE_SOFT_PRIOR = False -> Strategy A: Rigid Global 0.5 Threshold
# USE_SOFT_PRIOR = True  -> Strategy B: Soft Prior Probability Shift
# ==============================================================================
USE_SOFT_PRIOR         = True
PRIOR_SHIFT_MULTIPLIER = 0.2

# 預設儲存檔名 (你可以根據測試策略更改，例如 'submission_soft_0.2.csv')
SUB_FILENAME           = "submission_20percent.csv"

# ── 其他設定 ──────────────────────────────────────────────────────────────────
BATCH_SIZE  = 1024 * 2  # 推論時不需要 backward pass，可以加大 Batch Size 
NUM_WORKERS = 8
PIN_MEMORY  = True

DATA_DIR    = os.path.join(ROOT, "data", "processed")
TRAIN_CSV   = os.path.join(DATA_DIR, "train_processed.csv")
TEST_CSV    = os.path.join(DATA_DIR, "test_processed.csv")
PRIORS_CSV  = os.path.join(DATA_DIR, "region_priors.csv")
CKPT_DIR    = os.path.join(ROOT, "models", "v40")
SUB_PATH    = os.path.join(ROOT, SUB_FILENAME)

def main():
    print("=" * 60)
    print("[Inference] Starting stand-alone inference pipeline...")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. 讀取資料與建立映射 (Minimal Load)
    # ─────────────────────────────────────────────────────────────────────────
    # 為了建立與訓練時完全相同的 region_index，我們只需要 train_df 的 region_id 欄位
    train_regions = pd.read_csv(TRAIN_CSV, usecols=["region_id"])
    test_df       = pd.read_csv(TEST_CSV)
    priors_df     = pd.read_csv(PRIORS_CSV)

    all_region_ids = sorted(
        set(train_regions["region_id"].unique()) | set(test_df["region_id"].unique())
    )
    region_index: dict[str, int] = {rid: i for i, rid in enumerate(all_region_ids)}
    
    # Prior lookups
    region_zero_prob_dict: dict[str, float] = dict(
        zip(priors_df["region_id"], priors_df["region_zero_prob"])
    )
    GLOBAL_ZERO_PRIOR: float = float(priors_df["region_zero_prob"].mean())

    # ─────────────────────────────────────────────────────────────────────────
    # 2. 建立 Test Loader
    # ─────────────────────────────────────────────────────────────────────────
    test_ds = TestDataset(test_df, region_index)
    test_loader = DataLoader(
        test_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        persistent_workers = (NUM_WORKERS > 0),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. 自動尋找並載入最佳模型
    # ─────────────────────────────────────────────────────────────────────────
    ckpt_files = glob.glob(os.path.join(CKPT_DIR, "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in {CKPT_DIR}")
    
    best_ckpt_path = max(ckpt_files, key=os.path.getctime)
    print(f"\n[Inference] Loading model from: {best_ckpt_path}")
    
    model = CORALPatchTSTWrapper.load_from_checkpoint(best_ckpt_path)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. 執行推論 (Inference)
    # ─────────────────────────────────────────────────────────────────────────
    print("[Inference] Running forward pass on test set...")
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            probs = model.predict_proba(x_batch)  # (B, 5, 50)
            all_probs.append(probs.cpu().numpy())

    test_coral_probs = np.concatenate(all_probs, axis=0)  # (2248, 5, 50)
    test_ordered_regions: list[str] = test_ds.ordered_region_ids

    # 清理 VRAM
    del model, all_probs, x_batch
    gc.collect()
    torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Dual-Strategy Decoder 
    # ─────────────────────────────────────────────────────────────────────────
    all_fold_decoded: list[np.ndarray] = []

    if not USE_SOFT_PRIOR:
        print("\n==> Using Strategy A: Rigid Global 0.5 Decoder")
        decoded_fold = np.sum(test_coral_probs >= 0.5, axis=-1) * 0.1
        decoded_fold = np.clip(decoded_fold, 0.0, 5.0)
        all_fold_decoded.append(decoded_fold)

    else:
        print(f"\n==> Using Strategy B: Soft Prior Shift (Multiplier: {PRIOR_SHIFT_MULTIPLIER})")
        decoded_fold_list: list[np.ndarray] = []

        for i, region_id in enumerate(test_ordered_regions):
            p_zero_prior = region_zero_prob_dict.get(region_id, GLOBAL_ZERO_PRIOR)
            p_drought_prior = 1.0 - p_zero_prior

            adjusted_probs = (
                test_coral_probs[i]
                + (p_drought_prior - 0.5) * PRIOR_SHIFT_MULTIPLIER
            )
            region_pred = np.sum(adjusted_probs >= 0.5, axis=-1) * 0.1
            region_pred = np.clip(region_pred, 0.0, 5.0)
            decoded_fold_list.append(region_pred)

        decoded_fold = np.stack(decoded_fold_list, axis=0)
        all_fold_decoded.append(decoded_fold)

    final_preds = np.mean(all_fold_decoded, axis=0)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. 建立與儲存 Submission
    # ─────────────────────────────────────────────────────────────────────────
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

    submission.to_csv(SUB_PATH, index=False)
    print(f"\n[Inference] Submission saved → {SUB_PATH}")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. 顯示 6-Tier 統計資訊
    # ─────────────────────────────────────────────────────────────────────────
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

    print("\n[Inference] 6-Tier Binned Share Percentages:")
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
    print("\n[Inference] Pipeline complete. ✓")

if __name__ == "__main__":
    main()