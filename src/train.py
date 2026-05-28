"""
train.py -- Drought Score Forecasting Pipeline (v32 -- Robust Sequence Network)
======================================================================================
Usage:
    python src/train.py

Outputs:
    submission.csv                        -- Kaggle submission (2248 rows x 6 cols)
    models/dsn_fold{k}.pt                 -- DroughtSequenceNet checkpoint per fold
    _training_log_32nd.txt                -- Full console log

v32 Architecture (Robust Sequence Network -- Severity-Aware Training)
------------------------------------------------------------------------
  INPUT PIPELINE:
    Data reshaped into a 3D chronological time-series tensor of shape (B, 13, 27).
    Training-time Gaussian noise (sigma=0.05) injected at input for scale-shift immunity.

  MODEL: DroughtSequenceNet v32 (src/model.py)
    Branch A (Dual-Layer Dilated TCN):
      Conv1d(27->64, kernel=3, dilation=1) + GELU + InstanceNorm1d
      Conv1d(64->64, kernel=3, dilation=2) + GELU + InstanceNorm1d
      + AdaptiveAvgPool1d(1)  -> Multi-week anomaly vector (B, 64)
    Branch B (Spatial Dropout + BiLSTM):
      Dropout1d(p=0.1) zeroes entire time-step columns before LSTM
      LSTM(27->64, layers=3, bidirectional=True, batch_first=True)
      -> Final hidden state (B, 128)  [forward 64 + backward 64]
    Fusion MLP:
      Linear(192->128) -> GELU -> Dropout(0.2) -> Linear(128->5)
      -> (B, 5) raw unbounded regression  [NO final activation]

  TRAINING:
    Loss      : SeverityWeightedL1  mean(|pred-true| * (1.0 + true * 0.3))
                y=0 weight: 1.0x baseline  |  y=5 weight: 2.5x amplified
    Optimizer : AdamW(lr=1e-3, weight_decay=1e-3)
    Scheduler : CosineAnnealingLR(T_max=100, eta_min=1e-6)
    Epochs    : 100 (hard limit)
    Batch     : 1024 (fallback 512 if VRAM error)
    Early Stop: monitor validation SeverityWeightedL1; patience=10 epochs

  INFERENCE:
    All 5 fold models predict on test set independently.
    Median blending: np.median(fold_preds, axis=0)
    Micro-Zero Floor: np.where(preds < 0.05, 0.0, preds)  [noise wiper]
    Physical clip: np.clip(final, 0.0, 5.0)

  CV:
    5-Fold StratifiedGroupKFold, grouped by region_id,
    stratified on 10-quantile bins of per-region mean score.

v32 Changes over v31
------------------------
  [ABOLISH] nn.L1Loss() pure MAE objective
  [ABOLISH] Single-layer Conv1d Branch A (dilation=1 only)
  [ABOLISH] GroupNorm in Branch A
  [ABOLISH] No dropout before BiLSTM

  [INTRODUCE] SeverityWeightedL1: mean(|pred-true| * (1.0 + true*0.3))
  [INTRODUCE] Dual-layer Dilated TCN (dilation=1 then dilation=2) in Branch A
  [INTRODUCE] InstanceNorm1d per conv layer in Branch A
  [INTRODUCE] Dropout1d(p=0.1) spatial step dropout before BiLSTM in Branch B
  [INTRODUCE] Training-time Gaussian noise injection (sigma=0.05)
  [INTRODUCE] Epochs extended: 50 -> 100
  [INTRODUCE] Scheduler T_max extended: 50 -> 100
  [INTRODUCE] Micro-Zero Floor: np.where(preds < 0.05, 0.0, preds)

  [RETAIN] refine_features() feature pipeline from v26
  [RETAIN] 5-Fold StratifiedGroupKFold, group=region_id, 10-quantile strata
  [RETAIN] Leakage-free per-fold Target Encoding injection
  [RETAIN] 2248-row submission.csv with 6 columns
  [RETAIN] np.median(fold_preds, axis=0) zero-interference blending
"""

import os
import sys
import time
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc

# -- project root on sys.path --------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_stratified_group_cv_folds,
    build_sequence_dataset,
    build_sequence_test,
    DroughtSequenceDataset,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
    WF_NUM_FOLDS,
    GAP_WEEKS,
    N_TS_FOLDS,
    TS_SHIFT_WEEKS,
)
from src.model import DroughtSequenceNet

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


set_seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Cross-validation
N_FOLDS = 5

# Training hyperparameters (v32)
EPOCHS         = 100
BATCH_SIZE     = 1024
LR             = 1e-3
WEIGHT_DECAY   = 1e-3
ETA_MIN        = 1e-6
EARLY_STOP_PAT = 10   # patience in epochs for early stopping

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters (v32 DroughtSequenceNet)
SEQ_LEN     = WINDOW_SIZE        # 13
N_FEATURES  = len(FEATURE_COLS)  # 27
CONV_OUT    = 64
LSTM_HIDDEN = 64
LSTM_LAYERS = 3
MLP_HIDDEN  = 128
DROPOUT     = 0.2


# ---------------------------------------------------------------------------
# Severity-Weighted L1 Loss (v32)
# ---------------------------------------------------------------------------
def severity_weighted_l1_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Severity-Weighted L1 Loss: punishes errors on high-drought targets proportionally.

    Formula:
        Loss = mean( |pred - true| * (1.0 + true * 0.3) )

    Weights:
        y_true = 0.0  ->  weight = 1.0x  (baseline, no amplification)
        y_true = 1.0  ->  weight = 1.3x
        y_true = 2.0  ->  weight = 1.6x
        y_true = 3.0  ->  weight = 1.9x
        y_true = 4.0  ->  weight = 2.2x
        y_true = 5.0  ->  weight = 2.5x  (peak amplification)

    Physical meaning: counteracts the 60% zero-inflation gravitational sink
    by imposing 2.5x stronger gradient pressure on extreme drought predictions,
    forcing the network to break through zero-collapsing suppression.

    Parameters
    ----------
    preds   : (B, H) tensor -- raw model predictions
    targets : (B, H) tensor -- ground-truth drought scores in [0, 5]

    Returns
    -------
    scalar loss tensor
    """
    weights = 1.0 + targets * 0.3    # (B, H); range [1.0, 2.5] for targets in [0, 5]
    loss    = torch.abs(preds - targets) * weights
    return loss.mean()


# ---------------------------------------------------------------------------
# Target Encoding helpers (leakage-free, per-fold)
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
    result = []
    for entry in groups:
        if len(entry) == 4:
            group, i_min, i_max, _gap = entry
        else:
            group, i_min, i_max = entry
        g   = group.copy()
        rid = g["region_id"].iloc[0]
        mean_s, zero_p = te_map.get(rid, (global_mean, global_zero_prob))
        g["region_mean_score"] = np.float32(mean_s)
        g["region_zero_prob"]  = np.float32(zero_p)
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
# Binned Error Matrix diagnostic
# ---------------------------------------------------------------------------
INTERVAL_LABELS = [
    "Interval 0  [Absolute Zero     y == 0.0       ]",
    "Interval 1  [Mild Drought      0.0 < y <= 1.0 ]",
    "Interval 2  [Moderate Drought  1.0 < y <= 2.0 ]",
    "Interval 3  [Severe Drought    2.0 < y <= 3.0 ]",
    "Interval 4  [Extreme Drought   3.0 < y <= 4.0 ]",
    "Interval 5  [Exceptional       4.0 < y <= 5.0 ]",
]

def _interval_mask(y_true: np.ndarray, interval_idx: int) -> np.ndarray:
    if interval_idx == 0:
        return y_true == 0.0
    elif interval_idx == 1:
        return (y_true > 0.0) & (y_true <= 1.0)
    elif interval_idx == 2:
        return (y_true > 1.0) & (y_true <= 2.0)
    elif interval_idx == 3:
        return (y_true > 2.0) & (y_true <= 3.0)
    elif interval_idx == 4:
        return (y_true > 3.0) & (y_true <= 4.0)
    elif interval_idx == 5:
        return (y_true > 4.0) & (y_true <= 5.0)
    else:
        raise ValueError(f"Unknown interval index: {interval_idx}")


def print_binned_error_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                               fold_k: int, log_fn) -> None:
    log_fn("")
    log_fn("  " + "=" * 80)
    log_fn(f"  BINNED ERROR MATRIX  --  Fold {fold_k}  (OOF Predictions, SeverityWeightedL1)")
    log_fn("  " + "=" * 80)
    header = (
        f"  {'Interval':<50}  {'Count':>7}  {'AvgTrue':>8}  "
        f"{'AvgPred':>8}  {'MAE':>8}"
    )
    log_fn(header)
    log_fn("  " + "-" * 78)

    for idx, label in enumerate(INTERVAL_LABELS):
        mask = _interval_mask(y_true, idx)
        n    = int(mask.sum())
        if n == 0:
            log_fn(
                f"  {label:<50}  {n:>7}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}"
            )
            continue
        avg_true = float(y_true[mask].mean())
        avg_pred = float(y_pred[mask].mean())
        mae      = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
        log_fn(
            f"  {label:<50}  {n:>7,}  {avg_true:>8.4f}  "
            f"{avg_pred:>8.4f}  {mae:>8.4f}"
        )

    log_fn("  " + "=" * 80)
    log_fn("")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full pass over the training DataLoader, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)   # (B, 13, 27)
        y_batch = y_batch.to(device, non_blocking=True)   # (B, 5)

        optimizer.zero_grad()
        preds = model(X_batch)          # (B, 5)
        loss  = criterion(preds, y_batch)
        loss.backward()
        # Gradient clipping for LSTM stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _eval_one_epoch(model, loader, criterion, device):
    """Evaluate model on a DataLoader, return mean loss and all predictions."""
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        preds = model(X_batch)
        loss  = criterion(preds, y_batch)

        total_loss  += loss.item()
        n_batches   += 1
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    mean_loss   = total_loss / max(n_batches, 1)
    all_preds   = np.concatenate(all_preds,   axis=0)   # (N_val, 5)
    all_targets = np.concatenate(all_targets, axis=0)   # (N_val, 5)
    return mean_loss, all_preds, all_targets


@torch.no_grad()
def _predict(model, X_np, batch_size, device):
    """Run inference on a 3D NumPy array, return predictions as NumPy."""
    model.eval()
    dataset = DroughtSequenceDataset(X_np)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=(device.type == "cuda"))
    preds_list = []
    for X_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        preds_list.append(model(X_batch).cpu().numpy())
    return np.concatenate(preds_list, axis=0)   # (N, 5)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    # -- 0. Pipeline description -----------------------------------------------
    log("=" * 90)
    log("Drought Forecasting Pipeline  v32  (Robust Sequence Network)")
    log("DroughtSequenceNet  |  Dual Dilated TCN Branch  +  Spatial Dropout + BiLSTM Branch")
    log("Input  : (B, 13, 27) 3D chronological time-series tensor")
    log("Target : (B, 5) multi-step scores (next 5 weeks)")
    log("Loss   : SeverityWeightedL1  mean(|pred-true| * (1.0 + true*0.3))")
    log("         y=0 weight: 1.0x  |  y=5 weight: 2.5x  (drought severity amplification)")
    log("Blend  : np.median(fold_preds, axis=0)  (zero-interference)")
    log("Floor  : np.where(preds < 0.05, 0.0, preds)  (Micro-Zero noise wiper)")
    log("Clip   : np.clip(0.0, 5.0)  (physical constraint)")
    log("CV     : 5-Fold StratifiedGroupKFold  |  group=region_id  |  10-quantile strata")
    log("=" * 90)
    log("")
    log("v32 Configuration:")
    log(f"  Device         : {DEVICE}")
    log(f"  Epochs (max)   : {EPOCHS}")
    log(f"  Batch size     : {BATCH_SIZE}")
    log(f"  LR             : {LR}")
    log(f"  Weight decay   : {WEIGHT_DECAY}")
    log(f"  CosineAnnealing: T_max={EPOCHS}, eta_min={ETA_MIN}")
    log(f"  Early stopping : patience={EARLY_STOP_PAT} epochs (val SeverityWeightedL1)")
    log(f"  GRADIENT CLIP  : max_norm=1.0  (LSTM stability)")
    log(f"  Input shape    : (B, {SEQ_LEN}, {N_FEATURES})  [13 weeks x 27 features]")
    log(f"  Conv1d out     : {CONV_OUT}  |  LSTM hidden: {LSTM_HIDDEN}  |  layers: {LSTM_LAYERS}")
    log(f"  Fused dim      : {CONV_OUT + LSTM_HIDDEN * 2}  |  MLP hidden: {MLP_HIDDEN}")
    log(f"  Output shape   : (B, {HORIZON})  [5 forecast weeks]")

    # -- 1. Load data ----------------------------------------------------------
    log("\nLoading processed data ...")
    try:
        train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
        test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n  --> Run `python src/preprocess.py` to generate processed CSVs."
        )

    log(f"  train raw: {train_raw.shape}  |  test raw: {test_raw.shape}")

    n_train_regions = train_raw["region_id"].nunique()
    n_test_regions  = test_raw["region_id"].nunique()
    log(f"  train regions: {n_train_regions}  |  test regions: {n_test_regions}")
    assert n_train_regions == 2248, f"Expected 2248 train regions, got {n_train_regions}"
    assert n_test_regions  == 2248, f"Expected 2248 test regions,  got {n_test_regions}"

    # -- 1b. Feature Validation ------------------------------------------------
    log("\n[v32 Feature Validation]")
    assert "week_sin" in train_raw.columns, "week_sin missing -- run preprocess.py first"
    assert "week_cos" in train_raw.columns, "week_cos missing -- run preprocess.py first"
    log("  v week_sin, week_cos present.")

    bad_cols_8w_13w = [c for c in train_raw.columns if ("8w" in c or "13w" in c)]
    if bad_cols_8w_13w:
        log(f"  *** WARNING: 8w/13w features found: {bad_cols_8w_13w}")
    else:
        log("  v No 8w/13w rolling features (domain-shift columns absent).")

    lag_cols_check = [c for c in train_raw.columns if "lag" in c]
    if lag_cols_check:
        log(f"  [INFO] Lag columns in raw CSV (excluded by FEATURE_COLS): "
            f"{lag_cols_check[:6]}{'...' if len(lag_cols_check) > 6 else ''}")
    log(f"  v32 FEATURE_COLS: {len(FEATURE_COLS)} features. 3D input: (B, {SEQ_LEN}, {N_FEATURES})")

    # -- 2. Feature refinement -------------------------------------------------
    log("\nRefining features (drought proxy index + log1p prec + v22/v26 pruning) ...")
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    log(f"  train after refinement: {train_df.shape}  |  test: {test_df.shape}")

    # -- 3. Drop rows with NaN score -------------------------------------------
    before = len(train_df)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    dropped_nan = before - len(train_df)
    if dropped_nan:
        log(f"  [NaN drop] Removed {dropped_nan:,} rows with NaN score.")

    # -- 4. Target score distribution summary ---------------------------------
    log("\n[Training Target Distribution]")
    all_scores = train_df["score"].values
    zero_frac  = (all_scores == 0.0).mean()
    log(f"  mean={all_scores.mean():.4f}  std={all_scores.std():.4f}  "
        f"min={all_scores.min():.2f}  max={all_scores.max():.2f}")
    log(f"  Zero-inflation: {zero_frac:.2%} of training scores == 0.0")
    log(f"  Non-zero: {(all_scores > 0.0).sum():,}  ({(all_scores > 0.0).mean():.2%})  "
        f"vs  Zeroes: {(all_scores == 0.0).sum():,}  ({zero_frac:.2%})")
    log(f"  SeverityWeightedL1: zero targets get 1.0x weight; y=5 targets get 2.5x weight.")

    # -- 5. Model architecture print -------------------------------------------
    _tmp_model = DroughtSequenceNet(
        seq_len=SEQ_LEN, n_features=N_FEATURES, conv_out=CONV_OUT,
        lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
        mlp_hidden=MLP_HIDDEN, horizon=HORIZON, dropout=DROPOUT,
    )
    log(f"\n{_tmp_model.architecture_summary()}")
    del _tmp_model

    # -- 6. Build 5-Fold StratifiedGroupKFold CV splits -----------------------
    log(f"\n{'='*90}")
    log(f"5-Fold StratifiedGroupKFold CV  [v32 -- DroughtSequenceNet + SeverityWeightedL1]")
    log(f"  Group  : region_id  |  Strata : 10-quantile bins of per-region mean score")
    log(f"{'='*90}")

    folds = build_stratified_group_cv_folds(train_df, n_splits=N_FOLDS)
    log(f"\n  Folds built: {len(folds)}")
    for fi, (tg, vg) in enumerate(folds):
        log(f"  Fold {fi}: train_groups={len(tg):,}  val_groups={len(vg):,}")

    # -- 7. 5-Fold Deep Learning Training Loop ---------------------------------
    log(f"\n{'='*90}")
    log("5-Fold DroughtSequenceNet Training  [v32]")
    log(f"  Optimizer : AdamW(lr={LR}, weight_decay={WEIGHT_DECAY})")
    log(f"  Scheduler : CosineAnnealingLR(T_max={EPOCHS}, eta_min={ETA_MIN})")
    log(f"  Loss      : SeverityWeightedL1  mean(|pred-true| * (1.0 + true*0.3))")
    log(f"  Batch     : {BATCH_SIZE}  |  Max Epochs: {EPOCHS}")
    log(f"  Early Stop: patience={EARLY_STOP_PAT} (val SeverityWeightedL1)")
    log(f"{'='*90}")

    fold_results    = []    # (fold_k, week_maes, oof_mae, epoch_stopped)
    fold_test_preds = []    # list of np.ndarray (2248, 5) per fold

    criterion = severity_weighted_l1_loss

    for fold_k, (raw_train_groups, raw_val_groups) in enumerate(folds):

        log(f"\n{'='*90}")
        log(f"FOLD {fold_k + 1} / {N_FOLDS}  [v32 DroughtSequenceNet + SeverityWeightedL1]")
        log(f"  train_groups: {len(raw_train_groups):,}  |  "
            f"val_groups: {len(raw_val_groups):,}")
        log(f"{'='*90}")

        fold_t0 = time.time()

        # -- 7a. Compute fold-local Target Encoding (leakage-free) ------------
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
        log(f"  [OOF TE] Train regions: {len(train_region_ids_fold):,}  "
            f"| Val regions: {len(val_region_ids_fold):,}")

        # -- 7b. Augment groups with fold-local TE ----------------------------
        aug_train_groups = _augment_groups_with_te(
            raw_train_groups, te_map_fold, gm_fold, gzp_fold
        )
        aug_val_groups = _augment_groups_with_te(
            raw_val_groups, te_map_fold, gm_fold, gzp_fold
        )

        # -- 7c. Determine feature columns available --------------------------
        _sample_group = aug_train_groups[0][0]
        feat_cols = [c for c in FEATURE_COLS if c in _sample_group.columns]
        n_feat    = len(feat_cols)
        log(f"  feat_cols available: {n_feat} / {len(FEATURE_COLS)}")

        # -- 7d. Build 3D sequence arrays (N, 13, F) --------------------------
        log(f"  Building 3D sequence arrays ...")
        X_train_np, y_train_np, _ = build_sequence_dataset(aug_train_groups, feat_cols)
        X_val_np,   y_val_np,   _ = build_sequence_dataset(aug_val_groups,   feat_cols)

        log(f"  X_train: {X_train_np.shape}  |  y_train: {y_train_np.shape}")
        log(f"  X_val  : {X_val_np.shape}    |  y_val  : {y_val_np.shape}")

        # Sanity: verify 3D shape
        assert X_train_np.ndim == 3 and X_train_np.shape[1:] == (WINDOW_SIZE, n_feat), (
            f"X_train shape mismatch: got {X_train_np.shape}, "
            f"expected (N, {WINDOW_SIZE}, {n_feat})"
        )
        log(f"  v Input tensor shape confirmed: (N, {WINDOW_SIZE}, {n_feat}) = (B, 13, 27)")

        # -- 7e. Build test 3D array for this fold ----------------------------
        test_df_fold  = _merge_te_to_df(test_df, te_map_fold, gm_fold, gzp_fold)
        X_test_np, test_region_ids = build_sequence_test(test_df_fold, feat_cols)
        log(f"  X_test: {X_test_np.shape}")

        # -- 7f. Build DataLoaders --------------------------------------------
        train_ds = DroughtSequenceDataset(X_train_np, y_train_np)
        val_ds   = DroughtSequenceDataset(X_val_np,   y_val_np)

        effective_batch = BATCH_SIZE
        train_loader = DataLoader(
            train_ds, batch_size=effective_batch, shuffle=True,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"), drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=effective_batch * 2, shuffle=False,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        )

        # -- 7g. Instantiate model, optimizer, scheduler ----------------------
        ckpt_path = os.path.join(MODELS_DIR, f"dsn_fold{fold_k}.pt")

        model = DroughtSequenceNet(
            seq_len     = WINDOW_SIZE,
            n_features  = n_feat,
            conv_out    = CONV_OUT,
            lstm_hidden = LSTM_HIDDEN,
            lstm_layers = LSTM_LAYERS,
            mlp_hidden  = MLP_HIDDEN,
            horizon     = HORIZON,
            dropout     = DROPOUT,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=ETA_MIN
        )

        log(f"\n  Model params: {model.count_parameters():,}")

        # -- 7h. Resume from checkpoint if available --------------------------
        start_epoch  = 0
        best_val_mae = float("inf")
        patience_ctr = 0
        best_state   = None

        if os.path.exists(ckpt_path):
            log(f"  [RESUME] Checkpoint found: {ckpt_path}  -- loading.")
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                start_epoch  = ckpt.get("epoch", 0) + 1
                best_val_mae = ckpt.get("best_val_mae", float("inf"))
                patience_ctr = ckpt.get("patience_ctr", 0)
                log(f"  [RESUME] Resuming from epoch {start_epoch}  "
                    f"best_val_mae={best_val_mae:.4f}")
            except Exception as exc:
                log(f"  [WARN] Checkpoint load failed ({exc}). Starting from scratch.")
                start_epoch  = 0
                best_val_mae = float("inf")
                patience_ctr = 0

        # -- 7i. Epoch training loop ------------------------------------------
        epoch_stopped = start_epoch
        for epoch in range(start_epoch, EPOCHS):
            ep_t0 = time.time()

            try:
                train_loss = _train_one_epoch(
                    model, train_loader, optimizer, criterion, DEVICE
                )
            except RuntimeError as oom_err:
                if "out of memory" in str(oom_err).lower() and effective_batch > 512:
                    log(f"  [OOM] Batch {effective_batch} caused OOM. "
                        f"Rebuilding loaders with batch=512.")
                    effective_batch = 512
                    torch.cuda.empty_cache()
                    train_loader = DataLoader(
                        train_ds, batch_size=effective_batch, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
                        drop_last=False,
                    )
                    val_loader = DataLoader(
                        val_ds, batch_size=effective_batch * 2, shuffle=False,
                        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
                    )
                    train_loss = _train_one_epoch(
                        model, train_loader, optimizer, criterion, DEVICE
                    )
                else:
                    raise

            val_loss, val_preds, val_targets = _eval_one_epoch(
                model, val_loader, criterion, DEVICE
            )
            scheduler.step()

            ep_elapsed = time.time() - ep_t0
            current_lr = scheduler.get_last_lr()[0]

            log(f"  Epoch [{epoch+1:3d}/{EPOCHS}]  "
                f"train_SWL1={train_loss:.4f}  val_SWL1={val_loss:.4f}  "
                f"lr={current_lr:.2e}  t={ep_elapsed:.1f}s")

            epoch_stopped = epoch + 1

            # Early stopping & checkpoint saving
            if val_loss < best_val_mae:
                best_val_mae = val_loss
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
                torch.save(
                    {
                        "model_state":     {k: v.cpu() for k, v in model.state_dict().items()},
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch":           epoch,
                        "best_val_mae":    best_val_mae,
                        "patience_ctr":    patience_ctr,
                        "fold_k":          fold_k,
                    },
                    ckpt_path,
                )
                log(f"    [SAVED] New best val_SWL1={best_val_mae:.4f}  -> {ckpt_path}")
            else:
                patience_ctr += 1
                if patience_ctr >= EARLY_STOP_PAT:
                    log(f"  [Early Stop] No improvement for {EARLY_STOP_PAT} epochs. "
                        f"Stopping at epoch {epoch+1}.")
                    break

        # -- 7j. Load best model state for inference --------------------------
        if best_state is not None:
            model.load_state_dict(best_state)
            log(f"  [BEST] Loaded best model state (val_SWL1={best_val_mae:.4f})")

        # -- 7k. OOF validation predictions -----------------------------------
        val_loss_final, oof_preds, oof_targets = _eval_one_epoch(
            model, val_loader, criterion, DEVICE
        )
        oof_mae = float(np.mean(np.abs(oof_preds - oof_targets)))

        # Per-week OOF MAE
        week_maes = [
            float(np.mean(np.abs(oof_preds[:, w] - oof_targets[:, w])))
            for w in range(HORIZON)
        ]

        fold_elapsed = time.time() - fold_t0
        log(f"\n  [Fold {fold_k}] Week MAEs (OOF L1): "
            + "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes)))
        log(f"  [Fold {fold_k}] Overall OOF MAE   : {oof_mae:.4f}")
        log(f"  [Fold {fold_k}] Best Val SWL1 (ckpt): {best_val_mae:.4f}")
        log(f"  [Fold {fold_k}] Epochs run           : {epoch_stopped}/{EPOCHS}")
        log(f"  [Fold {fold_k}] Elapsed              : {fold_elapsed:.1f}s")

        # -- 7l. Binned Error Matrix -------------------------------------------
        print_binned_error_matrix(oof_targets.ravel(), oof_preds.ravel(), fold_k, log)

        fold_results.append((fold_k, week_maes, oof_mae, epoch_stopped))

        # -- 7m. Test-set inference for this fold -----------------------------
        test_preds_fold = _predict(model, X_test_np, effective_batch * 2, DEVICE)
        fold_test_preds.append(test_preds_fold)   # (2248, 5)

        log(f"  [Test inference] shape={test_preds_fold.shape}  "
            f"mean={test_preds_fold.mean():.4f}  "
            f"std={test_preds_fold.std():.4f}  "
            f"min={test_preds_fold.min():.4f}  "
            f"max={test_preds_fold.max():.4f}")

        # Cleanup
        del X_train_np, y_train_np, X_val_np, y_val_np
        del train_ds, val_ds, train_loader, val_loader
        del model, optimizer, scheduler
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # -- 8. Cross-fold summary -------------------------------------------------
    log(f"\n{'='*90}")
    log(f"5-Fold Cross-Validation Summary  [v32 DroughtSequenceNet + SeverityWeightedL1]")
    log(f"{'='*90}")
    fold_oof_maes = []
    for fold_k, week_maes, oof_mae, epoch_stopped in fold_results:
        wk_str = "  ".join(f"W{i+1}={m:.4f}" for i, m in enumerate(week_maes))
        log(f"  Fold {fold_k} [OOF L1]: {wk_str}  ->  Mean={oof_mae:.4f}"
            f"  (stopped epoch {epoch_stopped})")
        fold_oof_maes.append(oof_mae)

    overall_mean = float(np.mean(fold_oof_maes))
    overall_std  = float(np.std(fold_oof_maes))
    log(f"\n  Overall CV L1 MAE : {overall_mean:.4f}  +-  {overall_std:.4f}")
    log(f"  Best Fold  : Fold {int(np.argmin(fold_oof_maes))} "
        f"(MAE={min(fold_oof_maes):.4f})")
    log(f"  Worst Fold : Fold {int(np.argmax(fold_oof_maes))} "
        f"(MAE={max(fold_oof_maes):.4f})")

    log(f"\n  Per-week CV MAE breakdown:")
    for week_idx in range(HORIZON):
        wk_maes = [fold_results[k][1][week_idx] for k in range(N_FOLDS)]
        log(f"    Week {week_idx + 1}: mean={np.mean(wk_maes):.4f}  "
            f"std={np.std(wk_maes):.4f}")

    # -- 9. Median Blending + Micro-Zero Floor --------------------------------
    log(f"\n{'='*90}")
    log(f"[v32] Post-Ensemble Median Blending + Micro-Zero Floor  ({N_FOLDS} folds)")
    log(f"  Strategy      : np.median(all_fold_predictions, axis=0)")
    log(f"  Micro-Zero    : np.where(preds < 0.05, 0.0, preds)  [noise wiper]")
    log(f"  Physical clip : np.clip(0.0, 5.0)")
    log(f"{'='*90}")

    # Stack folds: (N_FOLDS, 2248, 5)
    preds_stack = np.stack(fold_test_preds, axis=0)
    log(f"  preds_stack shape : {preds_stack.shape}")

    # Per-fold stats
    for k in range(N_FOLDS):
        p = preds_stack[k]
        log(f"  Fold {k} preds: mean={p.mean():.4f}  std={p.std():.4f}  "
            f"min={p.min():.4f}  max={p.max():.4f}")

    # Median compression: (N_FOLDS, 2248, 5) -> (2248, 5)
    final_preds = np.median(preds_stack, axis=0)

    log(f"\n  Pre-floor/clip median blend stats:")
    log(f"    mean={final_preds.mean():.4f}  std={final_preds.std():.4f}  "
        f"min={final_preds.min():.4f}  max={final_preds.max():.4f}")

    # Micro-Zero Floor: cleanly wipe out trailing decimals under 0.05
    # Removes floating-point noise without harming legitimate light drought signals.
    final_preds = np.where(final_preds < 0.05, 0.0, final_preds)

    log(f"  Post-floor stats (< 0.05 zeroed):")
    log(f"    mean={final_preds.mean():.4f}  std={final_preds.std():.4f}  "
        f"min={final_preds.min():.4f}  max={final_preds.max():.4f}")

    # Physical constraint — clip to valid drought score range [0, 5]
    final_submission = np.clip(final_preds, 0.0, 5.0)

    log(f"  Post-clip stats (physical guard [0, 5]):")
    log(f"    mean={final_submission.mean():.4f}  std={final_submission.std():.4f}  "
        f"min={final_submission.min():.4f}  max={final_submission.max():.4f}")

    # -- 10. Submission prediction diagnostics ---------------------------------
    log("\n[v32 Submission Prediction Diagnostics]")
    all_sub_preds   = final_submission.ravel()
    zero_frac_final = float((all_sub_preds == 0.0).mean())
    near_zero_frac  = float((all_sub_preds < 0.05).mean())
    p50  = float(np.percentile(all_sub_preds, 50))
    p75  = float(np.percentile(all_sub_preds, 75))
    p90  = float(np.percentile(all_sub_preds, 90))
    p95  = float(np.percentile(all_sub_preds, 95))
    p99  = float(np.percentile(all_sub_preds, 99))
    pmax = float(np.max(all_sub_preds))
    log(f"  n={len(all_sub_preds):,}  "
        f"mean={all_sub_preds.mean():.4f}  std={all_sub_preds.std():.4f}")
    log(f"  p50={p50:.4f}  p75={p75:.4f}  p90={p90:.4f}  "
        f"p95={p95:.4f}  p99={p99:.4f}  max={pmax:.4f}")
    log(f"  Exact zero fraction  (==0.0): {zero_frac_final:.4f}")
    log(f"  Near-zero fraction  (<0.05) : {near_zero_frac:.4f}")
    log(f"  (Training zero baseline was {zero_frac:.2%})")

    if p99 < 2.0:
        log("  *** WARNING: p99 < 2.0 -- predictions may be under-dispersed! ***")
    else:
        log("  v p99 >= 2.0 -- prediction diversity is healthy.")

    # -- 11. Format & save submission.csv -------------------------------------
    all_region_ids = test_region_ids
    log("\nFormatting submission.csv ...")
    rows = []
    for i, region_id in enumerate(all_region_ids):
        preds_i = final_submission[i]
        rows.append({
            "region_id":  region_id,
            "pred_week1": float(preds_i[0]),
            "pred_week2": float(preds_i[1]),
            "pred_week3": float(preds_i[2]),
            "pred_week4": float(preds_i[3]),
            "pred_week5": float(preds_i[4]),
        })

    submission = pd.DataFrame(rows)
    sub_path   = os.path.join(ROOT, "submission.csv")
    submission.to_csv(sub_path, index=False)

    # -- 12. Sanity checks -----------------------------------------------------
    assert len(submission) == 2248, f"Expected 2248 rows, got {len(submission)}"
    assert list(submission.columns) == [
        "region_id", "pred_week1", "pred_week2",
        "pred_week3", "pred_week4", "pred_week5",
    ], f"Unexpected columns: {list(submission.columns)}"
    log("  v Submission assertion passed: 2248 rows, 6 columns.")

    assert not submission.isnull().any().any(), "NaN values found in submission!"
    log("  v No NaN values in submission.")

    test_regions  = set(test_df["region_id"].unique())
    train_regions = set(train_df["region_id"].unique())
    assert test_regions == train_regions, "Train/test region sets do not match!"
    log("  v Train/test regions match (2248).")

    pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]
    assert submission[pred_cols].max().max() <= 5.0 + 1e-6
    assert submission[pred_cols].min().min() >= 0.0 - 1e-6
    log("  v All predictions in [0, 5]  (np.clip physical guard enforced).")

    log(f"\n  submission.csv -> {sub_path}")
    log(f"  Rows (excl. header): {len(submission)}")
    log(f"  Columns: {list(submission.columns)}")
    log(f"\n  Preview:\n{submission.head(5).to_string(index=False)}")

    # -- 13. Total elapsed time ------------------------------------------------
    elapsed = time.time() - t0
    log(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    log_path = os.path.join(ROOT, "_training_log_32nd.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines))
    print(f"\nTraining log saved -> {log_path}")

    return {
        "fold_results":    fold_results,
        "overall_cv_mae":  overall_mean,
        "std_cv_mae":      overall_std,
        "input_shape":     (SEQ_LEN, N_FEATURES),
        "submission":      submission,
        "sub_p99":         p99,
        "zero_frac_final": zero_frac_final,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = main()
