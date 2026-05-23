"""
adversarial_witchhunt.py – Adversarial Feature Importance Witch Hunt
====================================================================
Diagnostic script for v25 / v26 pipeline.

Purpose: Train a LightGBM binary classifier to distinguish between the
Train and Test flattened tabular matrices (507-dim, v23 paradigm).
A perfect AUC = 1.0 means the model can flawlessly separate Train from
Test, indicating catastrophic Covariate Shift driven by "traitor features"
that leak chronological or synthetic boundary information.

Data source: data/processed/train_processed.csv  &  data/processed/test_processed.csv

Outputs: Top 20 traitor features ranked by Gain importance.

Usage:
    python src/adversarial_witchhunt.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path so src.* imports work
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_tabular_dataset,
    build_full_train_groups,
    build_tabular_test,
    make_flat_col_names,
    FEATURE_COLS,
    WINDOW_SIZE,
    HORIZON,
)

print("=" * 80)
print("ADVERSARIAL FEATURE IMPORTANCE WITCH HUNT")
print("Objective: Unmask the Top 20 Traitor Features causing AUC = 1.0")
print(f"Data dir : data/processed/")
print("=" * 80)

# ---------------------------------------------------------------------------
# Step 1: Load already-processed CSVs directly
# ---------------------------------------------------------------------------
PROC_DIR       = os.path.join(ROOT, "data", "processed")
train_proc_path = os.path.join(PROC_DIR, "train_processed.csv")
test_proc_path  = os.path.join(PROC_DIR, "test_processed.csv")

print(f"\n[Step 1] Loading processed CSVs ...")
print(f"  train : {train_proc_path}")
print(f"  test  : {test_proc_path}")

train_w = pd.read_csv(train_proc_path)
test_w  = pd.read_csv(test_proc_path)

print(f"  train_w : {train_w.shape}  |  test_w : {test_w.shape}")
print(f"  train regions : {train_w['region_id'].nunique()}")
print(f"  test  regions : {test_w['region_id'].nunique()}")

# ---------------------------------------------------------------------------
# Step 2: Feature Refinement (drought index + log1p prec + v22 pruning)
# ---------------------------------------------------------------------------
print("\n[Step 2] Refining features (drought proxy + log1p prec + v22 pruning) ...")
train_df = refine_features(train_w, is_train=True)
test_df  = refine_features(test_w,  is_train=False)

# Drop rows without a valid score in train
before    = len(train_df)
train_df  = train_df.dropna(subset=["score"]).reset_index(drop=True)
dropped   = before - len(train_df)
print(f"  train after refinement + NaN-score drop : {len(train_df):,} rows  "
      f"(removed {dropped:,})")
print(f"  test  after refinement                  : {test_df.shape}")

# ---------------------------------------------------------------------------
# Step 3: Global Target Encoding (leakage-free at diagnostics level is fine)
# ---------------------------------------------------------------------------
print("\n[Step 3] Computing global Target Encoding stats ...")

def _zero_prob(x):
    return (x == 0.0).mean()

te_stats = (
    train_df.groupby("region_id")["score"]
    .agg(region_mean_score="mean", region_zero_prob=_zero_prob)
    .reset_index()
)
global_mean      = float(te_stats["region_mean_score"].mean())
global_zero_prob = float(te_stats["region_zero_prob"].mean())
te_map           = {
    row["region_id"]: (float(row["region_mean_score"]), float(row["region_zero_prob"]))
    for _, row in te_stats.iterrows()
}
print(f"  global_mean={global_mean:.4f}  global_zero_prob={global_zero_prob:.4f}")

def _inject_te(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["region_mean_score"] = df["region_id"].map(
        lambda rid: te_map.get(rid, (global_mean, global_zero_prob))[0]
    ).astype(np.float32)
    df["region_zero_prob"] = df["region_id"].map(
        lambda rid: te_map.get(rid, (global_mean, global_zero_prob))[1]
    ).astype(np.float32)
    return df

train_df = _inject_te(train_df)
test_df  = _inject_te(test_df)

feat_cols = [c for c in FEATURE_COLS if c in train_df.columns]
missing   = [c for c in FEATURE_COLS if c not in feat_cols]
print(f"  Available FEATURE_COLS : {len(feat_cols)} / {len(FEATURE_COLS)}")
if missing:
    print(f"  *** MISSING features   : {missing}")

# ---------------------------------------------------------------------------
# Step 4: Build Flattened Tabular Matrices  (507 wide, v23 paradigm)
# ---------------------------------------------------------------------------
n_flat = WINDOW_SIZE * len(feat_cols)
print(f"\n[Step 4] Building flattened tabular matrices  "
      f"({WINDOW_SIZE}w × {len(feat_cols)} feats = {n_flat} cols) ...")

train_groups                     = build_full_train_groups(train_df)
X_train_np, y_train_np, _        = build_tabular_dataset(train_groups, feat_cols)
X_test_np, test_rids             = build_tabular_test(test_df, feat_cols)
flat_col_names                   = make_flat_col_names(feat_cols, window=WINDOW_SIZE)

print(f"  X_train : {X_train_np.shape}  |  y_train : {y_train_np.shape}")
print(f"  X_test  : {X_test_np.shape}")
print(f"  flat column names : {len(flat_col_names)}  (expected {n_flat})")

# ---------------------------------------------------------------------------
# Step 5: Build adversarial binary matrix  (Train=0, Test=1)
# ---------------------------------------------------------------------------
print("\n[Step 5] Constructing adversarial binary classification matrix ...")

train_labels = np.zeros(len(X_train_np), dtype=np.int32)
test_labels  = np.ones( len(X_test_np),  dtype=np.int32)

X_all = np.vstack([X_train_np, X_test_np])
y_all = np.concatenate([train_labels, test_labels])

print(f"  Combined : {X_all.shape}")
print(f"  n_train={len(train_labels):,}  n_test={len(test_labels):,}  "
      f"test_frac={len(test_labels)/len(y_all):.2%}")

# ---------------------------------------------------------------------------
# Step 6: Train LightGBM adversarial classifier
# ---------------------------------------------------------------------------
print("\n[Step 6] Training LightGBM adversarial classifier  "
      "(is_test ~ 507 flattened features) ...")

try:
    from lightgbm import LGBMClassifier, log_evaluation
    lgbm_available = True
except ImportError:
    lgbm_available = False
    print("  *** LightGBM not found; falling back to sklearn RandomForestClassifier")

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss

# 80 / 20 stratified split for evaluation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for tr_idx, vl_idx in sss.split(X_all, y_all):
    X_tr, X_vl = X_all[tr_idx], X_all[vl_idx]
    y_tr, y_vl = y_all[tr_idx], y_all[vl_idx]

print(f"  Split : train={len(X_tr):,}  val={len(X_vl):,}")

if lgbm_available:
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    print("  Fitting LGBMClassifier (n_estimators=300, max_depth=5, lr=0.05) ...")
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[log_evaluation(period=50)],
    )
    y_prob    = clf.predict_proba(X_vl)[:, 1]
    gain_imp  = clf.booster_.feature_importance(importance_type="gain")
    split_imp = clf.booster_.feature_importance(importance_type="split")

else:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                 random_state=42, n_jobs=-1)
    print("  Fitting RandomForestClassifier (n_estimators=100) ...")
    clf.fit(X_tr, y_tr)
    y_prob    = clf.predict_proba(X_vl)[:, 1]
    gain_imp  = clf.feature_importances_
    split_imp = clf.feature_importances_

auc     = roc_auc_score(y_vl, y_prob)
logloss = log_loss(y_vl, y_prob)
acc     = ((y_prob >= 0.5).astype(int) == y_vl).mean()

print(f"\n  *** ADVERSARIAL VALIDATION RESULTS ***")
print(f"  AUC      = {auc:.6f}  (1.0 = perfect Train/Test separation)")
print(f"  LogLoss  = {logloss:.6f}")
print(f"  Accuracy = {acc:.4f}")

# ---------------------------------------------------------------------------
# Step 7: Extract Top 20 Traitor Features
# ---------------------------------------------------------------------------
print("\n[Step 7] Ranking features by importance ...")

imp_df = pd.DataFrame({
    "feature":   flat_col_names,
    "gain":      gain_imp,
    "split":     split_imp,
})
imp_df["gain_pct"]  = 100.0 * imp_df["gain"]  / imp_df["gain"].sum().clip(min=1e-9)
imp_df["split_pct"] = 100.0 * imp_df["split"] / imp_df["split"].sum().clip(min=1e-9)

top20_gain  = imp_df.nlargest(20, "gain").reset_index(drop=True)
top20_split = imp_df.nlargest(20, "split").reset_index(drop=True)

# ---- Print: by GAIN -------------------------------------------------------
print("\n")
print("=" * 80)
print("  TOP 20 TRAITOR FEATURES  (ranked by GAIN – information contribution)")
print("=" * 80)
print(f"  {'Rank':>4}  {'Feature':<48}  {'Gain%':>7}  {'Split%':>7}  {'Slot':>5}")
print("  " + "-" * 76)
for rank, row in top20_gain.iterrows():
    feat = row["feature"]
    # e.g. deficit_roll_cum_4w_w13 -> slot w13
    slot = "w" + feat.rsplit("_w", 1)[-1] if "_w" in feat else "  -"
    print(f"  {rank+1:>4}  {feat:<48}  {row['gain_pct']:>6.3f}%  "
          f"{row['split_pct']:>6.3f}%  {slot:>5}")
print("=" * 80)

# ---- Print: by SPLIT ------------------------------------------------------
print("\n")
print("=" * 80)
print("  TOP 20 TRAITOR FEATURES  (ranked by SPLIT – frequency in tree splits)")
print("=" * 80)
print(f"  {'Rank':>4}  {'Feature':<48}  {'Split%':>7}  {'Gain%':>7}  {'Slot':>5}")
print("  " + "-" * 76)
for rank, row in top20_split.iterrows():
    feat = row["feature"]
    slot = "w" + feat.rsplit("_w", 1)[-1] if "_w" in feat else "  -"
    print(f"  {rank+1:>4}  {feat:<48}  {row['split_pct']:>6.3f}%  "
          f"{row['gain_pct']:>6.3f}%  {slot:>5}")
print("=" * 80)

# ---------------------------------------------------------------------------
# Step 8: Category breakdown
# ---------------------------------------------------------------------------
print("\n[Step 8] Category breakdown in Top 20 (Gain) ...")

categories = {
    "cumulative/rolling (roll_cum|roll_sum|roll_mean)": ["roll_cum", "roll_sum", "roll_mean"],
    "deficit / PET (drought proxy)":                   ["deficit", "pet"],
    "target encoding (region_mean|zero_prob)":         ["region_mean_score", "region_zero_prob"],
    "lag features (lag1w|lag2w)":                      ["lag1w", "lag2w"],
    "cyclic calendar (week_sin|cos|day_ordinal)":      ["week_sin", "week_cos", "day_ordinal"],
    "intra-week std (_week_std)":                      ["_week_std"],
    "intra-week max (_week_max)":                      ["_week_max"],
    "intra-week min (_week_min)":                      ["_week_min"],
    "base weather":                                    ["prec", "tmp", "humidity",
                                                        "wind", "surf_pre", "surf_tmp"],
}

cat_cnt  = {k: 0   for k in categories}
cat_gain = {k: 0.0 for k in categories}

for _, row in top20_gain.iterrows():
    feat     = row["feature"]
    assigned = False
    for cat, kws in categories.items():
        if any(kw in feat for kw in kws) and not assigned:
            cat_cnt[cat]  += 1
            cat_gain[cat] += row["gain_pct"]
            assigned = True

print(f"\n  {'Category':<48}  {'Count':>5}  {'GainSum':>9}")
print("  " + "-" * 65)
for cat in categories:
    if cat_cnt[cat] > 0:
        print(f"  {cat:<48}  {cat_cnt[cat]:>5}  {cat_gain[cat]:>8.2f}%")

# ---------------------------------------------------------------------------
# Step 9: Window slot hotspot (w1=oldest … w13=most recent)
# ---------------------------------------------------------------------------
print("\n[Step 9] Gain % concentrated by window slot ...")

slot_gain = {}
for _, row in imp_df.iterrows():
    feat = row["feature"]
    if "_w" in feat:
        try:
            w = int(feat.rsplit("_w", 1)[-1])
        except ValueError:
            w = 0
    else:
        w = 0
    slot_gain[w] = slot_gain.get(w, 0.0) + row["gain_pct"]

print(f"\n  Gain % by slot (w1=oldest, w13=most recent context week):")
for w in sorted(slot_gain.keys()):
    bar = "█" * max(0, int(slot_gain[w] / 1.5))
    print(f"    w{w:>2}: {slot_gain[w]:>6.2f}%  {bar}")

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
print("\n")
print("=" * 80)
print("  WITCH HUNT SUMMARY")
print("=" * 80)
print(f"  Adversarial AUC          : {auc:.6f}")
print(f"  Adversarial LogLoss      : {logloss:.6f}")
print(f"  Adversarial Accuracy     : {acc:.4f}")
print(f"  Train samples (windows)  : {len(X_train_np):,}")
print(f"  Test  samples (regions)  : {len(X_test_np):,}")
print(f"  Features evaluated       : {len(flat_col_names):,}")
print("")
print("  TOP 5 TRAITOR FEATURES (Gain):")
for rank, row in top20_gain.head(5).iterrows():
    print(f"    #{rank+1}  {row['feature']}  [{row['gain_pct']:.3f}%]")
print("=" * 80)
print()
print("The Adversarial Feature Importance investigation is complete.")
print("Here are the Top 20 traitor features driving the AUC to 1.0.")
print("Please review these components so we can proceed with feature")
print("pruning for the V26 definitive checklist.")
