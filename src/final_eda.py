"""
final_eda.py — Complete EDA & Data Mining Pipeline (v54)
=========================================================
整合來源: adv.py, anatestpadding.py, eda.py, v45_anacluster.py,
          v51_explore.py, v52_feature.py, v54_explore.py,
          v54_feature.py, v54_feature2.py

各節說明:
  §1   Raw Data Reality Check     — raw train.csv / test.csv
  §2   Processed Data Overview    — 資料形狀、NaN、特徵驗證
  §3   Score Distribution         — 直方圖 + Zero-Inflation 三聯圖
  §4   Train vs Test Distribution — 分佈對比表 + 99th-pctile Danger Check + KDE 圖
  §5   Drought Feature Analysis   — 乾旱等級剖面 (Multiplier) + Spearman 表
  §6   Correlation Heatmaps       — Pearson & Spearman 熱圖 + 分數 bar chart
  §7   Region Time-Series         — 5 個隨機 region 氣象 + score 疊圖
  §8   Feature Boxplots           — 氣象特徵箱形圖
  §9   Cyclical Feature Coverage  — week_sin / week_cos 單位圓
  §10  Dataset Structure          — weeks/region + deployment gap + sliding windows
  §11  Climate Cluster Analysis   — 聚落分析 (物理特徵，無 score 洩漏)
  §12  Adversarial Validation     — Train vs Test LightGBM 二元分類 AUC

所有圖片 → plots/final_eda/
"""

import os
import sys
import random
import warnings
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
PROC_DIR  = os.path.join(DATA_DIR, "v54_processed")
PLOTS_DIR = os.path.join(ROOT, "plots", "final_eda")
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── Feature Constants (aligned with v54_dataset.py) ───────────────────────────
FEATURE_COLS = [
    "prec", "surf_pre", "humidity",
    "tmp", "tmp_max", "tmp_min", "tmp_range",
    "wind", "wind_min", "wind_range",
    "tmp_week_max", "tmp_week_min", "tmp_week_std",
    "humidity_week_max", "humidity_week_min", "humidity_week_std",
    "wind_week_max", "wind_week_min", "wind_week_std",
    "prec_week_max", "surf_pre_week_max", "tmp_roll_mean_4w",
    "humidity_roll_mean_4w",
    "week_sin", "week_cos",
    "pet", "prec_roll_sum_4w",
    "water_shortage", "water_shortage_roll_cum_4w",
    "aridity_index", "heat_shock", "tmp_anomaly",
    "water_shortage_roll_cum_8w",
    "cross_shortage_x_anomaly", "is_dry_spell", "shortage_momentum",
    "cluster_id",
]

DROUGHT_FEATS = [
    "prec", "tmp", "humidity",
    "water_shortage", "water_shortage_roll_cum_4w", "water_shortage_roll_cum_8w",
    "tmp_anomaly", "cross_shortage_x_anomaly", "is_dry_spell",
    "shortage_momentum", "aridity_index", "heat_shock",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {path}")

def _section(n: int, title: str) -> None:
    print(f"\n{'='*80}")
    print(f" §{n}  {title}")
    print(f"{'='*80}")

def _present(cols, df):
    return [c for c in cols if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# §1  Raw Data Reality Check
# ─────────────────────────────────────────────────────────────────────────────
def sec1_raw_reality_check(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    _section(1, "Raw Data Reality Check  (raw daily CSVs)")

    # Compute simple PET and VPD proxies on raw daily data
    for df in (df_train, df_test):
        df["pet_raw"] = 0.55 * df["tmp"].clip(lower=0)
        df["vpd_raw"] = df["tmp"] * (100.0 - df["humidity"]) / 100.0

    df_safe = df_train[df_train["score"] == 0.0]
    df_ext  = df_train[df_train["score"] > 3.5]

    print(f"\n[1.1] Train: Safe (score=0, n={len(df_safe):,}) vs Extreme (score>3.5, n={len(df_ext):,})")
    feats = ["prec", "tmp", "humidity", "wind", "pet_raw", "vpd_raw"]
    hdr = f"  {'Feature':<15} | {'Safe Mean':>10} | {'Extreme Mean':>13} | {'Ext Top10%':>11} | {'Shift':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for f in feats:
        if f not in df_train.columns:
            continue
        s_m = df_safe[f].mean()
        e_m = df_ext[f].mean()
        edge = df_ext[f].quantile(0.10 if f in ("prec", "humidity") else 0.90)
        shift = (e_m - s_m) / (abs(s_m) + 1e-5) * 100
        flag = "  ⚠" if abs(shift) > 40 else ""
        print(f"  {f:<15} | {s_m:>10.3f} | {e_m:>13.3f} | {edge:>11.3f} | {shift:>+6.1f}%{flag}")

    print(f"\n[1.2] Test vs Train Climate Reality Check")
    hdr2 = f"  {'Metric':<25} | {'Train':>12} | {'Test':>12}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    metrics = [
        ("Avg Temperature",    "tmp",     "mean"),
        ("Temp 99th pctile",   "tmp",     lambda x: x.quantile(0.99)),
        ("Avg Humidity",       "humidity","mean"),
        ("Avg Precipitation",  "prec",    "mean"),
        ("Avg VPD proxy",      "vpd_raw", "mean"),
        ("VPD 99th pctile",    "vpd_raw", lambda x: x.quantile(0.99)),
    ]
    for label, col, agg in metrics:
        if col not in df_train.columns:
            continue
        tr_v = df_train[col].agg(agg) if isinstance(agg, str) else agg(df_train[col])
        te_v = df_test[col].agg(agg)  if isinstance(agg, str) else agg(df_test[col])
        flag = "  🚨" if te_v > tr_v * 1.15 else ""
        print(f"  {label:<25} | {tr_v:>12.3f} | {te_v:>12.3f}{flag}")

    hot_thresh = df_ext["tmp"].mean()
    tr_hd = ((df_train["tmp"] > hot_thresh) & (df_train["prec"] < 1.0)).mean()
    te_hd = ((df_test["tmp"]  > hot_thresh) & (df_test["prec"]  < 1.0)).mean()
    print(f"\n  Hot & Dry Days (tmp>{hot_thresh:.1f}°C, prec<1mm): "
          f"Train={tr_hd:.2%}  Test={te_hd:.2%}")
    if te_hd > tr_hd * 2:
        print("  🚨 WARNING: Test 'Hot & Dry' ratio is 2× higher than Train — potential mega-drought scenario!")

    print(f"\n  Raw data shapes: train={df_train.shape}  test={df_test.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# §2  Processed Data Overview
# ─────────────────────────────────────────────────────────────────────────────
def sec2_processed_overview(train_w: pd.DataFrame, test_w: pd.DataFrame) -> None:
    _section(2, "Processed Data Overview  (data/v54_processed)")

    print(f"  train_w shape : {train_w.shape}")
    print(f"  test_w  shape : {test_w.shape}")

    present = _present(FEATURE_COLS, train_w)
    missing = [c for c in FEATURE_COLS if c not in train_w.columns]
    print(f"\n  FEATURE_COLS coverage: {len(present)}/{len(FEATURE_COLS)} present")
    if missing:
        print(f"  Missing features: {missing}")

    # Week counts
    tr_wk = train_w.groupby("region_id").size()
    te_wk = test_w.groupby("region_id").size()
    print(f"\n  Train weeks/region: min={tr_wk.min()}  max={tr_wk.max()}  "
          f"mean={tr_wk.mean():.1f}  (expected 782)")
    print(f"  Test  weeks/region: min={te_wk.min()}  max={te_wk.max()}  "
          f"(expected 13)")
    print(f"  Train regions: {train_w['region_id'].nunique():,}")
    print(f"  Test  regions: {test_w['region_id'].nunique():,}")

    # NaN check
    nan_train = train_w[present].isna().sum()
    nan_any   = nan_train[nan_train > 0]
    if len(nan_any):
        print(f"\n  NaN columns (train): {nan_any.to_dict()}")
    else:
        print("\n  No NaN in any FEATURE_COL (train). ✓")

    if "score" in train_w.columns:
        n_nan_score = train_w["score"].isna().sum()
        print(f"  NaN scores in train: {n_nan_score}  (expected 0 — ghost-week check)")

    sentinel_ok = "_v54_processed" in train_w.columns
    print(f"  _v54_processed sentinel: {'present ✓' if sentinel_ok else 'MISSING ✗'}")


# ─────────────────────────────────────────────────────────────────────────────
# §3  Score Distribution & Zero-Inflation
# ─────────────────────────────────────────────────────────────────────────────
def sec3_score_distribution(train_w: pd.DataFrame) -> None:
    _section(3, "Score Distribution & Zero-Inflation")

    if "score" not in train_w.columns:
        print("  [Skip] No 'score' column.")
        return

    sv = train_w["score"].dropna().values
    zero_frac = (sv == 0.0).mean()

    # ── console stats
    print(f"  Total score rows : {len(sv):,}")
    print(f"  Mean  : {sv.mean():.4f}   Std: {sv.std():.4f}")
    print(f"  Zero-inflation (score==0): {zero_frac:.2%}  ({(sv==0).sum():,})")
    print(f"\n  {'Bracket':<25} {'Count':>10} {'Pct':>8}")
    print("  " + "-" * 45)
    brackets = [
        ("Absolute Zero   (=0)",   lambda x: x == 0.0),
        ("Mild Drought    (0-1]",   lambda x: (x > 0) & (x <= 1)),
        ("Moderate Drought (1-2]",  lambda x: (x > 1) & (x <= 2)),
        ("Severe Drought   (2-3]",  lambda x: (x > 2) & (x <= 3)),
        ("Extreme Drought  (3-4]",  lambda x: (x > 3) & (x <= 4)),
        ("Exceptional      (4-5]",  lambda x: (x > 4) & (x <= 5)),
    ]
    for label, mask_fn in brackets:
        cnt = int(mask_fn(sv).sum())
        print(f"  {label:<25} {cnt:>10,} {cnt/len(sv):>7.2%}")

    # ── Plot 1: Score Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.25, 5.75, 0.5)
    ax.hist(sv, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Target Variable Distribution — Weekly Drought Score (v54)", fontsize=12)
    ax.set_xticks(range(6))
    total = len(sv)
    for v in range(6):
        cnt = int(((sv >= v - 0.25) & (sv < v + 0.25)).sum())
        if cnt > 0:
            ax.text(v, cnt + total * 0.005, f"{cnt/total:.1%}",
                    ha="center", va="bottom", fontsize=8, color="navy")
    _save(fig, "01_score_histogram.png")

    # ── Plot 2: Zero-Inflation Three-Panel
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pie
    axes[0].pie(
        [zero_frac, 1 - zero_frac],
        labels=[f"score=0\n({zero_frac:.1%})", f"score>0\n({1-zero_frac:.1%})"],
        colors=["#FF6B6B", "#4ECDC4"], explode=(0.05, 0),
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11},
    )
    axes[0].set_title("Zero-Inflation\n(motivates Hurdle model)", fontsize=12)

    # Bar per bracket
    cts = [(sv == 0).sum()] + [int(((sv > i) & (sv <= i+1)).sum()) for i in range(5)]
    axes[1].bar(range(6), cts, color=["#FF6B6B"] + ["#4ECDC4"] * 5,
                edgecolor="white", alpha=0.9)
    for i, c in enumerate(cts):
        axes[1].text(i, c + total * 0.003, f"{c/total:.1%}",
                     ha="center", va="bottom", fontsize=8, color="navy")
    axes[1].set_xticks(range(6))
    axes[1].set_xlabel("Score Bracket", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Count per Integer Score Bracket", fontsize=12)
    axes[1].axvline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

    # CDF
    ss = np.sort(sv)
    cdf = np.arange(1, len(ss)+1) / len(ss)
    axes[2].plot(ss, cdf, color="steelblue", linewidth=1.5)
    axes[2].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[2].axhline(zero_frac, color="red", linestyle=":", linewidth=1.0)
    axes[2].text(0.1, zero_frac + 0.01, f"CDF@0 = {zero_frac:.1%}",
                 color="red", fontsize=10)
    axes[2].set_xlabel("Score", fontsize=11)
    axes[2].set_ylabel("Cumulative Fraction", fontsize=11)
    axes[2].set_title("Empirical CDF of Score", fontsize=12)
    axes[2].set_xlim(-0.1, 5.1)

    fig.suptitle(f"Zero-Inflation Analysis — {zero_frac:.1%} of weekly scores = 0", fontsize=12)
    fig.tight_layout()
    _save(fig, "02_zero_inflation.png")


# ─────────────────────────────────────────────────────────────────────────────
# §4  Train vs Test Feature Distribution
# ─────────────────────────────────────────────────────────────────────────────
def sec4_train_test_distribution(train_w: pd.DataFrame, test_w: pd.DataFrame) -> None:
    _section(4, "Train vs Test Feature Distribution")

    compare_feats = _present([
        "prec", "tmp", "humidity", "wind",
        "prec_week_max", "tmp_week_max", "tmp_week_std",
        "water_shortage", "water_shortage_roll_cum_4w", "water_shortage_roll_cum_8w",
        "is_dry_spell", "shortage_momentum", "cross_shortage_x_anomaly",
        "aridity_index", "tmp_anomaly",
    ], train_w)
    compare_feats = [c for c in compare_feats if c in test_w.columns]

    # ── Distribution comparison table
    print(f"\n[4.1] Feature Distribution Comparison (Train vs Test)")
    hdr = f"  {'Feature':<28} | {'Tr Mean':>8} | {'Te Mean':>8} | " \
          f"{'Ratio':>6} | {'Tr 99%':>8} | {'Te 99%':>8} | {'Danger':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for f in compare_feats:
        tr_m = train_w[f].mean()
        te_m = test_w[f].mean()
        ratio = te_m / (tr_m + 1e-9)
        tr99  = train_w[f].quantile(0.99)
        te99  = test_w[f].quantile(0.99)
        pct99 = (te99 - tr99) / (abs(tr99) + 1e-9) * 100
        flag = "🚨" if pct99 > 15 else ("⚠" if pct99 > 8 else "  ")
        print(f"  {f:<28} | {tr_m:>8.3f} | {te_m:>8.3f} | "
              f"{ratio:>6.2f}x | {tr99:>8.3f} | {te99:>8.3f} | {flag} {pct99:+.1f}%")

    # ── Plot: KDE comparison for key features
    key_feats = _present(["prec", "tmp", "humidity", "water_shortage",
                           "water_shortage_roll_cum_4w", "tmp_anomaly",
                           "is_dry_spell", "shortage_momentum"], train_w)
    key_feats = [c for c in key_feats if c in test_w.columns]
    n = len(key_feats)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, f in enumerate(key_feats):
        ax = axes[i]
        tr_vals = train_w[f].dropna().sample(n=min(200_000, len(train_w)), random_state=RANDOM_SEED)
        te_vals = test_w[f].dropna()
        try:
            tr_vals.plot.kde(ax=ax, color="#2196F3", linewidth=1.8, label="Train", bw_method=0.3)
            te_vals.plot.kde(ax=ax, color="#FF5722", linewidth=1.8, label="Test",  bw_method=0.3)
        except Exception:
            ax.hist(tr_vals, bins=30, color="#2196F3", alpha=0.5, density=True, label="Train")
            ax.hist(te_vals, bins=30, color="#FF5722", alpha=0.5, density=True, label="Test")
        ax.set_title(f, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Train vs Test Feature Distributions (KDE) — v54", fontsize=13)
    fig.tight_layout()
    _save(fig, "03_train_test_dist.png")


# ─────────────────────────────────────────────────────────────────────────────
# §5  Drought Feature Analysis
# ─────────────────────────────────────────────────────────────────────────────
def sec5_drought_analysis(train_w: pd.DataFrame) -> None:
    _section(5, "Drought Feature Analysis  (Multiplier + Spearman)")

    if "score" not in train_w.columns:
        print("  [Skip] No 'score' column.")
        return

    df = train_w.dropna(subset=["score"]).copy()
    levels = {
        "Safe (0)":    df[df["score"] == 0.0],
        "Mild (0-1]":  df[(df["score"] > 0) & (df["score"] <= 1)],
        "Mod  (1-2]":  df[(df["score"] > 1) & (df["score"] <= 2)],
        "Sev  (2-3]":  df[(df["score"] > 2) & (df["score"] <= 3)],
        "Ext  (>3)":   df[df["score"] > 3],
    }
    sizes = {k: len(v) for k, v in levels.items()}
    print("\n  Sample counts per drought level:")
    for k, n in sizes.items():
        print(f"    {k:<15}: {n:>10,} ({n/len(df):.1%})")

    feats = _present(DROUGHT_FEATS, df)

    # ── Multiplier Table
    print(f"\n[5.1] Feature Multiplier Across Drought Levels (mean values)")
    col_w = 12
    hdr = f"  {'Feature':<28}" + "".join(f" | {k:>{col_w}}" for k in levels) + " | {'Ext/Safe':>9}"
    print(hdr)
    print("  " + "-" * len(hdr))
    for f in feats:
        row = f"  {f:<28}"
        safe_m = levels["Safe (0)"][f].mean()
        for lname, ldf in levels.items():
            m = ldf[f].mean()
            row += f" | {m:>{col_w}.3f}"
        ext_m = levels["Ext  (>3)"][f].mean()
        multiplier = ext_m / (abs(safe_m) + 1e-6)
        flag = "  🔥" if multiplier > 2.0 else ("  ⬆" if multiplier > 1.3 else "")
        row += f" | {multiplier:>9.2f}x{flag}"
        print(row)

    # ── Spearman Table
    print(f"\n[5.2] Spearman Rank Correlation with Score (sample 200k)")
    df_s = df.dropna(subset=feats).sample(n=min(200_000, len(df)), random_state=RANDOM_SEED)
    y_s  = df_s["score"].values
    results = []
    for f in feats:
        corr, pval = spearmanr(df_s[f].values, y_s)
        results.append((f, corr, pval))
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"  {'Feature':<28} | {'Spearman r':>11} | {'p-value':>10} | {'|r|':>6}")
    print("  " + "-" * 65)
    for f, r, p in results:
        sig = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
        print(f"  {f:<28} | {r:>11.4f} | {p:>10.2e} |{sig}")

    # ── Plot: Drought Profile Bar Chart
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    axes = axes.flatten()
    level_colors = ["#4CAF50", "#FFEB3B", "#FF9800", "#F44336", "#9C27B0"]
    level_names  = list(levels.keys())

    for idx, f in enumerate(feats[:12]):
        ax = axes[idx]
        means = [levels[k][f].mean() for k in level_names]
        ax.bar(range(len(level_names)), means, color=level_colors, edgecolor="white", alpha=0.9)
        ax.set_xticks(range(len(level_names)))
        ax.set_xticklabels([k.split("(")[0].strip() for k in level_names],
                           rotation=25, ha="right", fontsize=8)
        ax.set_title(f, fontsize=9, fontweight="bold")

    for j in range(len(feats[:12]), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Mean Value by Drought Severity Level (v54)", fontsize=13)
    fig.tight_layout()
    _save(fig, "04_drought_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# §6  Correlation Heatmaps
# ─────────────────────────────────────────────────────────────────────────────
def sec6_correlation_heatmaps(train_w: pd.DataFrame) -> None:
    _section(6, "Correlation Heatmaps  (Pearson & Spearman)")

    if "score" not in train_w.columns:
        print("  [Skip] No 'score' column.")
        return

    exclude = {"score", "day_ordinal", "week_idx", "week_end_date",
               "region_id", "_v54_processed"}
    num_cols = [c for c in train_w.select_dtypes(include=[np.number]).columns
                if c not in exclude and c in FEATURE_COLS]
    corr_df  = train_w[num_cols + ["score"]].dropna(subset=["score"])
    corr_s   = corr_df.sample(n=min(500_000, len(corr_df)), random_state=RANDOM_SEED)

    for method in ("pearson", "spearman"):
        print(f"  Computing {method} correlation matrix ({corr_s.shape[0]:,} rows)...")
        cm = corr_s.corr(method=method)
        n  = len(cm)

        # Full heatmap
        fig, ax = plt.subplots(figsize=(max(14, n * 0.48), max(12, n * 0.44)))
        mask = np.zeros_like(cm, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        sns.heatmap(cm, mask=mask, annot=False, cmap="coolwarm", center=0,
                    linewidths=0.25, ax=ax, cbar_kws={"shrink": 0.6})
        ax.set_title(f"{method.capitalize()} Correlation Heatmap (v54 Features)", fontsize=12)
        _save(fig, f"0{'5' if method=='pearson' else '6'}_corr_heatmap_{method}.png")

        # Score-only bar
        sc = cm["score"].drop("score").sort_values(key=abs, ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, max(6, len(sc) * 0.32)))
        colors = ["tomato" if v < 0 else "steelblue" for v in sc]
        sc.plot(kind="barh", ax=ax2, color=colors)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel(f"{method.capitalize()} correlation with score")
        ax2.set_title(f"Feature vs Score — {method.capitalize()} (v54)", fontsize=12)
        _save(fig2, f"0{'7' if method=='pearson' else '8'}_corr_score_{method}.png")


# ─────────────────────────────────────────────────────────────────────────────
# §7  Region Time-Series
# ─────────────────────────────────────────────────────────────────────────────
def sec7_region_timeseries(train_w: pd.DataFrame, n_regions: int = 5) -> None:
    _section(7, "Region Time-Series Visualisation")

    all_regions = train_w["region_id"].unique().tolist()
    sampled     = random.sample(all_regions, min(n_regions, len(all_regions)))
    weather_vars = _present(["tmp", "humidity", "prec", "wind"], train_w)
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    labels  = {"tmp": "Temp (°C)", "humidity": "Humidity (%)",
                "prec": "Precip (mm)", "wind": "Wind (m/s)"}

    for rid in sampled:
        sub = train_w[train_w["region_id"] == rid].sort_values("week_idx").copy()
        fig, axes = plt.subplots(len(weather_vars), 1,
                                 figsize=(16, 3 * len(weather_vars)), sharex=True)
        if len(weather_vars) == 1:
            axes = [axes]
        x = sub["week_idx"].values if "day_ordinal" not in sub.columns else sub["day_ordinal"].values

        for idx, (var, ax) in enumerate(zip(weather_vars, axes)):
            ax.plot(x, sub[var].values, color=colors[idx % 4],
                    linewidth=0.9, alpha=0.85, label=labels.get(var, var))
            ax.set_ylabel(labels.get(var, var), fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            if "score" in sub.columns:
                ax2 = ax.twinx()
                sub_s = sub.dropna(subset=["score"])
                x_s   = (sub_s["week_idx"].values if "day_ordinal" not in sub_s.columns
                         else sub_s["day_ordinal"].values)
                ax2.scatter(x_s, sub_s["score"].values,
                            color="red", s=12, alpha=0.65, zorder=5)
                ax2.set_ylabel("Score", color="red", fontsize=8)
                ax2.tick_params(axis="y", labelcolor="red", labelsize=7)
                ax2.set_ylim(-0.5, 5.5)

        axes[-1].set_xlabel("Week Index", fontsize=9)
        fig.suptitle(f"Weekly Weather & Score — Region {rid} (v54)", fontsize=12, y=1.01)
        fig.tight_layout()
        _save(fig, f"09_timeseries_{rid}.png")


# ─────────────────────────────────────────────────────────────────────────────
# §8  Feature Boxplots
# ─────────────────────────────────────────────────────────────────────────────
def sec8_feature_boxplots(train_w: pd.DataFrame) -> None:
    _section(8, "Feature Boxplots  (weekly aggregated post-clip)")

    boxplot_feats = _present([
        "prec", "surf_pre", "humidity", "tmp", "tmp_max", "tmp_min",
        "wind", "wind_min", "tmp_week_max", "tmp_week_min", "tmp_week_std",
        "humidity_week_max", "humidity_week_std", "wind_week_max", "wind_week_std",
        "prec_week_max",
    ], train_w)

    df_s   = train_w.sample(n=min(500_000, len(train_w)), random_state=RANDOM_SEED)
    n      = len(boxplot_feats)
    ncols  = 4
    nrows  = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(boxplot_feats):
        axes[i].boxplot(
            df_s[col].dropna(), vert=True, patch_artist=True,
            boxprops=dict(facecolor="#90CAF9", color="navy"),
            medianprops=dict(color="red", linewidth=1.5),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Meteorological Feature Boxplots — Weekly Aggregated (v54)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "10_feature_boxplots.png")


# ─────────────────────────────────────────────────────────────────────────────
# §9  Cyclical Feature Coverage
# ─────────────────────────────────────────────────────────────────────────────
def sec9_cyclical_features(train_w: pd.DataFrame) -> None:
    _section(9, "Cyclical Feature Coverage  (week_sin / week_cos)")

    if "week_sin" not in train_w.columns or "week_cos" not in train_w.columns:
        print("  [Skip] week_sin / week_cos not found.")
        return

    df_s = train_w.sample(n=min(200_000, len(train_w)), random_state=RANDOM_SEED)
    print(f"  week_sin range: [{train_w['week_sin'].min():.4f}, {train_w['week_sin'].max():.4f}]")
    print(f"  week_cos range: [{train_w['week_cos'].min():.4f}, {train_w['week_cos'].max():.4f}]")
    radius_mean = np.sqrt(train_w["week_sin"]**2 + train_w["week_cos"]**2).mean()
    print(f"  Mean radius (should ≈1.0): {radius_mean:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Unit circle
    axes[0].scatter(df_s["week_sin"], df_s["week_cos"],
                    s=1, alpha=0.05, color="steelblue", rasterized=True)
    th = np.linspace(0, 2 * np.pi, 300)
    axes[0].plot(np.sin(th), np.cos(th), "r-", linewidth=1.5, alpha=0.6)
    axes[0].set_aspect("equal"); axes[0].set_xlim(-1.2, 1.2); axes[0].set_ylim(-1.2, 1.2)
    axes[0].set_title("Cyclical Coverage (unit circle)", fontsize=12)
    axes[0].set_xlabel("week_sin"); axes[0].set_ylabel("week_cos")

    axes[1].hist(df_s["week_sin"], bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[1].set_title("Distribution of week_sin", fontsize=12)

    axes[2].hist(df_s["week_cos"], bins=60, color="#4CAF50", edgecolor="white", alpha=0.85)
    axes[2].set_title("Distribution of week_cos", fontsize=12)

    fig.suptitle("Cyclical Time Encoding (doy/365.25 ratio) — v54", fontsize=12)
    fig.tight_layout()
    _save(fig, "11_cyclical_features.png")


# ─────────────────────────────────────────────────────────────────────────────
# §10  Dataset Structure Analysis
# ─────────────────────────────────────────────────────────────────────────────
def sec10_dataset_structure(train_w: pd.DataFrame, test_w: pd.DataFrame) -> None:
    _section(10, "Dataset Structure Analysis")

    WINDOW_SIZE = 13
    HORIZON     = 5

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Weeks per train region
    wk_tr = train_w.groupby("region_id").size()
    axes[0, 0].hist(wk_tr.values, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0, 0].axvline(wk_tr.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"Mean={wk_tr.mean():.0f}")
    axes[0, 0].set_title("Train: Weeks per Region  (expected 782)", fontsize=11)
    axes[0, 0].set_xlabel("Weeks"); axes[0, 0].set_ylabel("# Regions")
    axes[0, 0].legend()

    # Weeks per test region
    wk_te = test_w.groupby("region_id").size()
    axes[0, 1].hist(wk_te.values, bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
    axes[0, 1].axvline(wk_te.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"Mean={wk_te.mean():.0f}")
    axes[0, 1].set_title("Test: Weeks per Region  (expected 13)", fontsize=11)
    axes[0, 1].set_xlabel("Weeks"); axes[0, 1].set_ylabel("# Regions")
    axes[0, 1].legend()

    # Deployment gap (in weeks)
    gap_computed = False
    if "day_ordinal" in train_w.columns and "day_ordinal" in test_w.columns:
        tr_max = train_w.groupby("region_id")["day_ordinal"].max()
        te_min = test_w.groupby("region_id")["day_ordinal"].min()
        common = tr_max.index.intersection(te_min.index)
        gaps_w = np.round((te_min[common] - tr_max[common]).values / 7).astype(int)
        axes[1, 0].hist(gaps_w, bins=30, color="#FF9800", edgecolor="white", alpha=0.85)
        axes[1, 0].axvline(gaps_w.mean(), color="red", linestyle="--", linewidth=1.5,
                           label=f"Mean={gaps_w.mean():.1f} wks")
        axes[1, 0].set_title("Deployment Gap per Region  (train_end → test_start)", fontsize=11)
        axes[1, 0].set_xlabel("Gap (weeks)"); axes[1, 0].set_ylabel("# Regions")
        axes[1, 0].legend()
        axes[1, 0].text(0.98, 0.95,
                        f"min={gaps_w.min()}  med={int(np.median(gaps_w))}  max={gaps_w.max()}",
                        transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7))
        median_gap = int(np.median(gaps_w))
        gap_computed = True
        print(f"  Deployment gap: min={gaps_w.min()}  median={median_gap}  max={gaps_w.max()} weeks")
    else:
        axes[1, 0].text(0.5, 0.5, "day_ordinal not available",
                        ha="center", va="center", transform=axes[1, 0].transAxes)
        median_gap = 4

    # Available sliding windows
    rl = train_w.groupby("region_id").size()
    seq_counts = np.maximum(0, rl.values - WINDOW_SIZE - (median_gap if gap_computed else 4) - HORIZON)
    axes[1, 1].hist(seq_counts, bins=30, color="#9C27B0", edgecolor="white", alpha=0.85)
    axes[1, 1].axvline(seq_counts.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"Mean={seq_counts.mean():.0f}")
    axes[1, 1].set_title(f"Available Sliding Windows (W={WINDOW_SIZE}, H={HORIZON})\n"
                          f"total≈{seq_counts.sum():,}", fontsize=11)
    axes[1, 1].set_xlabel("# Windows per Region"); axes[1, 1].set_ylabel("# Regions")
    axes[1, 1].legend()

    fig.suptitle(f"Dataset Structure — v54  |  {train_w['region_id'].nunique():,} regions", fontsize=13)
    fig.tight_layout()
    _save(fig, "12_dataset_structure.png")

    print(f"  Train: {train_w['region_id'].nunique():,} regions × ~{wk_tr.mean():.0f} wks = {len(train_w):,} rows")
    print(f"  Test:  {test_w['region_id'].nunique():,} regions × ~{wk_te.mean():.0f} wks = {len(test_w):,} rows")
    print(f"  Total training samples (est.): {int(seq_counts.sum()):,}")


# ─────────────────────────────────────────────────────────────────────────────
# §11  Climate Cluster Analysis
# ─────────────────────────────────────────────────────────────────────────────
def sec11_cluster_analysis(cluster_df: pd.DataFrame, train_w: pd.DataFrame) -> None:
    _section(11, "Climate Cluster Analysis  (physical features, no score leakage)")

    phys_feats = [c for c in ["tmp_mean", "tmp_std", "prec_mean", "prec_std",
                               "humidity_mean", "humidity_std", "wind_mean", "wind_std"]
                  if c in cluster_df.columns]
    if not phys_feats:
        print("  [Skip] No physical cluster features found in region_stats.csv.")
        return

    # Join post-hoc score stats from training data (for interpretive display only)
    if "score" in train_w.columns:
        score_stats = (
            train_w.dropna(subset=["score"])
            .groupby("region_id")["score"]
            .agg(score_mean="mean", score_zero_prob=lambda x: (x == 0).mean(), score_std="std")
            .reset_index()
        )
        df = cluster_df.merge(score_stats, on="region_id", how="left")
    else:
        df = cluster_df.copy()

    # Centroid table — sorted by prec_mean descending (wetter → drier)
    print("\n[11.1] Cluster Centroids (physical features):")
    agg_cols = phys_feats + (["score_mean", "score_zero_prob"] if "score_mean" in df.columns else [])
    summary = (
        df.groupby("cluster_id")
          .agg(region_count=("region_id", "count"),
               **{c: (c, "mean") for c in agg_cols})
          .round(4)
    )
    if "prec_mean" in summary.columns:
        summary = summary.sort_values("prec_mean", ascending=False)
    print(summary.to_string())

    # Homogeneity
    print("\n[11.2] Cluster Homogeneity (within-cluster std — lower = more homogeneous):")
    std_sum = df.groupby("cluster_id")[phys_feats].std().round(4)
    std_sum = std_sum.loc[summary.index]
    print(std_sum.to_string())

    # ── Plot: Boxplots for physical features by cluster
    n   = len(phys_feats)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = axes.flatten()
    order = summary.index.tolist()

    for i, feat in enumerate(phys_feats):
        sns.boxplot(x="cluster_id", y=feat, data=df,
                    ax=axes[i], order=order, palette="viridis")
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Cluster ID")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Physical Climate Cluster Distributions (v54, leakage-free)", fontsize=13)
    fig.tight_layout()
    _save(fig, "13_cluster_distributions.png")

    # ── Plot: Pairplot (4 key mean values)
    pair_feats = _present(["tmp_mean", "prec_mean", "humidity_mean", "wind_mean"], df)
    if len(pair_feats) >= 2:
        df_plot = df.copy()
        df_plot["cluster_id"] = df_plot["cluster_id"].astype(str)
        pp = sns.pairplot(df_plot, vars=pair_feats, hue="cluster_id",
                          corner=True, palette="tab10",
                          plot_kws={"alpha": 0.6, "s": 15})
        pp.figure.suptitle("Climate Ecosystem Feature Interactions", y=1.01, fontsize=14)
        pp.savefig(os.path.join(PLOTS_DIR, "14_cluster_pairplot.png"), dpi=150)
        plt.close(pp.figure)
        print(f"    Saved → {os.path.join(PLOTS_DIR, '14_cluster_pairplot.png')}")


# ─────────────────────────────────────────────────────────────────────────────
# §12  Adversarial Validation
# ─────────────────────────────────────────────────────────────────────────────
def sec12_adversarial_validation(train_w: pd.DataFrame, test_w: pd.DataFrame) -> None:
    _section(12, "Adversarial Validation  (Train vs Test LightGBM AUC)")

    exclude = {"score", "week_idx", "day_ordinal", "week_end_date",
               "region_id", "_v54_processed"}
    feat_cols = [c for c in train_w.select_dtypes(include=[np.number]).columns
                 if c not in exclude and c in test_w.columns]

    print(f"  Features used: {len(feat_cols)}")

    tr_sample = train_w[feat_cols].dropna().sample(n=min(60_000, len(train_w)), random_state=RANDOM_SEED)
    te_sample = test_w[feat_cols].dropna()

    X = pd.concat([tr_sample, te_sample], ignore_index=True)
    y = np.array([0] * len(tr_sample) + [1] * len(te_sample))

    print(f"  Train sample: {len(tr_sample):,}  |  Test rows: {len(te_sample):,}")

    skf  = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    oof  = np.zeros(len(X))
    fi   = np.zeros(len(feat_cols))

    for tr_idx, va_idx in skf.split(X, y):
        clf = lgb.LGBMClassifier(
            max_depth=4, num_leaves=15, learning_rate=0.05,
            n_estimators=500, n_jobs=-1, random_state=RANDOM_SEED, verbose=-1,
        )
        clf.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
        fi += clf.feature_importances_ / 3.0

    auc = roc_auc_score(y, oof)
    print(f"\n  {'='*60}")
    print(f"  ADVERSARIAL AUC: {auc:.4f}")
    if auc < 0.55:
        print("  ✓ LOW SHIFT — Train and Test are well-aligned (AUC < 0.55)")
    elif auc < 0.65:
        print("  ⚠ MODERATE SHIFT — Some distributional difference (0.55-0.65)")
    else:
        print("  🚨 HIGH SHIFT — Significant Train/Test covariate shift (AUC > 0.65)")
    print(f"  {'='*60}")

    fi_df = pd.DataFrame({"Feature": feat_cols, "Importance": fi})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    print(f"\n  Top 20 Most Shifted Features:")
    print(f"  {'Feature':<30} {'Importance':>12}")
    print("  " + "-" * 45)
    for _, row in fi_df.head(20).iterrows():
        print(f"  {row['Feature']:<30} {row['Importance']:>12.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 80)
    print("  final_eda.py — Complete EDA & Data Mining Pipeline (v54)")
    print(f"  Output directory: {PLOTS_DIR}")
    print("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading raw data (train.csv, test.csv) ...")
    raw_train_path = os.path.join(DATA_DIR, "train.csv")
    raw_test_path  = os.path.join(DATA_DIR, "test.csv")
    core_cols      = ["date", "region_id", "prec", "tmp", "humidity", "wind", "score"]
    raw_cols_tr    = set(pd.read_csv(raw_train_path, nrows=0).columns)
    raw_cols_te    = set(pd.read_csv(raw_test_path,  nrows=0).columns)
    raw_train = pd.read_csv(raw_train_path,
                            usecols=[c for c in core_cols if c in raw_cols_tr])
    raw_test  = pd.read_csv(raw_test_path,
                            usecols=[c for c in core_cols if c in raw_cols_te and c != "score"])

    print("Loading processed data (v54_processed) ...")
    train_path = os.path.join(PROC_DIR, "train_processed.csv")
    test_path  = os.path.join(PROC_DIR, "test_processed.csv")
    cluster_path = os.path.join(PROC_DIR, "region_stats.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Processed CSVs not found in {PROC_DIR}.\n"
            "Run `python src/v54_preprocess.py` first."
        )

    train_w   = pd.read_csv(train_path)
    test_w    = pd.read_csv(test_path)
    print(f"  train_w: {train_w.shape}  |  test_w: {test_w.shape}")

    cluster_df = pd.read_csv(cluster_path) if os.path.exists(cluster_path) else None

    # ── Run sections ───────────────────────────────────────────────────────────
    sec1_raw_reality_check(raw_train, raw_test)
    sec2_processed_overview(train_w, test_w)
    sec3_score_distribution(train_w)
    sec4_train_test_distribution(train_w, test_w)
    sec5_drought_analysis(train_w)
    sec6_correlation_heatmaps(train_w)
    sec7_region_timeseries(train_w, n_regions=5)
    sec8_feature_boxplots(train_w)
    sec9_cyclical_features(train_w)
    sec10_dataset_structure(train_w, test_w)

    if cluster_df is not None:
        sec11_cluster_analysis(cluster_df, train_w)
    else:
        print("\n[Skip §11] region_stats.csv not found.")

    sec12_adversarial_validation(train_w, test_w)

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    saved   = sorted(f for f in os.listdir(PLOTS_DIR) if f.endswith(".png"))
    print(f"\n{'='*80}")
    print(f"  All done in {elapsed:.1f}s. {len(saved)} plots saved to {PLOTS_DIR}/")
    for f in saved:
        print(f"    {f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
