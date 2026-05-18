"""
Drought Prediction - Exploratory Data Analysis (v9)
====================================================
Generates and saves to /plots:
  1. Pearson & Spearman correlation heatmaps (features vs. score)
  2. Time-series line plots for 5 randomly sampled region_ids
  3. Histogram of the target variable `score`
  4. Boxplots of meteorological features (outlier inspection)
  5. [v9 NEW] Zero-Inflation Analysis – fraction of scores == 0
  6. [v9 NEW] Cyclical Time Feature Coverage (week_sin vs week_cos circle)
  7. [v9 NEW] Region Target Encoding Distribution
     (region_mean_score & region_zero_prob histograms)

Run after preprocess.py has been executed (or import its helpers directly).
Calling this script ALSO regenerates data/processed/ CSVs with v9 features
(week_sin, week_cos replacing month/week_of_year).

Date columns are STRING-based (years 3004-3020) so all x-axis plotting
uses the integer ``day_ordinal`` column instead of pd.Timestamp objects.
"""

import os
import random
import warnings

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Import preprocessing helpers
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_data,
    impute_met_features,
    handle_outliers,
    align_labels_strategy_a,
    aggregate_test_weekly,
    preprocess_data,
    export_processed,
    MET_COLS,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# RNG seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Seaborn theme
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ---------------------------------------------------------------------------
# Helper: save figure
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 1. Correlation heatmaps (Pearson & Spearman)
# ---------------------------------------------------------------------------
def plot_correlation_heatmaps(df: pd.DataFrame) -> None:
    """
    Compute Pearson and Spearman correlation matrices between all numeric
    engineered features and ``score``, then save heatmaps.
    Rows with NaN score are dropped before computing correlations.
    Downsamples to 500 k rows for Spearman performance.
    v9: week_sin and week_cos appear instead of month/week_of_year.
    """
    df_valid   = df.dropna(subset=["score"]).copy()
    df_sampled = df_valid.sample(n=min(500_000, len(df_valid)), random_state=RANDOM_SEED)

    # Exclude non-feature columns
    exclude = {"score", "day_ordinal", "year"}
    numeric_cols = [
        c for c in df_sampled.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    corr_data = df_sampled[numeric_cols + ["score"]]

    for method in ("pearson", "spearman"):
        corr_matrix = corr_data.corr(method=method)

        # --- full lower-triangle heatmap ---
        n = len(corr_matrix)
        fig, ax = plt.subplots(figsize=(max(14, n * 0.5), max(12, n * 0.45)))
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            center=0,
            linewidths=0.3,
            ax=ax,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(f"{method.capitalize()} Correlation Heatmap (Weekly Features) — v9",
                     fontsize=14, pad=12)
        _save(fig, f"corr_heatmap_{method}.png")

        # --- score-only bar chart ---
        score_corr = (
            corr_matrix["score"]
            .drop("score")
            .sort_values(key=abs, ascending=False)
        )
        fig2, ax2 = plt.subplots(figsize=(10, max(6, len(score_corr) * 0.35)))
        colors = ["tomato" if v < 0 else "steelblue" for v in score_corr]
        score_corr.plot(kind="barh", ax=ax2, color=colors)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel(f"{method.capitalize()} correlation with score")
        ax2.set_title(f"Feature vs. Score ({method.capitalize()}) — Weekly v9", fontsize=13)
        _save(fig2, f"corr_score_bar_{method}.png")


# ---------------------------------------------------------------------------
# 2. Time-series plots for sampled regions
# ---------------------------------------------------------------------------
def plot_region_timeseries(df: pd.DataFrame, n_regions: int = 5) -> None:
    """
    For n randomly sampled region_ids, plot weekly trends of:
      - Temperature (tmp)
      - Humidity
      - Precipitation (prec)
      - Wind speed (wind)
      - Score (right axis, scatter)

    X-axis uses ``day_ordinal`` (integer weeks since epoch) and is labelled
    with the ``week_key`` string every ~26 ticks for readability.
    """
    all_regions = df["region_id"].unique().tolist()
    n_regions   = min(n_regions, len(all_regions))
    sampled     = random.sample(all_regions, n_regions)

    weather_vars = [c for c in ["tmp", "humidity", "prec", "wind"] if c in df.columns]
    colors       = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    labels       = {
        "tmp":      "Temp (°C)",
        "humidity": "Humidity (%)",
        "prec":     "Precip (mm)",
        "wind":     "Wind (m/s)",
    }

    for rid in sampled:
        sub = df[df["region_id"] == rid].sort_values("day_ordinal").copy()

        fig, axes = plt.subplots(
            len(weather_vars), 1,
            figsize=(16, 3 * len(weather_vars)),
            sharex=True,
        )
        if len(weather_vars) == 1:
            axes = [axes]

        x = sub["day_ordinal"].values

        for idx, (var, ax) in enumerate(zip(weather_vars, axes)):
            ax.plot(x, sub[var].values, color=colors[idx % len(colors)],
                    linewidth=0.9, alpha=0.85, label=labels.get(var, var))
            ax.set_ylabel(labels.get(var, var), fontsize=9)
            ax.legend(loc="upper right", fontsize=8)

            # Overlay score on secondary axis
            if "score" in sub.columns:
                ax2   = ax.twinx()
                sub_s = sub.dropna(subset=["score"])
                ax2.scatter(
                    sub_s["day_ordinal"].values, sub_s["score"].values,
                    color="red", s=12, alpha=0.65, label="score", zorder=5,
                )
                ax2.set_ylabel("Score", color="red", fontsize=8)
                ax2.tick_params(axis="y", labelcolor="red", labelsize=7)
                ax2.set_ylim(-0.5, 5.5)
                ax2.legend(loc="upper left", fontsize=7)

        # Tick labels: sample ~10 week_key strings along x-axis
        ax_bot = axes[-1]
        step   = max(1, len(sub) // 10)
        tick_x = x[::step]
        if "week_key" in sub.columns:
            tick_labels = sub["week_key"].values[::step]
        else:
            tick_labels = [str(v) for v in tick_x]
        ax_bot.set_xticks(tick_x)
        ax_bot.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7)
        ax_bot.set_xlabel("Week", fontsize=9)

        fig.suptitle(f"Weekly Weather & Score — Region {rid}", fontsize=13, y=1.01)
        fig.tight_layout()
        _save(fig, f"timeseries_{rid}.png")


# ---------------------------------------------------------------------------
# 3. Score histogram
# ---------------------------------------------------------------------------
def plot_score_histogram(df: pd.DataFrame) -> None:
    """Histogram of the target variable score (0–5 integer scale)."""
    score_vals = df["score"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.25, 5.75, 0.5)
    ax.hist(score_vals, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Target Variable (score) — Weekly Aggregation", fontsize=13)
    ax.set_xticks(range(6))

    total = len(score_vals)
    for v in range(6):
        cnt = ((score_vals >= v - 0.25) & (score_vals < v + 0.25)).sum()
        if cnt > 0:
            ax.text(v, cnt + total * 0.005, f"{cnt/total:.1%}",
                    ha="center", va="bottom", fontsize=8, color="navy")

    _save(fig, "score_histogram.png")


# ---------------------------------------------------------------------------
# 4. Boxplots for outlier inspection
# ---------------------------------------------------------------------------
def plot_feature_boxplots(df: pd.DataFrame) -> None:
    """Boxplots of all meteorological features (post-clip weekly values)."""
    df = df.sample(n=min(500_000, len(df)), random_state=RANDOM_SEED)
    cols  = [c for c in MET_COLS if c in df.columns]
    n     = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].boxplot(
            df[col].dropna(), vert=True, patch_artist=True,
            boxprops=dict(facecolor="#90CAF9", color="navy"),
            medianprops=dict(color="red", linewidth=1.5),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Meteorological Feature Boxplots — Weekly Aggregated (Post-Clip)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "feature_boxplots.png")


# ---------------------------------------------------------------------------
# 5. [v9 NEW] Zero-Inflation Analysis
# ---------------------------------------------------------------------------
def plot_zero_inflation(df: pd.DataFrame) -> None:
    """
    v9 new plot: analyse and visualise the zero-inflation in the score column.
    Shows:
      - Pie chart: fraction of scores == 0 vs > 0
      - Bar chart: count per integer score bracket
      - Cumulative distribution up to score 5
    This directly motivates the Two-Stage architecture in v9.
    """
    score_vals = df["score"].dropna().values
    zero_mask  = score_vals == 0.0
    zero_frac  = zero_mask.mean()
    nonzero_frac = 1.0 - zero_frac

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- Panel 1: Pie chart ----
    ax1 = axes[0]
    wedge_colors = ["#FF6B6B", "#4ECDC4"]
    wedge_explode = (0.05, 0)
    ax1.pie(
        [zero_frac, nonzero_frac],
        labels=[f"score = 0\n({zero_frac:.1%})", f"score > 0\n({nonzero_frac:.1%})"],
        colors=wedge_colors,
        explode=wedge_explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax1.set_title("Zero-Inflation in Score Distribution", fontsize=13)

    # ---- Panel 2: Bar chart of integer score brackets ----
    ax2 = axes[1]
    brackets = np.arange(0, 6)
    counts = []
    for v in brackets:
        if v == 0:
            cnt = (score_vals == 0.0).sum()
        elif v == 5:
            cnt = (score_vals >= 4.5).sum()
        else:
            cnt = ((score_vals >= v - 0.5) & (score_vals < v + 0.5)).sum()
        counts.append(cnt)
    total = len(score_vals)
    bar_colors = ["#FF6B6B"] + ["#4ECDC4"] * 5
    bars = ax2.bar(brackets, counts, color=bar_colors, edgecolor="white", alpha=0.9)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.003,
                 f"{cnt/total:.1%}", ha="center", va="bottom", fontsize=9, color="navy")
    ax2.set_xlabel("Score Bracket", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_xticks(brackets)
    ax2.set_title("Score Distribution by Integer Bracket", fontsize=13)
    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                label="Zero/Non-zero boundary")
    ax2.legend(fontsize=9)

    # ---- Panel 3: CDF ----
    ax3 = axes[2]
    sorted_scores = np.sort(score_vals)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax3.plot(sorted_scores, cdf, color="steelblue", linewidth=1.5)
    ax3.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax3.axhline(zero_frac, color="red", linestyle=":", linewidth=1.0, alpha=0.7)
    ax3.text(0.05, zero_frac + 0.01, f"CDF at 0 = {zero_frac:.1%}",
             color="red", fontsize=10, va="bottom")
    ax3.set_xlabel("Score", fontsize=12)
    ax3.set_ylabel("Cumulative Fraction", fontsize=12)
    ax3.set_title("Empirical CDF of Score", fontsize=13)
    ax3.set_xlim(-0.1, 5.1)
    ax3.grid(True, alpha=0.4)

    fig.suptitle(
        f"v9 Zero-Inflation Analysis  —  {zero_frac:.1%} of weekly scores are exactly 0\n"
        f"Motivates Two-Stage Architecture: P(drought) × Severity",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "zero_inflation_analysis_v9.png")


# ---------------------------------------------------------------------------
# 6. [v9 NEW] Cyclical Time Feature Coverage
# ---------------------------------------------------------------------------
def plot_cyclical_features(df: pd.DataFrame) -> None:
    """
    v9 new plot: verify that week_sin and week_cos form a proper unit circle
    (covering all 53 weeks uniformly), and show their individual distributions.
    """
    if "week_sin" not in df.columns or "week_cos" not in df.columns:
        print("  [Skip] week_sin / week_cos not found in dataframe.")
        return

    # Sample for performance
    df_s = df.sample(n=min(200_000, len(df)), random_state=RANDOM_SEED)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- Panel 1: Unit circle scatter ----
    ax1 = axes[0]
    ax1.scatter(df_s["week_sin"].values, df_s["week_cos"].values,
                s=1, alpha=0.05, color="steelblue", rasterized=True)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax1.plot(np.sin(theta), np.cos(theta), "r-", linewidth=1.5, alpha=0.6,
             label="Unit circle")
    ax1.set_xlabel("week_sin", fontsize=11)
    ax1.set_ylabel("week_cos", fontsize=11)
    ax1.set_title("Cyclical Feature Coverage\n(should fill unit circle)", fontsize=12)
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: week_sin distribution ----
    ax2 = axes[1]
    ax2.hist(df_s["week_sin"].values, bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("week_sin value", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of week_sin", fontsize=12)

    # ---- Panel 3: week_cos distribution ----
    ax3 = axes[2]
    ax3.hist(df_s["week_cos"].values, bins=60, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax3.set_xlabel("week_cos value", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Distribution of week_cos", fontsize=12)

    fig.suptitle(
        "v9 Cyclical Time Encoding: week_sin / week_cos\n"
        "(replaces linear month + week_of_year to preserve temporal continuity)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "cyclical_features_v9.png")


# ---------------------------------------------------------------------------
# 7. [v9 NEW] Region Target Encoding Distribution
# ---------------------------------------------------------------------------
def plot_region_te_stats(df: pd.DataFrame) -> None:
    """
    v9 new plot: compute and show the distribution of per-region target
    encoding statistics (region_mean_score & region_zero_prob) that will be
    used as features in train.py (leakage-free, fold-specific).
    """
    if "score" not in df.columns:
        print("  [Skip] No score column for TE stats plot.")
        return

    def _zero_prob(x):
        return (x == 0.0).mean()

    te_stats = (
        df.groupby("region_id")["score"]
        .agg(region_mean_score="mean", region_zero_prob=_zero_prob)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- Panel 1: region_mean_score histogram ----
    ax1 = axes[0]
    ax1.hist(te_stats["region_mean_score"], bins=50, color="#FF9800", edgecolor="white",
             alpha=0.85)
    ax1.axvline(te_stats["region_mean_score"].mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"Global mean = {te_stats['region_mean_score'].mean():.3f}")
    ax1.set_xlabel("region_mean_score", fontsize=11)
    ax1.set_ylabel("# Regions", fontsize=11)
    ax1.set_title("Per-Region Mean Drought Score", fontsize=12)
    ax1.legend(fontsize=9)

    # ---- Panel 2: region_zero_prob histogram ----
    ax2 = axes[1]
    ax2.hist(te_stats["region_zero_prob"], bins=50, color="#9C27B0", edgecolor="white",
             alpha=0.85)
    ax2.axvline(te_stats["region_zero_prob"].mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"Global mean = {te_stats['region_zero_prob'].mean():.3f}")
    ax2.set_xlabel("region_zero_prob", fontsize=11)
    ax2.set_ylabel("# Regions", fontsize=11)
    ax2.set_title("Per-Region Zero-Score Probability", fontsize=12)
    ax2.legend(fontsize=9)

    # ---- Panel 3: 2D scatter ----
    ax3 = axes[2]
    sc = ax3.scatter(
        te_stats["region_mean_score"], te_stats["region_zero_prob"],
        s=4, alpha=0.5, c=te_stats["region_mean_score"],
        cmap="RdYlGn_r", vmin=0, vmax=2,
    )
    plt.colorbar(sc, ax=ax3, label="region_mean_score")
    ax3.set_xlabel("region_mean_score", fontsize=11)
    ax3.set_ylabel("region_zero_prob", fontsize=11)
    ax3.set_title("Mean Score vs. Zero Probability\n(per region)", fontsize=12)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "v9 Target Encoding Feature Distributions (Full Training Set)\n"
        f"  2248 regions  |  Global mean score = {te_stats['region_mean_score'].mean():.3f}"
        f"  |  Global zero_prob = {te_stats['region_zero_prob'].mean():.3f}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "region_te_stats_v9.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 65)
    print("Drought Prediction — EDA Pipeline v9 (Cyclical + Zero-Inflation)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load & preprocess (Strategy A: weekly aggregation, no interpolation)
    # ------------------------------------------------------------------
    print("\nLoading raw data …")
    train_raw = load_data("train.csv")
    test_raw  = load_data("test.csv")
    print(f"  Raw train: {train_raw.shape}  |  Raw test: {test_raw.shape}")

    print("Imputing …")
    train = impute_met_features(train_raw)
    test  = impute_met_features(test_raw)

    print("Outlier clipping (Z=3.5) …")
    train = handle_outliers(train, z_thresh=3.5)
    test  = handle_outliers(test,  z_thresh=3.5)

    print("Weekly aggregation (v9: produces week_sin, week_cos; drops month, week_of_year) …")
    train_w = align_labels_strategy_a(train)
    test_w  = aggregate_test_weekly(test)

    # Verify v9 cyclical features
    assert "week_sin" in train_w.columns, "week_sin missing from weekly data"
    assert "week_cos" in train_w.columns, "week_cos missing from weekly data"
    assert "month" not in train_w.columns, "month should be dropped (v9)"
    assert "week_of_year" not in train_w.columns, "week_of_year should be dropped (v9)"
    print(f"  ✓ week_sin range: [{train_w['week_sin'].min():.3f}, {train_w['week_sin'].max():.3f}]")
    print(f"  ✓ week_cos range: [{train_w['week_cos'].min():.3f}, {train_w['week_cos'].max():.3f}]")

    print("Feature engineering …")
    train_w = preprocess_data(train_w)
    test_w  = preprocess_data(test_w)
    print(f"  train_w: {train_w.shape}  |  test_w: {test_w.shape}")

    # ------------------------------------------------------------------
    # Region count assertion
    # ------------------------------------------------------------------
    n_train = train_w["region_id"].nunique()
    n_test  = test_w["region_id"].nunique()
    assert n_train == 2248, f"Expected 2248 train regions, got {n_train}"
    assert n_test  == 2248, f"Expected 2248 test regions, got {n_test}"
    print(f"  ✓ Region counts: train={n_train}, test={n_test}")

    # Zero-inflation stats
    score_vals = train_w["score"].dropna()
    zero_frac  = (score_vals == 0.0).mean()
    print(f"\n[v9 Zero-Inflation Check]")
    print(f"  Score mean  : {score_vals.mean():.4f}")
    print(f"  Score == 0  : {zero_frac:.2%}  ({(score_vals == 0.0).sum():,} / {len(score_vals):,} rows)")
    print(f"  Score >  0  : {1 - zero_frac:.2%}")
    print(f"  ==> Two-Stage architecture directly addresses this zero-inflation.")

    # ------------------------------------------------------------------
    # Export processed data for PyTorch
    # ------------------------------------------------------------------
    print("\nExporting processed data to data/processed/ …")
    export_processed(train_w, test_w, fmt="csv")

    # ------------------------------------------------------------------
    # EDA plots
    # ------------------------------------------------------------------
    print("\n[1/7] Correlation heatmaps …")
    plot_correlation_heatmaps(train_w)

    print("[2/7] Region time-series plots (5 regions) …")
    plot_region_timeseries(train_w, n_regions=5)

    print("[3/7] Score histogram …")
    plot_score_histogram(train_w)

    print("[4/7] Feature boxplots …")
    plot_feature_boxplots(train_w)

    print("[5/7] Zero-Inflation analysis (v9 new) …")
    plot_zero_inflation(train_w)

    print("[6/7] Cyclical feature coverage (v9 new) …")
    plot_cyclical_features(train_w)

    print("[7/7] Region target encoding stats (v9 new) …")
    plot_region_te_stats(train_w)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    print("\n[Validation] NaN counts per column (train_w):")
    nan_report = train_w.isna().sum()
    nonzero_nans = nan_report[nan_report > 0]
    if len(nonzero_nans) > 0:
        print(nonzero_nans.to_string())
    else:
        print("  ✓ No NaNs detected in any column.")

    met_check = [c for c in MET_COLS if c in train_w.columns]
    nan_met   = train_w[met_check].isna().sum().sum()
    print(f"\nNaN in meteorological feature columns: {nan_met}")
    assert nan_met == 0, f"Unexpected NaNs in met features: {nan_met}"

    print("\n✓ EDA pipeline v9 completed. All plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
