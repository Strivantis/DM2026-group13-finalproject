import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns

def run_v17_eda():
    print("="*50)
    print("v17 Diagnostics: Fold 2 Anomaly & Zero-Inflation")
    print("="*50)

    # 1. 模擬 v17 的 5-Fold 分割 (與 train.py 相同的 seed=42)
    train_df = pd.read_csv("data/processed/train_processed.csv")
    test_df = pd.read_csv("data/processed/test_processed.csv")
    
    # 取得每個 region 的靜態屬性
    region_stats = train_df.groupby("region_id").agg(
        mean_score=("score", "mean"),
        zero_prob=("score", lambda x: (x == 0).mean())
    ).reset_index()

    # 計算真實 Gap
    test_start = test_df.groupby("region_id")["day_ordinal"].min().reset_index(name="test_start")
    train_end = train_df.groupby("region_id")["day_ordinal"].max().reset_index(name="train_end")
    gaps = pd.merge(test_start, train_end, on="region_id")
    gaps["gap_weeks"] = (gaps["test_start"] - gaps["train_end"]) / 7.0
    region_stats = pd.merge(region_stats, gaps[["region_id", "gap_weeks"]], on="region_id")

    # GroupKFold 分割
    gkf = GroupKFold(n_splits=5)
    region_stats["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(region_stats, groups=region_stats["region_id"])):
        region_stats.loc[val_idx, "fold"] = fold

    # 檢視各 Fold 的統計差異 (鎖定 Fold 2)
    print("\n[Region Characteristics per Fold]")
    fold_analysis = region_stats.groupby("fold").agg(
        region_count=("region_id", "count"),
        avg_mean_score=("mean_score", "mean"),
        avg_zero_prob=("zero_prob", "mean"),
        avg_gap_weeks=("gap_weeks", "mean"),
        max_gap=("gap_weeks", "max")
    )
    print(fold_analysis)
    
    # 2. Submission 的軟塌陷分析
    sub = pd.read_csv("submission_17th.csv")
    pred_cols = [c for c in sub.columns if "pred" in c]
    all_preds = sub[pred_cols].values.flatten()
    
    print("\n[Submission Predictions Distribution]")
    print(f"Total Predictions: {len(all_preds)}")
    print(f"Exactly 0.0: {(all_preds == 0.0).sum()} ({(all_preds == 0.0).mean()*100:.2f}%)")
    print(f"< 0.1: {(all_preds < 0.1).sum()} ({(all_preds < 0.1).mean()*100:.2f}%)")
    print(f"< 0.5: {(all_preds < 0.5).sum()} ({(all_preds < 0.5).mean()*100:.2f}%)")
    print(f"Mean Prediction: {all_preds.mean():.4f}")

    # 繪製分佈圖
    plt.figure(figsize=(10, 5))
    sns.histplot(all_preds, bins=50, kde=True, color='red', alpha=0.5, label='v17 Predictions')
    plt.axvline(x=0.05, color='black', linestyle='--', label='Near Zero Threshold')
    plt.title("v17 Submission Values Distribution (Missing Zeros?)")
    plt.legend()
    plt.savefig("v17_submission_dist.png")
    print("\nDistribution plot saved to v17_submission_dist.png")

if __name__ == "__main__":
    run_v17_eda()