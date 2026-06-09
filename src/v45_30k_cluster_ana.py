"""
v45_30k_cluster_analysis.py - OOF Error Analysis by Climate Ecosystem

This script merges the robust OOF predictions from V45.1 with the K-Means 
cluster definitions to pinpoint exactly which climate zones are causing 
the highest Mean Absolute Error (MAE).
"""

import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else os.getcwd()
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
OOF_FILE = os.path.join(ROOT, "v45.1_meta_train_oof.csv")

def main():
    print("=" * 80)
    print(" 🌍 V45.1 CLIMATE ECOSYSTEM (CLUSTER) ERROR ANALYSIS ")
    print("=" * 80)

    # 1. 檢查檔案是否存在
    if not os.path.exists(OOF_FILE):
        raise FileNotFoundError(f"Cannot find OOF file at: {OOF_FILE}")
    
    # 2. 載入資料
    print("\n[1] Loading OOF predictions and Cluster mapping...")
    oof_df = pd.read_csv(OOF_FILE)
    
    # 載入 train_processed.csv 來獲取 region 的 cluster_id
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"), usecols=["region_id", "cluster_id"])
    cluster_map = train_df.drop_duplicates(subset=["region_id"]).set_index("region_id")["cluster_id"].to_dict()

    # 3. 將 cluster_id 映射到 OOF 資料中
    oof_df["cluster_id"] = oof_df["region_id"].map(cluster_map)

    # 確保沒有 mapping 失敗的資料
    missing_clusters = oof_df["cluster_id"].isna().sum()
    if missing_clusters > 0:
        print(f"Warning: {missing_clusters} rows could not be mapped to a cluster.")

    # 4. 模擬神聖四捨五入 (因為你的 OOF file 中的 final_hurdle_pred 是浮點數)
    print("\n[2] Applying np.round() to OOF predictions to match LB scoring...")
    oof_df["rounded_pred"] = np.round(np.clip(oof_df["final_hurdle_pred"], 0.0, 5.0))
    oof_df["absolute_error"] = np.abs(oof_df["rounded_pred"] - oof_df["true_score"])

    # 5. 計算總體 MAE 作為基準線
    global_mae = oof_df["absolute_error"].mean()
    print(f"\n🌍 Global OOF Rounded MAE: {global_mae:.4f}")

    # 6. 依據 Cluster 進行分組診斷
    print("\n[3] MAE Breakdown by Climate Cluster:")
    
    cluster_stats = oof_df.groupby("cluster_id").agg(
        n_samples=("region_id", "count"),
        n_regions=("region_id", "nunique"),
        mean_true_score=("true_score", "mean"),
        zero_ratio=("true_score", lambda x: (x == 0).mean()),
        cluster_mae=("absolute_error", "mean")
    ).reset_index()

    # 依據 MAE 從最爛 (最高) 排到最好 (最低)
    cluster_stats = cluster_stats.sort_values(by="cluster_mae", ascending=False)

    print("-" * 80)
    print(f"{'Cluster':<8} | {'Regions':<8} | {'Samples':<9} | {'Mean True':<10} | {'Zero %':<8} | {'Rounded MAE':<12}")
    print("-" * 80)
    for _, row in cluster_stats.iterrows():
        c_id = int(row['cluster_id'])
        # 標記表現比平均差的 Cluster
        alert = " ⚠️ (Worse than Avg)" if row['cluster_mae'] > global_mae else ""
        print(f"Cluster {c_id:<1} | {int(row['n_regions']):<8} | {int(row['n_samples']):<9,} | {row['mean_true_score']:<10.4f} | {row['zero_ratio']:<8.1%} | {row['cluster_mae']:<10.4f}{alert}")
    print("-" * 80)

    # 7. 進一步分析：最爛 Cluster 的死因在哪？(False Positives vs False Negatives)
    worst_cluster = int(cluster_stats.iloc[0]["cluster_id"])
    print(f"\n[4] Deep Dive: Why is Cluster {worst_cluster} performing so poorly?")
    
    worst_df = oof_df[oof_df["cluster_id"] == worst_cluster]
    
    # 計算在真實為 0 的情況下，模型預測非 0 的比例與平均預測值 (False Positives)
    fp_mask = (worst_df["true_score"] == 0) & (worst_df["rounded_pred"] > 0)
    fp_count = fp_mask.sum()
    fp_avg_pred = worst_df.loc[fp_mask, "rounded_pred"].mean() if fp_count > 0 else 0
    
    # 計算在真實 > 0 的情況下，模型預測偏低 (Under-prediction)
    under_mask = (worst_df["true_score"] > 0) & (worst_df["rounded_pred"] < worst_df["true_score"])
    under_count = under_mask.sum()
    under_avg_diff = (worst_df.loc[under_mask, "true_score"] - worst_df.loc[under_mask, "rounded_pred"]).mean() if under_count > 0 else 0

    total_worst = len(worst_df)
    print(f"  -> True Zeros falsely predicted as Drought (FP): {fp_count:,} cases ({(fp_count/total_worst):.1%}) | Avg Pred: {fp_avg_pred:.2f}")
    print(f"  -> True Droughts under-predicted (Cowardice):    {under_count:,} cases ({(under_count/total_worst):.1%}) | Avg Miss: {under_avg_diff:.2f} points")

    print("\n💡 ACTIONABLE INSIGHT:")
    if fp_count > under_count:
        print(f"  Model B (Classifier) is too aggressive in Cluster {worst_cluster}. Consider increasing the Hurdle Threshold specifically for this ecosystem.")
    else:
        print(f"  Model A (Regressor) is too conservative in Cluster {worst_cluster}. It lacks the features to confidently predict extreme values here.")

if __name__ == "__main__":
    main()