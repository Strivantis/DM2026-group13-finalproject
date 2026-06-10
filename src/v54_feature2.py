"""
v54_feature_validator2.py
深度挖掘特徵交互作用 (Interactions) 與長期滾動記憶 (Long-term Rolling)
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

def print_header(title):
    print("\n" + "="*80)
    print(f" 🚀 {title}")
    print("="*80)

def main():
    t0 = time.time()
    
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    print("Loading RAW train data for deep simulation...")
    core_cols = ["region_id", "date", "prec", "tmp", "humidity", "wind", "score"]
    df = pd.read_csv(train_path, usecols=core_cols).dropna(subset=["score"])
    
    # 確保按時間排序
    df = df.sort_values(by=["region_id", "date"])

    print("Calculating Base Features...")
    df["pet"] = 0.55 * df["tmp"].clip(lower=0)
    df["water_shortage"] = (df["pet"] - df["prec"]).clip(lower=0)
    
    stats = df.groupby("region_id")["tmp"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].replace(0, 1.0)
    df = df.merge(stats, on="region_id", how="left", suffixes=("", "_reg"))
    df["tmp_anomaly"] = (df["tmp"] - df["mean"]) / df["std"]

    # =========================================================================
    # [探勘 1] 長期時間記憶 (Long-term Rolling)
    # 之前我們只做到 4w (28天)，這次我們看看 8w (56天) 和 12w (84天)
    # =========================================================================
    print("Calculating Long-term Memory Features...")
    # 假設一週 7 天，我們直接用日資料算 rolling，這樣更精確
    df["shortage_roll_4w"] = df.groupby("region_id")["water_shortage"].transform(lambda s: s.rolling(28, min_periods=1).sum())
    df["shortage_roll_8w"] = df.groupby("region_id")["water_shortage"].transform(lambda s: s.rolling(56, min_periods=1).sum())
    df["shortage_roll_12w"] = df.groupby("region_id")["water_shortage"].transform(lambda s: s.rolling(84, min_periods=1).sum())

    # =========================================================================
    # [探勘 2] 交互作用 (Feature Crosses)
    # 尋找 1+1 > 2 的破壞力
    # =========================================================================
    print("Calculating Interaction Features...")
    # 交互 1: 長期缺水 * 當週高溫異常 (旱上加熱)
    df["cross_shortage_x_anomaly"] = df["shortage_roll_4w"] * np.clip(df["tmp_anomaly"], 0, None)
    
    # 交互 2: 降雨赤字率 (缺的佔應該下的比例)
    # 如果 pet 是 10，下了 2，赤字率是 8/10 = 0.8。如果 pet 是 2，下了 0，赤字率是 2/2 = 1.0
    df["drought_ratio"] = df["water_shortage"] / (df["pet"] + 1e-5)

    # 交互 3: 連續無雨天數的 Proxy (用 rolling max 近似)
    df["prec_roll_max_2w"] = df.groupby("region_id")["prec"].transform(lambda s: s.rolling(14, min_periods=1).max())
    df["is_dry_spell"] = (df["prec_roll_max_2w"] < 0.5).astype(int)

    # =========================================================================
    # 評估區塊
    # =========================================================================
    df_safe = df[df["score"] == 0.0]
    df_mild = df[(df["score"] > 0.0) & (df["score"] <= 1.5)]
    df_ext  = df[df["score"] > 3.5]
    
    feats_to_eval = [
        "water_shortage", "shortage_roll_4w", "shortage_roll_8w", "shortage_roll_12w",
        "cross_shortage_x_anomaly", "drought_ratio", "is_dry_spell"
    ]
    
    print_header("Module 1: The Multiplier Check (Interactions & Long-term)")
    print(f"{'Feature':<25} | {'Safe':<10} | {'Mild (1分)':<12} | {'Extreme (5分)':<15} | {'Ext/Mild 倍數':<15}")
    print("-" * 85)
    
    for feat in feats_to_eval:
        safe_m = df_safe[feat].mean()
        mild_m = df_mild[feat].mean()
        ext_m  = df_ext[feat].mean()
        
        multiplier = ext_m / (mild_m + 1e-5) 
        indicator = "🔥" if multiplier > 1.5 else ""
        print(f"{feat:<25} | {safe_m:<10.3f} | {mild_m:<12.3f} | {ext_m:<15.3f} | {multiplier:<10.2f}x {indicator}")

    print_header("Module 2: Spearman Rank Correlation")
    df_sample = df.dropna().sample(n=100000, random_state=42)
    y_sample = df_sample["score"].values
    
    print(f"{'Feature':<25} | {'Spearman':<15}")
    print("-" * 45)
    for feat in feats_to_eval:
        corr, _ = spearmanr(df_sample[feat], y_sample)
        print(f"{feat:<25} | {corr:<15.4f}")

    print("\n💡 挖掘思路：")
    print("- 看看 8w 或 12w 的累積是不是比 4w 更能分辨極端乾旱。")
    print("- 看看 `cross_shortage_x_anomaly` 這個組合技有沒有產生巨大的 Multiplier。")

if __name__ == "__main__":
    main()