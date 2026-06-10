"""
v54_feature_validator.py
在記憶體中即時計算 V54 新特徵，並驗證它們對抗「極端乾旱」的爆發力。
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

    print("Loading RAW train data for feature simulation...")
    core_cols = ["region_id", "date", "prec", "tmp", "humidity", "wind", "score"]
    df = pd.read_csv(train_path, usecols=core_cols).dropna(subset=["score"])
    
    # 確保按時間排序，以利於計算 Rolling/Lag
    df = df.sort_values(by=["region_id", "date"])

    print("Simulating V52/V54 Features in memory...")
    
    # 1. 基礎水份短缺 (V52)
    df["pet"] = 0.55 * df["tmp"].clip(lower=0)
    df["water_shortage"] = (df["pet"] - df["prec"]).clip(lower=0)
    
    # 2. 歷史均溫與異常度 (V51/V52)
    stats = df.groupby("region_id")["tmp"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].replace(0, 1.0)
    df = df.merge(stats, on="region_id", how="left", suffixes=("", "_reg"))
    df["tmp_anomaly"] = (df["tmp"] - df["mean"]) / df["std"]

    # 3. 模擬 Lag 特徵 (簡化版，用 shift)
    df["water_shortage_lag1"] = df.groupby("region_id")["water_shortage"].shift(7) # 假設一週是 7 天
    df["water_shortage_lag2"] = df.groupby("region_id")["water_shortage"].shift(14)

    # =========================================================================
    # V54 新特徵實裝測試
    # =========================================================================
    
    # [新] 特徵 1: 異常度平方 (強迫樹模型重視極端熱浪)
    df["tmp_anomaly_sq"] = np.clip(df["tmp_anomaly"], 0.0, None) ** 2
    
    # [新] 特徵 2: 蒸發動能指數爆發 (熱力學模擬)
    temp_factor = np.exp(np.clip((df["tmp"] - 25.0) / 10.0, 0.0, None))
    df["vpd_exp"] = temp_factor * ((100.0 - df["humidity"]) / 100.0)
    
    # [新] 特徵 3: 連續乾旱動能 (近期缺水加權平方)
    momentum = (df["water_shortage"] * 4.0 + 
                df["water_shortage_lag1"].fillna(0) * 2.0 + 
                df["water_shortage_lag2"].fillna(0) * 1.0)
    df["shortage_momentum"] = momentum ** 2


    # =========================================================================
    # Module 1: 破防力測試 (Multiplier Analysis)
    # =========================================================================
    print_header("Module 1: Feature Multiplier (Safe vs Extreme)")
    print("我們來看看 V54 新特徵在遇到 5 分大旱時，能不能產生『爆炸性』的數值變化。")
    
    df_safe = df[df["score"] == 0.0]
    df_mild = df[(df["score"] > 0.0) & (df["score"] <= 1.5)] # 輕微乾旱
    df_ext  = df[df["score"] > 3.5] # 極端乾旱
    
    feats_to_eval = [
        "water_shortage", "tmp_anomaly", # 舊特徵
        "tmp_anomaly_sq", "vpd_exp", "shortage_momentum" # V54 炸彈
    ]
    
    print(f"{'Feature':<20} | {'Safe':<10} | {'Mild (1分)':<12} | {'Extreme (5分)':<15} | {'Ext/Mild 倍數':<15}")
    print("-" * 80)
    
    for feat in feats_to_eval:
        safe_m = df_safe[feat].mean()
        mild_m = df_mild[feat].mean()
        ext_m  = df_ext[feat].mean()
        
        # 我們關心的是：特徵能不能把 1 分和 5 分拉開？
        multiplier = ext_m / (mild_m + 1e-5) 
        
        indicator = "🔥" if multiplier > 1.5 else ""
        print(f"{feat:<20} | {safe_m:<10.3f} | {mild_m:<12.3f} | {ext_m:<15.3f} | {multiplier:<10.2f}x {indicator}")

    # =========================================================================
    # Module 2: 排序相關性 (Spearman Correlation)
    # =========================================================================
    print_header("Module 2: Non-linear Correlation (Spearman Rank)")
    
    # 隨機抽樣加速
    df_sample = df.dropna().sample(n=100000, random_state=42)
    y_sample = df_sample["score"].values
    
    print(f"{'Feature':<20} | {'Spearman Correlation':<20}")
    print("-" * 45)
    for feat in feats_to_eval:
        corr, _ = spearmanr(df_sample[feat], y_sample)
        print(f"{feat:<20} | {corr:<20.4f}")

    print("\n💡 決策指南：")
    print("1. 如果 V54 炸彈的『Ext/Mild 倍數』遠大於舊特徵，代表它們能成功幫助樹模型把 5 分從 1 分中切出來！")
    print("2. 只要確認倍數夠高，我們就可以安心把這三顆炸彈寫進 V54_preprocess.py。")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()