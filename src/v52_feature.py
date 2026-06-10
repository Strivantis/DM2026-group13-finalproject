"""
v52_feature_explorer.py
核心目標：用數據說話。探勘 Train/Test 的分佈差異，並尋找能觸發極端乾旱的物理特徵邊界。
"""

import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "v51_processed")

# 我們要重點監控的核心物理特徵
TARGET_FEATURES = [
    "prec", "tmp", "humidity", 
    "tmp_anomaly", "aridity_index", "heat_shock", "deficit_roll_cum_4w"
]

def print_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def main():
    train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test_processed.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"找不到處理過的資料。請確認路徑：{PROCESSED_DIR}")
        return

    print("載入資料中 (這可能需要幾十秒)...")
    train_df = pd.read_csv(train_path, usecols=TARGET_FEATURES + ["score"])
    test_df  = pd.read_csv(test_path, usecols=TARGET_FEATURES)

    # -------------------------------------------------------------------
    # 報告 1：Train vs Test 的整體氣候分佈對比 (尋找 Covariate Shift)
    # -------------------------------------------------------------------
    print_section("報告 1：Train vs Test 特徵分佈對比 (Covariate Shift Check)")
    print(f"{'Feature':<20} | {'Train Mean':<12} | {'Test Mean':<12} | {'Train Max':<12} | {'Test Max':<12}")
    print("-" * 75)
    
    for feat in TARGET_FEATURES:
        tr_mean = train_df[feat].mean()
        te_mean = test_df[feat].mean()
        tr_max  = train_df[feat].max()
        te_max  = test_df[feat].max()
        
        # 標記危險的偏移 (如果 Test 的平均值或最大值遠高於 Train)
        warning_flag = "⚠️" if te_mean > tr_mean * 1.2 or te_max > tr_max else ""
        print(f"{feat:<20} | {tr_mean:<12.4f} | {te_mean:<12.4f} | {tr_max:<12.4f} | {te_max:<12.4f} {warning_flag}")

    # -------------------------------------------------------------------
    # 報告 2：極端乾旱 (Score 4, 5) 的物理觸發邊界
    # -------------------------------------------------------------------
    print_section("報告 2：極端乾旱 (Score 4, 5) 的物理特徵邊界")
    
    # 擷取不同乾旱等級的子集
    df_safe   = train_df[train_df["score"] == 0.0]
    df_mild   = train_df[(train_df["score"] > 0.0) & (train_df["score"] <= 2.0)]
    df_severe = train_df[(train_df["score"] > 2.0) & (train_df["score"] <= 3.0)]
    df_extreme= train_df[train_df["score"] > 3.0]

    print(f"資料筆數分佈 -> 安全: {len(df_safe):,}, 輕/中旱: {len(df_mild):,}, 嚴重旱: {len(df_severe):,}, 極端旱: {len(df_extreme):,}\n")
    
    print(f"{'Feature':<20} | {'Safe (Score 0) Mean':<20} | {'Extreme (Score >3) Mean':<20} | {'Multiplier (倍數)':<10}")
    print("-" * 80)
    
    for feat in TARGET_FEATURES:
        safe_mean = df_safe[feat].mean()
        ext_mean  = df_extreme[feat].mean()
        
        # 避免除以零
        if abs(safe_mean) > 1e-5:
            multiplier = ext_mean / safe_mean
        else:
            multiplier = np.nan
            
        print(f"{feat:<20} | {safe_mean:<20.4f} | {ext_mean:<20.4f} | {multiplier:<10.2f}x")

    # -------------------------------------------------------------------
    # 報告 3：特徵分位數深度分析 (揭露樹模型的盲點)
    # -------------------------------------------------------------------
    print_section("報告 3：特徵的 99% 分位數分析 (樹模型的切割極限)")
    print("說明：樹模型很難外推。如果 Test 的 99% 分位數遠大於 Train 的 99%，樹模型會瞎眼。")
    print(f"{'Feature':<20} | {'Train 99%':<12} | {'Test 99%':<12} | {'Diff (%)':<10}")
    print("-" * 65)

    for feat in TARGET_FEATURES:
        tr_99 = train_df[feat].quantile(0.99)
        te_99 = test_df[feat].quantile(0.99)
        
        if tr_99 > 1e-5:
            diff_pct = ((te_99 - tr_99) / tr_99) * 100
        else:
            diff_pct = 0.0
            
        alert = "🚨 DANGER" if diff_pct > 15.0 else ""
        print(f"{feat:<20} | {tr_99:<12.4f} | {te_99:<12.4f} | {diff_pct:>8.1f}%  {alert}")

    print("\n" + "="*80)
    print("探勘結束。請將這些數據回報，我們將基於此設計 v52_preprocess.py 的新特徵！")
    print("="*80)

if __name__ == "__main__":
    main()