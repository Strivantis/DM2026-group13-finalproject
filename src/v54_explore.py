"""
v54_raw_explorer.py
從最原始的 raw data 出發，尋找真實的物理極端值。
"""

import os
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

def print_header(title):
    print("\n" + "="*80)
    print(f" 🚀 {title}")
    print("="*80)

def main():
    t0 = time.time()
    
    # 讀取最原始的資料
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path  = os.path.join(DATA_DIR, "test.csv")
    
    print("Loading RAW data...")
    # 只取核心氣象欄位，以節省時間和記憶體
    core_cols = ["date", "region_id", "prec", "tmp", "humidity", "wind", "score"]
    df_train = pd.read_csv(train_path, usecols=core_cols).dropna(subset=["score"])
    df_test  = pd.read_csv(test_path, usecols=[c for c in core_cols if c != "score"])

    # 簡單模擬 PET 和 VPD，以理解原始物理狀態
    df_train["pet_raw"] = 0.55 * df_train["tmp"].clip(lower=0)
    df_train["vpd_raw"] = df_train["tmp"] * (100 - df_train["humidity"]) / 100.0
    
    df_test["pet_raw"] = 0.55 * df_test["tmp"].clip(lower=0)
    df_test["vpd_raw"] = df_test["tmp"] * (100 - df_test["humidity"]) / 100.0

    # =========================================================================
    # Module 1: The Raw Extremes (物理極端值的真實面貌)
    # =========================================================================
    print_header("Module 1: The Raw Extremes (Train Set)")
    df_safe    = df_train[df_train["score"] == 0.0]
    df_extreme = df_train[df_train["score"] > 3.5]
    
    feats_to_check = ["prec", "tmp", "humidity", "wind", "pet_raw", "vpd_raw"]
    print(f"{'Feature':<15} | {'Safe Mean':<12} | {'Extreme Mean':<14} | {'Extreme Top 10%':<15}")
    print("-" * 65)
    for feat in feats_to_check:
        safe_m = df_safe[feat].mean()
        ext_m  = df_extreme[feat].mean()
        # 對於乾燥指標，我們看最高的前 10%；對於降雨/濕度，看最低的 10% (Bottom 10%)
        if feat in ["prec", "humidity"]:
            ext_edge = df_extreme[feat].quantile(0.10)
        else:
            ext_edge = df_extreme[feat].quantile(0.90)
            
        indicator = "🔥" if abs(ext_m - safe_m)/(abs(safe_m)+1e-5) > 0.4 else ""
        print(f"{feat:<15} | {safe_m:<12.3f} | {ext_m:<14.3f} | {ext_edge:<15.3f} {indicator}")

    # =========================================================================
    # Module 2: The Test Set Reality (Test 到底有多熱？)
    # =========================================================================
    print_header("Module 2: The Test Set Reality (Are we facing a mega-drought?)")
    print("我們來對比 Train 的歷史紀錄和 Test 的真實氣候：")
    
    print(f"{'Metric':<25} | {'Train History':<15} | {'Test Reality':<15}")
    print("-" * 60)
    
    print(f"{'Average Temp':<25} | {df_train['tmp'].mean():<15.2f} | {df_test['tmp'].mean():<15.2f}")
    print(f"{'Max Temp (99th pctl)':<25} | {df_train['tmp'].quantile(0.99):<15.2f} | {df_test['tmp'].quantile(0.99):<15.2f}")
    
    print(f"{'Average VPD':<25} | {df_train['vpd_raw'].mean():<15.2f} | {df_test['vpd_raw'].mean():<15.2f}")
    print(f"{'Max VPD (99th pctl)':<25} | {df_train['vpd_raw'].quantile(0.99):<15.2f} | {df_test['vpd_raw'].quantile(0.99):<15.2f}")
    
    # 計算「高溫少雨日」的比例 (以 Train 的極端邊界為基準)
    # 假設高溫閾值 = Train 極端乾旱期的平均溫度，少雨閾值 = 1.0mm
    hot_thresh = df_extreme['tmp'].mean()
    
    train_hot_dry = ((df_train["tmp"] > hot_thresh) & (df_train["prec"] < 1.0)).mean()
    test_hot_dry  = ((df_test["tmp"] > hot_thresh) & (df_test["prec"] < 1.0)).mean()
    print(f"{'Hot & Dry Days %':<25} | {train_hot_dry:<15.2%} | {test_hot_dry:<15.2%}")
    if test_hot_dry > train_hot_dry * 2:
        print("\n🚨 警告：Test Set 的『高溫少雨日』比例遠超 Train！我們確實面臨大旱年，模型必須敢猜高分。")

    print("\n" + "="*80)
    print(f"✅ Raw Data Mining Complete. Elapsed time: {time.time() - t0:.1f}s")
    print("="*80)

if __name__ == "__main__":
    main()