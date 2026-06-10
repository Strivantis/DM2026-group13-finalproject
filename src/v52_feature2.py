"""
v52_feature2.py
針對「水份赤字 (Deficit)」與「溫度與濕度的交叉乘積」進行二次探勘，
尋找更適合樹模型的非線性特徵。
"""

import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "v51_processed")

def main():
    train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
    if not os.path.exists(train_path):
        print(f"找不到檔案：{train_path}")
        return

    print("載入 Train Data 中...")
    # 只載入需要的欄位以節省記憶體
    cols = ["score", "prec", "pet", "deficit", "deficit_roll_cum_4w", "tmp", "humidity"]
    df = pd.read_csv(train_path, usecols=cols)

    # 1. 探勘 Deficit 的負數深淵
    print("\n" + "="*80)
    print(" 探勘 1：水份赤字 (Deficit) 的極端負數分佈")
    print("="*80)
    df_safe   = df[df["score"] == 0.0]
    df_extreme= df[df["score"] > 3.0]

    # 計算 deficit < 0 的比例
    safe_neg_ratio = (df_safe["deficit"] < 0).mean()
    ext_neg_ratio  = (df_extreme["deficit"] < 0).mean()
    
    print(f"安全日 (Score 0)   | Deficit < 0 的比例: {safe_neg_ratio:.2%} | 平均值: {df_safe['deficit'].mean():.4f}")
    print(f"極端旱 (Score > 3) | Deficit < 0 的比例: {ext_neg_ratio:.2%} | 平均值: {df_extreme['deficit'].mean():.4f}")
    
    # 觀察最極端的 1% 負數
    print(f"\n安全日   Deficit 底端 1%: {df_safe['deficit'].quantile(0.01):.4f}")
    print(f"極端旱   Deficit 底端 1%: {df_extreme['deficit'].quantile(0.01):.4f}")
    print("💡 結論：如果極端旱的底端負數遠低於安全日，我們應該在 V52 新增一個絕對值反轉特徵：`water_shortage = max(0, pet - prec)`，讓它變成一個純粹的正向旱災指標！")

    # 2. 探勘溫濕度乘積 (蒸發動能)
    print("\n" + "="*80)
    print(" 探勘 2：高溫與低濕的複合打擊 (VPD Proxy)")
    print("="*80)
    # 模擬簡單的 VPD (Vapor Pressure Deficit) 概念：溫度高且濕度低時數值飆升
    df["vpd_proxy"] = df["tmp"] * (100 - df["humidity"]) / 100.0
    
    safe_vpd = df.loc[df["score"] == 0.0, "vpd_proxy"].mean()
    ext_vpd  = df.loc[df["score"] > 3.0, "vpd_proxy"].mean()
    
    print(f"安全日   平均 VPD Proxy: {safe_vpd:.4f}")
    print(f"極端旱   平均 VPD Proxy: {ext_vpd:.4f}")
    print(f"鑑別倍數 (Multiplier): {ext_vpd / safe_vpd:.2f}x")
    print("💡 結論：如果倍數超過 1.5x，這將是 V52 必加的黃金特徵。")
    print("="*80)

if __name__ == "__main__":
    main()