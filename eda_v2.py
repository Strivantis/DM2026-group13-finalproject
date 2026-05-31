import pandas as pd
import numpy as np

def diagnostic_eda():
    print("載入處理後的數據...")
    # 假設你的檔案路徑正確，若有不同請自行調整
    train = pd.read_csv("data/processed/train_processed.csv")
    test = pd.read_csv("data/processed/test_processed.csv")

    # 1. 區域完整性與列數統計
    train_counts = train.groupby("region_id").size()
    test_counts = test.groupby("region_id").size()

    print("\n=== [1] 區域與序列長度檢查 ===")
    print(f"Train 區域數量: {len(train_counts)}")
    print(f"Test 區域數量: {len(test_counts)}")
    
    missing_in_test = set(train_counts.index) - set(test_counts.index)
    if missing_in_test:
         print(f"警告: Test 遺失了 {len(missing_in_test)} 個 Train 中的區域")
    else:
         print("Train/Test 區域完全對齊。")

    print("\n[Train] 每區週數 (Rows) 統計:")
    print(train_counts.describe())
    
    print("\n[Test] 每區週數 (Rows) 統計:")
    print(test_counts.describe())

    # 2. Rolling Feature 的初始階段失真檢查 (Domain Shift)
    # 修改為配合資料中的 prec_roll_sum_4w 特徵
    print("\n=== [2] 特徵分佈偏移檢查 (Rolling Features) ===")
    
    if "prec_roll_sum_4w" in test.columns:
        # 取出 Train 中一段穩定的期間 (排除前 4 週，取第 4 到 17 週的平均作為基準)
        train_stable = train.groupby("region_id").apply(lambda x: x.iloc[4:18]["prec_roll_sum_4w"].mean()).mean()
        
        # 比較 Test 第一週與第四週 (滿 4 週) 的 rolling sum
        test_w1 = test.groupby("region_id").first()["prec_roll_sum_4w"].mean()
        # 抓取第 4 週的資料 (index 3) 來確認特徵穩定的狀態
        test_w4 = test.groupby("region_id").nth(3)["prec_roll_sum_4w"].mean() if test_counts.max() >= 4 else np.nan
        
        print(f"Train 穩定狀態 (4w sum 基準): {train_stable:.4f}")
        print(f"Test 第 1 週狀態 (實際僅 1 週 sum): {test_w1:.4f}")
        print(f"Test 第 4 週狀態 (實際滿 4 週 sum): {test_w4:.4f}")
        print("-> 若 Test 第 1 週顯著低於 Train 穩定狀態，代表測試集初期的 Rolling 特徵發生尺度塌陷。")
    else:
        print("未在資料中找到 'prec_roll_sum_4w' 特徵，跳過此檢查。")

    # 3. 檢查 NaN 狀態
    print("\n=== [3] NaN 殘留檢查 ===")
    print(f"Train NaN 數量:\n{train.isna().sum()[train.isna().sum() > 0]}")
    print(f"Test NaN 數量:\n{test.isna().sum()[test.isna().sum() > 0]}")

if __name__ == "__main__":
    diagnostic_eda()