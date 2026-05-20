import pandas as pd
import numpy as np

def diagnostic_eda():
    print("載入處理後的數據...")
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
    # 由於 min_periods=1，測試集前幾週的 rolling sum 會因數據不足而產生嚴重的人為偏差
    print("\n=== [2] 特徵分佈偏移檢查 (Rolling Features) ===")
    
    if "prec_roll_sum_13w" in test.columns:
        # 取出 Train 中任意一段連續 14 週的 13w rolling sum 平均值作為基準 (排除前 13 週)
        train_stable = train.groupby("region_id").apply(lambda x: x.iloc[13:27]["prec_roll_sum_13w"].mean()).mean()
        
        # 比較 Test 第一週與第十三週的 rolling sum
        test_w1 = test.groupby("region_id").first()["prec_roll_sum_13w"].mean()
        test_w13 = test.groupby("region_id").nth(12)["prec_roll_sum_13w"].mean() if test_counts.max() >= 13 else np.nan
        
        print(f"Train 穩定狀態 (13w sum): {train_stable:.4f}")
        print(f"Test 第 1 週狀態 (實際僅 1 週 sum): {test_w1:.4f}")
        print(f"Test 第 13 週狀態 (實際滿 13 週 sum): {test_w13:.4f}")
        print("-> 若 Test 第 1 週顯著低於 Train 穩定狀態，代表特徵在推論初期發生嚴重的尺度塌陷。")

    # 3. 檢查 NaN 狀態
    print("\n=== [3] NaN 殘留檢查 ===")
    print(f"Train NaN 數量:\n{train.isna().sum()[train.isna().sum() > 0]}")
    print(f"Test NaN 數量:\n{test.isna().sum()[test.isna().sum() > 0]}")

if __name__ == "__main__":
    diagnostic_eda()