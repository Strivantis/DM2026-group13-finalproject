import pandas as pd
import numpy as np
import os

# 設定路徑
PROC_DIR = "data/processed"
TRAIN_PATH = os.path.join(PROC_DIR, "train_processed.csv")
TEST_PATH = os.path.join(PROC_DIR, "test_processed.csv")

def parse_week_key(wk_str):
    """
    將 week_key (如 '023102-W20') 轉換為絕對週數
    年份 * 53 + 週數
    """
    if pd.isna(wk_str):
        return np.nan
    parts = str(wk_str).split("-W")
    year = int(parts[0])
    week = int(parts[1])
    return year * 53 + week

def main():
    print("Loading processed data (only reading region_id and week_key)...")
    train = pd.read_csv(TRAIN_PATH, usecols=["region_id", "week_key"])
    test = pd.read_csv(TEST_PATH, usecols=["region_id", "week_key"])

    print("Calculating absolute weeks...")
    train['abs_week'] = train['week_key'].apply(parse_week_key)
    test['abs_week'] = test['week_key'].apply(parse_week_key)

    print("Aggregating temporal boundaries per region...")
    # 抓出 Train 的最後一週
    train_max_abs = train.groupby("region_id")['abs_week'].max().rename("train_max_abs")
    train_max_key = train.groupby("region_id")['week_key'].max().rename("train_max_key")
    
    # 抓出 Test 的第一週
    test_min_abs = test.groupby("region_id")['abs_week'].min().rename("test_min_abs")
    test_min_key = test.groupby("region_id")['week_key'].min().rename("test_min_key")
    
    # 合併結果
    df = pd.concat([train_max_key, train_max_abs, test_min_key, test_min_abs], axis=1)
    
    # 計算時間斷層 Gap = Test 起點 - Train 終點
    # Gap = 1 代表 Test 緊接著 Train 的下一週
    df['gap_weeks'] = df['test_min_abs'] - df['train_max_abs']

    print("\n" + "="*50)
    print("Gap Analysis Summary (weeks between Train End and Test Start)")
    print("="*50)
    print(df['gap_weeks'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    
    print("\n" + "="*50)
    print("Top 10 Most Common Gaps")
    print("="*50)
    print(df['gap_weeks'].value_counts().head(10))

    # 檢查是否有 Test 早於 Train 的詭異狀況
    negative_gaps = df[df['gap_weeks'] <= 0]
    if len(negative_gaps) > 0:
        print(f"\n[WARNING] Found {len(negative_gaps)} regions where Test starts BEFORE Train ends!")
        print(negative_gaps.head())
    
    # 輸出成報表供你檢查
    report_path = "gap_analysis_report.csv"
    df.to_csv(report_path)
    print(f"\nDetailed report saved to {report_path}")

if __name__ == "__main__":
    main()