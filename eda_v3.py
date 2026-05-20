import pandas as pd
import numpy as np

def analyze_raw_data():
    print("="*50)
    print("Raw Data Zero-Trust Analysis")
    print("="*50)
    
    print("\n[Loading Raw Data...]")
    # 僅讀取關鍵欄位以優化記憶體使用 (1.2GB -> ~200MB)
    train = pd.read_csv("data/train.csv", usecols=["region_id", "date", "score"])
    test = pd.read_csv("data/test.csv", usecols=["region_id", "date"])

    # 1. 訓練集總體狀態
    print("\n--- [1] Train Set Raw Statistics ---")
    total_train_rows = len(train)
    valid_scores = train["score"].notna().sum()
    missing_scores = train["score"].isna().sum()
    
    print(f"Total Regions: {train['region_id'].nunique()}")
    print(f"Total Rows: {total_train_rows}")
    print(f"Rows with valid score: {valid_scores} ({(valid_scores/total_train_rows)*100:.2f}%)")
    print(f"Rows missing score: {missing_scores} ({(missing_scores/total_train_rows)*100:.2f}%)")

    # 2. 訓練集區域級別統計
    print("\n--- [2] Region Level Stats (Train) ---")
    train_region_stats = train.groupby("region_id").agg(
        total_days=("date", "count"),
        valid_scores=("score", "count")
    )
    print(train_region_stats.describe())
    
    # 檢查是否有區域的總天數或分數數量與其他區域不同
    unique_days_counts = train_region_stats["total_days"].unique()
    unique_score_counts = train_region_stats["valid_scores"].unique()
    print(f"\nUnique 'total_days' lengths across regions: {unique_days_counts}")
    print(f"Unique 'valid_scores' lengths across regions: {unique_score_counts}")

    # 3. 測試集區域級別統計
    print("\n--- [3] Region Level Stats (Test) ---")
    print(f"Total Regions: {test['region_id'].nunique()}")
    test_region_stats = test.groupby("region_id").agg(
        total_days=("date", "count")
    )
    print(test_region_stats.describe())
    print(f"Unique 'total_days' lengths across regions: {test_region_stats['total_days'].unique()}")

    # 4. 目標值頻率分析 (以 R1 為例)
    print("\n--- [4] Score Frequency Analysis (Region R1) ---")
    r1_data = train[train["region_id"] == "R1"].reset_index(drop=True)
    r1_scored = r1_data[r1_data["score"].notna()]
    
    print(f"R1 Total Rows (Days): {len(r1_data)}")
    print(f"R1 Scored Rows: {len(r1_scored)}")
    print("\nFirst 10 scored dates for R1:")
    print(r1_scored.head(10).to_string())
    
    # 計算相鄰分數之間的天數差距 (驗證是否嚴格為 7 天一測)
    r1_scored_indices = r1_scored.index.to_numpy()
    index_diffs = np.diff(r1_scored_indices)
    print(f"\nRow index differences between scores for R1: {np.unique(index_diffs, return_counts=True)}")

if __name__ == "__main__":
    analyze_raw_data()