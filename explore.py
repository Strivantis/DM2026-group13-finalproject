"""
v51_data_exploration.py (Fixed OutOfBoundsDatetime Version)
任務 1: 調查 Test Set 的真實長度與時間連續性 (是否存在 9 個月的隱藏資料)
任務 2: 評估原始日資料補值 (impute_met_features) 的災難程度
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else os.getcwd()
DATA_DIR = os.path.join(ROOT, "FinalProject/data")

def date_to_abs_days(date_str):
    """
    將 YYYY-MM-DD 轉換為從公元元年開始的絕對天數。
    避開 Pandas 的 datetime64[ns] 年份上限 (2262年) 限制。
    這裡採用簡化的閏年計算，足夠用來計算時間差。
    """
    y, m, d = map(int, date_str.split('-'))
    month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # 計算閏年數量 (不考慮 100/400 規則，因為是在 3000 年代，且為了快速估算足夠精準)
    leap_years = y // 4
    
    days = y * 365 + leap_years + sum(month_days[:m]) + d
    if m <= 2 and y % 4 == 0:
        days -= 1 # 如果當年是閏年但還沒過二月
    return days

def main():
    print("=" * 80)
    print(" 🕵️‍♂️ V51 DATA EXPLORATION: UNCOVERING THE TRUTH (No-Datetime Limit Version)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 讀取原始資料
    # -------------------------------------------------------------------------
    try:
        train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    except FileNotFoundError:
        print(f"Error: Could not find train.csv or test.csv in {DATA_DIR}")
        return

    # =========================================================================
    # 任務 1: 調查 Test Set 的真實長度與連續性
    # =========================================================================
    print("\n[MISSION 1] Test Set Continuity & Length Analysis")
    
    # 由於格式是 YYYY-MM-DD，可以直接用字串 min/max 找出日期極值
    train_end = train['date'].max()
    test_start = test['date'].min()
    test_end = test['date'].max()
    
    print(f"  Train Set End Date : {train_end}")
    print(f"  Test Set Start Date: {test_start}")
    print(f"  Test Set End Date  : {test_end}")
    
    gap_days = date_to_abs_days(test_start) - date_to_abs_days(train_end)
    print(f"  Gap between Train and Test: {gap_days} days")

    # 隨機抽樣 3 個 region 來檢查 test set 的實際行數
    sample_regions = test['region_id'].unique()[:3]
    for region in sample_regions:
        region_test = test[test['region_id'] == region]
        days_in_test = len(region_test)
        
        region_min_date = region_test['date'].min()
        region_max_date = region_test['date'].max()
        date_diff = date_to_abs_days(region_max_date) - date_to_abs_days(region_min_date) + 1
        
        print(f"\n  Region: {region}")
        print(f"    Rows in test.csv      : {days_in_test} days")
        print(f"    Actual Date Range Span: {date_diff} days")
        
        if days_in_test > 91: # 13 weeks = 91 days
            print("    🚨 ALERT: Test set contains MORE than 13 weeks of data!")
            if test_start < "3020-09-24":
                print("    🚨 ALERT: Test set contains data from the 'Gap' period!")
        else:
            print("    ✅ CONFIRMED: Test set contains EXACTLY 13 weeks (91 days). No hidden gap data.")

    # =========================================================================
    # 任務 2: 評估原始日資料補值 (impute_met_features) 的災難程度
    # =========================================================================
    print("\n[MISSION 2] Missing Data & Imputation Disaster Assessment")
    
    total_rows = len(train)
    missing_stats = train.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
    
    if len(missing_stats) == 0:
        print("  ✅ Amazing! train.csv has NO missing values (NaNs).")
        print("     The ffill/bfill bug didn't cause any damage because there was nothing to fill!")
    else:
        print("  ⚠️ Found missing values in train.csv:")
        print("-" * 50)
        print(f"  {'Column':<15} | {'Missing Count':<15} | {'Missing %'}")
        print("-" * 50)
        
        for col, count in missing_stats.items():
            perc = (count / total_rows) * 100
            print(f"  {col:<15} | {count:<15,} | {perc:.2f}%")
        print("-" * 50)
        
        prec_cols = ['prec', 'surf_pre']
        for col in prec_cols:
            if col in missing_stats:
                missing_count = missing_stats[col]
                print(f"\n  🚨 Disaster Assessment for '{col}':")
                print(f"     You had {missing_count:,} missing days.")
                print(f"     Because of `ffill()`, these days were artificially filled with previous rain.")
                print("     ✅ Fix: We must change `ffill` to `.fillna(0.0)` for these columns.")

    print("\n" + "=" * 80)
    print(" Next Step: Awaiting Results!")
    print("=" * 80)

if __name__ == "__main__":
    main()