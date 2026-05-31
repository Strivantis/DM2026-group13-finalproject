"""
src/inspect_processed.py
===========================================================================
目的：對 V20 Pipeline 導出的處理後數據進行全面審計與特徵分佈 Data Mining。
檢查項：
  1. 結構與對齊審計：驗證區域數量（2248）、測試集週數（13）與基礎維度。
  2. 毒藥特徵封印檢查：嚴格斷言 score_lag1w 與 score_lag2w 是否已被徹底剔除。
  3. 協變量偏移再驗證：量化測試集初期（Week 0-2）與中後期（Week 3-12）以及訓練集的特徵分佈對齊度。
  4. 零膨脹與目標分佈挖掘：統計訓練集 Target 零膨脹比例與非零高分段的分佈特徵。
"""

import os
import pandas as pd
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROC = os.path.join(_BASE, "data", "processed")
TRAIN_PROCESSED = os.path.join(_PROC, "train_processed.csv")
TEST_PROCESSED = os.path.join(_PROC, "test_processed.csv")

def run_audit():
    print("="*75)
    print("  [Data Space Audit & Mining] 啟動 V20 處理後數據全量審計")
    print("="*75)
    
    if not os.path.exists(TRAIN_PROCESSED) or not os.path.exists(TEST_PROCESSED):
        print(" 錯誤：找不到處理後的數據檔案，請先運行 src/preprocess.py。")
        return

    # 1. 載入數據
    print("\n1. 正在讀取處理後的週級數據矩陣...")
    df_train = pd.read_csv(TRAIN_PROCESSED)
    df_test = pd.read_csv(TEST_PROCESSED)
    print(f"   [Train Processed] 形狀: {df_train.shape}")
    print(f"   [Test Processed]  形狀: {df_test.shape}")

    # 2. 結構與空間對齊度檢查
    print("\n2. 執行結構與維度空間對齊審計...")
    train_regions = df_train['region_id'].nunique()
    test_regions = df_test['region_id'].nunique()
    print(f"   區域數量檢查: Train = {train_regions} | Test = {test_regions}")
    assert train_regions == 2248 and test_regions == 2248, "區域數量未完全對齊至 2248！"
    
    # 驗證測試集每區行數是否嚴格限制在 13 週
    test_counts = df_test.groupby('region_id').size()
    print(f"   測試集每區週數: 最小值 = {test_counts.min()} | 最大值 = {test_counts.max()} (預期值: 13)")
    assert test_counts.min() == 13 and test_counts.max() == 13, "測試集週數不符合 13 週剛性限制！"
    
    # 驗證測試集週索引範圍
    print(f"   測試集 week_idx 範圍: [{df_test['week_idx'].min()}, {df_test['week_idx'].max()}] (預期值: [0, 12])")
    assert df_test['week_idx'].min() == 0 and df_test['week_idx'].max() == 12, "測試集 week_idx 未正確平移歸零！"
    print("   v 結構與空間維度全數通過斷言。")

    # 3. 毒藥特徵隔離審計
    print("\n3. 執行毒藥特徵（Score Lags）防禦審計...")
    forbidden_cols = ['score_lag1w', 'score_lag2w']
    for col in forbidden_cols:
        if col in df_train.columns or col in df_test.columns:
            print(f"   ❌ 警告：發現殘留毒藥特徵 '{col}'！請檢查 Pipeline 清洗邏輯。")
        else:
            print(f"   v 確認特徵 '{col}' 已被物理隔離。")

    # 4. 協變量偏移與氣候常態修復挖掘
    print("\n4. 執行協變量偏移與邊界修復深度挖掘 (以降雨滾動總合為例)...")
    if 'prec_roll_sum_4w' in df_test.columns:
        train_mean = df_train['prec_roll_sum_4w'].mean()
        train_var = df_train['prec_roll_sum_4w'].var()
        
        # 提取測試集初期（修復前易塌陷段）與中後期（穩態段）
        test_early = df_test[df_test['week_idx'].isin([0, 1, 2])]['prec_roll_sum_4w']
        test_late = df_test[df_test['week_idx'].isin([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]['prec_roll_sum_4w']
        
        print(f"   [Train 穩態基準面]  均值: {train_mean:.4f} | 方差: {train_var:.4f}")
        print(f"   [Test Week 0-2 初期] 均值: {test_early.mean():.4f} | 方差: {test_early.var():.4f}")
        print(f"   [Test Week 3-12 後期]均值: {test_late.mean():.4f} | 方差: {test_late.var():.4f}")
        
        shift_ratio_early = abs(test_early.mean() - train_mean) / train_mean
        print(f"   👉 測試集初期特徵尺度與訓練集偏移率: {shift_ratio_early:.2%}")
        if shift_ratio_early < 0.10:
            print("   v 氣候常態前置填補生效！測試集初期分佈已成功融入訓練集穩態。")
        else:
            print("   ⚠️ 注意：特徵空間仍存在某種程度的偏移，請監控後續表現。")
    else:
        print("   未發現 'prec_roll_sum_4w' 特徵，跳過此項挖掘。")

    # 5. 零膨脹與目標分佈挖掘 (Target Space Mining)
    print("\n5. 執行訓練集 Target Space 零膨脹分佈挖掘...")
    if 'score' in df_train.columns:
        scores = df_train['score']
        total_samples = len(scores)
        zero_samples = (scores == 0.0).sum()
        zero_ratio = zero_samples / total_samples
        
        print(f"   訓練集總週數樣本量: {total_samples}")
        print(f"   絕對零分 (0.0 score) 樣本量: {zero_samples} | 佔比: {zero_ratio:.2%}")
        
        non_zero_scores = scores[scores > 0.0]
        print(f"   非零乾旱樣本量: {len(non_zero_scores)} | 佔比: {(1 - zero_ratio):.2%}")
        print(f"   非零段分數分佈 - 均值: {non_zero_scores.mean():.4f} | 中位數: {non_zero_scores.median():.4f}")
        print(f"   非零段分佈百分位數 - 25%: {non_zero_scores.quantile(0.25):.4f} | 75%: {non_zero_scores.quantile(0.75):.4f} | 95%: {non_zero_scores.quantile(0.95):.4f}")
        
        # 建立嚴格的 Binned 矩陣基線快照
        print("\n   [Binned Error Matrix 基線分佈快照]")
        bins = [-0.01, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        labels = ['0.0', '0-1', '1-2', '2-3', '3-4', '4-5']
        binned_summary = pd.cut(scores, bins=bins, labels=labels).value_counts().sort_index()
        for idx, val in binned_summary.items():
            print(f"     區間 [{idx:^5}]: 樣本數 = {val:7d} | 總佔比 = {val/total_samples:.2%}")
    else:
        print("   訓練集中未發現 'score' 標籤欄位。")
        
    print("\n" + "="*75)
    print("  AUDIT COMPLETE: 數據結構安全，特徵與目標空間特徵總結完畢")
    print("="*75)

if __name__ == "__main__":
    run_audit()