import os
import pandas as pd
import numpy as np

# 1. 設置路徑
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, "FinalProject/data", "processed", "train_processed.csv")
TEST_PATH = os.path.join(ROOT, "FinalProject/data", "processed", "test_processed.csv")

print("=" * 75)
print("啟動 2,248 獨立區域局部協變量偏移（Per-Region Covariate Shift）嚴密數學證明")
print("=" * 75)

# 2. 定義核心驗證欄位 (完全對齊你提供的特徵矩陣)
core_features = [
    "tmp", "prec", "humidity", "deficit", 
    "prec_roll_sum_4w", "deficit_roll_cum_4w"
]

print("正在載入數據欄位...")
# 訓練集載入核心欄位
train_df = pd.read_csv(TRAIN_PATH, usecols=["region_id"] + core_features)
# 測試集載入核心欄位與 week_idx
test_df = pd.read_csv(TEST_PATH, usecols=["region_id", "week_idx"] + core_features)

print(f"訓練集維度: {train_df.shape} | 測試集維度: {test_df.shape}")

# 3. 建立各區域歷史 15 年的全量穩態基線 (Train Baseline)
print("正在計算 2,248 個區域各自的歷史平均值與標準差...")
train_stats = train_df.groupby("region_id")[core_features].agg(["mean", "std"])

# 4. 建立各區域盲測當季 13 週的觀測均值 (Test Realised Mean)
print("正在計算各區域測試當季 13 週的觀測均值...")
test_stats = test_df.groupby("region_id")[core_features].mean()

# 5. 核心證明計算：計算每個區域獨立對齊後的 Local Z-Score Drift
print("\n[執行空間去中心化對齊] 計算 Test 相對於該區域自身歷史的局部偏離標準差...")
drift_records = []

for region_id in train_stats.index:
    if region_id not in test_stats.index:
        continue
        
    region_drift = {"region_id": region_id}
    for feat in core_features:
        mu_train = train_stats.loc[region_id, (feat, "mean")]
        std_train = train_stats.loc[region_id, (feat, "std")]
        mu_test = test_stats.loc[region_id, feat]
        
        # 物理防護欄：防止常年無雨區標準差為 0 導致除以零
        std_train = std_train if std_train > 1e-5 else 1e-5
        
        # 核心數學公式：計算該區域專屬的局部偏移量 (Local Z-Score)
        local_z_drift = (mu_test - mu_train) / std_train
        region_drift[f"{feat}_local_z"] = local_z_drift
        
    drift_records.append(region_drift)

drift_df = pd.DataFrame(drift_records)

# 6. 輸出統計審計報告
print("\n" + "=" * 75)
print("2,248 區域特異性局部偏移量化報告")
print("=" * 75)

for feat in core_features:
    col = f"{feat}_local_z"
    z_vals = drift_df[col].values
    
    macro_mean_drift = np.mean(z_vals)
    spatial_std_drift = np.std(z_vals)
    max_drift_val = np.max(np.abs(z_vals))
    
    # 判定局部時空失真 (偏離超過 1 個歷史標準差) 的區域比例
    severe_shift_pct = np.mean(np.abs(z_vals) > 1.0) * 100
    # 判定有多少比例的區域，其局部變異方向與全域總體平均方向完全相反 (空間撕裂度)
    opposite_dir_pct = np.mean(np.sign(z_vals) != np.sign(macro_mean_drift)) * 100
    
    print(f"特徵欄位 [{feat:<22}]:")
    print(f"  全域宏觀平均偏移 (Macro Drift) : {macro_mean_drift:>7.4f} σ")
    print(f"  區域在地變異方差 (Spatial Std) : {spatial_std_drift:>7.4f} σ  <-- [關鍵指標]")
    print(f"  單區局部極端最大偏離 (Max Drift) : {max_drift_val:>7.4f} σ")
    print(f"  在地時空嚴重異常區域佔比 (|Z|>1): {severe_shift_pct:>6.2f}%")
    print(f"  與全域宏觀趨勢背道而馳的區域佔比 : {opposite_dir_pct:>6.2f}%")
    print("-" * 75)

print("驗證完畢。")