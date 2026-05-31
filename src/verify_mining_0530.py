"""
src/verify_mining_fixed.py
======================================================================
結構對齊驗證腳本：確保與 preprocess.py 的週聚合、週滾動順序 100% 一致。
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(_BASE, "data", "train.csv")
TEST_PATH = os.path.join(_BASE, "data", "test.csv")

print("="*75)
print("  [Data Mining & Verification] 100% 結構對齊邊界特徵偏移度量")
print("="*75)

if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    print(" 錯誤：找不到原始 data/train.csv 或 data/test.csv，請確認路徑。")
    exit()

# ---------------------------------------------------------------------------
# [工具函數] 複製自 preprocess.py 的核心邏輯
# ---------------------------------------------------------------------------
_MONTH_OFFSET = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
def _parse_doy(date_str: str) -> int:
    parts = date_str.split("-")
    return _MONTH_OFFSET[int(parts[1]) - 1] + int(parts[2])

# 載入原始 Daily 資料
df_train_raw = pd.read_csv(TRAIN_PATH)
df_test_raw = pd.read_csv(TEST_PATH)

df_train_raw['doy'] = df_train_raw['date'].apply(_parse_doy)
df_test_raw['doy'] = df_test_raw['date'].apply(_parse_doy)

# ---------------------------------------------------------------------------
# 1. 建立 Train 穩態基準面 (完全比照 preprocess.py 產出的週數據)
# ---------------------------------------------------------------------------
print("1. 正在建立 Train 穩態週級特徵基準面...")
df_train_raw = df_train_raw.sort_values(["region_id", "date"], ignore_index=True)
keep_days_train = 5474
df_train_truncated = df_train_raw[df_train_raw.groupby("region_id").cumcount() < keep_days_train].copy()
df_train_truncated["week_idx"] = df_train_truncated.groupby("region_id").cumcount() // 7

# 週聚合 (只抽取降雨總和進行驗證)
train_weekly = df_train_truncated.groupby(["region_id", "week_idx"], sort=False).agg(
    prec=pd.NamedAgg(column="prec", aggfunc="sum")
).reset_index()

# 週滾動
train_weekly["prec_roll_sum_4w"] = train_weekly.groupby("region_id")["prec"].transform(
    lambda s: s.rolling(4, min_periods=1).sum()
)
# 排除前 4 週未飽和數據，取得真正穩態分佈
train_steady_features = train_weekly.loc[train_weekly["week_idx"] >= 4, "prec_roll_sum_4w"].values

# ---------------------------------------------------------------------------
# 2. 挖掘 Climatology 基線 (Per-Region, Per-Doy 氣候常態)
# ---------------------------------------------------------------------------
print("2. 正在從 Train Set 挖掘區域氣候常態基線 (Daily Climatology)...")
climatology = df_train_raw.groupby(['region_id', 'doy'])['prec'].mean().reset_index()
climatology.rename(columns={'prec': 'clim_prec'}, inplace=True)
global_daily_mean = df_train_raw['prec'].mean()

# ---------------------------------------------------------------------------
# 3. 模擬策略 A：現行 Pipeline 填零法 (直接對 13 週 Test 進行週聚合與週滾動)
# ---------------------------------------------------------------------------
print("3. 模擬現行 Pipeline (填零法) 的週級特徵...")
df_test_raw = df_test_raw.sort_values(["region_id", "date"], ignore_index=True)
df_test_raw["week_idx"] = df_test_raw.groupby("region_id").cumcount() // 7

test_weekly_current = df_test_raw.groupby(["region_id", "week_idx"], sort=False).agg(
    prec=pd.NamedAgg(column="prec", aggfunc="sum")
).reset_index()

test_weekly_current["prec_roll_sum_4w"] = test_weekly_current.groupby("region_id")["prec"].transform(
    lambda s: s.rolling(4, min_periods=1).sum()
)

# ---------------------------------------------------------------------------
# 4. 模擬策略 B：氣候常態前置填補法 (日級補 21 天 -> 週聚合 -> 週滾動 -> 裁切)
# ---------------------------------------------------------------------------
print("4. 模擬升級策略 (氣候常態前置填補法)...")
test_padded_list = []
pad_days = 21

for region, group in df_test_raw.groupby('region_id'):
    start_doy = group['doy'].iloc[0]
    
    # 構造前置 21 天的虛擬 DOY 序列
    pad_doys = [(start_doy - i - 1) % 366 for i in range(pad_days)][::-1]
    pad_doys = [366 if d == 0 else d for d in pad_doys]
    
    pad_df = pd.DataFrame({'region_id': region, 'doy': pad_doys})
    pad_df['date'] = [f"2999-12-{31 - pad_days + 1 + i:02d}" for i in range(pad_days)]
    
    # 對齊氣候常態
    pad_df = pad_df.merge(climatology[climatology['region_id'] == region], on=['region_id', 'doy'], how='left')
    pad_df['prec'] = pad_df['clim_prec'].fillna(global_daily_mean)
    pad_df = pad_df.drop(columns=['doy', 'clim_prec'])
    
    # 拼接 Dummy 緩衝區與真實 Test 日資料
    combined = pd.concat([pad_df, group[['region_id', 'date', 'prec']]], ignore_index=True)
    test_padded_list.append(combined)

df_test_padded_raw = pd.concat(test_padded_list).reset_index(drop=True)
df_test_padded_raw["week_idx"] = df_test_padded_raw.groupby("region_id").cumcount() // 7

# 週聚合 (此時會有 3 週虛擬 + 13 週真實 = 16 週)
test_weekly_fixed = df_test_padded_raw.groupby(["region_id", "week_idx"], sort=False).agg(
    prec=pd.NamedAgg(column="prec", aggfunc="sum")
).reset_index()

# 週滾動
test_weekly_fixed["prec_roll_sum_4w"] = test_weekly_fixed.groupby("region_id")["prec"].transform(
    lambda s: s.rolling(4, min_periods=1).sum()
)

# 剝離 Dummy 緩衝區 (前 3 週)，還原 13 週測試集
test_weekly_fixed = test_weekly_fixed[test_weekly_fixed['week_idx'] >= 3].copy()
test_weekly_fixed['week_idx'] = test_weekly_fixed['week_idx'] - 3
test_weekly_fixed = test_weekly_fixed.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 5. 指標計算與量化對比 (聚焦於 Test Set 前 3 週，即 week_idx 0, 1, 2)
# ---------------------------------------------------------------------------
early_week_mask = test_weekly_current["week_idx"].isin([0, 1, 2])

current_early_vals = test_weekly_current.loc[early_week_mask, "prec_roll_sum_4w"].values
fixed_early_vals = test_weekly_fixed.loc[early_week_mask, "prec_roll_sum_4w"].values

w_dist_current = wasserstein_distance(train_steady_features, current_early_vals)
w_dist_fixed = wasserstein_distance(train_steady_features, fixed_early_vals)

print("\n" + "="*75)
print("  STRUCTURAL ALIGNED MINING OUTPUT RESULTS")
print("="*75)
print(f"Train 穩態週降雨滾動總和均值: {train_steady_features.mean():.4f} | 方差: {train_steady_features.var():.4f}")
print(f"現行 Pipeline 測試集前3週均值: {current_early_vals.mean():.4f} (特徵尺度塌陷程度)")
print(f"常態前置修復後測試集前3週均值: {fixed_early_vals.mean():.4f} (修復後的特徵尺度)")
print("-"*75)
print(f"【現行填零法】與 Train 的分佈偏移度 (Wasserstein Distance): {w_dist_current:.4f}")
print(f"【氣候常態法】與 Train 的分佈偏移度 (Wasserstein Distance): {w_dist_fixed:.4f}")
print(f"  👉 結構對齊後的特徵偏移風險降幅: {((w_dist_current - w_dist_fixed) / w_dist_current):.2%}")
print("="*75)