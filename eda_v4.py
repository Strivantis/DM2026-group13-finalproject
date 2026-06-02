import os
import pandas as pd
import numpy as np

# 路徑設定
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, "FinalProject/data/processed", "train_processed.csv")
OUT_PATH = os.path.join(ROOT, "FinalProject/data/processed", "region_priors.csv")

print("="*60)
print("啟動區域先驗機率 (Target Encoding Priors) 挖掘")
print("="*60)

# 1. 讀取原始訓練集 (確保取得最真實的 15 年全量標籤)
print(f"正在讀取 {TRAIN_PATH} ...")
df = pd.read_csv(TRAIN_PATH, usecols=["region_id", "score"])

# 2. 過濾掉 score 為 NaN 的 ghost rows
df_valid = df.dropna(subset=["score"]).copy()
print(f"有效 Score 樣本數: {len(df_valid):,}")

# 3. 聚合計算 Per-Region 的統計特徵
print("正在計算各區域的 Mean Score 與 Zero Probability ...")
priors = df_valid.groupby("region_id").agg(
    region_mean_score=("score", "mean"),
    region_zero_prob=("score", lambda x: (x == 0.0).mean())
).reset_index()

# 4. 輸出全局統計以供驗證
global_mean = priors["region_mean_score"].mean()
global_zero = priors["region_zero_prob"].mean()
print(f"\n[全域統計驗證]")
print(f"Global Mean of Mean Scores : {global_mean:.4f}")
print(f"Global Mean of Zero Probs  : {global_zero:.4f}")

# 5. 顯示極端區域
print("\n[最常發生乾旱的 5 個區域 (Zero Prob 最低)]")
print(priors.sort_values("region_zero_prob").head(5).to_string(index=False))

print("\n[最少發生乾旱的 5 個區域 (Zero Prob 最高)]")
print(priors.sort_values("region_zero_prob", ascending=False).head(5).to_string(index=False))

# 6. 儲存至 processed 目錄
priors.to_csv(OUT_PATH, index=False)
print(f"\n區域先驗字典已成功匯出至: {OUT_PATH}")
print("="*60)