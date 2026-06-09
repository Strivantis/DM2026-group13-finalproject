import os
import pandas as pd
import numpy as np

# 建立絕對路徑
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_SUB  = os.path.join(ROOT, "FinalProject/submission_45th_30k_ABraw.csv") # 你的 0.8428 浮點數原檔
OUTPUT_SUB = os.path.join(ROOT, "FinalProject/submission_45th_30k_pure_rounded.csv")

print("=" * 70)
print(" V51 PURE ROUNDING VERIFIER ")
print("=" * 70)

if not os.path.exists(INPUT_SUB):
    raise FileNotFoundError(f"找不到原始浮點數檔案：{INPUT_SUB}")

# 1. 讀取原檔
df = pd.read_csv(INPUT_SUB)
pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]

print(f"成功載入檔案：{INPUT_SUB}")
print(f"原始浮點數均值：\n{df[pred_cols].mean()}")

# 2. 執行最純粹、毫無花招的神聖四捨五入
for col in pred_cols:
    df[col] = np.round(df[col]).clip(0.0, 5.0).astype(int)

# 3. 檢查分佈是否與歷史對齊
print("\n執行四捨五入後的整數均值：")
print(df[pred_cols].mean())

# 4. 匯出
df.to_csv(OUTPUT_SUB, index=False)
print(f"\n✅ 驗證檔案已產出：{OUTPUT_SUB}")
print("請直接將此檔案上傳 Kaggle。如果它回歸 0.8246，代表簡單四捨五入就是真理！")
print("=" * 70)