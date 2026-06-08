import os
import numpy as np
import pandas as pd

# 定義你的專案根目錄與路徑
ROOT = "/home/sean/KC3000/NYCU/114-2/Data Mining/FinalProject"
INPUT_SUB  = os.path.join(ROOT, "submission_45th_30k_ABraw.csv") # 確認是修改後的正確檔名
OUTPUT_SUB = os.path.join(ROOT, "submission_45th_30k_optimized.csv")

# 直接從你成功的 Log 中把最佳閾值拿過來用 (2.1000e-03 即為 0.0021)
best_thresholds = [0.0021, 1.3331, 2.2490, 3.2704, 6.1084]

print(">>> 正在載入原始 submission 檔案...")
sub_df = pd.read_csv(INPUT_SUB)

print(">>> 正在套用 Nelder-Mead 最佳化閾值...")
pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]

for col in pred_cols:
    # 使用 np.digitize 將連續型浮點數轉換為 0~5 的整數類別
    sub_df[col] = np.digitize(sub_df[col], best_thresholds)

print(">>> 正在匯出最終優化後的 Submission...")
sub_df.to_csv(OUTPUT_SUB, index=False)

print(f"✅ 成功！檔案已安全儲存至: {OUTPUT_SUB}")