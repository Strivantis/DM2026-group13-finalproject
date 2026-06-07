import pandas as pd
import numpy as np

# 1. 讀取你目前分數最高的 submission (V45.1 Raw)
# 請替換為你實際拿到 0.8428 的那個檔案名稱
input_file = "submission_45th_30k_ABraw.csv" 
output_file = "submission_45th_30k_ABraw_rounded.csv"

df = pd.read_csv(input_file)
pred_cols = ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]

# 2. 強制四捨五入成整數 (0, 1, 2, 3, 4, 5)
for col in pred_cols:
    df[col] = np.round(df[col])

df.to_csv(output_file, index=False)
print(f"✅ 整數化完成，請立即上傳 {output_file}")