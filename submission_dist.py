import os
import io
import pandas as pd

def calculate_prediction_proportions(file_path):
    # 1. 安全地讀取 Markdown 檔案內容
    if not os.path.exists(file_path):
        print(f"找不到檔案：{file_path}，請確認路徑是否正確。")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. 清理 Markdown 標籤，只保留包含逗號的 CSV 資料行
    csv_lines = [line for line in lines if ',' in line and not line.strip().startswith('```')]
    csv_content = "".join(csv_lines)

    # 3. 將文字轉換為 Pandas DataFrame
    try:
        df = pd.read_csv(io.StringIO(csv_content))
    except Exception as e:
        print(f"解析 CSV 資料時發生錯誤：{e}")
        return

    # 4. 取得所有預測欄位（排除 region_id）
    # 這裡自動選取所有名稱包含 'pred' 的欄位
    pred_cols = [col for col in df.columns if 'pred' in col]
    
    if not pred_cols:
        print("找不到任何預測欄位（欄位名需包含 'pred'）")
        return

    # 5. 將所有預測值攤平（Flatten）成一維陣列
    all_pred_values = df[pred_cols].values.flatten()

    # 6. 將數值進行四捨五入，並轉換為整數類型
    # 備註：Pandas 的 round() 在遇到 .5 時會採用「四捨六入五成雙」
    # 為了確保符合傳統四捨五入，我們加上 0.5 後向下取整，或直接使用內建 round
    rounded_series = pd.Series(all_pred_values).round().astype(int)

    # 7. 計算 0 到 5 的比例 (normalize=True 會直接算百分比)
    value_counts = rounded_series.value_counts(normalize=True)

    # 8. 美化並印出結果
    print("--- 預測儲存格數值分佈報告 ---")
    print(f"總計算儲存格數量: {len(all_pred_values)} 個\n")
    
    for target_val in range(6):
        # 取得該數字的比例，若沒出現過則預設為 0
        proportion = value_counts.get(target_val, 0.0)
        print(f"數字 {target_val} 的比例: {proportion:.2%}")

if __name__ == "__main__":
    # 指定 workplace 中的檔案路徑
    target_file = 'submission_27th.csv'
    calculate_prediction_proportions(target_file)