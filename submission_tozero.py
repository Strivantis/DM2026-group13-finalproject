import pandas as pd

def process_submission(file_path, output_path):
    print("開始讀取 CSV 檔案...")
    # 1. 讀取 CSV 檔案
    df = pd.read_csv(file_path)
    
    # 2. 指定需要處理的預測欄位
    pred_cols = ['pred_week1', 'pred_week2', 'pred_week3', 'pred_week4', 'pred_week5']
    
    print("正在將所有小於 0.5 的預測值歸零...")
    # 3. 使用條件定位 (.loc)，當欄位數值 < 0.5 時，強制替換為 0.0
    for col in pred_cols:
        df.loc[df[col] < 0.5, col] = 0.0
        
    # 4. 儲存處理後的檔案，index=False 代表不寫入 row 的索引序號
    df.to_csv(output_path, index=False)
    print(f"處理完成！新檔案已儲存至: {output_path}")

# 執行程式
if __name__ == "__main__":
    input_file = 'submission_27th.csv'
    output_file = 'submission_27th_processed.csv'
    
    process_submission(input_file, output_file)