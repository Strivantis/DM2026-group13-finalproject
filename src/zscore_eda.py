import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

def main():
    print("🌪️ 啟動 V34 Z-Score 異常空間 EDA & 特徵挖掘腳本 (連續時序滾動版)\n" + "="*60)
    
    # ---------------------------------------------------------
    # 準備階段：讀取資料
    # ---------------------------------------------------------
    train_path = 'data/processed/train_processed.csv'
    test_path = 'data/processed/test_processed.csv'
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("❌ 錯誤：找不到資料檔案。請確認檔案路徑正確。")
        return
        
    print("正在載入 Train 與 Test 資料集...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    base_features = ['prec', 'surf_pre', 'humidity', 'tmp', 'wind']
    epsilon = 1e-6
    
    # ---------------------------------------------------------
    # Step 1 & 2: 零變異數排雷與 Z-Score 轉換
    # ---------------------------------------------------------
    print("\n[Step 1 & 2] 歷史總體分佈 Z-Score 轉換與邊界掃描...")
    
    # 計算 Train Set 每個 region_id 的歷史統計量
    train_mean = train.groupby('region_id')[base_features].mean()
    train_std = train.groupby('region_id')[base_features].std().fillna(0)
    
    # 轉換函式
    def apply_long_zscore(df, mean_df, std_df, cols):
        df_z = df.copy()
        for col in cols:
            m = df_z['region_id'].map(mean_df[col])
            s = df_z['region_id'].map(std_df[col])
            df_z[f"{col}_z"] = (df_z[col] - m) / (s + epsilon)
        return df_z

    train = apply_long_zscore(train, train_mean, train_std, base_features)
    test = apply_long_zscore(test, train_mean, train_std, base_features)
    
    z_cols = [f"{col}_z" for col in base_features]
    print("Test Set 中 Z-Score 絕對值溢出臨界點 (|Z| > 5.0) 的比例：")
    for z_col in z_cols:
        outlier_share = (test[z_col].abs() > 5.0).mean() * 100
        print(f" - {z_col}: {outlier_share:.4f}%")

    # ---------------------------------------------------------
    # Step 3: V34 異常特徵工程 (使用 Shift 與 Rolling)
    # ---------------------------------------------------------
    print("\n[Step 3] 執行連續時序特徵工程 (Shift & Rolling)...")
    
    def inject_rolling_features(df):
        # 確保資料依照地區與時間順序排列，這是 Rolling 的絕對前提
        df = df.sort_values(by=['region_id', 'week_idx']).copy()
        
        # 1. 閃旱動能指標 (Momentum): 當前週 (t) 減去 4 週前 (t-4)
        # 由於 shift 無法避免前 4 週沒資料，我們維持原樣，後續 fillna 會補 0 (代表動能無變化)
        df['tmp_momentum'] = df['tmp_z'] - df.groupby('region_id')['tmp_z'].shift(4)
        df['prec_momentum'] = df['prec_z'] - df.groupby('region_id')['prec_z'].shift(4)
        
        # 2. 累積異常赤字 (Cumulative Z-Deficit): 過去 4 週降雨 Z-Score 總和
        # 💡 加入 min_periods=1，即使剛開局不到 4 週也能計算目前的總和
        df['recent_4w_prec_deficit'] = df.groupby('region_id')['prec_z'].transform(
            lambda x: x.rolling(window=4, min_periods=1).sum()
        )
        
        # 3. 極端異象計數器 (Extreme Event Counter): 過去 13 週溫度超標次數
        # 💡 同樣加入 min_periods=1，立刻開始計算超標次數
        df['_temp_extreme'] = (df['tmp_z'] > 1.5).astype(float)
        df['extreme_temp_counter'] = df.groupby('region_id')['_temp_extreme'].transform(
            lambda x: x.rolling(window=13, min_periods=1).sum()
        )
        df = df.drop(columns=['_temp_extreme'])
        
        return df
    
    train = inject_rolling_features(train)
    test = inject_rolling_features(test)
    
    new_features = ['tmp_momentum', 'prec_momentum', 'recent_4w_prec_deficit', 'extreme_temp_counter']
    
    if 'score' in train.columns:
        print("全新核武特徵與目標變數 (score) 的 Pearson 相關係數：")
        for feat in new_features:
            correlation = train[feat].corr(train['score'])
            print(f" - {feat}: {correlation:.4f}")

    # ---------------------------------------------------------
    # Step 4: 對抗性驗證終極測試 (Adversarial Validation)
    # ---------------------------------------------------------
    print("\n[Step 4] 對抗性驗證終極測試 (驗證時空斷層是否抹除)...")
    
    adv_features = z_cols + new_features
    
    adv_train = train[adv_features].copy()
    adv_test = test[adv_features].copy()
    
    adv_train['is_test'] = 0
    adv_test['is_test'] = 1
    
    # 合併並填補滾動計算初期產生的 NaN
    adv_master = pd.concat([adv_train, adv_test], axis=0).reset_index(drop=True)
    adv_master = adv_master.fillna(0)
    
    X = adv_master.drop(columns=['is_test'])
    y = adv_master['is_test']
    
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("正在訓練 LightGBM 對抗性分類器...")
    model = lgb.LGBMClassifier(max_depth=5, n_estimators=200, random_state=42, verbose=-1, learning_rate=0.05)
    model.fit(X_tr, y_tr)
    
    preds = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, preds)
    
    print(f"\n📊 對抗性驗證 AUC 分數: {auc:.4f}")
    if auc < 0.65:
        print("🎉 完美！AUC 接近 0.5 ~ 0.6，代表 Z-Score 成功抹除了 Train/Test 的時空斷層！")
    else:
        print("⚠️ 注意：AUC 仍然偏高 (> 0.75)，代表特徵中仍存有未被抹除的時空特徵。")

if __name__ == '__main__':
    main()