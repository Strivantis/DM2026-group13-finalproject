"""
v51_feature_exploration.py
快速探勘：在不需要跑完整 Pipeline 的情況下，測試新特徵在 Cluster 1 的潛力。
"""
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else os.getcwd()
PROCESSED_DIR = os.path.join(ROOT, "data", "v51_processed")

def main():
    print("=" * 70)
    print(" 🚀 V51 FEATURE POTENTIAL EXPLORER (CLUSTER 1 ONLY) ")
    print("=" * 70)

    # 1. 載入資料 (這需要在跑過 v51_preprocess 之後)
    print("Loading data...")
    try:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    except FileNotFoundError:
        print("Please run the updated preprocess.py first to generate new features.")
        return

    # 2. 聚焦重災區：只取 Cluster 1 且發生乾旱 (score > 0) 的資料
    # 這完美模擬了 V50 Model A 的運作環境
    c1_df = df[(df["cluster_id"] == 1) & (df["score"] > 0)].copy()
    print(f"Focusing on Cluster 1 (Drought only): {len(c1_df)} samples")

    # 定義特徵：原版特徵 + 3個新特徵
    base_feats = ["prec", "humidity", "tmp", "tmp_max", "tmp_week_max", "pet", "deficit"]
    new_feats = ["aridity_index", "heat_shock", "tmp_anomaly"]
    
    features = base_feats + new_feats
    
    # 確認新特徵存在
    missing = [f for f in features if f not in c1_df.columns]
    if missing:
        print(f"Missing features: {missing}. Run preprocess.py first!")
        return

    X = c1_df[features]
    y = c1_df["score"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 訓練一個輕量級模型
    print("\nTraining a quick validation model...")
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        max_depth=4, # 極淺樹，逼迫模型挑選最強特徵
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])

    # 4. 檢視特徵重要性 (Feature Importance - Split)
    print("\n[Feature Importance (Split / How many times used)]")
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    print("-" * 50)
    for _, row in importance.iterrows():
        marker = "⭐ NEW" if row["Feature"] in new_feats else ""
        print(f"{row['Feature']:<20} : {int(row['Importance']):>5} {marker}")
    print("-" * 50)

    print("\n💡 INTERPRETATION:")
    print("If the ⭐ NEW features appear in the top half of the list, it means they provided a 'shortcut' that the model preferred over raw features.")
    print("If they are at the bottom, your intuition was right: 30,000 trees didn't need our help, and we should discard them to save time.")

if __name__ == "__main__":
    main()