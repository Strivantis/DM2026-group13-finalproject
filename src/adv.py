"""
adv_true_406.py - True Flattened Feature Adversarial Validation

This script uses YOUR exact dataset builder to generate the 406D matrices,
then performs Adversarial Validation. This ensures we are testing exactly
what the LightGBM models see during training/inference.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import (
    refine_features,
    build_full_train_groups,
    build_tabular_dataset,
    build_tabular_test,
    FEATURE_COLS,
    WINDOW_SIZE,
    make_flat_col_names
)

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

def main():
    print("=" * 85)
    print("True 406D Adversarial Validation (Model's Exact Viewpoint)")
    print("=" * 85)

    # 1. Load & Refine Data exactly as in training
    print("\n[1] Loading and refining data...")
    train_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test_raw  = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))
    
    train_df = refine_features(train_raw, is_train=True)
    test_df  = refine_features(test_raw,  is_train=False)
    train_df = train_df.dropna(subset=["score"]).reset_index(drop=True)
    train_df["region_mean_score"] = 0.0
    train_df["region_zero_prob"] = 0.0

    # 2. Build Tabular Matrices (The 406D view)
    print("\n[2] Building 406D Flat Matrices...")
    train_groups = build_full_train_groups(train_df)
    # Using only a subset of train groups to balance the classes and speed up
    # Since Test is 2248 regions, we sample around 2248 * 5 windows from train
    import random
    random.seed(42)
    sample_train_groups = random.sample(train_groups, min(len(train_groups), 500))
    
    X_train_np, _, _ = build_tabular_dataset(sample_train_groups, FEATURE_COLS)
    
    # Test matrix requires TE placeholders just to pass the builder safely
    test_df["region_mean_score"] = 0.0
    test_df["region_zero_prob"] = 0.0
    X_test_np, _ = build_tabular_test(test_df, FEATURE_COLS)

    print(f"    X_train shape: {X_train_np.shape} (sampled)")
    print(f"    X_test shape : {X_test_np.shape}")

    # Generate the actual column names based on surviving features
    actual_features = [c for c in FEATURE_COLS if c in train_df.columns]
    flat_cols = make_flat_col_names(actual_features, WINDOW_SIZE)
    print(f"    Expected feature names matched: {len(flat_cols)}")

    # 3. Create Adversarial Dataset
    X_tr = pd.DataFrame(X_train_np, columns=flat_cols)
    X_tr["is_test"] = 0
    X_te = pd.DataFrame(X_test_np, columns=flat_cols)
    X_te["is_test"] = 1

    df_combined = pd.concat([X_tr, X_te], ignore_index=True)
    X = df_combined[flat_cols]
    y = df_combined["is_test"]

    # 4. Train Adversarial Model
    print("\n[3] Training Adversarial Classifier...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_combined))
    feature_importances = np.zeros(len(flat_cols))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            n_estimators=500,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        feature_importances += model.feature_importances_ / 3.0

    overall_auc = roc_auc_score(y, oof_preds)
    
    print("\n" + "=" * 85)
    print(f"TRUE 406D ADVERSARIAL AUC: {overall_auc:.4f}")
    print("=" * 85)

    fi_df = pd.DataFrame({
        "Feature": flat_cols,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    print("\n[ Top 15 Most Shifted Model Features ]")
    print(fi_df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()