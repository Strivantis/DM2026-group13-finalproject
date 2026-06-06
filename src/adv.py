"""
adversarial_validation.py - Train/Test Distribution Shift Detector

This script runs an Adversarial Validation by training an LGBMClassifier
to distinguish between the Training set (label 0) and Test set (label 1).
If the AUC > 0.7, it indicates a significant covariate shift.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# 特徵白名單 (只保留純物理量與相對趨勢，拔除所有絕對時間與季節特徵)
PHYSICAL_FEATURES = [
    "prec", "prec_week_max", "surf_pre", "surf_pre_week_max",
    "humidity", "humidity_week_max", "humidity_week_min", "humidity_week_std",
    "tmp", "tmp_week_max", "tmp_week_min", "tmp_week_std",
    "wind", "wind_week_max", "wind_week_min", "wind_week_std",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp", "wind_max", "wind_min", "wind_range",
    "prec_roll_sum_4w", "tmp_roll_mean_4w", "humidity_roll_mean_4w",
    "tmp_lag1w", "tmp_lag2w", "humidity_lag1w", "humidity_lag2w",
    "prec_lag1w", "prec_lag2w", "wind_lag1w", "wind_lag2w",
    "pet", "deficit", "deficit_roll_cum_4w"
]

def main():
    print("=" * 80)
    print("Adversarial Validation: Hunting for Covariate Shift")
    print("=" * 80)

    # 1. Load Data
    print("\n[1] Loading Processed Data...")
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"))

    # Assign adversarial targets
    train["is_test"] = 0
    test["is_test"] = 1

    # Ensure we only use available features
    features_to_use = [f for f in PHYSICAL_FEATURES if f in train.columns and f in test.columns]
    print(f"    Evaluating {len(features_to_use)} physical features.")

    # Combine datasets
    df_combined = pd.concat([train[features_to_use + ["is_test"]], test[features_to_use + ["is_test"]]], ignore_index=True)
    X = df_combined[features_to_use]
    y = df_combined["is_test"]

    # 2. Train Adversarial Model
    print("\n[2] Training Adversarial Classifier (5-Fold CV)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(df_combined))
    feature_importances = np.zeros(len(features_to_use))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        # Use a simple, constrained model to avoid overfitting noise
        model = lgb.LGBMClassifier(
            max_depth=5,
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=1000,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
        feature_importances += model.feature_importances_ / 5.0
        
        fold_auc = roc_auc_score(y_va, oof_preds[val_idx])
        print(f"    Fold {fold + 1} AUC: {fold_auc:.4f}")

    # 3. Evaluate Results
    overall_auc = roc_auc_score(y, oof_preds)
    print("\n" + "=" * 80)
    print(f"OVERALL ADVERSARIAL AUC: {overall_auc:.4f}")
    print("=" * 80)
    
    if overall_auc < 0.6:
        print("Diagnosis: EXCELLENT. Train and Test distributions are nearly indistinguishable.")
    elif overall_auc < 0.7:
        print("Diagnosis: WARNING. Mild shift detected. Models should generalize mostly fine.")
    else:
        print("Diagnosis: DANGER. Severe Covariate Shift! The model can easily tell Test data apart.")
        print("           You must aggressively drop or neutralize the top 'Ghost' features.")

    # 4. Feature Importance (The "Ghosts")
    fi_df = pd.DataFrame({
        "Feature": features_to_use,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    print("\n[ Top 10 'Ghost' Features (Causing the Shift) ]")
    print(fi_df.head(10).to_string(index=False))

    # Save plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=fi_df.head(20), hue="Feature", legend=False)
    plt.title(f"Adversarial Feature Importance (Overall AUC: {overall_auc:.3f})")
    plt.tight_layout()
    plot_path = os.path.join(ROOT, "adversarial_features.png")
    plt.savefig(plot_path)
    print(f"\n    Saved Importance Plot -> {plot_path}")

if __name__ == "__main__":
    main()