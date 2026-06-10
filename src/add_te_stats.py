"""
add_te_stats.py  —  1-time data script
Merges region_mean_score & region_zero_prob from v51_processed/region_stats.csv
into v54_processed/region_stats.csv so v55_train.py can read them as global priors.
"""
import os
import pandas as pd

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V51_PATH = os.path.join(ROOT, "data", "v51_processed", "region_stats.csv")
V54_PATH = os.path.join(ROOT, "data", "v54_processed", "region_stats.csv")

def main():
    v51 = pd.read_csv(V51_PATH, usecols=["region_id", "score_mean", "score_zero_prob"])
    v51 = v51.rename(columns={"score_mean": "region_mean_score",
                               "score_zero_prob": "region_zero_prob"})

    v54 = pd.read_csv(V54_PATH)
    v54 = v54.drop(columns=[c for c in ["region_mean_score", "region_zero_prob"]
                             if c in v54.columns])

    merged = v54.merge(v51, on="region_id", how="left")

    n_missing = merged["region_mean_score"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} regions have no v51 TE stats; filling with global mean/prob.")
        merged["region_mean_score"] = merged["region_mean_score"].fillna(
            merged["region_mean_score"].mean())
        merged["region_zero_prob"]  = merged["region_zero_prob"].fillna(
            merged["region_zero_prob"].mean())

    merged.to_csv(V54_PATH, index=False)
    print(f"Updated {V54_PATH}  shape={merged.shape}")
    print(merged[["region_id", "region_mean_score", "region_zero_prob"]].head())

if __name__ == "__main__":
    main()
