"""
predict.py — generate a Kaggle submission from the trained LightGBM models.

Loads test features, predicts the 5 forecast weeks, clips to [0, 5], and writes a
submission in sample_submission.csv format (all 2248 regions, in that file's order).

Usage:  python predict.py [output_name]      (default submission_v1.csv)
"""
import json, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

ART = "artifacts"
N_WEEKS = 5
OUT = sys.argv[1] if len(sys.argv) > 1 else "submission_v1.csv"


def main():
    Xt = np.load(os.path.join(ART, "test_X.npy"))
    regs = json.load(open(os.path.join(ART, "test_regions.json")))
    assert len(regs) == Xt.shape[0]
    print(f"[predict] test_X={Xt.shape} regions={len(regs)}", flush=True)

    preds = np.zeros((Xt.shape[0], N_WEEKS), dtype=np.float64)
    for w in range(N_WEEKS):
        booster = lgb.Booster(model_file=os.path.join(ART, f"lgb_week{w+1}.txt"))
        preds[:, w] = np.clip(booster.predict(Xt), 0, 5)

    df = pd.DataFrame({"region_id": regs})
    for w in range(N_WEEKS):
        df[f"pred_week{w+1}"] = preds[:, w]

    # align to sample_submission region order to guarantee format/order match
    sample = pd.read_csv("sample_submission.csv")
    df = sample[["region_id"]].merge(df, on="region_id", how="left")
    assert df.isna().sum().sum() == 0, "missing predictions for some regions"

    df.to_csv(OUT, index=False)
    print(f"[predict] wrote {OUT} rows={len(df)} "
          f"pred_means={[round(df[f'pred_week{w+1}'].mean(),3) for w in range(N_WEEKS)]}", flush=True)


if __name__ == "__main__":
    main()
