"""
train.py — train 5 LightGBM regressors (one per forecast week) with MAE objective.

- Time-based validation: the most recent anchors (by per-region row index) are held
  out, approximating the forward-forecast nature of the test set.
- Built-in early stopping, model checkpoint, and learning-rate shrinkage scheduler.
- Stdout is restricted to coarse iteration milestones + per-week summaries.

Outputs: artifacts/lgb_week{1..5}.txt , artifacts/val_metrics.json
Usage:  python train.py
"""
import json, os
import numpy as np
import lightgbm as lgb

ART = "artifacts"
SEED = 42
VAL_FRAC = 0.15          # most-recent fraction held out for validation
N_WEEKS = 5

PARAMS = dict(
    objective="regression_l1",   # optimise MAE directly
    metric="l1",
    learning_rate=0.03,          # shrinkage scheduler base rate
    num_leaves=63,
    min_child_samples=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_depth=-1,
    num_threads=0,               # use all cores
    verbosity=-1,
    seed=SEED,
)
NUM_ROUNDS = 3000
EARLY_STOP = 150


def main():
    X = np.load(os.path.join(ART, "train_X.npy"))
    Y = np.load(os.path.join(ART, "train_y.npy"))
    OD = np.load(os.path.join(ART, "train_anchord.npy"))   # per-region anchor row index ~ time
    meta = json.load(open(os.path.join(ART, "feature_meta.json")))
    feat_names = meta["feature_names"]
    print(f"[train] X={X.shape} Y={Y.shape} feats={len(feat_names)}", flush=True)

    # time-based split: hold out most-recent anchors (highest row index)
    thr = np.quantile(OD, 1.0 - VAL_FRAC)
    val_mask = OD >= thr
    tr_mask = ~val_mask
    print(f"[train] split thr_rowidx={thr:.0f} | train={tr_mask.sum()} val={val_mask.sum()}", flush=True)

    metrics = {"per_week": {}, "config": {"params": PARAMS, "num_rounds": NUM_ROUNDS,
                                          "early_stop": EARLY_STOP, "val_frac": VAL_FRAC,
                                          "stride": meta.get("stride")}}
    val_maes = []
    for w in range(N_WEEKS):
        ytr, yval = Y[tr_mask, w], Y[val_mask, w]
        dtr = lgb.Dataset(X[tr_mask], label=ytr, feature_name=feat_names, free_raw_data=False)
        dval = lgb.Dataset(X[val_mask], label=yval, reference=dtr, free_raw_data=False)
        evals = {}
        booster = lgb.train(
            PARAMS, dtr, num_boost_round=NUM_ROUNDS,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=300),          # coarse milestones only
                lgb.record_evaluation(evals),
            ],
        )
        best_it = booster.best_iteration
        pred = np.clip(booster.predict(X[val_mask], num_iteration=best_it), 0, 5)
        mae = float(np.mean(np.abs(pred - yval)))
        val_maes.append(mae)
        booster.save_model(os.path.join(ART, f"lgb_week{w+1}.txt"), num_iteration=best_it)
        metrics["per_week"][f"week{w+1}"] = {"val_mae": mae, "best_iteration": int(best_it)}
        print(f"[train] week{w+1}: best_it={best_it} val_MAE={mae:.4f}", flush=True)

    macro = float(np.mean(val_maes))
    metrics["macro_val_mae"] = macro
    with open(os.path.join(ART, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] DONE macro_val_MAE={macro:.4f} per_week={[round(m,4) for m in val_maes]}", flush=True)


if __name__ == "__main__":
    main()
