"""
submit_and_log.py — submit a file to Kaggle, retrieve the online MAE, and append a
complete experiment record to run_history.json.

Usage:  python submit_and_log.py <submission.csv> "<message>" [iteration_tag]
"""
import json, os, sys, time, subprocess, datetime, csv, io

COMP = "data-mining-2026-final-project"
HIST = "run_history.json"
ART = "artifacts"
# kaggle CLI lives next to the running python interpreter; fall back to PATH name
KAGGLE = os.path.join(os.path.dirname(sys.executable), "kaggle")
if not os.path.exists(KAGGLE):
    KAGGLE = "kaggle"


def run(cmd):
    cmd = [KAGGLE if c == "kaggle" else c for c in cmd]
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "submission_v1.csv"
    msg = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    tag = sys.argv[3] if len(sys.argv) > 3 else "v1"

    print(f"[submit] uploading {sub} -> {COMP}", flush=True)
    r = run(["kaggle", "competitions", "submit", "-c", COMP, "-f", sub, "-m", msg])
    print("[submit]", (r.stdout + r.stderr).strip()[:500], flush=True)

    print("[submit] waiting 45s for scoring...", flush=True)
    time.sleep(45)

    online_mae = None
    for attempt in range(6):
        r = run(["kaggle", "competitions", "submissions", "-c", COMP, "--csv"])
        # use csv parser: the description column contains commas
        rows = list(csv.DictReader(io.StringIO(r.stdout)))
        # find this submission's row (match on file name), else newest row
        match = next((x for x in rows if x.get("fileName") == os.path.basename(sub)), None)
        row = match or (rows[0] if rows else None)
        if row is not None:
            status = row.get("status", "")
            val = (row.get("publicScore") or "").strip()
            if "COMPLETE" in status and val not in ("", "None", "Pending"):
                try:
                    online_mae = float(val)
                    break
                except ValueError:
                    pass
        print(f"[submit] score not ready (attempt {attempt+1}), waiting 20s...", flush=True)
        time.sleep(20)
    print(f"[submit] online_mae = {online_mae}", flush=True)

    val_metrics = json.load(open(os.path.join(ART, "val_metrics.json")))
    meta = json.load(open(os.path.join(ART, "feature_meta.json")))

    entry = {
        "iteration": tag,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "submission_file": sub,
        "message": msg,
        "model": "LightGBM x5 (one per forecast week), objective=L1/MAE",
        "features": {
            "n_features": len(meta["feature_names"]),
            "window": meta["window"],
            "horizons": meta["horizons"],
            "stride": meta["stride"],
            "groups": "per-weather-col {mean,std,min,max,last,last7,last28,trend} "
                      "+ precip sums/dry-days + seasonality + per-region score climatology",
        },
        "hyperparameters": val_metrics.get("config", {}),
        "macro_val_mae": val_metrics.get("macro_val_mae"),
        "per_week_val_mae": {k: v["val_mae"] for k, v in val_metrics.get("per_week", {}).items()},
        "online_mae": online_mae,
    }

    hist = []
    if os.path.exists(HIST):
        try:
            hist = json.load(open(HIST))
        except Exception:
            hist = []
    hist.append(entry)
    with open(HIST, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[submit] appended iteration '{tag}' to {HIST} "
          f"(val={entry['macro_val_mae']}, online={online_mae})", flush=True)


if __name__ == "__main__":
    main()
