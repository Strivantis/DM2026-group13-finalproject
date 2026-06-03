"""
preprocess.py — build supervised features for drought-severity forecasting.

File structure (discovered): each region is split into TWO temporally-contiguous
blocks — block A (early ~4750 days) for all regions, then block B (later ~730 days)
for all regions. The two blocks join seamlessly (B starts the day after A ends), so
each region is reassembled by concatenating its blocks in file order = [A, B].
Region ids are sparse/non-sequential and each region lives in its own date range,
so absolute dates are meaningless across regions — the time axis is ROW POSITION
within a region, and only seasonality (month / day-of-year) is used calendar-wise.

Supervised construction mirrors the test setup exactly:
  * anchor = a weekly score-grid row t (score present every 7 rows)
  * features = aggregates over the 91-row weather window ending at t
               (the anchor's own score is NOT used — unavailable at test time)
  * targets  = score at rows t+7..t+35 (weeks 1..5); anchors missing any future
               score row are skipped (handles cadence/boundary gaps).

Outputs (artifacts/): train_X.npy, train_y.npy, train_anchord.npy, train_regidx.npy,
  test_X.npy, test_regions.json, region_stats.json, feature_meta.json

Usage:  python preprocess.py [stride]   (stride in score-weeks between anchors, default 3)
"""
import csv, sys, json, os, warnings
import numpy as np

warnings.filterwarnings("ignore")

DATA_DIR = "data"
ART = "artifacts"
os.makedirs(ART, exist_ok=True)

WEATHER_COLS = ["prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
                "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
                "wind", "wind_max", "wind_min", "wind_range"]
NW = len(WEATHER_COLS)
WINDOW = 91
HORIZONS = np.array([7, 14, 21, 28, 35])
PREC_IDX = WEATHER_COLS.index("prec")
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 3

AGG = ["mean", "std", "min", "max", "last", "last7", "last28", "trend"]
FEAT_NAMES = [f"{c}_{a}" for c in WEATHER_COLS for a in AGG]
FEAT_NAMES += ["prec_sum91", "prec_sum28", "prec_sum7", "prec_drydays91"]
FEAT_NAMES += ["month", "doy_sin", "doy_cos"]
FEAT_NAMES += ["reg_score_mean", "reg_score_std", "reg_score_median"]
F = len(FEAT_NAMES)
_CUM = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def md_of(datestr):
    """Robust month/day parse for variable-width years (e.g. 3004- or 58044-)."""
    p = datestr.split("-")
    return int(p[-2]), int(p[-1])


def window_features(Wm, mon, doy, reg_mean, reg_std, reg_med):
    """Wm: (k,91,NW); mon,doy: (k,). Returns (k,F) float32."""
    k = Wm.shape[0]
    mean = np.nanmean(Wm, axis=1); std = np.nanstd(Wm, axis=1)
    mn = np.nanmin(Wm, axis=1); mx = np.nanmax(Wm, axis=1)
    last = Wm[:, -1, :]
    last7 = np.nanmean(Wm[:, -7:, :], axis=1)
    last28 = np.nanmean(Wm[:, -28:, :], axis=1)
    trend = np.nanmean(Wm[:, -14:, :], axis=1) - np.nanmean(Wm[:, :14, :], axis=1)
    agg = np.stack([mean, std, mn, mx, last, last7, last28, trend], axis=2).reshape(k, NW * len(AGG))

    prec = Wm[:, :, PREC_IDX]
    precf = np.hstack([
        np.nansum(prec, axis=1, keepdims=True),
        np.nansum(prec[:, -28:], axis=1, keepdims=True),
        np.nansum(prec[:, -7:], axis=1, keepdims=True),
        (np.nan_to_num(prec, nan=0.0) <= 0.01).sum(axis=1, keepdims=True).astype(np.float64),
    ])
    season = np.stack([mon.astype(np.float64),
                       np.sin(2 * np.pi * doy / 365.25),
                       np.cos(2 * np.pi * doy / 365.25)], axis=1)
    clim = np.tile([reg_mean, reg_std, reg_med], (k, 1))
    return np.hstack([agg, precf, season, clim]).astype(np.float32)


def iter_blocks(path, has_score):
    """Yield (region, mon, day, W, sc) numpy arrays for each contiguous block."""
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        ci = {c: i for i, c in enumerate(header)}
        widx = [ci[c] for c in WEATHER_COLS]
        sidx = ci.get("score", -1); didx = ci["date"]; ridx = ci["region_id"]
        cur = None; mons = []; days = []; ws = []; scs = []
        for row in r:
            reg = row[ridx]
            if reg != cur:
                if cur is not None:
                    yield cur, mons, days, ws, scs
                cur = reg; mons = []; days = []; ws = []; scs = []
            m, d = md_of(row[didx])
            mons.append(m); days.append(d)
            ws.append([float(row[j]) if row[j] != "" else np.nan for j in widx])
            scs.append(float(row[sidx]) if (has_score and row[sidx] != "") else np.nan)
        if cur is not None:
            yield cur, mons, days, ws, scs


def load_region_series(path, has_score):
    """Reassemble each region's full series by concatenating its blocks in file order."""
    data = {}; order = []
    for reg, mons, days, ws, scs in iter_blocks(path, has_score):
        blk = (np.asarray(mons, np.int16), np.asarray(days, np.int16),
               np.asarray(ws, np.float32), np.asarray(scs, np.float32))
        if reg not in data:
            data[reg] = []; order.append(reg)
        data[reg].append(blk)
    out = {}
    for reg in order:
        blks = data[reg]
        mon = np.concatenate([b[0] for b in blks])
        day = np.concatenate([b[1] for b in blks])
        W = np.concatenate([b[2] for b in blks]).astype(np.float64)
        sc = np.concatenate([b[3] for b in blks]).astype(np.float64)
        out[reg] = (mon, day, W, sc)
    return out, order


def build_train():
    series, order = load_region_series(os.path.join(DATA_DIR, "train.csv"), True)
    Xs, ys, ods, ridxs = [], [], [], []
    region_stats = {}; GLOBAL_S = []; n_samp = 0
    for ridx, reg in enumerate(order):
        mon, day, W, sc = series[reg]
        T = len(sc)
        valid = sc[~np.isnan(sc)]
        if valid.size == 0:
            continue
        rmean = float(valid.mean()); rstd = float(valid.std()); rmed = float(np.median(valid))
        region_stats[reg] = [rmean, rstd, rmed, int(valid.size)]
        GLOBAL_S.append(valid)

        score_pos = np.where(~np.isnan(sc))[0]
        anchors = score_pos[::STRIDE]
        anchors = anchors[(anchors >= WINDOW - 1) & (anchors + HORIZONS[-1] < T)]
        if anchors.size == 0:
            continue
        tpos = anchors[:, None] + HORIZONS[None, :]
        good = anchors[~np.isnan(sc[tpos]).any(axis=1)]
        if good.size == 0:
            continue
        idx = good[:, None] - np.arange(WINDOW - 1, -1, -1)[None, :]
        Wm = W[idx]
        doy = _CUM[mon[good].astype(int) - 1] + day[good]
        X = window_features(Wm, mon[good].astype(float), doy, rmean, rstd, rmed)
        Y = sc[good[:, None] + HORIZONS[None, :]].astype(np.float32)
        Xs.append(X); ys.append(Y)
        ods.append(good.astype(np.int32)); ridxs.append(np.full(good.size, ridx, np.int32))
        n_samp += good.size
        if (ridx + 1) % 400 == 0:
            print(f"[train] regions={ridx+1} samples={n_samp}", flush=True)

    X = np.concatenate(Xs); Y = np.concatenate(ys)
    OD = np.concatenate(ods); RI = np.concatenate(ridxs)
    np.save(os.path.join(ART, "train_X.npy"), X)
    np.save(os.path.join(ART, "train_y.npy"), Y)
    np.save(os.path.join(ART, "train_anchord.npy"), OD)
    np.save(os.path.join(ART, "train_regidx.npy"), RI)
    gall = np.concatenate(GLOBAL_S)
    region_stats["__global__"] = [float(gall.mean()), float(gall.std()), float(np.median(gall)), int(gall.size)]
    with open(os.path.join(ART, "region_stats.json"), "w") as f:
        json.dump(region_stats, f)
    print(f"[train] DONE regions={len(order)} samples={X.shape[0]} feats={X.shape[1]} "
          f"y_mean={Y.mean():.4f}", flush=True)
    return region_stats


def build_test(region_stats):
    gmean, gstd, gmed = region_stats["__global__"][:3]
    series, order = load_region_series(os.path.join(DATA_DIR, "test.csv"), False)
    rows_out = []; regs = []
    for reg in order:
        mon, day, W, sc = series[reg]
        if W.shape[0] != WINDOW:
            W = W[-WINDOW:]; mon = mon[-WINDOW:]; day = day[-WINDOW:]
        rs = region_stats.get(reg, [gmean, gstd, gmed])
        doy = np.array([_CUM[int(mon[-1]) - 1] + int(day[-1])])
        X = window_features(W[None, :, :], np.array([float(mon[-1])]), doy, rs[0], rs[1], rs[2])
        rows_out.append(X[0]); regs.append(reg)
    Xt = np.vstack(rows_out).astype(np.float32)
    np.save(os.path.join(ART, "test_X.npy"), Xt)
    with open(os.path.join(ART, "test_regions.json"), "w") as f:
        json.dump(regs, f)
    print(f"[test] DONE regions={len(regs)} feats={Xt.shape[1]}", flush=True)


if __name__ == "__main__":
    print(f"[preprocess] stride={STRIDE} window={WINDOW} F={F}", flush=True)
    rs = build_train()
    build_test(rs)
    with open(os.path.join(ART, "feature_meta.json"), "w") as f:
        json.dump({"feature_names": FEAT_NAMES, "window": WINDOW,
                   "horizons": HORIZONS.tolist(), "stride": STRIDE}, f)
    print("[preprocess] all artifacts written to artifacts/", flush=True)
