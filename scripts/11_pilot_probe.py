"""D0 pilot probes: choose the idr0119 channel mapping and pyramid level, and give the
provisional F-Decode values on vehicle (training) wells.

Rule (fixed before any probe result is computed, committed with this script):
  * targets per (field, clip, tile): idr0119 log(1 + mean cells per frame in the tile) and the
    G1 fraction (cells over the clip's 16 frames whose log HDHB nuclear/ring ratio exceeds the
    threshold T); idr0064 the mean Erk C/N ratio over the clip's cells in the tile;
  * T is the density valley of a two-component Gaussian mixture of all pilot cells' log ratios
    (seed 0, 200,000 cells sampled with seed 0) and is recorded;
  * probe = ridge regression on the standardised tile-mean latent (alpha by inner 5-fold CV over
    10^-1 .. 10^5), evaluated by 5-fold GroupKFold over plates; R^2 of the pooled held-out
    predictions; tiles without a cell are dropped for the G1 and Erk targets;
  * the idr0119 configuration (mapping x level) with the highest mean of the two R^2 values is
    chosen; ties go to the cheaper transfer (phase only), then to level 1;
  * provisional F-Decode (plan section 3.4) for the chosen configuration on these vehicle wells:
    count R^2 >= 0.8 and G1 R^2 >= 0.5 (idr0119), Erk C/N R^2 >= 0.5 (idr0064).

Writes results/d0/pilot_probe.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drugworld import REPO
from drugworld import readouts as R

PILOT = REPO / "data" / "pilot"
OUT = REPO / "results" / "d0" / "pilot_probe.json"
NF = 16
L0 = [(y, x, 256) for y in (8, 264, 520, 776) for x in (64, 320, 576, 832, 1088)]
L1 = [(2 * y, 2 * x, 512) for y in (4, 260) for x in (96, 352)]
E0 = [(y, x, 256) for y in (0, 256) for x in (0, 256)]
BOXES = {"phase_L0": L0, "rgb_L0": L0, "phase_L1": L1, "rgb_L1": L1, "erk_L0": E0}


def targets(cells, starts, boxes, thr=None):
    """cells (n, 4): t, cy, cx, value -> per (clip, tile): log count, fraction > thr / mean."""
    t, cy, cx, v = cells.T
    n_c, n_t = len(starts), len(boxes)
    cnt = np.zeros((n_c, n_t))
    val = np.full((n_c, n_t), np.nan)
    for j, (y, x, s) in enumerate(boxes):
        inb = (cy >= y) & (cy < y + s) & (cx >= x) & (cx < x + s)
        for i, s0 in enumerate(starts):
            m = inb & (t >= s0) & (t < s0 + NF)
            cnt[i, j] = m.sum() / NF
            vv = v[m]
            vv = vv[np.isfinite(vv)]
            if len(vv):
                val[i, j] = (vv > thr).mean() if thr is not None else vv.mean()
    return np.log1p(cnt), val


def probe(X, y, groups):
    ok = np.isfinite(y)
    X, y, groups = X[ok], y[ok], groups[ok]
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return {"r2": float(r2_score(y, pred)), "n": int(len(y)),
            "n_plates": int(len(np.unique(groups))), "target_sd": float(y.std())}


def main() -> int:
    res = {"rule": __doc__.split("Rule")[1].split("Writes")[0].strip()}
    # ---- idr0119
    files = sorted((PILOT / "idr0119").glob("*.npz"))
    D = [np.load(f) for f in files]
    plates = [f.name.split("_")[0] for f in files]
    allv = np.concatenate([d["cells"][:, 3] for d in D])
    allv = allv[np.isfinite(allv)]
    rng = np.random.default_rng(0)
    thr = R.gmm_valley(rng.choice(allv, min(200_000, len(allv)), replace=False))
    r0119 = {"fields": len(files), "plates": len(set(plates)), "g1_log_ratio_threshold": thr,
             "cells": int(sum(len(d["cells"]) for d in D)), "configs": {}}
    for cfg in ("phase_L0", "phase_L1", "rgb_L0", "rgb_L1"):
        X, yc, yg, g = [], [], [], []
        for d, p in zip(D, plates):
            lc, gf = targets(d["cells"], d["starts"], BOXES[cfg], thr)
            X.append(d[cfg].reshape(-1, d[cfg].shape[-1]).astype(np.float32))
            yc.append(lc.ravel())
            yg.append(gf.ravel())
            g += [p] * lc.size
        X, yc, yg, g = np.concatenate(X), np.concatenate(yc), np.concatenate(yg), np.array(g)
        c, gg = probe(X, yc, g), probe(X, yg, g)
        r0119["configs"][cfg] = {"log_count": c, "g1_fraction": gg,
                                 "mean_r2": (c["r2"] + gg["r2"]) / 2}
        print(cfg, round(c["r2"], 3), round(gg["r2"], 3))
    cost = {"phase_L1": 0, "phase_L0": 1, "rgb_L1": 2, "rgb_L0": 3}
    best = max(r0119["configs"], key=lambda k: (round(r0119["configs"][k]["mean_r2"], 6),
                                                -cost[k]))
    b = r0119["configs"][best]
    r0119["chosen"] = best
    r0119["f_decode_provisional"] = {
        "count_r2": b["log_count"]["r2"], "g1_r2": b["g1_fraction"]["r2"],
        "pass": bool(b["log_count"]["r2"] >= 0.8 and b["g1_fraction"]["r2"] >= 0.5)}
    res["idr0119"] = r0119
    # ---- idr0064
    files = sorted((PILOT / "idr0064").glob("*.npz"))
    D = [np.load(f) for f in files]
    plates = [f.name.split("_")[0] for f in files]
    X, ye, g = [], [], []
    for d, p in zip(D, plates):
        _, e = targets(d["cells"], d["starts"], BOXES["erk_L0"])
        X.append(d["erk_L0"].reshape(-1, d["erk_L0"].shape[-1]).astype(np.float32))
        ye.append(e.ravel())
        g += [p] * e.size
    pe = probe(np.concatenate(X), np.concatenate(ye), np.array(g))
    res["idr0064"] = {"fields": len(files), "plates": len(set(plates)), "erk_cn": pe,
                      "f_decode_provisional": {"erk_cn_r2": pe["r2"],
                                               "pass": bool(pe["r2"] >= 0.5)}}
    print("idr0064 erk", round(pe["r2"], 3))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k].get("f_decode_provisional") for k in ("idr0119", "idr0064")}),
          "chosen", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
