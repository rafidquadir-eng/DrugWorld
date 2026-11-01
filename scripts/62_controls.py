"""Controls (plan section 9), reported and never used to change a gate; fixed before any
evaluation output. --part val (D4) or test (D5, DW_D5_TEST=1). Writes results/d4/controls_val.json
(or results/d5/controls_test.json).

1. Static-clip control: the selected dw_full forecasts (3 seeds) and copy-last / slot-mean are
   scored against the static-clip latents (each clip = its middle frame repeated, embedded in
   D1, normalised with the same statistics); the model / copy-last error ratio is reported for
   measured and static targets.
2. Context swap: dw_full rolled out from the own first clip of a field of another condition on
   another plate (seeded, seed 0), against its own-first-clip forecast (`pred_own`).
3. Evaluation-time action shuffle: H1 dw_full with doses permuted within each drug x line (seed
   99), H2b dw_full:<src> with encodings permuted across compounds (seed 99); error change.
4. Leakage: (a) identical frame-0 SHA-1 in wells of different split roles; (b) fingerprint
   pairs with Pearson r > 0.98 across split roles (GeneWorld rule); (c) parent InChIKeys in more
   than one idr0064 split; (d) scaffold integrity: no test or validation scaffold in training.
5. Plate-batch probe: for H1 training conditions with >= 3 replicate plates, logistic
   regression (standardised 48 h normalised tile-mean latent, PCA 64) predicting the plate
   from fields, 4-fold stratified CV; mean accuracy against chance (1 / plates).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drugworld import REPO, backup, latents
from drugworld import evaluate as E
from drugworld import train as TR

PRIMARY = {"h1": ["48h", "72h"], "h2a": ["24h", "48h"], "h2b": ["2h", "3h", "4h"]}
SRC = ("ecfp4", "molformer", "target")


def static_z(d):
    f, arr, _ = latents.load(d.which, arrays=("static_mean",))
    info = f.merge(d.fields[["plate", "well", "field"]].reset_index(), on=["plate", "well",
                                                                           "field"])
    X = arr["static_mean"].astype(np.float32)
    if d.which == "idr0119":
        n = np.load(REPO / "norm" / "idr0119_line_vehicle.npz")
        li = np.array([list(n["lines"]).index(x) for x in d.fields.line.iloc[info["index"]]])
        X = (X - n["mean"][li][:, :, None]) / n["sd"][li][:, :, None]
    else:
        n = np.load(REPO / "norm" / "idr0064_plate_median.npz")
        pi = np.array([list(n["plates"]).index(p) for p in d.fields.plate.iloc[info["index"]]])
        X = (X - n["median"][pi][:, :, None]) / n["sd"]
    out = np.zeros_like(X)
    out[info["index"].to_numpy()] = X
    return out


def runs_of(sel, ms, fam):
    rs = [json.loads(p.read_text()) for p in (REPO / "results" / "d3" / "train").glob("*.json")]
    return sorted([r for r in rs if r["set"] == ms and r["family"] == fam and r["norm"] ==
                   "primary" and r["config"] == sel[f"{ms}|{fam}"]], key=lambda r: r["seed"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["val", "test"], required=True)
    args = ap.parse_args()
    if args.part == "test" and os.environ.get("DW_D5_TEST") != "1":
        sys.exit("test controls are D5 only")
    prot = TR.protocol()
    sel = json.loads((REPO / "results" / "d3" / "selection.json").read_text())["selected"]
    out = {"part": args.part, "static": {}, "context_swap": {}, "action_shuffle": {}}
    for ms, fams in (("h1", ["dw_full"]), ("h2a", ["dw_full"]),
                     ("h2b", [f"dw_full:{s}" for s in SRC])):
        which = TR.dataset_of(ms)
        d = TR.load(which, "primary", "vehicle")
        d_own = TR.load(which, "primary", "own")
        hz = list(prot["sets"][which]["clips"]["horizons"])
        clips = [prot["sets"][which]["clips"]["horizons"][h]["clip_index"] for h in hz]
        pp = [hz.index(h) for h in PRIMARY[ms]]
        role = TR.roles(d, ms).to_numpy()
        idx = np.flatnonzero(role == args.part)
        zs = static_z(d)
        f = d.fields
        for fam in fams:
            rs = runs_of(sel, ms, fam)
            P = [dict(np.load(REPO / "data" / "preds" / args.part / f"{r['name']}.npz"))
                 for r in rs]
            pred = np.mean([p["pred"].astype(np.float32) for p in P], 0)
            copy = np.repeat(d.z[idx, :1].astype(np.float32), len(clips), 1)
            slot = E.common_baselines(d, role, idx, clips)["slot_mean"]
            meas, stat = d.z[idx][:, clips], zs[idx][:, clips]
            res = {}
            for name, truth in (("measured", meas), ("static", stat)):
                e = {k: float(E.errors(v, truth)[:, pp].mean()) for k, v in
                     (("model", pred), ("copy_last", copy), ("slot_mean", slot))}
                e["model_over_copy"] = e["model"] / e["copy_last"]
                res[name] = e
            out["static"][f"{ms}|{fam}"] = res
            # context swap and evaluation-time shuffle (GPU)
            rng = np.random.default_rng(0)
            cond = (f.drug1 + "|" + f.dose1.astype(str) + "|" + f.drug2 + "|" +
                    f.dose2.astype(str)) if which == "idr0119" else f.compound_key.astype(str)
            swap_src = []
            for i in idx:
                cand = np.flatnonzero((cond != cond.iloc[i]).to_numpy() &
                                      (f.plate != f.plate.iloc[i]).to_numpy() &
                                      ((f.line == f.line.iloc[i]).to_numpy() if which ==
                                       "idr0119" else True))
                swap_src.append(cand[rng.integers(len(cand))])
            zsw = d_own.z.copy()
            zsw[idx, 0] = d_own.z[np.array(swap_src), 0]
            sw, own, shuf = [], [], []
            for r, p in zip(rs, P):
                model, _ = E.load_model(REPO / "checkpoints" / "d3" / r["checkpoint"])
                act = TR.Actions(d, "dw_full" if ms != "h2b" else fam, r["seed"])
                sw.append(E.errors(E.forecast(model, zsw, act, idx, clips), meas)[:, pp].mean())
                own.append(E.errors(p["pred_own"].astype(np.float32), meas)[:, pp].mean())
                a2 = shuffled(act, f, which, idx)
                shuf.append(E.errors(E.forecast(model, d.z, a2, idx, clips), meas)[:, pp]
                            .mean())
                del model
                torch.cuda.empty_cache()
            base = float(E.errors(pred, meas)[:, pp].mean())
            out["context_swap"][f"{ms}|{fam}"] = {"own_start": float(np.mean(own)),
                                                 "swapped_start": float(np.mean(sw))}
            out["action_shuffle"][f"{ms}|{fam}"] = {"model": base,
                                                   "shuffled_actions": float(np.mean(shuf)),
                                                   "kind": "dose within drug x line"
                                                   if which == "idr0119" else
                                                   "encodings across compounds"}
        print(ms, out["static"].get(f"{ms}|{fams[0]}"))
    out["leakage"] = leakage()
    out["plate_batch_probe"] = plate_probe(prot)
    path = REPO / "results" / ("d4" if args.part == "val" else "d5") / \
        f"controls_{args.part}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    backup.backup(path)
    print(json.dumps(out["leakage"], indent=1, default=str)[:2000])
    return 0


def shuffled(act, f, which, idx):
    a = TR.Actions.__new__(TR.Actions)
    a.toks, a.tmask = act.toks.copy(), act.tmask
    a.vehicle, a.redose = act.vehicle, act.redose
    rng = np.random.default_rng(99)
    if which == "idr0119":
        for (line, drug), g in f.iloc[idx].groupby(["line", "drug1"]):
            rows = g.index.to_numpy()
            doses = act.toks[rows, 0, -1]
            u = np.unique(doses)
            perm = dict(zip(u, rng.permutation(u)))
            a.toks[rows, 0, -1] = [perm[x] for x in doses]
    else:
        rows = idx[act.tmask[idx, 0]]
        a.toks[rows, 0] = act.toks[rows[rng.permutation(len(rows))], 0]
    return a


def leakage():
    out = {}
    for which, col in (("idr0119", "h1_role"), ("idr0064", "role")):
        f, arr, _ = latents.load(which, arrays=("fingerprint",))
        w = pd.read_csv(REPO / "splits" / f"{which}_wells.csv", dtype={"plate": str},
                        keep_default_na=False)
        info = f.merge(w[["plate", "well", col]], on=["plate", "well"], how="left")
        r = info[col].to_numpy()
        sha = info.groupby("frame0_sha1")
        out[f"{which}_frame0_sha1_across_roles"] = int(sum(g[col].nunique() > 1 for _, g in sha))
        x = arr["fingerprint"].astype(np.float64)
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12)
        n = 0
        for i in range(0, len(x), 512):
            c = x[i:i + 512] @ x.T / x.shape[1]
            ii, jj = np.nonzero(c > 0.98)
            gi = ii + i
            m = (gi < jj) & (r[gi] != r[jj]) & ~((info.plate.to_numpy()[gi] ==
                                                   info.plate.to_numpy()[jj]) &
                                                  (info.well.to_numpy()[gi] ==
                                                   info.well.to_numpy()[jj]))
            n += int(m.sum())
        out[f"{which}_fingerprint_pairs_r_gt_0.98_across_roles"] = n
    w = pd.read_csv(REPO / "splits" / "idr0064_wells.csv", dtype={"plate": str},
                    keep_default_na=False)
    c = w[w.role.isin(["train", "val", "test"])]
    out["idr0064_parent_inchikey_in_two_splits"] = sorted(
        k for k, g in c.groupby("inchikey_parent") if g.role.nunique() > 1)
    tr = set(c.scaffold[c.role == "train"])
    out["idr0064_heldout_scaffolds_in_train"] = sorted(
        s for s in set(c.scaffold[c.role.isin(["val", "test"])]) if s in tr)
    return out


def plate_probe(prot):
    d = TR.load("idr0119", "primary", "vehicle")
    c48 = prot["sets"]["idr0119"]["clips"]["horizons"]["48h"]["clip_index"]
    f = d.fields
    tr = f[f.h1_role == "train"]
    accs = []
    for _, g in tr.groupby(["line", "drug1", "dose1"]):
        if g.plate.nunique() < 3 or g.plate.value_counts().min() < 4:
            continue
        X = d.z[g.index.to_numpy(), c48].astype(np.float32).mean(1)
        y = g.plate.to_numpy()
        pca = PCA(min(64, (3 * len(y)) // 4 - 1), random_state=0)   # < training-fold size
        pipe = make_pipeline(StandardScaler(), pca,
                             LogisticRegression(max_iter=2000))
        acc = cross_val_score(pipe, X, y, cv=StratifiedKFold(4, shuffle=True, random_state=0))
        accs.append((float(acc.mean()), 1 / g.plate.nunique()))
    a = np.array(accs)
    return {"n_conditions": int(len(a)), "mean_accuracy": float(a[:, 0].mean()) if len(a)
            else None, "mean_chance": float(a[:, 1].mean()) if len(a) else None}


if __name__ == "__main__":
    raise SystemExit(main())
