"""Forecasts of trained models for evaluation (D4: --part val; D5: --part test, only with
DW_D5_TEST=1 after the author's "go").

Runs evaluated: per model set and family, the selected configuration's 3 seeds
(results/d3/selection.json), the config-matched references, H1c dw_line_k and the global
sensitivity runs. For each run: forecasts at all four horizon clips of every field of the part
(H1c: the held-out line's single-drug fields of drugs measured in all lines; validation fold =
the validation line, test folds = the test lines); dw_none runs also forecast every fit field
(for the residual baseline); H1 dw_full runs also forecast each (plate, drug x line) of the
part at 41 log-spaced candidate doses over that drug x line's measured range (dose inversion,
G-Plan a). Primary-normalisation dw_full and dw_none runs also forecast from the field's own
first clip (secondary start, A1 C1: `pred_own`). Outputs data/preds/<part>/<run>.npz (not
committed; backed up).

  PYTHONPATH=src python scripts/60_predict.py --part val --gpus 0-7
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys

import numpy as np

from drugworld import REPO, backup
from drugworld import evaluate as E
from drugworld import train as TR

print = functools.partial(print, flush=True)
H1C_DRUGS = ("Doxorubicin", "Gemcitabine", "Paclitaxel", "Palbociclib")
N_CAND = 41


def runs():
    sel = json.loads((REPO / "results" / "d3" / "selection.json").read_text())["selected"]
    out = []
    for p in sorted((REPO / "results" / "d3" / "train").glob("*.json")):
        r = json.loads(p.read_text())
        key = f"{r['set']}|{r['family']}"
        own = key in sel and r["config"] == sel[key]
        matched = (r["set"] in ("h1", "h2a") and r["config"] == sel[f"{r['set']}|dw_full"]) \
            or (r["set"] == "h2b" and any(r["config"] == sel[f"h2b|dw_full:{s}"] and
                                          (r["family"] in ("dw_none", f"dw_structshuf:{s}"))
                                          for s in ("ecfp4", "molformer", "target")))
        if own or matched or r["set"].startswith("line:"):
            out.append(r)
    return out


def part_mask(d, model_set, role, part):
    f = d.fields
    if model_set.startswith("line:"):
        L = model_set.split(":", 1)[1]
        val_line = TR.protocol()["splits"]["idr0119"]["val_line"]
        if (part == "val") != (L == val_line):
            return np.zeros(len(f), bool)
        return ((f.line == L) & (f.drug2 == "") & f.drug1.isin(H1C_DRUGS)).to_numpy()
    return role == part


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["val", "test"], required=True)
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--nworkers", type=int, default=1)
    args = ap.parse_args()
    if args.part == "test" and os.environ.get("DW_D5_TEST") != "1":
        sys.exit("test forecasts are D5 only (set DW_D5_TEST=1 after the author's go)")
    rs = runs()
    if args.worker < 0:                          # launcher: one worker per GPU
        a, b = map(int, args.gpus.split("-"))
        n = b - a + 1
        procs = [subprocess.Popen([sys.executable, __file__, "--part", args.part, "--worker",
                                   str(k), "--nworkers", str(n)], cwd=REPO,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(a + k)})
                 for k in range(n)]
        rc = [p.wait() for p in procs]
        print("workers rc", rc, "runs", len(rs))
        return int(any(rc))
    prot = TR.protocol()
    out_dir = REPO / "data" / "preds" / args.part
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for r in rs[args.worker::args.nworkers]:
        path = out_dir / f"{r['name']}.npz"
        if path.exists():
            continue
        which = TR.dataset_of(r["set"])
        key = (which, r["norm"])
        if key not in cache:
            cache[key] = TR.load(which, r["norm"], "vehicle")
        d = cache[key]
        hz = prot["sets"][which]["clips"]["horizons"]
        clips = [v["clip_index"] for v in hz.values()]
        role = TR.roles(d, r["set"]).to_numpy()
        model, ck = E.load_model(REPO / "checkpoints" / "d3" / r["checkpoint"])
        fam = "dw_full" if r["family"] == "dw_line" else r["family"]
        act = TR.Actions(d, fam, r["seed"])
        idx = np.flatnonzero(part_mask(d, r["set"], role, args.part))
        res = {"idx": idx, "pred": E.forecast(model, d.z, act, idx, clips),
               "horizons": np.array(list(hz))}
        if fam == "dw_none":
            fit = np.flatnonzero(role == "fit")
            res["fit_idx"], res["fit_pred"] = fit, E.forecast(model, d.z, act, fit, clips)
        if r["set"] == "h1" and fam == "dw_full" and r["norm"] == "primary":
            res.update(candidates(d, act, model, idx, clips))
        if fam in ("dw_full", "dw_none") and r["norm"] == "primary":
            ko = (which, "own")
            if ko not in cache:
                cache[ko] = TR.load(which, "primary", "own")
            res["pred_own"] = E.forecast(model, cache[ko].z, act, idx, clips)
        np.savez(path, **res)
        backup.backup(path)
        print(f"worker {args.worker}: {r['name']} {len(idx)} fields")
    return 0


def candidates(d, act, model, idx, clips):
    """Forecasts at N_CAND log-spaced doses per (plate, line, drug) of the part's fields."""
    f = d.fields
    keys = sorted({(f.plate[i], f.line[i], f.drug1[i]) for i in idx})
    s = f[(f.drug1 != "") & (f.drug2 == "")]
    grid = s.groupby(["line", "drug1"]).dose1.agg(lambda x: sorted({float(v) for v in x}))
    e = np.load(REPO / "encodings" / "idr0119.npz", allow_pickle=True)
    dmax = dict(zip(e["drugs"], e["dose_max_nM"]))
    rep, doses, preds = [], [], []
    for plate, line, drug in keys:
        g = grid[(line, drug)]
        cand = np.logspace(np.log10(g[0]), np.log10(g[-1]), N_CAND)
        i0 = next(i for i in idx if f.plate[i] == plate and f.drug1[i] == drug)
        a = TR.Actions.__new__(TR.Actions)
        a.toks = np.repeat(act.toks[[i0]], N_CAND, 0)
        a.toks[:, 0, -1] = np.log10(cand / dmax[drug])
        a.tmask, a.vehicle, a.redose = (np.repeat(x[[i0]], N_CAND, 0) for x in
                                        (act.tmask, act.vehicle, act.redose))
        z0 = np.repeat(d.z[[i0]], N_CAND, 0)
        preds.append(E.forecast(model, z0, a, np.arange(N_CAND), clips))
        rep.append(f"{plate}|{line}|{drug}")
        doses.append(cand)
    return {"cand_keys": np.array(rep), "cand_doses": np.array(doses),
            "cand_pred": np.stack(preds)}


if __name__ == "__main__":
    raise SystemExit(main())
