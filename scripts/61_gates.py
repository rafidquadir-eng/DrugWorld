"""Gates on one part (D4: --part val -> results/d4/gates_val.json; D5: --part test, only with
DW_D5_TEST=1 -> results/d5/gates_test.json). Rules: plan sections 7-8, manifests/
drugworld_protocol.json, src/drugworld/gates.py (all fixed before this script first ran).

Per model set, units are the held-out conditions (H1 drug x line x dose, H2a combination
condition, H2b compound, H1c drug x dose of the held-out line). A model's unit error is the mean
over its 3 seeds of the unit's primary error (`evaluate.condition_errors`). Comparisons use the
config-matched references (dw_none and shuffled families at dw_full's configuration); each
family's own selected configuration is reported beside. Baseline strength (L3) is computed on
validation and, in D5, read from results/d4/gates_val.json (fixed on validation).

Readout space (idr0119, only if F-Decode passed, results/d2/feasibility.json): measured field
readouts are tile means of the frozen targets; forecasts are decoded with the frozen decoders.
Hill (H1) on measured training readouts; Bliss on decoded cell-count fractions; the ODE model
(results/d3/ode_fit.json) scaled by the plate's measured vehicle count at clip 0. G-Combo+ is on
the 72 h decoded log cell count (tile mean of log(1 + cells per frame)).
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drugworld import REPO, backup, latents, ode
from drugworld import evaluate as E
from drugworld import gates as G
from drugworld import targets as T
from drugworld import train as TR

print = functools.partial(print, flush=True)
PRIMARY = {"h1": ["48h", "72h"], "h2a": ["24h", "48h"], "h2b": ["2h", "3h", "4h"]}
UNITS = {"h1": ["line", "drug1", "dose1"], "h2a": ["drug1", "dose1", "drug2", "dose2"],
         "h2b": ["compound_key"], "line": ["drug1", "dose1"]}
SRC = ("ecfp4", "molformer", "target")


class Ctx:
    def __init__(self, part, norm):
        self.part, self.norm = part, norm
        self.prot = TR.protocol()
        self.sel = json.loads((REPO / "results" / "d3" / "selection.json").read_text())[
            "selected"]
        self.runs = [json.loads(p.read_text()) for p in
                     sorted((REPO / "results" / "d3" / "train").glob("*.json"))]
        self.data = {}

    def d(self, which):
        if which not in self.data:
            self.data[which] = TR.load(which, self.norm, "vehicle")
        return self.data[which]

    def preds(self, ms, fam, config):
        rs = [r for r in self.runs if r["set"] == ms and r["family"] == fam and
              r["config"] == config and r["norm"] == self.norm]
        out = {}
        for r in sorted(rs, key=lambda r: r["seed"]):
            p = REPO / "data" / "preds" / self.part / f"{r['name']}.npz"
            out[r["seed"]] = dict(np.load(p, allow_pickle=True))
        return out


def unit_err(d, idx, pred, clips, pp, units):
    truth = d.z[idx][:, clips]
    e = E.errors(pred, truth)[:, pp].mean(1, keepdims=True)
    ce = E.condition_errors(d.fields.iloc[idx].reset_index(drop=True), e, units)
    return ce.set_index(units)["h0"]


def model_err(ctx, d, ms, fam, config, idx, clips, pp, units, key="pred"):
    P = ctx.preds(ms, fam, config)
    if len(P) != 3:
        return None, len(P)
    s = [unit_err(d, idx, P[k][key].astype(np.float32), clips, pp, units) for k in sorted(P)]
    return pd.concat(s, axis=1).mean(1), 3


def residual_baseline(ctx, d, ms, role, idx, clips, pp, units, config):
    """GeneWorld R2 form, per dw_none seed; lambda chosen on validation."""
    P = ctx.preds(ms, "dw_none", config)
    if len(P) != 3:
        return None, None
    f = d.fields
    if ms == "h2b":
        comps, bits, train = E.knn_table(d, role)
        tb = bits[[comps.index(c) for c in train]]
    per_lam = defaultdict(list)
    for k in sorted(P):
        p = P[k]
        fit_idx, fit_pred = p["fit_idx"], p["fit_pred"].astype(np.float32)
        res = d.z[fit_idx][:, clips].astype(np.float32) - fit_pred
        ff = f.iloc[fit_idx].reset_index(drop=True)       # row i <-> res[i]
        if ms == "h2b":
            tab = {c: res[(ff.compound_key == c).to_numpy()].mean(0) for c in train}
        else:
            single = ff[(ff.drug1 != "") & (ff.drug2 == "")]
            tab = {}
            for (L, dr, ds), g in single.groupby(["line", "drug1", "dose1"]):
                tab.setdefault((L, dr), {})[float(ds)] = res[g.index.to_numpy()].mean(0)
        near = np.zeros((len(idx),) + res.shape[1:], np.float32)
        for i, r in enumerate(f.iloc[idx].itertuples()):
            if ms == "h2b":
                s = E.tanimoto(bits[[comps.index(r.compound_key)]], tb)[0]
                near[i] = tab[train[sorted(range(len(train)), key=lambda j: (-s[j], train[j]))
                                    [0]]]
            else:
                near[i] = E.nearest(tab[(r.line, r.drug1)], float(r.dose1))[0]
        for lam in E.LAMBDAS:
            per_lam[lam].append(unit_err(d, idx, p["pred"].astype(np.float32) + lam * near,
                                         clips, pp, units))
    means = {lam: pd.concat(v, axis=1).mean(1) for lam, v in per_lam.items()}
    return means, {lam: float(m.mean()) for lam, m in means.items()}


# ---------------------------------------------------------------- readouts (idr0119)

_CELLS = {}


def measured_readouts(ctx, d, idx, clips):
    """(n, H) log count and G1 fraction per field (tile means) from the per-cell readouts."""
    if not _CELLS:
        _CELLS["v"] = latents.load("idr0119", arrays=(), cells=True)
    f, arr, meta = _CELLS["v"]
    key = {(p, w, fl): i for i, (p, w, fl) in enumerate(zip(f.plate, f.well, f.field))}
    bx = T.boxes("idr0119", meta["tiles"], meta["level"])
    thr = ctx.prot["sets"]["idr0119"]["decoder_params"]["g1_threshold"]
    starts = [meta["starts"][c] for c in clips]
    lc = np.zeros((len(idx), len(clips)))
    g1 = np.zeros_like(lc)
    for n, i in enumerate(idx):
        r = d.fields.iloc[i]
        o = T.idr0119_targets(arr["cells"][key[(r.plate, r.well, r.field)]], starts, bx, thr)
        lc[n] = o["log_count"].mean(1)
        g1[n] = np.nanmean(o["g1_fraction"], 1)
    return {"log_count": lc, "g1_fraction": g1}


def dec(pred):
    return E.decode(pred, "idr0119", "log_count").mean(-1)


def readout_err(d, idx, pred_ro, truth_ro, pos, units):
    e = ((pred_ro - truth_ro) ** 2)[:, pos].mean(1, keepdims=True)
    ce = E.condition_errors(d.fields.iloc[idx].reset_index(drop=True), e, units)
    return ce.set_index(units)["h0"]


# ---------------------------------------------------------------- main

def evaluate_part(ctx, weak_from=None):
    prot, sel = ctx.prot, ctx.sel
    out = {"part": ctx.part, "norm": ctx.norm, "sets": {}}
    feas = json.loads((REPO / "results" / "d2" / "feasibility.json").read_text())
    decoded_ok = feas["f_decode"]["idr0119"]["pass"]
    weak_all = {}
    for ms in ("h1", "h2a", "h2b", "line"):
        which = "idr0064" if ms == "h2b" else "idr0119"
        d = ctx.d(which)
        hz = list(prot["sets"][which]["clips"]["horizons"])
        clips = [prot["sets"][which]["clips"]["horizons"][h]["clip_index"] for h in hz]
        pp = [hz.index(h) for h in PRIMARY["h1" if ms == "line" else ms]]
        units = UNITS[ms]
        if ms == "line":
            L = prot["splits"]["idr0119"]["val_line"] if ctx.part == "val" else None
            folds = [L] if L else prot["splits"]["idr0119"]["test_lines"]
            out["sets"]["h1c"] = {}
            for L in folds:
                out["sets"]["h1c"][L] = h1c(ctx, d, L, clips, pp, units)
            continue
        role = TR.roles(d, ms).to_numpy()
        idx = np.flatnonzero(role == ctx.part)
        fields = d.fields.iloc[idx].reset_index(drop=True)
        res = {"n_fields": int(len(idx)),
               "n_units": int(fields.drop_duplicates(units).shape[0]), "methods": {}}
        M = {}
        fams = {"h1": ["dw_full", "dw_none", "dw_doseshuf", "dw_doseonly"],
                "h2a": ["dw_full", "dw_none"],
                "h2b": [f"dw_full:{s}" for s in SRC] + [f"dw_structshuf:{s}" for s in SRC] +
                       ["dw_none"]}[ms]
        for fam in fams:
            own = sel[f"{ms}|{fam}"]
            e, n = model_err(ctx, d, ms, fam, own, idx, clips, pp, units)
            if e is not None:
                M[f"{fam}@own"] = e
        refcfg = {"h1": sel["h1|dw_full"], "h2a": sel["h2a|dw_full"]}
        if ms in refcfg:
            for fam in fams[1:]:
                e, _ = model_err(ctx, d, ms, fam, refcfg[ms], idx, clips, pp, units)
                if e is not None:
                    M[f"{fam}@matched"] = e
        else:
            for s in SRC:
                c = sel[f"h2b|dw_full:{s}"]
                for fam in ("dw_none", f"dw_structshuf:{s}"):
                    e, _ = model_err(ctx, d, ms, fam, c, idx, clips, pp, units)
                    if e is not None:
                        M[f"{fam}@matched:{s}"] = e
        B = E.common_baselines(d, role, idx, clips)
        if ms == "h1":
            B.update(E.h1_baselines(d, role, idx, clips))
        elif ms == "h2a":
            B.update(E.h2a_baselines(d, role, idx, clips, pp, B["vehicle_traj"]))
        else:
            kb, nn = E.h2b_baselines(d, role, idx, clips)
            B.update(kb)
            res["knn_neighbours"] = nn
        for k, v in B.items():
            M[k] = unit_err(d, idx, v, clips, pp, units)
        if ms in ("h1", "h2b"):
            # agnostic forecast: h1 dw_none at dw_full's configuration; h2b (source-agnostic)
            # dw_none at its own selected configuration
            rcfg = refcfg["h1"] if ms == "h1" else sel["h2b|dw_none"]
            rb, lam_tab = residual_baseline(ctx, d, ms, role, idx, clips, pp, units, rcfg)
            if rb is not None:
                lam = (min(lam_tab, key=lambda x: (lam_tab[x], x)) if weak_from is None else
                       weak_from["sets"][ms]["residual_lambda"])
                M["residual"] = rb[lam]
                res["residual_lambda"], res["residual_lambda_table"] = lam, lam_tab
        drug = None
        if ms in ("h1",):
            drug = pd.Series([u[1] for u in M["copy_last"].index], index=M["copy_last"].index)
        # readout-space methods (idr0119)
        R = {}
        if which == "idr0119" and decoded_ok:
            R = readout_methods(ctx, d, ms, role, idx, clips, hz, pp, units, M, B)
        # baseline strength (L3)
        lookups = {"h1": ["nearest_dose", "interp", "residual"], "h2a": ["hsa", "additive"],
                   "h2b": ["knn", "residual"]}[ms]
        if weak_from is None:
            strength = {b: G.compare(M[b], M["copy_last"]) for b in lookups if b in M}
            strength.update({b: G.compare(R[b], R["copy_last"]) for b in ("hill", "bliss")
                             if b in R})
            weak = {b: not v["beats"] for b, v in strength.items()}
            res["baseline_strength"] = strength
        else:
            weak = weak_from["weak"].get(ms, {})
        weak_all[ms] = weak
        res["weak"] = weak
        res["methods"] = {k: {"mean": float(v.mean()), "n_units": int(v.notna().sum())}
                          for k, v in M.items()}
        res["readout_methods"] = {k: {"mean": float(v.mean()), "n_units": int(v.notna().sum())}
                                  for k, v in R.items()}
        res["gates"] = set_gates(ms, M, R, weak, drug, decoded_ok)
        res["unit_errors"] = {k: {"|".join(map(str, i if isinstance(i, tuple) else (i,))): float(x)
                                  for i, x in v.items()} for k, v in {**M, **{
                                      f"readout:{a}": b for a, b in R.items()}}.items()}
        if ms == "h1" and ctx.norm == "primary":
            res["g_plan_a"] = plan_a(ctx, d, role, idx, clips, pp)
        if ms == "h2a" and decoded_ok and ctx.norm == "primary":
            res["g_plan_b"] = plan_b(ctx, d, role, idx, clips, hz, R)
        if ctx.norm == "primary":
            res["own_start"] = own_start(ctx, d, ms, idx, clips, pp, units)
        out["sets"][ms] = res
        print(ms, {k: (v["pass"] if isinstance(v, dict) and "pass" in v else None)
                   for k, v in res["gates"].items()})
    out["weak"] = weak_all
    out["summary"] = summary(out, decoded_ok)
    return out


def set_gates(ms, M, R, weak, drug, decoded_ok):
    g = {}
    if ms == "h1":
        b, chain = G.substitute("nearest_dose", weak)
        comps = {"dw_none": M.get("dw_none@matched"), b: (R if b == "hill" else M).get(b),
                 "dw_doseshuf": M.get("dw_doseshuf@matched")}
        full = M["dw_full@own"]
        c = {k: G.compare(full if k != "hill" else R["dw_full@own"], v, drug)
             for k, v in comps.items() if v is not None}
        g["G-Dose"] = {"comparisons": c, "lookup_used": b, "chain": chain,
                       "pass": len(c) == 3 and all(x["beats"] for x in c.values())}
        b2, chain2 = G.substitute("interp", weak)
        src = R if b2 == "hill" else M
        c2 = G.compare(R["dw_full@own"] if b2 == "hill" else full, src[b2], drug) \
            if b2 in src else None
        g["G-Dose+"] = {"comparison": c2, "lookup_used": b2, "chain": chain2,
                        "pass": bool(g["G-Dose"]["pass"] and c2 and c2["beats"])}
        g["own_config_dw_doseshuf"] = G.compare(full, M["dw_doseshuf@own"]) \
            if "dw_doseshuf@own" in M else None
        g["own_config_dw_none"] = G.compare(full, M["dw_none@own"]) if "dw_none@own" in M \
            else None
        g["diagnostic_dw_doseonly"] = G.compare(full, M["dw_doseonly@own"]) \
            if "dw_doseonly@own" in M else None
    elif ms == "h2a":
        full = M["dw_full@own"]
        c = {}
        used = {}
        for name in ("hsa", "additive"):
            b, _ = G.substitute(name, weak)
            used[name] = b
            src = R if b == "bliss" else M
            c[name] = G.compare(R["dw_full@own:72h"] if b == "bliss" else full, src[b]) \
                if b in src else None
        g["G-Combo"] = {"comparisons": c, "lookups_used": used,
                        "pass": all(v and v["beats"] for v in c.values())}
        c2 = {}
        if decoded_ok:
            for b in ("bliss", "ode"):
                if f"{b}:72h" in R:
                    c2[b] = G.compare(R["dw_full@own:72h"], R[f"{b}:72h"])
        g["G-Combo+"] = {"comparisons": c2, "decoded_readouts_used": decoded_ok,
                         "pass": bool(g["G-Combo"]["pass"] and decoded_ok and len(c2) == 2 and
                                      all(v["beats"] for v in c2.values()))}
        g["without_Lapatinib+Palbociclib"] = {
            k: G.compare(full[[("Lapatinib" not in str(i) or "Palbociclib" not in str(i))
                               for i in full.index]], M[k]) for k in ("hsa", "additive")
            if k in M}
        g["own_config_dw_none"] = G.compare(full, M["dw_none@own"]) if "dw_none@own" in M \
            else None
    else:
        per = {}
        for s in SRC:
            full = M.get(f"dw_full:{s}@own")
            if full is None:
                continue
            b, chain = G.substitute("knn", weak)
            c = {"dw_structshuf": G.compare(full, M[f"dw_structshuf:{s}@matched:{s}"]),
                 b: G.compare(full, M[b]),
                 "dw_none": G.compare(full, M[f"dw_none@matched:{s}"])}
            per[s] = {"comparisons": c, "lookup_used": b,
                      "p_max": max(x["p_one_sided"] for x in c.values()),
                      "all_beat": all(x["beats"] for x in c.values())}
        h = G.holm({s: v["p_max"] for s, v in per.items()})
        g["G-Compound"] = {"per_source": per, "holm": h,
                           "pass": any(v["reject"] for v in h.values())}
    return g


def readout_methods(ctx, d, ms, role, idx, clips, hz, pp, units, M, B):
    """Readout-space errors: dw_full decoded, latent baselines decoded, Hill (H1),
    Bliss / ODE / HSA / additive at 72 h (H2a)."""
    truth = measured_readouts(ctx, d, idx, clips)["log_count"]
    R = {}
    full = ctx.preds(ms, "dw_full", ctx.sel[f"{ms}|dw_full"])
    fd = np.mean([dec(full[k]["pred"].astype(np.float32)) for k in sorted(full)], 0)
    R["dw_full@own"] = readout_err(d, idx, fd, truth, pp, units)
    R["copy_last"] = readout_err(d, idx, dec(B["copy_last"]), truth, pp, units)
    p72 = [hz.index("72h")]
    if ms == "h1":
        R["hill"] = readout_err(d, idx, hill_pred(ctx, d, role, idx, clips), truth, pp, units)
    if ms == "h2a":
        R["dw_full@own:72h"] = readout_err(d, idx, fd, truth, p72, units)
        for k in ("hsa", "additive"):
            R[f"{k}:72h"] = readout_err(d, idx, dec(B[k]), truth, p72, units)
        R["bliss:72h"] = readout_err(d, idx, bliss_pred(d, role, idx, clips, B), truth, p72,
                                     units)
        R["bliss"] = R["bliss:72h"]
        op = ode_pred(ctx, d, role, idx, clips)
        if op is not None:
            R["ode:72h"] = readout_err(d, idx, op, truth, p72, units)
    return R


def hill_pred(ctx, d, role, idx, clips):
    """Per drug x line x horizon Hill fit on measured training readouts (log count)."""
    f = d.fields
    fit = np.flatnonzero((role == "fit") & (f.drug2 == "").to_numpy())
    ro = measured_readouts(ctx, d, fit, clips)["log_count"]
    ff = f.iloc[fit].reset_index(drop=True)
    out = np.zeros((len(idx), len(clips)))
    for i, r in enumerate(f.iloc[idx].itertuples()):
        m = ((ff.line == r.line) & ((ff.drug1 == r.drug1) | (ff.drug1 == ""))).to_numpy()
        doses = np.array([0.0 if x == "" else float(x) for x in ff.dose1[m]])
        for h in range(len(clips)):
            u = pd.Series(ro[m, h]).groupby(doses).mean()
            p = G.hill_fit(u.index.to_numpy(), u.to_numpy())
            out[i, h] = G.hill_fn(float(r.dose1), *p) if p is not None else \
                u.iloc[np.argmin(np.abs(np.log10(np.maximum(u.index, 1e-9)) -
                                        np.log10(float(r.dose1))))]
    return out


def bliss_pred(d, role, idx, clips, B):
    tab = E.dose_tables(d, role, clips)
    f = d.fields
    vfit = np.flatnonzero((role == "fit") & (f.h1_role == "vehicle").to_numpy() &
                          (f.line == "AU565").to_numpy())
    cv = np.expm1(dec(d.z[vfit][:, clips].astype(np.float32).mean(0, keepdims=True)))[0]
    vt = np.expm1(dec(B["vehicle_traj"]))
    out = np.zeros((len(idx), len(clips)))
    for i, r in enumerate(f.iloc[idx].itertuples()):
        fa = np.expm1(dec(tab[("AU565", r.drug1)][float(r.dose1)][None]))[0] / cv
        fb = np.expm1(dec(tab[("AU565", r.drug2)][float(r.dose2)][None]))[0] / cv
        out[i] = np.log1p(np.maximum(vt[i] * fa * fb, 0))
    return out


def ode_pred(ctx, d, role, idx, clips):
    p = REPO / "results" / "d3" / "ode_fit.json"
    if not p.exists():
        return None
    fit = json.loads(p.read_text())
    P = ode.Params(fit["drugs"], {k: tuple(v) for k, v in fit["dose_range"].items()})
    v = np.array(fit["params"])
    f = d.fields
    starts = [d.starts[c] for c in clips]
    veh = np.flatnonzero((f.h1_role == "vehicle").to_numpy() & (role == "fit"))
    c0 = measured_readouts(ctx, d, veh, [0])["log_count"][:, 0]
    plate_c0 = pd.Series(np.expm1(c0), index=f.plate.iloc[veh].to_numpy()).groupby(level=0) \
        .mean()
    out = np.zeros((len(idx), len(clips)))
    for i, r in enumerate(f.iloc[idx].itertuples()):
        n = ode.predict(P, v, [(r.drug1, float(r.dose1)), (r.drug2, float(r.dose2))],
                        193).sum(1)
        rel = np.array([n[s:s + T.NF].mean() / n[0] for s in starts])
        out[i] = np.log1p(plate_c0[r.plate] * rel)
    return out


def plan_a(ctx, d, role, idx, clips, pp):
    """Dose inversion on the part's held-out-dose wells (well = mean over fields)."""
    full = ctx.preds("h1", "dw_full", ctx.sel["h1|dw_full"])
    f = d.fields
    tab = E.dose_tables(d, role, clips)
    rows = []
    wells = f.iloc[idx].groupby(["plate", "well"])
    for (plate, well), g in wells:
        r = g.iloc[0]
        obs = d.z[g.index][:, clips].astype(np.float32).mean(0)[pp]
        key = f"{plate}|{r.line}|{r.drug1}"
        errs = []
        for k in sorted(full):
            P = full[k]
            j = list(P["cand_keys"]).index(key)
            cand, pred = P["cand_doses"][j], P["cand_pred"][j].astype(np.float32)[:, pp]
            errs.append(((pred - obs) ** 2).mean((1, 2, 3)))
        e = np.mean(errs, 0)
        d_model = cand[int(np.argmin(e))]
        t = tab[(r.line, r.drug1)]
        ds = sorted(t)
        d_near = ds[int(np.argmin([((t[x][pp] - obs) ** 2).mean() for x in ds]))]
        ie = [((E.interp(t, c)[pp] - obs) ** 2).mean() for c in cand]
        d_int = cand[int(np.argmin(ie))]
        true = float(r.dose1)
        rows.append({"plate": plate, "well": well, "line": r.line, "drug": r.drug1,
                     "dose": true, "model": abs(np.log10(d_model / true)),
                     "nearest": abs(np.log10(d_near / true)),
                     "interp": abs(np.log10(d_int / true))})
    df = pd.DataFrame(rows)
    out = {"n_wells": len(df), "median_abs_log10_error": {k: float(df[k].median())
                                                          for k in ("model", "nearest",
                                                                    "interp")}}
    for b in ("nearest", "interp"):
        t = G.sign_flip_mc((df.model - df[b]).to_numpy())
        out[f"vs_{b}"] = {**t, "median_below": bool(df.model.median() < df[b].median())}
    out["pass"] = all(out[f"vs_{b}"]["median_below"] and out[f"vs_{b}"]["p_one_sided"] is not None
                      and out[f"vs_{b}"]["p_one_sided"] < G.ALPHA for b in ("nearest", "interp"))
    return out


def plan_b(ctx, d, role, idx, clips, hz, R):
    """Combination ranking by predicted 72 h cell-count reduction (decoded), Spearman with the
    measured reduction, vs Bliss, HSA (and ODE); bootstrap over conditions (10,000, seed 0)."""
    f = d.fields
    h = hz.index("72h")
    truth = measured_readouts(ctx, d, idx, clips)["log_count"][:, h]
    B = E.common_baselines(d, role, idx, clips)
    B.update(E.h2a_baselines(d, role, idx, clips, [hz.index("24h"), hz.index("48h")],
                             B["vehicle_traj"]))
    vt_meas = measured_readouts(ctx, d, np.flatnonzero((f.h1_role == "vehicle").to_numpy() &
                                                       (role == "fit")), clips)["log_count"][:, h]
    vplate = pd.Series(np.expm1(vt_meas), index=f.plate[(f.h1_role == "vehicle").to_numpy() &
                                                        (role == "fit")].to_numpy()) \
        .groupby(level=0).mean()
    full = ctx.preds("h2a", "dw_full", ctx.sel["h2a|dw_full"])
    pm = np.mean([dec(full[k]["pred"].astype(np.float32))[:, h] for k in sorted(full)], 0)
    vdec = dec(B["vehicle_traj"])[:, h]
    preds = {"model": 1 - np.expm1(pm) / np.expm1(vdec),
             "bliss": 1 - np.expm1(bliss_pred(d, role, idx, clips, B)[:, h]) / np.expm1(vdec),
             "hsa": 1 - np.expm1(dec(B["hsa"])[:, h]) / np.expm1(vdec)}
    op = ode_pred(ctx, d, role, idx, clips)
    if op is not None:
        preds["ode"] = 1 - np.expm1(op[:, h]) / np.expm1(vdec)
    meas = 1 - np.expm1(truth) / np.array([vplate[p] for p in f.plate.iloc[idx]])
    df = pd.DataFrame({"cond": [f"{r.drug1} {r.dose1} + {r.drug2} {r.dose2}"
                                for r in f.iloc[idx].itertuples()], "meas": meas, **preds})
    c = df.groupby("cond").mean()
    rho = {k: float(spearmanr(c[k], c.meas).statistic) for k in preds}
    rng = np.random.default_rng(0)
    boot = defaultdict(list)
    for _ in range(10_000):
        s = c.iloc[rng.integers(len(c), size=len(c))]
        if s.meas.nunique() < 2:
            continue
        r_m = spearmanr(s.model, s.meas).statistic
        for b in ("bliss", "hsa", "ode"):
            if b in preds:
                boot[b].append(r_m - spearmanr(s[b], s.meas).statistic)
    out = {"n_conditions": len(c), "rho": rho}
    for b, v in boot.items():
        v = np.array(v)
        v = v[np.isfinite(v)]
        out[f"vs_{b}"] = {"p_le_0": float((v <= 0).mean()) if len(v) else None,
                          "n_boot": int(len(v))}
    out["pass"] = all(rho["model"] > rho[b] and out[f"vs_{b}"]["p_le_0"] is not None and
                      out[f"vs_{b}"]["p_le_0"] < G.ALPHA for b in ("bliss", "hsa"))
    return out


def own_start(ctx, d, ms, idx, clips, pp, units):
    out = {}
    if ("own", d.which) not in ctx.data:
        ctx.data[("own", d.which)] = TR.load(d.which, "primary", "own")
    d_own = ctx.data[("own", d.which)]
    for fam in ("dw_full", "dw_none") if ms != "h2b" else [f"dw_full:{s}" for s in SRC] + \
            ["dw_none"]:
        P = ctx.preds(ms, fam, ctx.sel[f"{ms}|{fam}"])
        if len(P) == 3 and all("pred_own" in p for p in P.values()):
            e = pd.concat([unit_err(d_own, idx, P[k]["pred_own"].astype(np.float32), clips,
                                    pp, units) for k in sorted(P)], axis=1).mean(1)
            out[fam] = float(e.mean())
    out["copy_last_own"] = float(unit_err(d_own, idx, np.repeat(
        d_own.z[idx, :1].astype(np.float32), len(clips), 1), clips, pp, units).mean())
    return out


def h1c(ctx, d, L, clips, pp, units):
    ms = f"line:{L}"
    role = TR.roles(d, ms).to_numpy()
    f = d.fields
    idx = np.flatnonzero((f.line == L).to_numpy() & (f.drug2 == "").to_numpy() &
                         f.drug1.isin(["Doxorubicin", "Gemcitabine", "Paclitaxel",
                                       "Palbociclib"]).to_numpy())
    B, excl = E.h1c_baselines(d, role, idx, clips, L)
    keep = np.setdiff1d(np.arange(len(idx)), excl)
    out = {"n_fields": int(len(idx)), "n_excluded_fields": int(len(excl)),
           "excluded_conditions": sorted({f"{f.drug1.iloc[idx[i]]} {f.dose1.iloc[idx[i]]}"
                                          for i in excl})}
    P = ctx.preds(ms, "dw_line", ctx.sel["h1|dw_full"])
    if len(P) != 3:
        out["missing_runs"] = True
        return out
    sub = idx[keep]
    pos = {i: n for n, i in enumerate(P[0]["idx"])}
    me = pd.concat([unit_err(d, sub, P[k]["pred"][[pos[i] for i in sub]].astype(np.float32),
                             clips, pp, units) for k in sorted(P)], axis=1).mean(1)
    M = {"dw_line": me}
    for k, v in B.items():
        M[k] = unit_err(d, sub, v[keep], clips, pp, units)
    M["copy_last"] = unit_err(d, sub, np.repeat(d.z[sub, :1].astype(np.float32), len(clips),
                                                1), clips, pp, units)
    drug = pd.Series([u[0] for u in me.index], index=me.index)
    out["methods"] = {k: float(v.mean()) for k, v in M.items()}
    out["n_units"] = int(len(me))
    out["vs_line_offset"] = G.compare(me, M["line_offset"], drug)
    out["vs_lines_mean"] = G.compare(me, M["lines_mean"], drug)
    return out


def summary(out, decoded_ok):
    s = out["sets"]
    g = {"G-Dose": s["h1"]["gates"]["G-Dose"]["pass"],
         "G-Dose+": s["h1"]["gates"]["G-Dose+"]["pass"],
         "G-Combo": s["h2a"]["gates"]["G-Combo"]["pass"],
         "G-Combo+": s["h2a"]["gates"]["G-Combo+"]["pass"],
         "G-Compound": s["h2b"]["gates"]["G-Compound"]["pass"]}
    if "g_plan_a" in s["h1"]:
        g["G-Plan(a)"] = s["h1"]["g_plan_a"]["pass"]
    if "g_plan_b" in s["h2a"]:
        g["G-Plan(b)"] = s["h2a"]["g_plan_b"]["pass"]
    g["H1c"] = {L: v.get("vs_line_offset", {}).get("beats") for L, v in s["h1c"].items()}
    g["decoded_readouts_used"] = decoded_ok
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["val", "test"], required=True)
    args = ap.parse_args()
    if args.part == "test" and os.environ.get("DW_D5_TEST") != "1":
        sys.exit("test gates are D5 only (set DW_D5_TEST=1 after the author's go)")
    weak = None
    if args.part == "test":
        weak = json.loads((REPO / "results" / "d4" / "gates_val.json").read_text())
    res = {"design_commit": TR.protocol()["design_commit"]}
    for norm in ("primary", "global"):
        ctx = Ctx(args.part, norm)
        res[norm] = evaluate_part(ctx, weak[norm] if weak else None)
    out = REPO / "results" / ("d4" if args.part == "val" else "d5") / \
        f"gates_{args.part}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist")
                              else str(o)))
    backup.backup(out)
    print(json.dumps(res["primary"]["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
