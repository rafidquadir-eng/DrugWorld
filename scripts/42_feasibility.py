"""Feasibility gates after the D2 freeze (plan section 3.4; A1 O1; rules in
manifests/drugworld_protocol.json "f_decode", "f_dose", "f_power"; written before they are
computed).

F-Decode: the decoder training fields of each set (training-split wells), the same probe as the
D0 pilot (standardised raw tile-mean latent, RidgeCV alphas 10^-1..10^5, pooled held-out R^2 of
5-fold GroupKFold over plates); also reported on the normalised latent. F-Dose (idr0119): per
drug x line over training-split doses, Spearman rho of the 48 h displacement from the line's
vehicle mean vs dose. F-Power: counts from the frozen split files.
Writes results/d2/feasibility.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drugworld import REPO, backup, latents
from drugworld import targets as T

PROT = json.loads((REPO / "manifests" / "drugworld_protocol.json").read_text())


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


def wells(which):
    return pd.read_csv(REPO / "splits" / f"{which}_wells.csv", dtype={"plate": str},
                       keep_default_na=False)


def normalised(which, f, X):
    if which == "idr0119":
        n = np.load(REPO / "norm" / "idr0119_line_vehicle.npz")
        info = f.merge(wells(which)[["plate", "well", "line"]], on=["plate", "well"])
        li = np.array([list(n["lines"]).index(x) for x in info.line])
        return (X.astype(np.float32) - n["mean"][li][:, :, None]) / n["sd"][li][:, :, None]
    n = np.load(REPO / "norm" / "idr0064_plate_median.npz")
    pi = np.array([list(n["plates"]).index(p) for p in f.plate])
    return (X.astype(np.float32) - n["median"][pi][:, :, None]) / n["sd"]


def train_mask(which, f):
    w = wells(which)
    info = f.merge(w, on=["plate", "well"], how="left")
    if which == "idr0119":
        h1, h2 = info.h1_role, info.h2a_role
        return (((h1.isin(["train", "vehicle"])) & h2.isin(["", "train", "vehicle"])) |
                ((h1 == "combo") & (h2 == "train"))).to_numpy()
    return (info.role == "train").to_numpy()


def f_decode(which):
    s = PROT["sets"][which]
    f, arr, meta = latents.load(which, arrays=("mean",), cells=True)
    m = train_mask(which, f)
    bx = T.boxes(which, meta["tiles"], meta["level"])
    ys = {}
    for i in np.flatnonzero(m):
        o = (T.idr0119_targets(arr["cells"][i], meta["starts"], bx,
                               s["decoder_params"]["g1_threshold"]) if which == "idr0119"
             else T.idr0064_targets(arr["cells"][i], meta["starts"], bx,
                                    s["decoder_params"]["pulse_prominence"]))
        for k, v in o.items():
            ys.setdefault(k, []).append(v.ravel())
    X = arr["mean"]
    groups = np.repeat(f.plate[m].to_numpy(), X.shape[1] * X.shape[2])
    raw = X[m].reshape(-1, X.shape[-1]).astype(np.float32)
    Z = normalised(which, f, X)[m].reshape(-1, X.shape[-1])
    res = {}
    for k, v in ys.items():
        y = np.concatenate(v)
        res[k] = {"raw": probe(raw, y, groups), "normalised": probe(Z, y, groups)}
        print(which, k, round(res[k]["raw"]["r2"], 3), round(res[k]["normalised"]["r2"], 3))
    return res, f, X


def f_dose(f, X):
    s = PROT["sets"]["idr0119"]
    c48 = s["clips"]["horizons"]["48h"]["clip_index"]
    w = wells("idr0119")
    info = f.merge(w, on=["plate", "well"], how="left")
    Z = normalised("idr0119", f, X[:, c48:c48 + 1])[:, 0].mean(1)   # fields x 1024
    df = pd.DataFrame({"plate": info.plate, "well": info.well, "line": info.line,
                       "drug": info.drug1, "dose": info.dose1.replace("", np.nan).astype(float),
                       "role": info.h1_role})
    rows = []
    for (line, drug), g in df[df.role.isin(["train", "train_sel"])].groupby(["line", "drug"]):
        veh = Z[(df.line == line).to_numpy() & (df.role == "vehicle").to_numpy()]
        vmean = veh.mean(0)
        # fields -> wells -> condition
        disp = {}
        for dose, gd in g.groupby("dose"):
            wm = [Z[gw.index].mean(0) for _, gw in gd.groupby(["plate", "well"])]
            disp[dose] = float(np.linalg.norm(np.mean(wm, 0) - vmean))
        vw = [Z[gw.index].mean(0) for _, gw in df[(df.line == line) & (df.role == "vehicle")]
              .groupby(["plate", "well"])]
        d0 = float(np.linalg.norm(np.mean(vw, 0) - vmean))
        doses = sorted(disp)
        rho = spearmanr(doses, [disp[d] for d in doses]).statistic
        rho0 = spearmanr([0.0] + doses, [d0] + [disp[d] for d in doses]).statistic
        rows.append({"line": line, "drug": drug, "n_doses": len(doses), "doses_nM": doses,
                     "displacement": [disp[d] for d in doses], "rho": float(rho),
                     "rho_with_vehicle": float(rho0)})
    frac = float(np.mean([r["rho"] >= 0.6 for r in rows]))
    frac0 = float(np.mean([r["rho_with_vehicle"] >= 0.6 for r in rows]))
    print("F-Dose", f"{frac:.3f} of {len(rows)} pairs (with vehicle: {frac0:.3f})")
    return {"pairs": rows, "fraction_rho_ge_0.6": frac, "pass": bool(frac >= 0.5),
            "fraction_with_vehicle_point": frac0}


def f_power():
    w19 = wells("idr0119")
    cond = w19[w19.h1_role.isin(["test", "test_extrap"])].drop_duplicates(
        ["line", "drug1", "dose1"])
    h1 = {"test_interior": int((cond.h1_role == "test").sum()),
          "test_extrap": int((cond.h1_role == "test_extrap").sum())}
    h1["total"] = h1["test_interior"] + h1["test_extrap"]
    combo = w19[w19.h2a_role == "test"].drop_duplicates(["drug1", "dose1", "drug2", "dose2"])
    w64 = wells("idr0064")
    h2b = int(w64[w64.role == "test"].compound_key.nunique())
    out = {"H1": {**h1, "pass": h1["total"] >= 20},
           "H2a": {"test_conditions": len(combo), "pass": len(combo) >= 6},
           "H2b": {"test_compounds": h2b, "pass": h2b >= 60}}
    print("F-Power", out)
    return out


def main() -> int:
    pilot = json.loads((REPO / "results" / "d0" / "pilot_probe.json").read_text())
    out = {"rules": {k: PROT["rules"][k] for k in ("f_decode", "f_dose", "f_power")},
           "design_commit": PROT["design_commit"]}
    dec19, f19, X19 = f_decode("idr0119")
    out["f_dose"] = f_dose(f19, X19)
    del X19
    dec64, _, _ = f_decode("idr0064")
    out["f_decode"] = {
        "idr0119": {"targets": dec19, "pass": bool(dec19["log_count"]["raw"]["r2"] >= 0.8 and
                                                   dec19["g1_fraction"]["raw"]["r2"] >= 0.5),
                    "provisional_vehicle": pilot["idr0119"]["f_decode_provisional"]},
        "idr0064": {"targets": dec64, "pass": bool(dec64["erk_cn"]["raw"]["r2"] >= 0.5),
                    "provisional_vehicle": pilot["idr0064"]["f_decode_provisional"]}}
    out["f_power"] = f_power()
    path = REPO / "results" / "d2" / "feasibility.json"
    path.write_text(json.dumps(out, indent=1))
    backup.backup(path)
    print(json.dumps({"F-Decode": {k: v["pass"] for k, v in out["f_decode"].items()},
                      "F-Dose": out["f_dose"]["pass"],
                      "F-Power": {k: v["pass"] for k, v in out["f_power"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
