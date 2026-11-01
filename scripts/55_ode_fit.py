"""Fit the Gross et al. phase-specific rate model (src/drugworld/ode.py) to AU565 training
conditions of the H2a model set (fit-role single-drug wells and vehicle wells; never a
combination), from the measured per-cell readouts (G1 = log HDHB ratio above the frozen
threshold). One condition = (drug, dose) or vehicle; its series is the mean over its fields of
the G1 and S/G2 cell numbers per frame. Writes results/d3/ode_fit.json.
"""
from __future__ import annotations

import json
import time

import numpy as np

from drugworld import REPO, backup, idr0119, latents, ode
from drugworld import train as TR


def main() -> int:
    prot = TR.protocol()
    thr = prot["sets"]["idr0119"]["decoder_params"]["g1_threshold"]
    f, arr, _ = latents.load("idr0119", arrays=(), cells=True)
    w = TR.wells("idr0119")
    info = f.merge(w, on=["plate", "well"], how="left")
    d = TR.Data("idr0119", info, np.zeros((len(info), 1, 1, 1), np.float16), [])
    role = TR.roles(d, "h2a").to_numpy()
    m = (role == "fit") & (info.line == "AU565").to_numpy() & (info.drug2 == "").to_numpy()
    conds = {}
    for key, g in info[m].groupby([info.drug1[m], info.dose1[m]]):
        drug, dose = key
        k = ("", 0.0) if drug == "" else (drug, float(dose))
        obs = ode.observed([arr["cells"][i] for i in g.index], thr, idr0119.N_GRID)
        conds[k] = ode.prepare(obs)
    drugs = sorted({k[0] for k in conds if k[0]})
    rng = {dr: (min(c for x, c in conds if x == dr), max(c for x, c in conds if x == dr))
           for dr in drugs}
    t0 = time.time()
    P, res = ode.fit(conds, drugs, rng, idr0119.N_GRID, workers=100, maxiter=3000)
    out = {"conditions": [list(k) for k in sorted(conds)], "drugs": drugs, "dose_range": rng,
           "params": res.x.tolist(), "loss": float(res.fun), "nfev": int(res.nfev),
           "success": bool(res.success), "message": str(res.message),
           "minutes": (time.time() - t0) / 60, "n_fields": int(m.sum()),
           "design_commit": prot["design_commit"]}
    path = REPO / "results" / "d3" / "ode_fit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    backup.backup(path)
    print({k: out[k] for k in ("loss", "nfev", "success", "minutes", "n_fields")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
