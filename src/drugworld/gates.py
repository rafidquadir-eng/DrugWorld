"""Gate helpers (plan section 8; fixed before any evaluation output).

* `compare(full, other)`: per-unit primary errors (aligned) -> one-sided Monte Carlo sign flip
  over units of d = full - other (100,000 draws, seed 0; GeneWorld `sign_flip_mc`), plus the
  per-drug breakdown when units carry a drug.
* "beats" = p < 0.05. Baseline-strength check (L3): a lookup baseline beats copy-last iff its
  sign-flip p < 0.05 against copy-last on validation; otherwise it is "weak", and a gate that
  names it uses the next one in its chain (CHAINS), skipping weak ones; the last link is used
  even if weak (recorded).
* Holm (step-down) over a dict of p-values.
* Measured readouts per field and horizon clip: idr0119 log count and G1 fraction (tile means of
  `targets.idr0119_targets`; G1 a nan-mean), idr0064 Erk C/N and pulse rate.
* Hill: four-parameter curve top + (bottom - top) C^n / (C^n + EC50^n) per drug x line x horizon
  on the training doses and vehicle (dose 0 sets the top asymptote's data point); bounds
  n in [0.3, 5], EC50 in [min dose / 10, max dose x 10]; least squares (scipy curve_fit,
  trust region), starting at n = 1 and EC50 = the geometric mean of the training doses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from geneworld.eval.h1 import sign_flip_mc

ALPHA = 0.05
CHAINS = {"nearest_dose": ["nearest_dose", "interp", "hill"],
          "interp": ["interp", "hill"],
          "hsa": ["hsa", "additive", "bliss"],
          "additive": ["additive", "bliss"],
          "knn": ["knn", "residual"]}


def compare(full: pd.Series, other: pd.Series, drug: pd.Series | None = None) -> dict:
    j = full.index.intersection(other.index)
    d = (full[j] - other[j]).to_numpy(float)
    out = sign_flip_mc(d)
    out["beats"] = bool(out["p_one_sided"] is not None and out["p_one_sided"] < ALPHA)
    out["mean_full"], out["mean_other"] = float(full[j].mean()), float(other[j].mean())
    out["rel_change"] = float(full[j].mean() / other[j].mean() - 1) if other[j].mean() else None
    if drug is not None:
        dd = pd.Series(d, index=j).groupby(drug[j]).agg(["mean", "size"])
        out["per_drug"] = {k: {"mean_diff": float(v["mean"]), "n": int(v["size"])}
                           for k, v in dd.iterrows()}
    return out


def holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: (p[k] is None, p[k] if p[k] is not None else 1))
    m, out, stop = len(keys), {}, False
    for i, k in enumerate(keys):
        thr = ALPHA / (m - i)
        ok = (not stop) and p[k] is not None and p[k] < thr
        stop = stop or not ok
        out[k] = {"p": p[k], "threshold": thr, "reject": bool(ok)}
    return out


def substitute(name: str, weak: dict) -> tuple[str, list]:
    chain = CHAINS.get(name, [name])
    for b in chain:
        if not weak.get(b, False):
            return b, chain
    return chain[-1], chain


def hill_fn(c, top, bottom, ec50, n):
    c = np.asarray(c, float)
    h = np.where(c > 0, c ** n / (c ** n + ec50 ** n), 0.0)
    return top + (bottom - top) * h


def hill_fit(doses, values):
    """doses include 0 (vehicle) -> params (top, bottom, ec50, n) or None."""
    doses, values = np.asarray(doses, float), np.asarray(values, float)
    pos = doses[doses > 0]
    p0 = [values[doses == 0].mean() if (doses == 0).any() else values[0], values[-1],
          float(np.exp(np.log(pos).mean())), 1.0]
    lo = [-np.inf, -np.inf, pos.min() / 10, 0.3]
    hi = [np.inf, np.inf, pos.max() * 10, 5.0]
    try:
        p, _ = curve_fit(hill_fn, doses, values, p0=p0, bounds=(lo, hi), method="trf",
                         maxfev=20000)
        return p
    except (RuntimeError, ValueError):
        return None
