"""D4/D5 evaluation core (plan sections 7-9; rules in manifests/drugworld_protocol.json).

Everything is in the frozen normalised space (or the global sensitivity space) with the
counterfactual start state at step 0. A forecast of a field is its latent at each horizon clip
(H, K, D); its error at a horizon = MSE over tiles and dimensions against the measured clip;
the primary error = mean over the model set's primary horizons. Condition error = mean over
replicate wells (each the mean over its fields); model errors are then averaged over 3 seeds.

Latent baselines (plan section 7; inputs from fit-role fields of the same model set only):
  copy_last      step 0 (the start state) at every horizon
  vehicle_traj   mean of the plate's fit vehicle fields at the horizon clip (per tile position)
  slot_mean      mean of all fit fields at the horizon clip
  nearest_dose   (H1) mean of the same drug x line at the nearest training dose in log dose
                 (equidistant: mean of both)
  interp         (H1) linear in log dose between the neighbouring training doses; beyond the
                 highest (lowest) training dose, the line through the last (first) two
  hsa            (H2a) the single-drug component with the larger mean displacement from the
                 line's vehicle mean over the primary horizons, at its dose
  additive       (H2a) plate vehicle trajectory + (single A - vehicle) + (single B - vehicle)
  knn            (H2b) similarity-weighted mean of the 5 training compounds with the highest
                 ECFP4 Tanimoto (raw 1024 bits; ties broken by compound name)
  lines_mean, line_offset (H1c; A1 C5) see `h1c_baselines`
  residual       (H1, H2b; GeneWorld R2 form) dw_none forecast + lambda x residual of the
                 nearest training condition (nearest dose / nearest compound by Tanimoto),
                 residual = mean over its fit fields of (measured - dw_none forecast), lambda
                 in {0, 0.25, 0.5, 0.75, 1} chosen on validation, per seed of dw_none
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from drugworld import REPO
from drugworld import train as TR
from drugworld.model import DrugPredictor

DEV = "cuda"
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------- model forecasts

def load_model(ck_path):
    ck = torch.load(ck_path, map_location=DEV, weights_only=False)
    c = ck["config"]
    m = DrugPredictor(ck["dim"], ck["tok_dim"], ck["mode"], width=c["width"], depth=c["depth"],
                      n_steps=ck["n_steps"], drop=c["dropout"]).to(DEV)
    m.load_state_dict(ck["model"])
    return m.eval(), ck


@torch.no_grad()
def forecast(model, z0, act: TR.Actions, idx, clips, bs=256):
    """Rollout from step 0 for fields `idx` -> (n, len(clips), K, D) float16."""
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        zz = torch.from_numpy(z0[j, :1].astype(np.float32)).to(DEV)
        a = [torch.from_numpy(x[j]).to(DEV) for x in (act.toks, act.tmask, act.vehicle,
                                                         act.redose)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model.rollout(zz, *a, max(clips)).float()
        out.append(p[:, [c - 1 for c in clips]].cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, len(clips)) + z0.shape[2:], np.float16)


def errors(pred, truth):
    """(n, H, K, D) x2 -> (n, H) MSE over K and D."""
    return ((pred.astype(np.float32) - truth.astype(np.float32)) ** 2).mean((2, 3))


def condition_errors(fields: pd.DataFrame, err: np.ndarray, unit_cols) -> pd.DataFrame:
    """Per-field errors (n, H) -> per condition (unit) mean over wells of field means."""
    df = fields[unit_cols + ["plate", "well"]].copy()
    cols = [f"h{i}" for i in range(err.shape[1])]
    df[cols] = err
    w = df.groupby(unit_cols + ["plate", "well"], dropna=False)[cols].mean().reset_index()
    return w.groupby(unit_cols, dropna=False)[cols].mean().reset_index()


# ---------------------------------------------------------------- latent baselines

def _mean(z, idx, clips):
    return z[idx][:, clips].astype(np.float32).mean(0)


def common_baselines(d: TR.Data, role, idx, clips):
    """copy_last, vehicle_traj, slot_mean for fields idx -> {name: (n, H, K, D)}."""
    f = d.fields
    fit = np.flatnonzero(role == "fit")
    veh = (f.h1_role == "vehicle").to_numpy() if d.which == "idr0119" else \
        (f.role == "vehicle").to_numpy()
    fitveh = np.flatnonzero(veh & (role == "fit"))
    out = {"copy_last": np.repeat(d.z[idx, :1].astype(np.float32), len(clips), 1),
           "slot_mean": np.broadcast_to(_mean(d.z, fit, clips), (len(idx), len(clips)) +
                                        d.z.shape[2:]).copy()}
    vt = np.zeros((len(idx), len(clips)) + d.z.shape[2:], np.float32)
    plates = f.plate.to_numpy()
    for p in np.unique(plates[idx]):
        src = fitveh[plates[fitveh] == p]
        if not len(src):                      # same fallback as the start state (A1 C1)
            fb = TR.protocol()["sets"][d.which]["start_state"]["fallback"].get(p, [])
            src = fitveh[np.isin(plates[fitveh], fb)]
        vt[plates[idx] == p] = _mean(d.z, src, clips)
    out["vehicle_traj"] = vt
    return out


def dose_tables(d: TR.Data, role, clips):
    """(line, drug) -> {dose: mean trajectory (H, K, D)} over fit single-drug fields."""
    f = d.fields
    fit = (role == "fit") & (f.drug1 != "").to_numpy() & (f.drug2 == "").to_numpy()
    tab = {}
    for (line, drug, dose), g in f[fit].groupby(["line", "drug1", "dose1"]):
        tab.setdefault((line, drug), {})[float(dose)] = _mean(d.z, g.index.to_numpy(), clips)
    return tab


def nearest(tab, x):
    ds = np.array(sorted(tab))
    dist = np.abs(np.log10(ds) - np.log10(x))
    best = ds[np.isclose(dist, dist.min())]
    return np.mean([tab[b] for b in best], 0), [float(b) for b in best]


def interp(tab, x):
    ds = np.array(sorted(tab))
    lx, l = np.log10(x), np.log10(ds)
    if len(ds) == 1:
        return tab[ds[0]]
    if lx <= l[0]:
        i = 0
    elif lx >= l[-1]:
        i = len(ds) - 2
    else:
        i = int(np.searchsorted(l, lx) - 1)
    w = (lx - l[i]) / (l[i + 1] - l[i])
    return (1 - w) * tab[ds[i]] + w * tab[ds[i + 1]]


def h1_baselines(d, role, idx, clips):
    tab = dose_tables(d, role, clips)
    f = d.fields.iloc[idx]
    nd = np.zeros((len(idx), len(clips)) + d.z.shape[2:], np.float32)
    ip = np.zeros_like(nd)
    for i, r in enumerate(f.itertuples()):
        t = tab[(r.line, r.drug1)]
        nd[i] = nearest(t, float(r.dose1))[0]
        ip[i] = interp(t, float(r.dose1))
    return {"nearest_dose": nd, "interp": ip}


def h2a_singles(d, role, clips):
    """AU565 single-drug and vehicle references for the H2a baselines (author decision on
    status report 2): per (drug, dose), fit-role single-drug fields on plates without the 48 h
    re-dose (redose_48h 0) when any exist, otherwise those on re-dosed plates; each single's
    effect is taken against the AU565 fit vehicle fields of the same re-dose class (all AU565 fit
    vehicle fields if that class has none). Returns ({(drug, dose): mean trajectory},
    {(drug, dose): vehicle mean trajectory}, {"drug|dose": source})."""
    f = d.fields
    fit = (role == "fit") & (f.line == "AU565").to_numpy()
    rd = f.redose_48h.astype(int).to_numpy()
    veh = fit & (f.h1_role == "vehicle").to_numpy()
    vall = _mean(d.z, np.flatnonzero(veh), clips)
    vcls = {c: (_mean(d.z, np.flatnonzero(veh & (rd == c)), clips) if (veh & (rd == c)).any()
                else vall) for c in (0, 1)}
    sing = fit & (f.drug1 != "").to_numpy() & (f.drug2 == "").to_numpy()
    tab, vref, src = {}, {}, {}
    for (drug, dose), g in f[sing].groupby(["drug1", "dose1"]):
        i = g.index.to_numpy()
        c = 0 if (rd[i] == 0).any() else 1
        k = (drug, float(dose))
        tab[k] = _mean(d.z, i[rd[i] == c], clips)
        vref[k] = vcls[c]
        src[f"{drug}|{float(dose)}"] = "not_redosed" if c == 0 else "redosed"
    return tab, vref, src


def h2a_baselines(d, role, idx, clips, primary_pos, veh_traj):
    """HSA and additive composition for combination fields idx (AU565); singles from
    h2a_singles (non-re-dosed plates preferred)."""
    tab, vref, _ = h2a_singles(d, role, clips)
    f = d.fields
    hsa = np.zeros((len(idx), len(clips)) + d.z.shape[2:], np.float32)
    add = np.zeros_like(hsa)
    for i, r in enumerate(f.iloc[idx].itertuples()):
        ka, kb = (r.drug1, float(r.dose1)), (r.drug2, float(r.dose2))
        sa, sb = tab[ka], tab[kb]
        da, db = sa - vref[ka], sb - vref[kb]
        ea = np.linalg.norm(da[primary_pos].reshape(len(primary_pos), -1), axis=1)
        eb = np.linalg.norm(db[primary_pos].reshape(len(primary_pos), -1), axis=1)
        hsa[i] = sa if ea.mean() >= eb.mean() else sb
        add[i] = veh_traj[i] + da + db
    return {"hsa": hsa, "additive": add}


def tanimoto(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = (a[:, None] & b[None]).sum(-1)
    union = (a[:, None] | b[None]).sum(-1)
    return inter / np.maximum(union, 1)


def knn_table(d, role):
    e = np.load(REPO / "encodings" / "idr0064.npz", allow_pickle=True)
    comps = list(e["compounds"])
    f = d.fields
    train = sorted(set(f.compound_key[(role == "fit") & (f.role == "train").to_numpy()]))
    bits = e["ecfp4_bits"]
    return comps, bits, train


def h2b_baselines(d, role, idx, clips, k=5):
    comps, bits, train = knn_table(d, role)
    f = d.fields
    traj = {c: _mean(d.z, f.index[(f.compound_key == c) & (role == "fit")].to_numpy(), clips)
            for c in train}
    tb = bits[[comps.index(c) for c in train]]
    out = np.zeros((len(idx), len(clips)) + d.z.shape[2:], np.float32)
    nn = {}
    for i, c in enumerate(f.compound_key.iloc[idx]):
        s = tanimoto(bits[[comps.index(c)]], tb)[0]
        order = sorted(range(len(train)), key=lambda j: (-s[j], train[j]))[:k]
        w = s[order]
        w = w / w.sum() if w.sum() > 0 else np.full(k, 1 / k)
        out[i] = sum(wj * traj[train[j]] for wj, j in zip(w, order))
        nn[c] = [(train[j], float(s[j])) for j in order]
    return {"knn": out}, nn


def h1c_baselines(d, role_line, idx, clips, line):
    """A1 C5: each training line's response interpolated in log dose at the held-out dose
    (training doses only, inside that line's training-dose range); lines_mean = mean over the
    eligible lines; line_offset = held-out plate vehicle trajectory + mean over eligible lines
    of (interpolated response - that line's fit-vehicle mean). Fields with no eligible line are
    returned in `excluded`."""
    tab = dose_tables(d, role_line, clips)
    f = d.fields
    veh = {}
    for L in f.line.unique():
        m = np.flatnonzero((role_line == "fit") & (f.h1_role == "vehicle").to_numpy() &
                           (f.line == L).to_numpy())
        if len(m):
            veh[L] = _mean(d.z, m, clips)
    # anchor: the held-out line's own plate vehicle (plan: "added to the held-out line's
    # vehicle"); its vehicle wells are role "heldout_line" in this model set, not "fit"
    rv = np.where((f.h1_role == "vehicle").to_numpy() & (f.line == line).to_numpy(), "fit",
                  role_line)
    base = common_baselines(d, rv, idx, clips)["vehicle_traj"]
    lm = np.full((len(idx), len(clips)) + d.z.shape[2:], np.nan, np.float32)
    lo = lm.copy()
    excluded = []
    for i, r in enumerate(f.iloc[idx].itertuples()):
        resp = []
        for (L, drug), t in tab.items():
            if L == line or drug != r.drug1:
                continue
            ds = sorted(t)
            if ds[0] <= float(r.dose1) <= ds[-1]:
                resp.append((interp(t, float(r.dose1)), veh[L]))
        if not resp:
            excluded.append(i)
            continue
        lm[i] = np.mean([x for x, _ in resp], 0)
        lo[i] = base[i] + np.mean([x - v for x, v in resp], 0)
    return {"lines_mean": lm, "line_offset": lo}, excluded


# ---------------------------------------------------------------- decoded readouts

def decode(pred, which, target):
    """(n, H, K, D) normalised latents -> (n, H, K) readout with the frozen decoder."""
    z = np.load(REPO / "decoders" / f"{which}_{target}.npz")
    x = (pred.astype(np.float32) - z["scaler_mean"]) / z["scaler_scale"]
    return x @ z["coef"] + z["intercept"]
