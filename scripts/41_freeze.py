"""D2 freeze (plan section 5; amendment A1, design_commit eefeeb0): splits, protocol,
encodings, normalisation statistics, counterfactual start states and decoders, written in one
run and committed in one commit before any model is trained.

Outputs (all committed):
  splits/idr0119_wells.csv, splits/idr0119_split.json, splits/idr0064_wells.csv,
  splits/idr0064_split.json
  encodings/idr0119.npz, encodings/idr0064.npz, encodings/encodings.json
  norm/<set>_<kind>.npz (normalisation statistics, start states)
  decoders/<set>_<target>.npz (ridge, fitted on training wells in the normalised space)
  manifests/drugworld_protocol.json (every rule below, with the SHA-256 of every file above)
Each file is backed up with SHA-256.

  PYTHONPATH=src python scripts/41_freeze.py
  (testing only: --dry DIR [--sets idr0064] writes everything under DIR, no backups)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drugworld import REPO, backup, idr0064, idr0119, latents, splits
from drugworld import targets as T

DESIGN_COMMIT = "eefeeb0"
SEED = splits.SEED
ALPHAS = np.logspace(-1, 5, 13)
OUT = {k: REPO / k for k in ("splits", "encodings", "norm", "decoders")}
WRITTEN: list = []


def save_npz(path, **arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arr)
    WRITTEN.append(path)


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    WRITTEN.append(path)


# ---------------------------------------------------------------- clips and horizons

def horizon_clip(starts, frame_h, h):
    """The stored clip whose last frame is the latest at or before h hours."""
    ok = [s for s in starts if (s + T.NF - 1) * frame_h <= h + 1e-9]
    return max(ok)


def clip_table(which, starts):
    fh = T.FRAME_H[which]
    hs = {"idr0119": [24, 48, 72, 96], "idr0064": [1, 2, 3, 4]}[which]
    return {f"{h}h": {"start": horizon_clip(starts, fh, h),
                      "clip_index": starts.index(horizon_clip(starts, fh, h)),
                      "window_h": [round(horizon_clip(starts, fh, h) * fh, 3),
                                   round((horizon_clip(starts, fh, h) + T.NF - 1) * fh, 3)]}
            for h in hs}


# ---------------------------------------------------------------- splits

def idr0119_split():
    ann = idr0119.annotations()
    sp = splits.idr0119_splits(ann, SEED)
    sel = splits.selection_plates(ann, np.random.default_rng(SEED + 1))
    w = splits.well_roles_idr0119(ann, sp, sel)
    redose = json.loads((REPO / "results" / "d2" / "au565_redose.json").read_text())
    w["redose_48h"] = [int(r.line == "AU565" and redose["plates"][r.plate]["kind"] ==
                           "single-drug only") for r in w.itertuples()]
    w = w.sort_values(["plate", "well"]).reset_index(drop=True)
    save_text(OUT["splits"] / "idr0119_wells.csv", w.to_csv(index=False))
    js = {"seed": SEED, "selection_seed": SEED + 1, "val_line": sp["val_line"],
          "test_lines": sp["test_lines"], "test_pairs": sp["test_pairs"],
          "val_pair": sp["val_pair"], "pair_conditions": sp["pair_conditions"],
          "selection_plates": sorted(sel),
          "single": [{"line": k[0], "drug": k[1], "dose_nM": k[2], "role": v}
                     for k, v in sorted(sp["single"].items())]}
    save_text(OUT["splits"] / "idr0119_split.json", json.dumps(js, indent=1))
    return w, js


def idr0064_split():
    ann = idr0064.annotations()
    comp = pd.read_csv(REPO / "data" / "structures" / "compounds_idr0064.csv",
                       dtype={"plate": str})
    supp = sorted((REPO / "data" / "annotations" / "goglia_supp").glob("*"))
    if supp:
        raise SystemExit("data/annotations/goglia_supp/ is not empty: parse Table S1 "
                         "(not yet implemented; stop and ask before freezing)")
    sp = splits.idr0064_splits(ann, comp, None, SEED)
    a = ann.copy()
    key = comp.set_index(["plate", "well"]).compound
    a["compound_key"] = [key.get((p, w), "") for p, w in zip(a.plate, a.well)]
    chk = set(map(tuple, sp["dmso_check"]))
    role = []
    for r in a.itertuples():
        if r.compound == "DMSO":
            role.append("dmso_check" if (r.plate, r.well) in chk else "vehicle")
        elif r.compound_key:
            role.append(sp["compound_split"][r.compound_key])
        else:
            role.append("excluded")
    a["role"] = role
    a = a.merge(comp[["compound", "scaffold", "inchikey_parent"]].rename(
        columns={"compound": "compound_key"}), on="compound_key", how="left")
    a = a.sort_values(["plate", "well"]).reset_index(drop=True)
    save_text(OUT["splits"] / "idr0064_wells.csv", a.to_csv(index=False))
    js = {"seed": sp["seed"], "stratification": sp["stratification"],
          "o2_fallback_applied": True,
          "counts": a.role.value_counts().to_dict(), "dmso_check": sp["dmso_check"]}
    save_text(OUT["splits"] / "idr0064_split.json", json.dumps(js, indent=1))
    return a, js


# ---------------------------------------------------------------- encodings

def encodings(w19, w64):
    out = {}
    e19 = np.load(REPO / "data" / "structures" / "ecfp4_idr0119.npz", allow_pickle=True)
    m19 = np.load(REPO / "data" / "structures" / "molformer_idr0119.npz", allow_pickle=True)
    assert e19["compounds"].tolist() == m19["compounds"].tolist()
    single = w19[(w19.drug1 != "")]
    dmax = {}
    for k in ("1", "2"):
        for d, x in zip(single[f"drug{k}"], single[f"dose{k}"]):
            if d:
                dmax[d] = max(dmax.get(d, 0.0), float(x))
    save_npz(OUT["encodings"] / "idr0119.npz", drugs=e19["compounds"],
             ecfp4=e19["bits"].astype(np.float32), molformer=m19["emb"].astype(np.float32),
             dose_max_nM=np.array([dmax[d] for d in e19["compounds"]]))
    out["idr0119"] = {"drug": "ECFP4 (1024 bits, as is) and MoLFormer-XL (768, as is); A1 C4",
                      "dose": "log10(dose / max dose of that drug over all lines), plus a "
                              "vehicle indicator", "dose_max_nM": dmax}
    e64 = np.load(REPO / "data" / "structures" / "ecfp4_idr0064.npz", allow_pickle=True)
    m64 = np.load(REPO / "data" / "structures" / "molformer_idr0064.npz", allow_pickle=True)
    assert e64["compounds"].tolist() == m64["compounds"].tolist()
    names = e64["compounds"].tolist()
    comp = pd.read_csv(REPO / "data" / "structures" / "compounds_idr0064.csv",
                       dtype={"plate": str}).set_index(
        "compound").loc[names]
    train = set(w64.compound_key[w64.role == "train"])
    tr = np.array([n in train for n in names])
    pe = PCA(128, random_state=0).fit(e64["bits"][tr].astype(np.float32))
    pm = PCA(128, random_state=0).fit(m64["emb"][tr])
    classes = sorted(comp.target_class.unique())
    onehot = np.array([[c == k for k in classes] for c in comp.target_class], np.float32)
    save_npz(OUT["encodings"] / "idr0064.npz", compounds=np.array(names),
             ecfp4_pca128=pe.transform(e64["bits"].astype(np.float32)).astype(np.float32),
             molformer_pca128=pm.transform(m64["emb"]).astype(np.float32),
             target_class_onehot=onehot, target_classes=np.array(classes),
             ecfp4_bits=e64["bits"])
    out["idr0064"] = {"sources": ["ecfp4_pca128", "molformer_pca128", "target_class_onehot"],
                      "pca": "fitted on the training compounds only (n = %d), random_state 0"
                             % tr.sum(),
                      "ecfp4_pca_variance_explained": float(pe.explained_variance_ratio_.sum()),
                      "molformer_pca_variance_explained":
                          float(pm.explained_variance_ratio_.sum()),
                      "target_classes": classes,
                      "dose": "single dose (2.5 uM): dose term 0, plus a vehicle indicator"}
    save_text(OUT["encodings"] / "encodings.json", json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- normalisation, start states

def batch_of(plate):
    return plate[:5] if not plate[0].isdigit() or len(plate) > 4 else plate.split(".")[0]


def norm_idr0119(f, X, w):
    """A1 C3: per line and clip, mean and SD over the line's vehicle fields (tiles pooled);
    SD floored at the line's 5th percentile over clips x dimensions."""
    info = f.merge(w[["plate", "well", "line", "h1_role"]], on=["plate", "well"], how="left")
    lines = sorted(info.line.unique())
    mu = np.zeros((len(lines), X.shape[1], X.shape[3]), np.float32)
    sd = np.zeros_like(mu)
    nveh = {}
    for i, L in enumerate(lines):
        v = X[(info.line == L).to_numpy() & (info.h1_role == "vehicle").to_numpy()]
        nveh[L] = int(len(v))
        x = v.astype(np.float32).transpose(1, 0, 2, 3).reshape(X.shape[1], -1, X.shape[3])
        mu[i], sd[i] = x.mean(1), x.std(1)
        sd[i] = np.maximum(sd[i], np.percentile(sd[i], 5))
    save_npz(OUT["norm"] / "idr0119_line_vehicle.npz", lines=np.array(lines), mean=mu, sd=sd)
    return {"lines": lines, "vehicle_fields": nveh}, lines, mu, sd, info


def norm_idr0064(f, X, w):
    """A1 C3: per plate and clip, centre on the median over all compound wells of the plate
    (tiles pooled); scale by the global SD (per dimension) of training compound wells."""
    info = f.merge(w[["plate", "well", "role"]], on=["plate", "well"], how="left")
    plates = sorted(info.plate.unique())
    comp = info.role.isin(["train", "val", "test"]).to_numpy()
    med = np.zeros((len(plates), X.shape[1], X.shape[3]), np.float32)
    for i, p in enumerate(plates):
        x = X[comp & (info.plate == p).to_numpy()].astype(np.float32)
        med[i] = np.median(x.transpose(1, 0, 2, 3).reshape(X.shape[1], -1, X.shape[3]), axis=1)
    pidx = np.array([plates.index(p) for p in info.plate])
    trn = (info.role == "train").to_numpy()
    c = X[trn].astype(np.float32) - med[pidx[trn]][:, :, None, :]
    sd = c.reshape(-1, X.shape[3]).std(0).astype(np.float32)
    save_npz(OUT["norm"] / "idr0064_plate_median.npz", plates=np.array(plates), median=med,
             sd=sd)
    return {"plates": plates, "compound_wells_per_plate":
            {p: int((comp & (info.plate == p).to_numpy()).sum()) for p in plates}}, \
        plates, med, sd, info


def apply_19(X, info, lines, mu, sd):
    li = np.array([lines.index(L) for L in info.line])
    return (X.astype(np.float32) - mu[li][:, :, None, :]) / sd[li][:, :, None, :]


def apply_64(X, info, plates, med, sd):
    pi = np.array([plates.index(p) for p in info.plate])
    return (X.astype(np.float32) - med[pi][:, :, None, :]) / sd


def start_states(Z, info, veh_mask, which):
    """A1 C1: per plate and tile position, the mean first-clip state of the plate's vehicle
    wells (normalised space); fallback: same line (idr0119) and imaging batch."""
    plates = sorted(info.plate.unique())
    out = np.zeros((len(plates), Z.shape[2], Z.shape[3]), np.float32)
    fallback = {}
    for i, p in enumerate(plates):
        m = veh_mask & (info.plate == p).to_numpy()
        if not m.any():
            same = (info.plate.map(batch_of) == batch_of(p)).to_numpy() & veh_mask
            if which == "idr0119":
                same &= (info.line == info.line[info.plate == p].iloc[0]).to_numpy()
            m = same
            fallback[p] = sorted(info.plate[m].unique())
        out[i] = Z[m, 0].mean(0)
    save_npz(OUT["norm"] / f"{which}_start_state.npz", plates=np.array(plates), start=out)
    return {"plates": len(plates), "fallback": fallback}


def sensitivity_stats(which, f, X, info, veh, train):
    """Per-plate vehicle (idr0119) / DMSO (idr0064) mean per clip, and global per-dimension
    mean and SD over training fields (all clips and tiles)."""
    plates = sorted(info.plate.unique())
    pm = np.zeros((len(plates), X.shape[1], X.shape[3]), np.float32)
    for i, p in enumerate(plates):
        m = veh & (info.plate == p).to_numpy()
        pm[i] = X[m].astype(np.float32).mean((0, 2)) if m.any() else np.nan
    x = X[train].astype(np.float32).reshape(-1, X.shape[3])
    save_npz(OUT["norm"] / f"{which}_sensitivity.npz", plates=np.array(plates),
             plate_vehicle_mean=pm, global_mean=x.mean(0), global_sd=x.std(0))
    return {"plate_vehicle_mean": "per plate and clip, tiles pooled",
            "global": "per dimension over training fields", "plates_without_vehicle":
            [p for i, p in enumerate(plates) if np.isnan(pm[i]).all()]}


# ---------------------------------------------------------------- decoders

def fit_decoder(Xf, y, groups):
    ok = np.isfinite(y)
    pipe = make_pipeline(StandardScaler(), Ridge())
    gs = GridSearchCV(pipe, {"ridge__alpha": ALPHAS}, cv=GroupKFold(5),
                      scoring="neg_mean_squared_error", n_jobs=13)
    gs.fit(Xf[ok], y[ok], groups=groups[ok])
    best = gs.best_estimator_
    sc, rd = best.named_steps["standardscaler"], best.named_steps["ridge"]
    return {"scaler_mean": sc.mean_, "scaler_scale": sc.scale_, "coef": rd.coef_,
            "intercept": np.array(rd.intercept_), "alpha": np.array(gs.best_params_[
                "ridge__alpha"]), "n": np.array(ok.sum())}


def decoders(which, Z, f, arr, meta, train_mask, params):
    bx = T.boxes(which, meta["tiles"], meta["level"])
    ys = {}
    for i in np.flatnonzero(train_mask):
        c = arr["cells"][i]
        o = (T.idr0119_targets(c, meta["starts"], bx, params["g1_threshold"])
             if which == "idr0119" else
             T.idr0064_targets(c, meta["starts"], bx, params["pulse_prominence"]))
        for k, v in o.items():
            ys.setdefault(k, []).append(v.ravel())
    Xf = Z[train_mask].reshape(-1, Z.shape[-1])
    groups = np.repeat(f.plate[train_mask].to_numpy(), Z.shape[1] * Z.shape[2])
    out = {}
    for k, v in ys.items():
        d = fit_decoder(Xf, np.concatenate(v), groups)
        save_npz(OUT["decoders"] / f"{which}_{k}.npz", **d)
        out[k] = {"alpha": float(d["alpha"]), "n": int(d["n"])}
        print(which, k, out[k])
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry")
    ap.add_argument("--sets", nargs="+", default=["idr0119", "idr0064"])
    args = ap.parse_args()
    if args.dry:
        from pathlib import Path
        for k in OUT:
            OUT[k] = Path(args.dry) / k
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    w19, s19 = idr0119_split()
    w64, s64 = idr0064_split()
    enc = encodings(w19, w64)
    pilot = json.loads((REPO / "results" / "d0" / "pilot_probe.json").read_text())
    g1 = pilot["idr0119"]["g1_log_ratio_threshold"]
    prot = {"design_commit": DESIGN_COMMIT, "code_commit": head,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sets": {}}
    for which in args.sets:
        f, arr, meta = latents.load(which, arrays=("mean",), cells=True)
        X = arr["mean"]
        if which == "idr0119":
            nm, lines, mu, sd, info = norm_idr0119(f, X, w19)
            Z = apply_19(X, info, lines, mu, sd)
            veh = (info.h1_role == "vehicle").to_numpy()
            h1, h2 = info.h1_role, info.merge(w19[["plate", "well", "h2a_role"]],
                                              on=["plate", "well"], how="left").h2a_role
            # training in every model set: vehicle and H1 training singles (all AU565 singles
            # are H2a training), plus H2a training-pair combinations
            both = ((h1.isin(["train", "vehicle"]) & h2.isin(["", "train", "vehicle"])) |
                    ((h1 == "combo") & (h2 == "train"))).tolist()
            params = {"g1_threshold": g1}
        else:
            nm, plates, med, sd, info = norm_idr0064(f, X, w64)
            Z = apply_64(X, info, plates, med, sd)
            veh = (info.role == "vehicle").to_numpy()
            both = (info.role == "train").to_numpy().tolist()
            dm = [arr["cells"][i] for i in np.flatnonzero(veh)]
            params = {"pulse_prominence": T.pulse_prominence(dm, meta["starts"])}
        st = start_states(Z, info, veh, which)
        sens = sensitivity_stats(which, f, X, info, veh, np.array(both))
        dec = decoders(which, Z, f, arr, meta, np.array(both), params)
        prot["sets"][which] = {"clips": {"starts": meta["starts"], "frames_per_clip": T.NF,
                                         "frame_h": T.FRAME_H[which],
                                         "horizons": clip_table(which, meta["starts"])},
                               "embedding": meta, "normalisation": nm, "start_state": st,
                               "decoders": dec, "decoder_params": params,
                               "sensitivity_normalisations": sens,
                               "decoder_training_fields": int(np.sum(both))}
    prot["rules"] = RULES
    prot["splits"] = {"idr0119": {k: s19[k] for k in ("seed", "selection_seed", "val_line",
                                                      "test_lines", "test_pairs", "val_pair",
                                                      "selection_plates")},
                      "idr0064": s64}
    prot["encodings"] = enc
    prot["files_sha256"] = {str(p.relative_to(OUT["splits"].parent)): backup.sha256(p)
                            for p in WRITTEN}
    path = (OUT["splits"].parent if args.dry else REPO / "manifests") / \
        "drugworld_protocol.json"
    prot["files_sha256"] = {str(p.relative_to(OUT["splits"].parent)): v for p, v in zip(
        WRITTEN, prot["files_sha256"].values())} if args.dry else prot["files_sha256"]
    path.write_text(json.dumps(prot, indent=1, default=str))
    if args.dry:
        print("dry run:", len(WRITTEN) + 1, "files under", args.dry)
        return 0
    for p in WRITTEN + [path]:
        backup.backup(p)
    print("frozen:", len(WRITTEN) + 1, "files")
    return 0


RULES = {
    "start_state": "A1 C1: every forecast starts from the plate-matched vehicle state (mean "
                   "first-clip latent of the plate's vehicle wells per tile position, "
                   "normalised space; idr0064 vehicle = DMSO wells not kept for checks). "
                   "Fallback when a plate has no usable vehicle well: the same line's vehicle "
                   "wells on plates of the same imaging batch (listed). Secondary: the well's "
                   "own first clip, reported beside.",
    "horizons": "Horizon h = the stored clip whose last frame is the latest at or before h "
                "after drug addition (idr0119) or after the first frame (idr0064); windows "
                "in sets.<set>.clips.horizons. H1 primary: mean of 48 h and 72 h; 24 h and "
                "96 h secondary. H2a primary (A1 C2): mean of 24 h and 48 h (both clips end "
                "before the 48 h re-dose); 72 h secondary. H2b primary: mean of 2, 3 and 4 h; "
                "1 h secondary. H1c and H3 use the H1 horizons. The clip grid was fixed in "
                "D1 (starts every 6 frames), so A1's 16-24 h and 40-48 h windows are realised "
                "as 15-22.5 h and 39-46.5 h (frame times).",
    "redose_covariate": "A1 C2: a per-plate 0/1 covariate 'medium change and re-dose at 48 h' "
                        "given to every model (action-agnostic included); 1 for the six AU565 "
                        "single-drug-only plates (study text), 0 for all other plates, "
                        "including the three mixed AU565 plates holding both single-drug and "
                        "combination wells (no 48 h signature in their vehicle wells, "
                        "results/d2/au565_redose.json).",
    "h2a_baseline_singles": "Author decision on status report 2: H2a baselines (additive "
                            "composition, HSA, Bliss) and the ODE fit inputs use, for each "
                            "component (drug, dose), the fit-role AU565 single-drug wells on "
                            "plates without the 48 h re-dose (redose_48h 0) when any exist, "
                            "otherwise those on re-dosed plates; each single's effect is "
                            "taken against the AU565 fit vehicle wells of the same re-dose "
                            "class (all AU565 fit vehicle wells if that class has none; the "
                            "ODE vehicle condition follows the same preference). The source "
                            "used per component is recorded (results/d4 gates "
                            "baseline_single_sources; results/d3/ode_fit.json sources). Only "
                            "the secondary 72 h H2a horizon can be affected: primary H2a "
                            "horizons end at or before 48 h.",
    "normalisation": "A1 C3. idr0119: per line and clip, mean and SD over the line's vehicle "
                     "fields (tiles pooled), SD floored at the line's 5th percentile over "
                     "clips x dimensions; sensitivity: per-plate vehicle centring. idr0064: "
                     "per plate and clip, centred on the median over all compound wells of "
                     "the plate; scaled by the per-dimension SD of training compound wells; "
                     "sensitivity: DMSO-centred. Global standardisation (per-dimension mean "
                     "and SD over training wells) is the plan's second sensitivity analysis.",
    "error": "Primary error of a forecast = mean squared error over the 1,024 dimensions, "
             "averaged over tiles and fields of the well, over the primary horizon clips, then "
             "over replicate wells of the condition, then over 3 seeds.",
    "units": "A1 C6: H1 drug x line x dose (test: interior test doses and test-line "
             "extrapolation doses); H2a combination condition (also reported without "
             "Lapatinib+Palbociclib); H2b compound; H1c drug x dose in the held-out line for "
             "drugs measured in all four lines (Doxorubicin, Gemcitabine, Paclitaxel, "
             "Palbociclib), all doses of that line, models trained on the other lines' H1 "
             "training wells; conditions outside every training line's measured dose range "
             "are excluded and counted (A1 C5).",
    "model_sets": "H1 models: wells with h1_role train (selection: train_sel; validation: val "
                  "and val_extrap). H2a models (A1 O4): AU565 wells with h2a_role train, "
                  "selection train_sel, validation val. H1 models are never evaluated on H2a "
                  "sets nor H2a models on H1 sets. H1c: dw_line_k per line k. H2b: idr0064 "
                  "train / val / test compounds; DMSO vehicle wells are training; dmso_check "
                  "wells are never fitted.",
    "decoders": "Ridge (standardised inputs, alpha from 10^-1..10^5 by 5-fold GroupKFold "
                "over plates) from the normalised tile-mean latent of one clip to each "
                "readout target (src/drugworld/targets.py) of that clip and tile, fitted on "
                "fields that are training in every model set: idr0119 vehicle and H1 "
                "training wells outside H2a held-out sets, plus H2a training-pair "
                "combinations; idr0064 training compounds. G1 threshold = the D0 pilot GMM "
                "valley; pulse prominence = 3 x the median absolute frame-to-frame C/N "
                "change of DMSO tracks.",
    "f_decode": "A1 O1: on training-split wells (the decoder training fields), ridge on the "
                "standardised raw tile-mean latent, alpha by inner CV over 10^-1..10^5, R^2 of "
                "pooled held-out predictions from 5-fold GroupKFold over plates (as in the "
                "D0 pilot); idr0119 log count >= 0.8 and G1 fraction >= 0.5; idr0064 Erk C/N "
                ">= 0.5. The provisional vehicle value is reported beside. A failure drops "
                "that set's decoded readouts (and for idr0119 the ODE comparisons) only.",
    "f_dose": "Per drug x line, over its training-split doses (lowest and training interior "
              "doses; wells train and train_sel): displacement = Euclidean norm of the "
              "condition's mean normalised latent (fields, tiles and replicate wells "
              "averaged) at the 48 h clip minus the line's vehicle mean at that clip; "
              "Spearman rho of displacement vs dose. Pass if rho >= 0.6 for >= 50% of drug x "
              "line pairs. Vehicle is not a point (its displacement from itself is not a "
              "dose response); the variant with vehicle as dose 0 is reported beside.",
    "f_power": "From the frozen split: H1 test held-out-dose conditions >= 20 (interior test "
               "plus test extrapolation), H2a test combination conditions >= 6, H2b test "
               "compounds >= 60. A shortfall drops that hypothesis before training.",
    "baselines": "Plan section 7 unchanged; H1c other-lines baseline per A1 C5 (each training "
                 "line's response interpolated in log dose at the held-out dose, inside its "
                 "measured range). The H2a additive and HSA baselines use the same single-drug "
                 "wells as the H2a models (A1 O4).",
    "statistics": "Plan section 8 unchanged: one-sided Monte Carlo sign flip over held-out "
                  "conditions, 100,000 draws, per-drug results; Holm across G-Dose, G-Combo, "
                  "G-Compound, G-Plan on test; G-Compound Holm across structure sources "
                  "(ECFP4, MoLFormer-XL, target class).",
    "grid": "Plan section 6 unchanged: width {512, 768} x dropout {0.1, 0.3} x weight decay "
            "{0.05, 0.2}, lr 3e-4, batch 256, <= 12,000 steps, early stopping on the "
            "validation primary error every 250 steps, patience 3,000; 3 seeds (0, 1, 2) per "
            "family; dw_doseshuf / dw_structshuf permutation seed = 10,000 + model seed.",
}


if __name__ == "__main__":
    raise SystemExit(main())
