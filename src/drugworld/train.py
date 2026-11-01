"""Model sets, families and action content for D3 (plan section 6; A1 O4, C1, C2, C4).

Model sets (rows = fields; roles from splits/<set>_wells.csv):
  h1        idr0119; fit: h1_role train + vehicle; selection-only: train_sel; validation: val,
            val_extrap; test: test, test_extrap.
  h2a       idr0119 AU565 (A1 O4); fit: h2a_role train + vehicle; validation: val; test: test.
  line:<L>  idr0119 H1c, dw_line_k without line L; fit: h1 train + vehicle of the other lines;
            early stopping on the other lines' val / val_extrap.
  h2b       idr0064; fit: role train + vehicle (DMSO); validation: val; test: test.
Families: dw_full, dw_none, dw_doseshuf, dw_doseonly (idr0119), dw_line (H1c, dw_full action);
idr0064 dw_full:<src>, dw_structshuf:<src> (src = ecfp4 | molformer | target), dw_none.
Shuffles are fixed permutations with seed 10,000 + model seed, identical in training and
evaluation (`permutation_record` gives their SHA-256).

Sequence per field: step 0 = the plate's counterfactual vehicle start state (A1 C1, primary;
`start="own"` uses the field's own first clip), steps 1..T-1 = the measured clips; all in the
frozen normalised space (`norm="primary"`, or "global" for the sensitivity analysis, where the
start state is rebuilt by the same rule in that space).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from drugworld import REPO, latents

PROT_PATH = REPO / "manifests" / "drugworld_protocol.json"
SOURCES = {"ecfp4": "ecfp4_pca128", "molformer": "molformer_pca128",
           "target": "target_class_onehot"}


def protocol() -> dict:
    return json.loads(PROT_PATH.read_text())


def wells(which: str) -> pd.DataFrame:
    return pd.read_csv(REPO / "splits" / f"{which}_wells.csv", dtype={"plate": str},
                       keep_default_na=False)


def dataset_of(model_set: str) -> str:
    return "idr0064" if model_set == "h2b" else "idr0119"


@dataclass
class Data:
    which: str
    fields: pd.DataFrame          # one row per field with roles and condition columns
    z: np.ndarray                 # (N, T, K, D) float16, step 0 = start state
    starts: list


def load(which: str, norm: str = "primary", start: str = "vehicle") -> Data:
    f, arr, meta = latents.load(which, arrays=("mean",))
    X = arr["mean"].astype(np.float32)
    w = wells(which)
    info = f.merge(w, on=["plate", "well"], how="left", validate="many_to_one")
    if norm == "primary" and which == "idr0119":
        n = np.load(REPO / "norm" / "idr0119_line_vehicle.npz")
        li = np.array([list(n["lines"]).index(x) for x in info.line])
        X = (X - n["mean"][li][:, :, None]) / n["sd"][li][:, :, None]
    elif norm == "primary":
        n = np.load(REPO / "norm" / "idr0064_plate_median.npz")
        pi = np.array([list(n["plates"]).index(p) for p in info.plate])
        X = (X - n["median"][pi][:, :, None]) / n["sd"]
    elif norm == "global":
        n = np.load(REPO / "norm" / f"{which}_sensitivity.npz")
        X = (X - n["global_mean"]) / n["global_sd"]
    else:
        raise ValueError(norm)
    if start == "vehicle" and norm == "primary":
        s = np.load(REPO / "norm" / f"{which}_start_state.npz")
        pi = np.array([list(s["plates"]).index(p) for p in info.plate])
        X[:, 0] = s["start"][pi]
    elif start == "vehicle":
        # same rule as the frozen start states (A1 C1), in this normalised space
        veh = (info.h1_role == "vehicle") if which == "idr0119" else (info.role == "vehicle")
        x0 = X[:, 0].copy()
        for p in info.plate.unique():
            m = (info.plate == p).to_numpy()
            src = veh.to_numpy() & m
            if not src.any():               # the frozen fallback plates for this plate
                fb = json.loads(PROT_PATH.read_text())["sets"][which]["start_state"]["fallback"]
                src = veh.to_numpy() & info.plate.isin(fb[p]).to_numpy()
            X[m, 0] = x0[src].mean(0)
    return Data(which, info, X.astype(np.float16), meta["starts"])


# ---------------------------------------------------------------- roles per model set

def roles(d: Data, model_set: str) -> pd.Series:
    """'fit' | 'sel' | 'val' | 'test' | '' per field."""
    f = d.fields
    if model_set == "h1" or model_set.startswith("line:"):
        r = f.h1_role.map({"train": "fit", "vehicle": "fit", "train_sel": "sel", "val": "val",
                           "val_extrap": "val", "test": "test",
                           "test_extrap": "test"}).fillna("")
        if model_set.startswith("line:"):
            L = model_set.split(":", 1)[1]
            r = r.where(f.line != L, "heldout_line")
        return r
    if model_set == "h2a":
        return f.h2a_role.map({"train": "fit", "vehicle": "fit", "train_sel": "sel",
                               "val": "val", "test": "test"}).fillna("")
    if model_set == "h2b":
        return f.role.map({"train": "fit", "vehicle": "fit", "val": "val",
                           "test": "test"}).fillna("")
    raise ValueError(model_set)


# ---------------------------------------------------------------- action content

def _perm(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(10_000 + seed).permutation(n)


class Actions:
    """Per-field action content: toks (N, 2, tok_dim), tmask (N, 2), vehicle (N,), redose (N,)."""

    def __init__(self, d: Data, family: str, seed: int):
        self.family, self.seed = family, seed
        self.record = {}
        f = d.fields
        n = len(f)
        if d.which == "idr0119":
            e = np.load(REPO / "encodings" / "idr0119.npz", allow_pickle=True)
            drugs = list(e["drugs"])
            enc = np.concatenate([e["ecfp4"], e["molformer"]], 1)
            dmax = dict(zip(drugs, e["dose_max_nM"]))
            dose_map = self._dose_map(f, family, seed)
            self.toks = np.zeros((n, 2, enc.shape[1] + 1), np.float32)
            self.tmask = np.zeros((n, 2), bool)
            for i, r in enumerate(f.itertuples()):
                for j, (dr, ds) in enumerate(((r.drug1, r.dose1), (r.drug2, r.dose2))):
                    if dr:
                        ds = dose_map.get((r.line, dr, float(ds)), float(ds))
                        if family != "dw_doseonly":
                            self.toks[i, j, :-1] = enc[drugs.index(dr)]
                        self.toks[i, j, -1] = np.log10(ds / dmax[dr])
                        self.tmask[i, j] = True
            self.vehicle = (f.drug1 == "").to_numpy(np.float32)
            self.redose = f.redose_48h.astype(float).to_numpy(np.float32)
        else:
            src = family.split(":")[1] if ":" in family else "ecfp4"
            e = np.load(REPO / "encodings" / "idr0064.npz", allow_pickle=True)
            comps = list(e["compounds"])
            enc = e[SOURCES[src]].astype(np.float32)
            if family.startswith("dw_structshuf"):
                p = _perm(len(comps), seed)
                enc = enc[p]
                self.record["structure_permutation_sha256"] = hashlib.sha256(
                    p.astype(np.int64).tobytes()).hexdigest()
            self.toks = np.zeros((n, 2, enc.shape[1] + 1), np.float32)
            self.tmask = np.zeros((n, 2), bool)
            for i, c in enumerate(f.compound_key):
                if c:
                    self.toks[i, 0, :-1] = enc[comps.index(c)]
                    self.tmask[i, 0] = True
            self.vehicle = (f.compound == "DMSO").to_numpy(np.float32)
            self.redose = np.zeros(n, np.float32)
        self.tok_dim = self.toks.shape[-1]

    def _dose_map(self, f, family, seed):
        """dw_doseshuf: within each drug x line, its measured single-drug doses permuted."""
        if family != "dw_doseshuf":
            return {}
        out, rec = {}, []
        s = f[(f.drug1 != "") & (f.drug2 == "")]
        grids = s.groupby(["line", "drug1"]).dose1.agg(lambda x: sorted({float(v) for v in x}))
        for k, ((line, drug), grid) in enumerate(sorted(grids.items())):
            p = np.random.default_rng([10_000 + seed, k]).permutation(len(grid))
            for a, b in zip(grid, np.array(grid)[p]):
                out[(line, drug, a)] = float(b)
            rec.append(f"{line}|{drug}|{','.join(map(str, p))}")
        self.record["dose_permutation_sha256"] = hashlib.sha256(
            "\n".join(rec).encode()).hexdigest()
        return out


def mode_of(family: str) -> str:
    return "none" if family == "dw_none" else "content"
