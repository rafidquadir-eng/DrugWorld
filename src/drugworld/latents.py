"""Load D1 latent and readout shards (after the shard guard), one table per set.

idr0064 has one shard per plate; idr0119 one per plate and worker (`<plate>.w<k>of<n>.npz`).
`load(which)` returns (fields, arrays): `fields` has one row per embedded field (plate, well,
field, frame0_sha1, n_filled, n_errors) in a fixed order (plate, well, field); arrays hold
`mean` (fields, clips, tiles, 1024) float16 and, on request, `grid`, `static_mean`,
`fingerprint`, `frame_mean`, and `cells` (list of (n, 4) arrays: frame, cy, cx, value).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from drugworld import REPO, shards

KEYS = ("mean", "grid", "static_mean", "static_grid", "fingerprint", "frame_mean")


def shard_paths(which: str):
    return sorted((REPO / "data" / "latents" / which).glob("*.npz"))


def load(which: str, arrays=("mean",), cells: bool = False, check: bool = True):
    if check:
        shards.check()
    rows, out, cl = [], {k: [] for k in arrays}, []
    meta = {}
    for p in shard_paths(which):
        plate = p.stem.split(".w")[0]
        z = np.load(p, allow_pickle=True)
        meta.setdefault("starts", z["starts"].tolist())
        meta.setdefault("tiles", z["tiles"].tolist())
        meta.setdefault("mapping", str(z["mapping"]))
        meta.setdefault("level", int(z["level"]))
        if z["starts"].tolist() != meta["starts"] or z["tiles"].tolist() != meta["tiles"]:
            raise ValueError(f"{p.name}: clip starts or tiles differ from the other shards")
        filled = list(z["filled"]) if z["filled"].ndim == 1 else [list(x) for x in z["filled"]]
        errors = list(z["errors"]) if z["errors"].ndim == 1 else [list(x) for x in z["errors"]]
        for i, key in enumerate(z["keys"]):
            well, field = str(key).split("/")
            rows.append({"plate": plate, "well": well, "field": int(field),
                         "frame0_sha1": str(z["frame0_sha1"][i]),
                         "n_filled": len(filled[i]), "n_errors": len(errors[i])})
        for k in arrays:
            out[k].append(z[k])
        if cells:
            r = np.load(REPO / "data" / "readouts" / which / p.name, allow_pickle=True)
            if r["keys"].tolist() != z["keys"].tolist():
                raise ValueError(f"{p.name}: readout keys differ from latent keys")
            cl += [np.asarray(c, dtype=np.float32) for c in r["cells"]]
    fields = pd.DataFrame(rows)
    arr = {k: np.concatenate(v) for k, v in out.items()}
    order = fields.sort_values(["plate", "well", "field"]).index.to_numpy()
    fields = fields.loc[order].reset_index(drop=True)
    arr = {k: v[order] for k, v in arr.items()}
    if cells:
        arr["cells"] = [cl[i] for i in order]
    if fields.duplicated(["plate", "well", "field"]).any():
        raise ValueError(f"{which}: a field appears in two shards")
    return fields, arr, meta
