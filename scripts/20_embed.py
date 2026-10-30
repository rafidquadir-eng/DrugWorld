"""D1 embedding (plan section 4): stream every well field, embed with frozen V-JEPA 2, keep
latents and per-cell readouts, drop the raw frames.

idr0119: every field of every well (760 wells, 3,026 fields); channel mapping and pyramid
level from the D0 pilot choice (`results/d0/pilot_probe.json`); 16-frame clips (8 h) starting
every 6 frames (3 h): 0, 6, ..., 174 and 177. idr0064: every well (450); mapping (Erk-KTR, H2B,
mean), level 0, 4 tiles; clips of 16 frames (48 min) every 4 frames: 0, 4, ..., 104 and 105.
The tile grid is fixed per set (below) before embedding.

Per (field, clip, tile): `mean` = tile-mean pooled tokens (D = 1024) and `grid` = time-mean 4 x 4
token grid, float16; the static-clip control (each clip = its middle frame repeated, plan
section 9) is embedded the same way (`static_mean`, `static_grid`). Per field: the leakage
fingerprint (GeneWorld `fingerprint`, on the phase channel for idr0119 and H2B for idr0064),
SHA-1 of frame 0, per-frame mean intensity, filled slots and fetch errors. Per-cell readouts go
to data/readouts/<set>/<plate>.npz (frame, centroid y/x, value; value = log HDHB nuclear/ring
ratio for idr0119, Erk C/N for idr0064).

One shard per plate (data/latents/<set>/<plate>.npz), written once, listed with SHA-256 in
manifests/latent_shards.tsv and backed up. Failures (fetch, or a channel missing > 10% of its
slots) are listed by cause in results/d1/embed_failures_<set>.json. Re-running skips plates
whose shard exists.

  PYTHONPATH=src python scripts/20_embed.py --set idr0119 --worker 0 --nworkers 8
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import queue
import threading
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from drugworld import REPO, backup, idr0064, idr0119, shards
from drugworld import stream as S
from drugworld.embed import NF, TILE, VJEPA2, compose, downsample, scale
from drugworld.fetch import Fetcher
from geneworld.preprocess.fingerprint import fingerprint

print = functools.partial(print, flush=True)
STARTS = {"idr0119": list(range(0, 175, 6)) + [177], "idr0064": list(range(0, 105, 4)) + [105]}
TILES = {"idr0119_L0": [(y, x) for y in (8, 264, 520, 776) for x in (64, 320, 576, 832, 1088)],
         "idr0119_L1": [(y, x) for y in (4, 260) for x in (96, 352)],
         "idr0064_L0": [(y, x) for y in (0, 256) for x in (0, 256)]}


def config(which: str) -> tuple[str, int]:
    if which == "idr0064":
        return "erk", 1
    chosen = json.loads((REPO / "results" / "d0" / "pilot_probe.json").read_text())[
        "idr0119"]["chosen"]
    mapping, lvl = chosen.split("_L")
    return mapping, 2 ** int(lvl)


def fields(which: str) -> pd.DataFrame:
    if which == "idr0119":
        fm = pd.read_parquet(REPO / "data" / "annotations" / "idr0119_files.parquet")
        return fm[["plate", "well", "field"]].drop_duplicates().sort_values(
            ["plate", "well", "field"]).reset_index(drop=True)
    a = idr0064.annotations()
    return a[["plate", "well"]].assign(field=1).sort_values(["plate", "well"]).reset_index(
        drop=True)


def embed_field(model, sc, mapping, k, tiles, starts):
    x = downsample(compose(sc, mapping), k)
    out = {"mean": [], "grid": [], "static_mean": [], "static_grid": []}
    for s in starts:
        clips = torch.stack([x[s:s + NF, :, y:y + TILE, xx:xx + TILE] for (y, xx) in tiles])
        m, g = model(clips)
        out["mean"].append(m)
        out["grid"].append(g)
        mid = clips[:, NF // 2:NF // 2 + 1].expand(-1, NF, -1, -1, -1)
        m, g = model(mid)
        out["static_mean"].append(m)
        out["static_grid"].append(g)
    return {kk: np.stack(v) for kk, v in out.items()}       # (clips, tiles, ...)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["idr0119", "idr0064"], required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--cpus", type=int, default=14)
    args = ap.parse_args()
    which = args.set
    mapping, k = config(which)
    tiles = TILES[f"{which}_L{k.bit_length() - 1}"]
    starts = STARTS[which]
    F_all = fields(which)
    plates = sorted(F_all.plate.unique())[args.worker::args.nworkers]
    lat_dir = REPO / "data" / "latents" / which
    ro_dir = REPO / "data" / "readouts" / which
    lat_dir.mkdir(parents=True, exist_ok=True)
    ro_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in plates if not (lat_dir / f"{p}.npz").exists()]
    print(f"worker {args.worker}: {which} {mapping} level {k.bit_length() - 1}, "
          f"{len(tiles)} tiles, {len(starts)} clips; plates {plates}, to do {todo}")
    fm = pd.read_parquet(REPO / "data" / "annotations" / "idr0119_files.parquet") \
        if which == "idr0119" else None
    Fch = Fetcher(args.streams)
    q: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        for p in todo:
            for r in F_all[F_all.plate == p].itertuples():
                t0 = time.time()
                got = (S.fetch_idr0119(Fch, fm, r.plate, r.well, r.field) if which == "idr0119"
                       else S.fetch_idr0064(Fch, r.plate, r.well))
                q.put((r, got, time.time() - t0))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    model = VJEPA2()
    pool = Pool(args.cpus)
    fail_path = REPO / "results" / "d1" / f"embed_failures_{which}_w{args.worker}.json"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    failures = json.loads(fail_path.read_text()) if fail_path.exists() else []
    store: dict = {}
    expected = {p: int((F_all.plate == p).sum()) for p in todo}
    seen = {p: 0 for p in todo}

    def flush(p):
        s = store.pop(p, None)
        if s and s["keys"]:
            path = lat_dir / f"{p}.npz"
            arr = {kk: np.stack(s[kk]) for kk in ("mean", "grid", "static_mean", "static_grid",
                                                  "fingerprint", "frame_mean")}
            np.savez(path, keys=np.array(s["keys"]), starts=np.array(starts),
                     tiles=np.array(tiles), mapping=mapping, level=k.bit_length() - 1,
                     frame0_sha1=np.array(s["sha1"]), filled=np.array(s["filled"], dtype=object),
                     errors=np.array(s["errors"], dtype=object), **arr)
            rpath = ro_dir / f"{p}.npz"
            np.savez(rpath, keys=np.array(s["keys"]),
                     cells=np.array(s["cells"], dtype=object))
            for q_ in (path, rpath):
                shards.register(q_)
                backup.backup(q_)
        fail_path.write_text(json.dumps(failures, indent=1))
        print(f"{p}: shard written ({len(s['keys']) if s else 0} fields)")

    while (item := q.get()) is not None:
        r, (ch, have, errs), tf = item
        seen[r.plate] += 1
        t0 = time.time()
        filled, cause = S.fill(ch, have)
        key = f"{r.well}/{r.field}"
        if cause:
            failures.append({"plate": r.plate, "well": r.well, "field": int(r.field),
                             "cause": cause, "fetch_errors": errs[:20]})
        else:
            if which == "idr0119":
                cells = pool.map(S.cells_idr0119, [(t, ch["nuclear"][t], ch["hdhb"][t])
                                                   for t in range(idr0119.N_GRID)])
                fp_ch = ch["phase"]
            else:
                cells = pool.map(S.cells_idr0064, [(t, ch["h2b"][t], ch["ktr"][t])
                                                   for t in range(idr0064.N_T)])
                fp_ch = ch["h2b"]
            fp = fingerprint(fp_ch)
            sc = {c: scale(x) for c, x in ch.items()}
            emb = embed_field(model, sc, mapping, k, tiles, starts)
            del sc
            torch.cuda.empty_cache()
            s = store.setdefault(r.plate, {kk: [] for kk in (
                "keys", "mean", "grid", "static_mean", "static_grid", "fingerprint",
                "frame_mean", "sha1", "filled", "errors", "cells")})
            s["keys"].append(key)
            for kk in ("mean", "grid", "static_mean", "static_grid"):
                s[kk].append(emb[kk])
            s["fingerprint"].append(fp["vector"])
            s["sha1"].append(fp["frame0_sha1"])
            s["frame_mean"].append(np.stack([x.reshape(len(x), -1).mean(1)
                                             for x in ch.values()]).astype(np.float32))
            s["filled"].append(filled)
            s["errors"].append(errs)
            s["cells"].append(np.concatenate(cells))
        print(f"{r.plate} {key}: fetch {tf:.0f}s, process {time.time() - t0:.0f}s, "
              f"filled {len(filled)}, errors {len(errs)}{', FAILED ' + cause if cause else ''}")
        if seen[r.plate] >= expected[r.plate]:
            flush(r.plate)
    pool.close()
    print("done")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    raise SystemExit(main())
