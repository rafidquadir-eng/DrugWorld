"""D0 embedding pilot on vehicle wells only (plan section 3, item 3; the split is not frozen
yet, so no drug well is read).

idr0119: field 1 of every vehicle well (83 wells, 34 plates, 4 lines). Each field is streamed
(193 slots x 3 channels, missing slots filled from the nearest present slot and listed), the
channels are scaled per field, and clips (16 frames, starts 0, 6, ..., 174 and 177) are embedded
with frozen V-JEPA 2 under two channel mappings (phase grey; nuclear/HDHB/phase as RGB) at two
pyramid levels (level 0: 20 tiles of 256 px over the central 1280 x 1024; level 1, 2x mean-pooled:
4 tiles over the central 512 x 512 of 704 x 520). Per-cell readouts (nucleus centroid, log HDHB
nuclear/ring ratio) are kept for every frame.

idr0064: all 18 DMSO wells; clips of 16 frames, starts 0, 4, ..., 104 and 105; mapping
(Erk-KTR, H2B, mean); level 0, 4 tiles; per-cell Erk C/N ratio per frame.

  PYTHONPATH=src python scripts/10_pilot.py --set idr0119 --worker 0 --nworkers 8
Output: data/pilot/<set>/<plate>_<well>_f<field>.npz (gitignored; backed up).
"""
from __future__ import annotations

import argparse
import functools
import io
import os
import threading
import time
import queue
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile
import torch

from drugworld import REPO, backup, idr0064, idr0119
from drugworld import readouts as R
from drugworld.embed import VJEPA2, compose, downsample, scale, NF, TILE
from drugworld.fetch import Fetcher

print = functools.partial(print, flush=True)
OUT = REPO / "data" / "pilot"
STARTS = {"idr0119": list(range(0, 175, 6)) + [177], "idr0064": list(range(0, 105, 4)) + [105]}
L0_TILES = [(y, x) for y in (8, 264, 520, 776) for x in (64, 320, 576, 832, 1088)]
L1_TILES = [(y, x) for y in (4, 260) for x in (96, 352)]
E_TILES = [(y, x) for y in (0, 256) for x in (0, 256)]
CONFIGS = {"idr0119": [("phase", 1, L0_TILES), ("phase", 2, L1_TILES),
                       ("rgb", 1, L0_TILES), ("rgb", 2, L1_TILES)],
           "idr0064": [("erk", 1, E_TILES)]}


def jobs(which: str) -> pd.DataFrame:
    if which == "idr0119":
        a = idr0119.annotations()
        v = a[a.drug1 == ""][["plate", "well"]].assign(field=1)
    else:
        a = idr0064.annotations()
        v = a[a.control == "vehicle control"][["plate", "well"]].assign(field=1)
    return v.sort_values(["plate", "well"]).reset_index(drop=True)


def fetch_idr0119(F: Fetcher, fm: pd.DataFrame, plate, well, field):
    f = fm[(fm.plate == plate) & (fm.well == well) & (fm.field == field) & (fm.slot >= 0)]
    f = f.drop_duplicates(["channel", "slot"])
    ch = {c: np.zeros((idr0119.N_GRID, 1040, 1408), np.uint16) for c in idr0119.CHANNEL.values()}
    have = {c: np.zeros(idr0119.N_GRID, bool) for c in ch}
    errs = []
    for r, (u, b, e) in zip(f.itertuples(), F.map(f.url)):
        if e:
            errs.append(f"{r.channel}/{r.slot}: {e}")
            continue
        ch[r.channel][r.slot] = tifffile.imread(io.BytesIO(b))
        have[r.channel][r.slot] = True
    return ch, have, errs


def fetch_idr0064(F: Fetcher, plate, well, field):
    urls = idr0064.chunk_urls(plate, well)
    ch = {"ktr": np.zeros((idr0064.N_T, 512, 512), np.uint16),
          "h2b": np.zeros((idr0064.N_T, 512, 512), np.uint16)}
    have = {c: np.zeros(idr0064.N_T, bool) for c in ch}
    errs = []
    for (t, c, _), (u, b, e) in zip(urls, F.map(u for _, _, u in urls)):
        name = ("ktr", "h2b")[c]
        if e:
            errs.append(f"{name}/{t}: {e}")
            continue
        ch[name][t] = idr0064.decode(b)
        have[name][t] = True
    return ch, have, errs


def fill(ch, have) -> list[str]:
    """Missing slots take the nearest present slot of the same channel (listed)."""
    filled = []
    for c, x in ch.items():
        idx = np.flatnonzero(have[c])
        for t in np.flatnonzero(~have[c]):
            if len(idx):
                x[t] = x[idx[np.argmin(np.abs(idx - t))]]
                filled.append(f"{c}/{t}")
    return filled


def _cells_0119(args):
    t, red, green = args
    lab = R.nuclei(red)
    cy, cx, r = R.ring_ratio(lab, green)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(r)
    return np.column_stack([np.full(len(cy), t), cy, cx, lr]).astype(np.float32)


def _cells_0064(args):
    t, h2b, ktr = args
    lab = R.nuclei(h2b, min_distance=3)
    cy, cx, r = R.ring_ratio(lab, ktr)
    with np.errstate(divide="ignore", invalid="ignore"):
        cn = 1.0 / r
    return np.column_stack([np.full(len(cy), t), cy, cx, cn]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["idr0119", "idr0064"], required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--cpus", type=int, default=14)
    args = ap.parse_args()
    which = args.set
    J = jobs(which)
    J = J.iloc[args.worker::args.nworkers]
    out = OUT / which
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in J.itertuples()
            if not (out / f"{r.plate}_{r.well}_f{r.field}.npz").exists()]
    print(f"worker {args.worker}: {len(J)} fields, {len(todo)} to do")
    fm = pd.read_parquet(REPO / "data" / "annotations" / "idr0119_files.parquet") \
        if which == "idr0119" else None
    F = Fetcher(args.streams)
    q: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        for r in todo:
            t0 = time.time()
            if which == "idr0119":
                got = fetch_idr0119(F, fm, r.plate, r.well, r.field)
            else:
                got = fetch_idr0064(F, r.plate, r.well, r.field)
            q.put((r, got, time.time() - t0))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    model = VJEPA2()
    pool = Pool(args.cpus)
    starts = STARTS[which]
    while (item := q.get()) is not None:
        r, (ch, have, errs), tf = item
        t0 = time.time()
        filled = fill(ch, have)
        if which == "idr0119":
            cells = pool.map(_cells_0119, [(t, ch["nuclear"][t], ch["hdhb"][t])
                                           for t in range(idr0119.N_GRID)])
        else:
            cells = pool.map(_cells_0064, [(t, ch["h2b"][t], ch["ktr"][t])
                                           for t in range(idr0064.N_T)])
        tc = time.time() - t0
        sc = {c: scale(x) for c, x in ch.items()}
        res = {}
        for mapping, k, tiles in CONFIGS[which]:
            x = downsample(compose(sc, mapping), k)
            means = []
            for s in starts:
                clips = torch.stack([x[s:s + NF, :, y:y + TILE, xx:xx + TILE]
                                     for (y, xx) in tiles])
                m, _ = model(clips)
                means.append(m)
            res[f"{mapping}_L{k.bit_length() - 1}"] = np.stack(means)   # (clips, tiles, D)
        del sc
        torch.cuda.empty_cache()
        path = out / f"{r.plate}_{r.well}_f{r.field}.npz"
        np.savez(path, cells=np.concatenate(cells), starts=np.array(starts),
                 filled=np.array(filled), errors=np.array(errs), **res)
        backup.backup(path)
        print(f"{r.plate} {r.well}: fetch {tf:.0f}s, cells {tc:.0f}s, embed "
              f"{time.time() - t0 - tc:.0f}s; filled {len(filled)}, errors {len(errs)}")
    pool.close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    raise SystemExit(main())
