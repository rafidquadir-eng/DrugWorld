"""Stream one well field into memory (no raw data kept) and compute per-cell readouts.

Same steps as the D0 pilot (`scripts/10_pilot.py`), shared by D1 (`scripts/20_embed.py`).
Missing time slots take the nearest present slot of the same channel and are listed; a field
whose channel misses more than MAX_MISSING_FRAC of its slots after retries is not embedded and
is listed as failed with the cause (fixed before D1 ran).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import tifffile

from drugworld import idr0064, idr0119
from drugworld import readouts as R
from drugworld.fetch import Fetcher

MAX_MISSING_FRAC = 0.10


def fetch_idr0119(F: Fetcher, fm: pd.DataFrame, plate, well, field):
    f = fm[(fm.plate == plate) & (fm.well == well) & (fm.field == field) & (fm.slot >= 0)]
    f = f.drop_duplicates(["channel", "slot"])
    ch = {c: np.zeros((idr0119.N_GRID, 1040, 1408), np.uint16)
          for c in idr0119.CHANNEL.values()}
    have = {c: np.zeros(idr0119.N_GRID, bool) for c in ch}
    errs = []
    for r, (u, b, e) in zip(f.itertuples(), F.map(f.url)):
        if e:
            errs.append(f"{r.channel}/{r.slot}: {e}")
            continue
        try:
            ch[r.channel][r.slot] = tifffile.imread(io.BytesIO(b))
            have[r.channel][r.slot] = True
        except Exception as exc:                       # corrupt file: listed
            errs.append(f"{r.channel}/{r.slot}: decode {type(exc).__name__}")
    return ch, have, errs


def fetch_idr0064(F: Fetcher, plate, well):
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


def fill(ch, have) -> tuple[list[str], str | None]:
    """Fill missing slots from the nearest present one; returns (filled, failure cause)."""
    filled, cause = [], None
    for c, x in ch.items():
        n_miss = int((~have[c]).sum())
        if n_miss > MAX_MISSING_FRAC * len(have[c]):
            cause = f"channel {c}: {n_miss} of {len(have[c])} slots missing"
        idx = np.flatnonzero(have[c])
        for t in np.flatnonzero(~have[c]):
            if len(idx):
                x[t] = x[idx[np.argmin(np.abs(idx - t))]]
                filled.append(f"{c}/{t}")
    return filled, cause


def cells_idr0119(args):
    t, red, green = args
    lab = R.nuclei(red)
    cy, cx, r = R.ring_ratio(lab, green)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(r)
    return np.column_stack([np.full(len(cy), t), cy, cx, lr]).astype(np.float32)


def cells_idr0064(args):
    t, h2b, ktr = args
    lab = R.nuclei(h2b, min_distance=3)
    cy, cx, r = R.ring_ratio(lab, ktr)
    with np.errstate(divide="ignore", invalid="ignore"):
        cn = 1.0 / r
    return np.column_stack([np.full(len(cy), t), cy, cx, cn]).astype(np.float32)
