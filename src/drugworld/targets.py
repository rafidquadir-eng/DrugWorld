"""Decoder targets per (field, clip, tile) from the per-cell readouts (plan section 5 Decoders).

idr0119: log(1 + mean cells per frame in the tile), G1 fraction (cells over the clip's 16
frames whose log HDHB nuclear/ring ratio exceeds G1_THRESHOLD, the D0 pilot's vehicle-cell GMM
valley, results/d0/pilot_probe.json) and S/G2/M fraction (1 - G1 fraction). Tile boxes are in
full-resolution pixels (level-1 tiles cover 512 x 512).
idr0064: Erk C/N mean over the clip's cells in the tile, and Erk pulse rate: cells are linked
frame to frame by mutual nearest centroid within LINK_PX; for each track spanning >= MIN_TRACK
of the clip's 16 frames, peaks of its C/N series with prominence >= PULSE_PROMINENCE are counted;
pulse rate = peaks per track-hour over all such tracks in the tile. PULSE_PROMINENCE is 3 x the
median absolute frame-to-frame C/N change of DMSO-well tracks (vehicle only), computed by
`pulse_prominence` and fixed in the protocol.
Tiles without a cell (G1, S/G2/M, Erk) or without a qualifying track (pulse rate) are NaN.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

NF = 16
LINK_PX = 4.0
MIN_TRACK = 12
FRAME_H = {"idr0119": 0.5, "idr0064": 0.05}


def boxes(which: str, tiles, level: int):
    k = 2 ** level
    return [(k * y, k * x, 256 * k) for y, x in tiles]


def _in_box(cy, cx, box):
    y, x, s = box
    return (cy >= y) & (cy < y + s) & (cx >= x) & (cx < x + s)


def idr0119_targets(cells, starts, bx, thr):
    t, cy, cx, v = cells.T
    out = np.full((3, len(starts), len(bx)), np.nan, np.float32)
    for j, b in enumerate(bx):
        inb = _in_box(cy, cx, b)
        for i, s0 in enumerate(starts):
            m = inb & (t >= s0) & (t < s0 + NF)
            out[0, i, j] = np.log1p(m.sum() / NF)
            vv = v[m]
            vv = vv[np.isfinite(vv)]
            if len(vv):
                out[1, i, j] = (vv > thr).mean()
                out[2, i, j] = 1.0 - out[1, i, j]
    return {"log_count": out[0], "g1_fraction": out[1], "sg2m_fraction": out[2]}


def tracks(cells, s0):
    """Mutual-nearest-centroid tracks over frames s0..s0+15: list of (frames, values)."""
    t, cy, cx, v = cells.T
    frames = [np.flatnonzero(t == s0 + k) for k in range(NF)]
    open_, done = {}, []
    for k, idx in enumerate(frames):
        pts = np.column_stack([cy[idx], cx[idx]])
        new = {}
        if open_ and len(idx):
            prev_ids = list(open_)
            prev = np.array([open_[i]["pos"] for i in prev_ids])
            d1, j1 = cKDTree(pts).query(prev)
            d2, j2 = cKDTree(prev).query(pts)
            for a, (dd, b) in enumerate(zip(d1, j1)):
                if dd <= LINK_PX and j2[b] == a:
                    tr = open_.pop(prev_ids[a])
                    tr["pos"] = pts[b]
                    tr["v"].append(v[idx[b]])
                    new[b] = tr
        done += list(open_.values())
        open_ = {}
        for b in range(len(idx)):
            open_[b] = new.get(b, {"pos": pts[b], "v": [v[idx[b]]], "y": pts[b][0],
                                   "x": pts[b][1]})
    done += list(open_.values())
    return done


def pulse_prominence(cell_lists, starts) -> float:
    d = []
    for cells in cell_lists:
        for s0 in starts:
            for tr in tracks(cells, s0):
                x = np.asarray(tr["v"], float)
                if len(x) >= MIN_TRACK and np.isfinite(x).all():
                    d.append(np.abs(np.diff(x)))
    return float(3 * np.median(np.concatenate(d)))


def idr0064_targets(cells, starts, bx, prominence):
    t, cy, cx, v = cells.T
    out = np.full((2, len(starts), len(bx)), np.nan, np.float32)
    for i, s0 in enumerate(starts):
        trs = [tr for tr in tracks(cells, s0) if len(tr["v"]) >= MIN_TRACK
               and np.isfinite(tr["v"]).all()]
        for j, b in enumerate(bx):
            m = _in_box(cy, cx, b) & (t >= s0) & (t < s0 + NF)
            vv = v[m]
            vv = vv[np.isfinite(vv)]
            if len(vv):
                out[0, i, j] = vv.mean()
            sel = [tr for tr in trs if _in_box(np.array([tr["y"]]), np.array([tr["x"]]), b)[0]]
            if sel:
                peaks = sum(len(find_peaks(np.asarray(tr["v"]), prominence=prominence)[0])
                            for tr in sel)
                hours = sum(len(tr["v"]) for tr in sel) * FRAME_H["idr0064"]
                out[1, i, j] = peaks / hours
    return {"erk_cn": out[0], "pulse_rate": out[1]}
