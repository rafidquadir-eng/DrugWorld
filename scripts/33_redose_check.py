"""A1 C2 confirmation on all D1 vehicle fields (author decision on status report 2): is there a
48 h re-dose signature in the vehicle wells of the three mixed AU565 plates?

Vehicle fields only (no drug-treated well is used). Per field, four step ratios at the 48 h
medium change (slot 95 to 96): |step| of log(1 + cell count) and of the log frame mean of each of
the 3 channels, each divided by the median |step| over slots 80-109 excluding that one. If slot
96 is missing, the step is taken over the gap (slot 95 to 97, from the readouts' frame grid).

Reference: the vehicle fields of the other three lines (never re-dosed). A field shows the
signature on a measure if its ratio exceeds the 99th percentile of that measure in the
reference; a plate shows it if the median over its vehicle fields of any measure does, or if
its acquisition has a gap of >= 60 min at slots 89-102 (results/d2/au565_redose.json).

Rule, fixed before computing: confirmed (covariate 0 for the mixed plates) if no mixed plate
shows the signature and at least 4 of the 6 single-drug-only plates do (the check has power);
otherwise ambiguous (stop and ask). Writes results/d2/au565_redose_check.json.
"""
from __future__ import annotations

import json

import numpy as np

from drugworld import REPO, backup, idr0119, latents

REF = slice(80, 110)
SLOT = 95


def ratio(series):
    s = np.asarray(series, float)
    ok = np.isfinite(s)
    if not ok[SLOT] or not (ok[SLOT + 1] or ok[SLOT + 2]):
        return None
    d = np.abs(np.diff(s))
    nxt = SLOT if ok[SLOT + 1] else SLOT + 1      # missing slot 96: step over the gap
    step = abs(s[nxt + 1] - s[SLOT]) if nxt != SLOT else d[SLOT]
    ref = np.r_[d[REF.start:SLOT], d[SLOT + 2:REF.stop]]
    ref = ref[np.isfinite(ref)]
    return float(step / np.median(ref)) if len(ref) and np.median(ref) > 0 else None


def main() -> int:
    a = idr0119.annotations()
    veh = a[a.drug1 == ""][["plate", "well", "line"]]
    f, arr, _ = latents.load("idr0119", arrays=("frame_mean",), cells=True)
    m = f.merge(veh, on=["plate", "well"], how="inner")
    idx = f.reset_index().merge(veh, on=["plate", "well"], how="inner")["index"].to_numpy()
    rows = []
    for i, r in zip(idx, m.itertuples()):
        c = arr["cells"][i]
        n = np.bincount(c[:, 0].astype(int), minlength=idr0119.N_GRID).astype(float)
        present = np.zeros(idr0119.N_GRID, bool)
        present[np.unique(c[:, 0].astype(int))] = True
        cnt = np.where(present, np.log1p(n), np.nan)
        fm = arr["frame_mean"][i].astype(float)
        fm = np.where(fm > 0, np.log(np.maximum(fm, 1e-12)), np.nan)
        fm[:, ~present] = np.nan
        rows.append({"plate": r.plate, "well": r.well, "field": int(r.field), "line": r.line,
                     "count": ratio(cnt), **{f"ch{k}": ratio(fm[k]) for k in range(3)}})
    meas = ["count", "ch0", "ch1", "ch2"]
    ref = {k: [x[k] for x in rows if x["line"] != "AU565" and x[k] is not None] for k in meas}
    thr = {k: float(np.percentile(v, 99)) for k, v in ref.items()}
    rec = json.loads((REPO / "results" / "d2" / "au565_redose.json").read_text())["plates"]
    plates = {}
    for p in sorted({x["plate"] for x in rows if x["line"] == "AU565"}):
        fs = [x for x in rows if x["plate"] == p]
        med = {k: (float(np.median([x[k] for x in fs if x[k] is not None]))
                   if any(x[k] is not None for x in fs) else None) for k in meas}
        gap = rec[p]["max_frame_gap_min_slots_90_102"] >= 60
        sig = gap or any(med[k] is not None and med[k] > thr[k] for k in meas)
        plates[p] = {"kind": rec[p]["kind"], "n_vehicle_fields": len(fs),
                     "median_ratio": med, "gap_ge_60_min": bool(gap), "signature": bool(sig)}
    mixed = [p for p, v in plates.items() if v["kind"] == "mixed"]
    single = [p for p, v in plates.items() if v["kind"] == "single-drug only"]
    n_single_sig = sum(plates[p]["signature"] for p in single)
    confirmed = not any(plates[p]["signature"] for p in mixed) and n_single_sig >= 4
    out = {"rule": __doc__.split("\n\n")[-1].split(" Writes")[0].replace("\n", " "),
           "reference_99th_percentile": thr,
           "reference_fields": {k: len(v) for k, v in ref.items()}, "plates": plates,
           "single_only_plates_with_signature": n_single_sig,
           "mixed_plates_with_signature": [p for p in mixed if plates[p]["signature"]],
           "outcome": "confirmed" if confirmed else "ambiguous"}
    path = REPO / "results" / "d2" / "au565_redose_check.json"
    path.write_text(json.dumps(out, indent=1))
    backup.backup(path)
    for p, v in plates.items():
        print(p, v["kind"], v["signature"], v["gap_ge_60_min"],
              {k: None if x is None else round(x, 2) for k, x in v["median_ratio"].items()})
    print("thresholds", {k: round(v, 2) for k, v in thr.items()}, "outcome", out["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
