"""A1 C2: record the AU565 48 h re-dose status per plate from the study files.

Evidence: (1) the study text (idr0119-study.txt, treatment protocol); (2) plate composition from
the IDR annotation (combination, single-drug and vehicle wells per plate); (3) acquisition
timestamps from the companion OME-XML (gap at slot 96); (4) the vehicle-well pilot fields
(D0, vehicle only): |step in log(1 + cell count)| from slot 95 to 96, divided by the median step
over slots 80-109 excluding that one. No drug-treated well is used.
Writes results/d2/au565_redose.json.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from drugworld import REPO, backup, idr0119

TEXT = ("For single drug treatments of AU565 cells only, at 48H the media was replaced in all "
        "wells including the control wells, and fresh media and drug were added.")


def main() -> int:
    study = (REPO / "data/annotations/idr0119-gross-cellresponse/idr0119-study.txt").read_text()
    assert TEXT in study
    a = idr0119.annotations()
    fm = pd.read_parquet(REPO / "data/annotations/idr0119_files.parquet")
    out = {"study_text": TEXT, "plates": {}}
    steps_other = []
    for f in sorted(glob.glob(str(REPO / "data/pilot/idr0119/*.npz"))):
        c = np.load(f, allow_pickle=True)["cells"]
        n = np.bincount(c[:, 0].astype(int), minlength=idr0119.N_GRID).astype(float)
        d = np.abs(np.diff(np.log(n + 1)))
        ref = np.r_[d[80:95], d[96:110]]
        name = f.split("/")[-1][:-4]
        plate = name.split("_")[0]
        r = float(d[95] / np.median(ref))
        if plate.startswith("AU"):
            out["plates"].setdefault(plate, {}).setdefault("vehicle_step_ratio_slot96", {})[name] = r
        else:
            steps_other.append(r)
    out["vehicle_step_ratio_slot96_other_lines_median"] = float(np.median(steps_other))
    for p, g in a[a.line == "AU565"].groupby("plate"):
        s = fm[(fm.plate == p) & (fm.slot >= 0)].groupby("slot").minutes.median().sort_index()
        n_combo = int((g.drug2 != "").sum())
        n_single = int(((g.drug2 == "") & (g.drug1 != "")).sum())
        kind = ("single-drug only" if n_combo == 0 else
                "combination only" if n_single == 0 else "mixed")
        out["plates"].setdefault(p, {}).update({
            "kind": kind, "wells_combination": n_combo, "wells_single": n_single,
            "wells_vehicle": int((g.drug1 == "").sum()),
            "slot96_present": bool(96 in s.index),
            "max_frame_gap_min_slots_90_102": float(np.diff(s.loc[89:102].values).max())})
    path = REPO / "results/d2/au565_redose.json"
    path.write_text(json.dumps(out, indent=1))
    backup.backup(path)
    for p, v in out["plates"].items():
        print(p, v["kind"], v["slot96_present"], v["max_frame_gap_min_slots_90_102"],
              {k: round(x, 1) for k, x in v.get("vehicle_step_ratio_slot96", {}).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
