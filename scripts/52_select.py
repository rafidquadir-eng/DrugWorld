"""D3 selection (plan section 6): per model set x family, the grid configuration with the lowest
mean over its 3 seeds of the best validation primary error (primary normalisation, vehicle
start). Ties (equal to 1e-9): fewer parameters (width 512), then dropout 0.1, then weight decay
0.05. Requires all 3 seeds of all 8 configurations. Writes results/d3/selection.json, which is
committed before any evaluation output or gate.
"""
from __future__ import annotations

import collections
import json
import time

from drugworld import REPO, backup

TRAIN = REPO / "results" / "d3" / "train"


def main() -> int:
    runs = [json.loads(p.read_text()) for p in sorted(TRAIN.glob("*.json"))]
    grid = [r for r in runs if r["norm"] == "primary" and r["start"] == "vehicle"
            and not r["set"].startswith("line:")]
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in grid:
        by[f"{r['set']}|{r['family']}"][r["config"]][r["seed"]] = r["best_val_primary"]
    sel, table = {}, {}
    for key, cfgs in sorted(by.items()):
        if len(cfgs) != 8 or any(len(s) != 3 for s in cfgs.values()):
            raise SystemExit(f"{key}: incomplete grid {({c: sorted(s) for c, s in cfgs.items()})}")
        mean = {c: sum(s.values()) / 3 for c, s in cfgs.items()}

        def rank(c):
            w, d, wd = c.split("_")
            return (round(mean[c], 9), int(w[1:]), float(d[1:]), float(wd[2:]))
        sel[key] = min(mean, key=rank)
        table[key] = {c: {"mean": mean[c], "seeds": cfgs[c]} for c in sorted(mean)}
    out = {"rule": __doc__.split("\n\n")[0].split("selection (plan section 6): ")[1],
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "selected": sel,
           "table": table}
    path = REPO / "results" / "d3" / "selection.json"
    path.write_text(json.dumps(out, indent=1))
    backup.backup(path)
    for k, v in sel.items():
        print(k, v, round(table[k][v]["mean"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
