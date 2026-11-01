"""G-Data (plan sections 4 and 8): >= 95% of the wells of each set embedded; failures by cause.

Rule (fixed before it is computed): a well counts as embedded if at least one of its fields
is in a shard that passes the shard guard (manifest and SHA-256); fields are also reported.
Expected wells and fields come from the annotations (idr0064: 450 wells, 1 field each) and the
idr0119 file map (760 wells, 3,026 fields). Also reported: fields with gap-filled frames,
failures by cause (results/d1/embed_failures_*.json), and leakage check 1 (identical frame-0
SHA-1 in two different wells). Writes results/d1/g_data.json.
"""
from __future__ import annotations

import collections
import json

import pandas as pd

from drugworld import REPO, backup, idr0064, latents, shards


def expected(which):
    if which == "idr0064":
        a = idr0064.annotations()
        return a[["plate", "well"]].assign(field=1)
    fm = pd.read_parquet(REPO / "data" / "annotations" / "idr0119_files.parquet")
    return fm[["plate", "well", "field"]].drop_duplicates()


def main() -> int:
    shards.check(hashes=True)
    out = {"rule": __doc__.split("Rule")[1].split("Writes")[0].strip(), "sets": {}}
    for which in ("idr0119", "idr0064"):
        exp = expected(which)
        f, _, meta = latents.load(which, arrays=(), check=False)
        got = set(map(tuple, f[["plate", "well"]].to_numpy()))
        wells = set(map(tuple, exp[["plate", "well"]].drop_duplicates().to_numpy()))
        fails = []
        for p in sorted((REPO / "results" / "d1").glob(f"embed_failures_{which}_w*.json")):
            fails += json.loads(p.read_text())
        # cause strings: "channel <name>: <n> of <N> slots missing" (scripts/20_embed.py)
        causes = collections.Counter(x["cause"].split(":")[0] + " slots missing"
                                     for x in fails)
        dup = f[f.frame0_sha1.duplicated(keep=False)]
        dup_wells = sum(g[["plate", "well"]].drop_duplicates().shape[0] > 1
                        for _, g in dup.groupby("frame0_sha1"))
        frac = len(got & wells) / len(wells)
        out["sets"][which] = {
            "wells_expected": len(wells), "wells_embedded": len(got & wells),
            "wells_fraction": frac, "fields_expected": len(exp), "fields_embedded": len(f),
            "fields_with_filled_frames": int((f.n_filled > 0).sum()),
            "filled_frames_total": int(f.n_filled.sum()),
            "failed_fields": len(fails),
            "failures": [{k: x[k] for k in ("plate", "well", "field", "cause")} for x in fails],
            "failure_causes": dict(causes),
            "missing_wells": sorted(map(list, wells - got)),
            "frame0_sha1_shared_across_wells": int(dup_wells),
            "config": meta, "pass": bool(frac >= 0.95)}
        print(which, f"wells {len(got & wells)}/{len(wells)} ({frac:.4f}), fields "
              f"{len(f)}/{len(exp)}, failed {len(fails)}, filled-frame fields "
              f"{int((f.n_filled > 0).sum())}, shared frame-0 hashes {int(dup_wells)}")
    out["pass"] = all(v["pass"] for v in out["sets"].values())
    path = REPO / "results" / "d1" / "g_data.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    backup.backup(path)
    print("G-Data", "PASS" if out["pass"] else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
