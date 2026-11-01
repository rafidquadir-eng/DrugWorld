"""D3 job runner: a queue of 50_train.py runs over the GPUs, `--slots` runs per GPU.

Stages:
  grid      every gated family x 8 configurations x 3 seeds (plan section 6):
            h1: dw_full, dw_none, dw_doseshuf, dw_doseonly; h2a: dw_full, dw_none;
            h2b: dw_full:<src>, dw_structshuf:<src> (src ecfp4, molformer, target), dw_none.
  after     (needs results/d3/selection.json, committed first): config-matched references
            (dw_none and the shuffled families at dw_full's selected configuration; for h2b at
            the selected configuration of dw_full of the same source), H1c dw_line_k at the h1
            dw_full configuration (4 lines x 3 seeds), and the global-normalisation
            sensitivity (gated families at their selected configurations, 3 seeds).
A run whose results/d3/train/<name>.json exists is skipped. Logs: logs/d3/<name>.log.

  PYTHONPATH=src python scripts/51_run_d3.py --stage grid --gpus 0-7 --slots 2
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import queue
import subprocess
import threading
import time

from drugworld import REPO

print = functools.partial(print, flush=True)
spec = importlib.util.spec_from_file_location("t50", REPO / "scripts" / "50_train.py")
T50 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T50)
SEEDS = (0, 1, 2)
SRC = ("ecfp4", "molformer", "target")
FAMILIES = {"h1": ["dw_full", "dw_none", "dw_doseshuf", "dw_doseonly"],
            "h2a": ["dw_full", "dw_none"],
            "h2b": [f"dw_full:{s}" for s in SRC] + [f"dw_structshuf:{s}" for s in SRC] +
                   ["dw_none"]}
LINES = ("21MT1", "AU565", "HCC1143", "MDAMB157")


def grid_jobs():
    return [(ms, fam, c["id"], s, "primary") for ms, fams in FAMILIES.items() for fam in fams
            for c in T50.configs() for s in SEEDS]


def after_jobs():
    sel = json.loads((REPO / "results" / "d3" / "selection.json").read_text())["selected"]
    jobs = []
    full = {"h1": sel["h1|dw_full"], "h2a": sel["h2a|dw_full"]}
    for ms, refs in (("h1", ["dw_none", "dw_doseshuf"]), ("h2a", ["dw_none"])):
        for fam in refs:
            jobs += [(ms, fam, full[ms], s, "primary") for s in SEEDS]
    for src in SRC:
        c = sel[f"h2b|dw_full:{src}"]
        jobs += [("h2b", f, c, s, "primary") for f in ("dw_none", f"dw_structshuf:{src}")
                 for s in SEEDS]
    jobs += [(f"line:{L}", "dw_line", full["h1"], s, "primary") for L in LINES for s in SEEDS]
    for key, c in sel.items():
        ms, fam = key.split("|")
        if fam != "dw_doseonly":
            jobs += [(ms, fam, c, s, "global") for s in SEEDS]
    out, seen = [], set()
    for j in jobs:
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def name(j):
    ms, fam, cid, seed, norm = j
    return T50.run_name(ms, fam, cid, seed, norm, "vehicle")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grid", "after"], required=True)
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--slots", type=int, default=2)
    args = ap.parse_args()
    a, b = map(int, args.gpus.split("-"))
    jobs = grid_jobs() if args.stage == "grid" else after_jobs()
    todo = [j for j in jobs if not (REPO / "results" / "d3" / "train" /
                                    f"{name(j)}.json").exists()]
    # longest first (width 768), so the tail is short
    todo.sort(key=lambda j: (not j[2].startswith("w768"), j[0], j[1], j[2], j[3]))
    print(f"stage {args.stage}: {len(jobs)} jobs, {len(todo)} to run on GPUs {a}-{b} x "
          f"{args.slots}")
    # width 768 needs ~20 GB, so two cannot share a 40 GB GPU: per GPU, slot 0 takes width-768
    # jobs first (then width 512), the other slots width 512 only
    big: queue.Queue = queue.Queue()
    small: queue.Queue = queue.Queue()
    for j in todo:
        (big if j[2].startswith("w768") else small).put(j)
    (REPO / "logs" / "d3").mkdir(parents=True, exist_ok=True)
    failed = []

    def worker(gpu, slot):
        while True:
            try:
                j = (big if slot == 0 and not big.empty() else small).get_nowait()
            except queue.Empty:
                return
            ms, fam, cid, seed, norm = j
            log = REPO / "logs" / "d3" / f"{name(j)}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": "src",
                   "OMP_NUM_THREADS": "4"}
            cmd = [os.path.expanduser("~/dwvenv/bin/python"), "-W", "ignore",
                   "scripts/50_train.py", "--set", ms, "--family", fam, "--config", cid,
                   "--seed", str(seed), "--norm", norm]
            t0 = time.time()
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT
                                    ).returncode
            print(f"gpu {gpu}: {name(j)} rc {rc} {(time.time() - t0) / 60:.1f} min "
                  f"({big.qsize() + small.qsize()} queued)")
            if rc:
                failed.append(name(j))

    th = [threading.Thread(target=worker, args=(g, k)) for g in range(a, b + 1)
          for k in range(args.slots)]
    for t in th:
        t.start()
        time.sleep(2)
    for t in th:
        t.join()
    (REPO / "logs" / "d3" / f"{args.stage}_done.txt").write_text(
        f"stage {args.stage} done {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}; "
        f"failed {failed}\n")
    print("done; failed:", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
