"""D4 checklist: the plan's 9 items (section 10), checked mechanically; written to
results/d4/checklist.json (and checklist_final.json with --final, after the gates commit) and
reported in reports/DW_GATES_VAL_SUMMARY.md.

Checklist:
 1. commit order: the freeze (manifests/drugworld_protocol.json) is an ancestor of the commit
    adding results/d3/selection.json, which is an ancestor of the first commit adding any
    results/d4 file other than gates_val.json, which is an ancestor of (or equal to) the commit
    adding gates_val.json; checked on HEAD after the gates commit (--final).
 2. 3 seeds per evaluated family and the config-matched references present.
 3. no learned ID embedding: every checkpoint's state-dict keys are DrugPredictor's content
    parameters only (no nn.Embedding weights).
 4. permutation hashes recorded for every dw_doseshuf / dw_structshuf run, equal across runs of
    the same seed and family.
 5. evaluated unit counts equal the frozen validation counts minus listed exclusions.
 6. the baseline-strength table and the substitutions it implies.
 7. no evaluation file refers to test conditions (unit keys and forecast indices).
 8. primary-normalised and global results both present.
 9. feasibility-gate results recorded (results/d2/feasibility.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import torch

from drugworld import REPO, latents
from drugworld import train as TR

ALLOWED = ("inp.", "step", "null", "tok.", "post.", "vehicle", "cov", "blocks.", "norm.", "out.")


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout


def first_commit(path):
    c = git("log", "--diff-filter=A", "--format=%H", "--", path).split()
    return c[-1] if c else None


def ancestor(a, b):
    return subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", a, b]
                          ).returncode == 0


def checklist(final):
    sel = json.loads((REPO / "results" / "d3" / "selection.json").read_text())["selected"]
    runs = [json.loads(p.read_text()) for p in (REPO / "results" / "d3" / "train").glob("*.json")]
    gv = json.loads((REPO / "results" / "d4" / "gates_val.json").read_text())
    c = {}
    # 1
    f_freeze = first_commit("manifests/drugworld_protocol.json")
    f_sel = first_commit("results/d3/selection.json")
    d4 = [p for p in git("log", "--diff-filter=A", "--name-only", "--format=", "--",
                         "results/d4").split() if not p.endswith("gates_val.json")]
    f_eval = [first_commit(p) for p in d4]
    f_gates = first_commit("results/d4/gates_val.json")
    ok1 = bool(f_freeze and f_sel and ancestor(f_freeze, f_sel) and f_freeze != f_sel)
    ok1 &= all(ancestor(f_sel, e) and e != f_sel for e in f_eval if e)
    ok1 &= (not final) or bool(f_gates and all(ancestor(e, f_gates) for e in f_eval if e))
    c["1_commit_order"] = {"pass": ok1, "freeze": f_freeze, "selection": f_sel,
                           "evaluation_outputs": sorted(set(e for e in f_eval if e)),
                           "gates_val": f_gates, "checked_after_gates_commit": final}
    # 2
    need = []
    for key, cfg in sel.items():
        ms, fam = key.split("|")
        need.append((ms, fam, cfg))
    for ms, refs in (("h1", ["dw_none", "dw_doseshuf"]), ("h2a", ["dw_none"])):
        need += [(ms, f, sel[f"{ms}|dw_full"]) for f in refs]
    for s in ("ecfp4", "molformer", "target"):
        need += [("h2b", f, sel[f"h2b|dw_full:{s}"]) for f in ("dw_none", f"dw_structshuf:{s}")]
    need += [(f"line:{L}", "dw_line", sel["h1|dw_full"]) for L in
             ("21MT1", "AU565", "HCC1143", "MDAMB157")]
    have = {}
    for r in runs:
        if r["norm"] == "primary":
            have.setdefault((r["set"], r["family"], r["config"]), set()).add(r["seed"])
    missing = [list(n) for n in need if have.get(n, set()) != {0, 1, 2}]
    c["2_seeds_and_references"] = {"pass": not missing, "missing": missing,
                                   "groups_checked": len(need)}
    # 3
    bad = []
    for p in sorted((REPO / "checkpoints" / "d3").glob("*.pt")):
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        bad += [f"{p.name}:{k}" for k in sd if not k.startswith(ALLOWED) or "embed" in k.lower()]
    c["3_no_id_embedding"] = {"pass": not bad, "violations": bad[:20],
                              "checkpoints": len(list((REPO / "checkpoints" / "d3").glob("*.pt")))}
    # 4
    perm, bad4 = {}, []
    for r in runs:
        if "shuf" in r["family"]:
            h = r["permutations"]
            if not h:
                bad4.append(r["name"])
            perm.setdefault((r["family"], r["seed"]), set()).add(json.dumps(h, sort_keys=True))
    bad4 += [f"{k}" for k, v in perm.items() if len(v) > 1]
    c["4_permutation_hashes"] = {"pass": not bad4, "problems": bad4,
                                 "hashes": {f"{k[0]}|s{k[1]}": json.loads(next(iter(v)))
                                            for k, v in perm.items()}}
    # 5
    counts = {}
    for ms, units in (("h1", ["line", "drug1", "dose1"]),
                      ("h2a", ["drug1", "dose1", "drug2", "dose2"]), ("h2b", ["compound_key"])):
        which = TR.dataset_of(ms)
        w = TR.wells(which)
        if ms == "h1":
            frozen = w[w.h1_role.isin(["val", "val_extrap"])].drop_duplicates(units).shape[0]
        elif ms == "h2a":
            frozen = w[w.h2a_role == "val"].drop_duplicates(units).shape[0]
        else:
            frozen = w[w.role == "val"].compound_key.nunique()
        got = gv["primary"]["sets"][ms]["n_units"]
        counts[ms] = {"frozen": int(frozen), "evaluated": got, "exclusions": 0}
    h1c = gv["primary"]["sets"]["h1c"]
    for L, v in h1c.items():
        counts[f"h1c:{L}"] = {"evaluated": v.get("n_units"),
                              "excluded_conditions": v.get("excluded_conditions")}
    c["5_unit_counts"] = {"pass": all(v["frozen"] - v["exclusions"] == v["evaluated"]
                                      for k, v in counts.items() if "frozen" in v),
                          "counts": counts}
    # 6
    st = {ms: {"strength": {b: {"p_vs_copy_last": v["p_one_sided"], "beats": v["beats"]}
                            for b, v in gv["primary"]["sets"][ms].get("baseline_strength", {})
                            .items()}, "weak": gv["primary"]["sets"][ms]["weak"]}
          for ms in ("h1", "h2a", "h2b")}
    subs = {"G-Dose lookup": gv["primary"]["sets"]["h1"]["gates"]["G-Dose"]["lookup_used"],
            "G-Dose+ lookup": gv["primary"]["sets"]["h1"]["gates"]["G-Dose+"]["lookup_used"],
            "G-Combo lookups": gv["primary"]["sets"]["h2a"]["gates"]["G-Combo"]["lookups_used"],
            "G-Compound lookup": {s: v["lookup_used"] for s, v in gv["primary"]["sets"]["h2b"]
                                  ["gates"]["G-Compound"]["per_source"].items()}}
    c["6_baseline_strength"] = {"pass": all("baseline_strength" in gv["primary"]["sets"][m]
                                            for m in ("h1", "h2a", "h2b")),
                                "table": st, "substitutions": subs}
    # 7
    test_keys = set()
    w19, w64 = TR.wells("idr0119"), TR.wells("idr0064")
    for r in w19[w19.h1_role.isin(["test", "test_extrap"])].itertuples():
        test_keys.add(f"{r.line}|{r.drug1}|{r.dose1}")
    for r in w19[w19.h2a_role == "test"].itertuples():
        test_keys.add(f"{r.drug1}|{r.dose1}|{r.drug2}|{r.dose2}")
    test_keys |= set(w64.compound_key[w64.role == "test"])
    hits = []
    for norm in ("primary", "global"):
        for ms in ("h1", "h2a", "h2b"):
            s = gv[norm]["sets"][ms]
            for m, ue in s["unit_errors"].items():
                hits += [f"{norm}:{ms}:{m}:{k}" for k in ue if k in test_keys]
            hits += [f"{norm}:knn:{k}" for k in s.get("knn_neighbours", {}) if k in test_keys]
    val_fields = {}
    for ms in ("h1", "h2a", "h2b"):
        which = TR.dataset_of(ms)
        f = latents.load(which, arrays=())[0].merge(TR.wells(which), on=["plate", "well"],
                                                     how="left")
        d = TR.Data(which, f, np.zeros((1,)), [])
        val_fields[ms] = set(np.flatnonzero(TR.roles(d, ms).to_numpy() == "val"))
    for p in (REPO / "data" / "preds" / "val").glob("*.npz"):
        ms = p.name.split("__")[0]
        if ms in val_fields and not set(np.load(p)["idx"].tolist()) <= val_fields[ms]:
            hits.append(f"forecast {p.name}: non-validation fields")
    c["7_no_test_reference"] = {"pass": not hits, "hits": hits[:20]}
    # 8
    c["8_normalised_and_global"] = {"pass": all(k in gv and gv[k]["sets"] for k in
                                                ("primary", "global"))}
    # 9
    fz = REPO / "results" / "d2" / "feasibility.json"
    c["9_feasibility_recorded"] = {"pass": fz.exists()}
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    c = checklist(args.final)
    path = REPO / "results" / "d4" / ("checklist_final.json" if args.final else "checklist.json")
    path.write_text(json.dumps(c, indent=1, default=str))
    for k, v in c.items():
        print(k, "PASS" if v["pass"] else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
