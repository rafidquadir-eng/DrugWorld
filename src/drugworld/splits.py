"""D2 splits (plan section 5), deterministic from the annotations and the seed.

idr0119 dose split, per drug x line over its measured single-drug doses (vehicle excluded):
  * the lowest dose is training;
  * the highest dose is the extrapolation set: validation in the validation line, test in the
    three test lines (the line folds of H1c);
  * of the interior doses, 3 are held out (1 validation, 2 test). Among all 3-subsets of the
    interior doses, those with the fewest adjacent held-out pairs (adjacency counted along the
    sorted grid, the highest dose included) are eligible ("where the grid allows"); one is drawn
    with the seed, then one of its 3 doses is drawn for validation;
  * all replicate wells of a held-out condition are held out; replicate wells of training
    conditions are split 80/20 by plate for selection.
idr0119 line folds (H1c): the seed draws the validation line; the other three are test folds.
idr0119 combination split (AU565): with >= 3 distinct pairs, leave-pair-out. Pairs are put in a
  seeded order; test takes pairs in that order until it holds >= 2 pairs and >= 6 conditions;
  validation is the next pair in the order with >= 2 conditions. All dose combinations of a
  held-out pair are held out.
idr0119 selection wells (plan: "replicate wells of training conditions are split 80/20 by
  plate for selection only"): within each line, plates with the same drug layout form a group;
  in every group of >= 3 plates the seed picks one plate as the selection plate. Wells of
  training conditions (single-drug training doses, training-pair combinations) on selection
  plates are `train_sel` (selection only, never fitted), unless that would leave the condition
  with < 2 fitted replicate wells; then they stay `train`. Vehicle wells are always `train`.
idr0064: Bemis-Murcko scaffold split of compounds 70/10/20 (`scaffold_split`). Compounds with
  the same parent structure share a scaffold and so a split. Each scaffold gets the majority
  stratum of its compounds (ties: seeded). Stratum = vendor target class, crossed with Goglia's
  Erk-phenotype class when Table S1 is present (A1 O2; otherwise target class only, recorded).
  Within each stratum, scaffolds are taken largest first (ties in seeded order) and each goes to
  the split furthest below its target count (fraction x stratum size; ties: train, val, test).
  DMSO: training, except a plate-stratified 20% kept as `dmso_check` for normalisation checks:
  ceil(0.2 x 18) = 4 DMSO wells, one on each of 4 seeded plates (one well drawn per plate). Untreated wells (plate 3.3) have no
  action encoding and are `excluded`.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

SEED = 20261030


def _adjacent_pairs(held: set[int], n: int) -> int:
    return sum(1 for i in range(n - 1) if i in held and i + 1 in held)


def dose_split(doses: list[float], rng: np.random.Generator) -> dict[float, str]:
    """Sorted dose grid -> {dose: 'train' | 'val' | 'test' | 'extrap'} (extrap = highest)."""
    d = sorted(doses)
    n = len(d)
    out = {d[0]: "train", d[-1]: "extrap"}
    interior = list(range(1, n - 1))
    if len(interior) < 3:
        raise ValueError(f"fewer than 3 interior doses: {d}")
    subsets = list(itertools.combinations(interior, 3))
    score = [_adjacent_pairs(set(s) | {n - 1}, n) for s in subsets]
    eligible = [s for s, sc in zip(subsets, score) if sc == min(score)]
    held = eligible[rng.integers(len(eligible))]
    val = held[rng.integers(3)]
    for i in interior:
        out[d[i]] = "val" if i == val else ("test" if i in held else "train")
    return out


def idr0119_splits(ann: pd.DataFrame, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    lines = sorted(ann.line.unique())
    val_line = lines[rng.integers(len(lines))]
    single = ann[(ann.drug2 == "") & (ann.drug1 != "")]
    cond = {}
    for (line, drug), g in single.groupby(["line", "drug1"], sort=True):
        sp = dose_split(sorted(g.dose1.unique()), rng)
        for dose, role in sp.items():
            if role == "extrap":
                role = "val_extrap" if line == val_line else "test_extrap"
            cond[(line, drug, float(dose))] = role
    combo = ann[ann.drug2 != ""]
    pairs = combo.assign(pair=[" + ".join(sorted([a, b])) for a, b in
                               zip(combo.drug1, combo.drug2)])
    pairs["cond"] = [tuple(sorted([(a, float(x)), (b, float(y))])) for a, x, b, y in
                     zip(pairs.drug1, pairs.dose1, pairs.drug2, pairs.dose2)]
    n_cond = pairs.groupby("pair").cond.nunique()
    order = list(n_cond.index[rng.permutation(len(n_cond))])
    test_pairs, i = [], 0
    while i < len(order) and (len(test_pairs) < 2 or n_cond[test_pairs].sum() < 6):
        test_pairs.append(order[i])
        i += 1
    val_pair = next((p for p in order[i:] if n_cond[p] >= 2), None)
    return {"seed": seed, "val_line": val_line,
            "test_lines": [x for x in lines if x != val_line],
            "single": cond, "test_pairs": test_pairs, "val_pair": val_pair,
            "pair_conditions": n_cond.to_dict()}


def selection_plates(ann: pd.DataFrame, rng: np.random.Generator) -> set[str]:
    """One seeded selection plate per (line, drug layout) group of >= 3 plates."""
    drugs = ann[ann.drug1 != ""].assign(d=lambda d: d.drug1 + "|" + d.drug2)
    layout = drugs.groupby("plate").d.unique().apply(lambda x: ";".join(sorted(x)))
    line = ann.groupby("plate").line.first()
    out = set()
    groups = pd.DataFrame({"layout": layout, "line": line.reindex(layout.index)})
    for _, g in groups.groupby(["line", "layout"], sort=True):
        plates = sorted(g.index)
        if len(plates) >= 3:
            out.add(plates[rng.integers(len(plates))])
    return out


def well_roles_idr0119(ann: pd.DataFrame, sp: dict, sel_plates: set[str]) -> pd.DataFrame:
    """Per well: role in the H1 dose split and in the H2a combination split.

    h1_role: train | train_sel | val | test | val_extrap | test_extrap | vehicle | combo
    h2a_role (AU565 only, A1 O4): train (all single-drug wells and training-pair combinations)
      | train_sel | val | test | vehicle; other lines: '' (not in the H2a model set).
    """
    a = ann.copy()
    pair = [" + ".join(sorted([x, y])) if y else "" for x, y in zip(a.drug1, a.drug2)]
    a["pair"] = pair
    h1, h2 = [], []
    for r in a.itertuples():
        if r.drug1 == "":
            h1.append("vehicle")
            h2.append("vehicle" if r.line == "AU565" else "")
        elif r.drug2 == "":
            h1.append(sp["single"][(r.line, r.drug1, float(r.dose1))])
            h2.append("train" if r.line == "AU565" else "")
        else:
            h1.append("combo")
            h2.append("test" if r.pair in sp["test_pairs"] else
                      "val" if r.pair == sp["val_pair"] else "train")
    a["h1_role"], a["h2a_role"] = h1, h2
    for col, key in (("h1_role", ["line", "drug1", "dose1"]),
                     ("h2a_role", ["line", "drug1", "dose1", "drug2", "dose2"])):
        tr = a[a[col] == "train"]
        for _, g in tr.groupby(key):
            on_sel = g[g.plate.isin(sel_plates)]
            if len(on_sel) and len(g) - len(on_sel) >= 2:
                a.loc[on_sel.index, col] = "train_sel"
    return a


def scaffold_split(comp: pd.DataFrame, rng: np.random.Generator,
                   frac=(("train", 0.7), ("val", 0.1), ("test", 0.2))) -> dict[str, str]:
    """comp: columns compound, scaffold, stratum -> {compound: split}."""
    sc = comp.groupby("scaffold").stratum.agg(lambda x: x.value_counts().index.tolist())
    major = {}
    for scaf, ranked in sc.items():
        counts = comp[comp.scaffold == scaf].stratum.value_counts()
        top = sorted(counts[counts == counts.max()].index)
        major[scaf] = top[rng.integers(len(top))] if len(top) > 1 else top[0]
    size = comp.groupby("scaffold").size()
    out = {}
    for stratum in sorted(set(major.values())):
        scafs = [x for x in sorted(major) if major[x] == stratum]
        order = rng.permutation(len(scafs))
        scafs = [scafs[i] for i in order]
        scafs.sort(key=lambda x: -size[x])            # stable: ties keep the seeded order
        n = sum(size[x] for x in scafs)
        got = {k: 0 for k, _ in frac}
        for x in scafs:
            k = max(frac, key=lambda kf: kf[1] * n - got[kf[0]])[0]
            got[k] += size[x]
            for c in comp.compound[comp.scaffold == x]:
                out[c] = k
    return out


def idr0064_splits(ann: pd.DataFrame, comp: pd.DataFrame, erk_class: dict | None,
                   seed: int = SEED) -> dict:
    """ann: idr0064.annotations(); comp: data/structures/compounds_idr0064.csv."""
    rng = np.random.default_rng(seed + 64)
    c = comp.copy()
    c["scaffold"] = c.scaffold.fillna("")          # acyclic compounds share the empty scaffold
    c["stratum"] = c.target_class if erk_class is None else \
        c.target_class + " | " + c.compound.map(erk_class).fillna("unclassified")
    split = scaffold_split(c, rng)
    dmso = ann[ann.compound == "DMSO"].sort_values(["plate", "well"])
    plates = sorted(dmso.plate.unique())
    k = int(np.ceil(0.2 * len(dmso)))
    chk_plates = [plates[i] for i in sorted(rng.choice(len(plates), k, replace=False))]
    check = []
    for p in chk_plates:
        w = sorted(dmso.well[dmso.plate == p])
        check.append((p, w[rng.integers(len(w))]))
    return {"seed": seed + 64, "stratification": "target class" if erk_class is None
            else "target class x Goglia Erk-phenotype class", "compound_split": split,
            "dmso_check": check}
