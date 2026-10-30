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
idr0064: Bemis-Murcko scaffold split of compounds 70/10/20, stratified (see `scaffold_split`).
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
