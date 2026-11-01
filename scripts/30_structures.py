"""Compound structures, ECFP4 and the target-class table (plan section 3 data table; A1 C4).

Source per compound (plan: "SMILES from the IDR annotations, else PubChem by name or CAS"):
  * idr0119: the SMILES in the IDR annotation (`smiles1`/`smiles2`), checked against PubChem
    by the annotated InChIKey (agreement of the RDKit parent InChIKey is recorded);
  * idr0064: the IDR annotation has no SMILES but gives the PubChem compound URL (CID);
    PubChem's SMILES for that CID (data/structures/idr0064_pubchem.csv, fetched in D0).
Parent structure: RDKit largest fragment (salts and counter-ions removed), canonical SMILES.
The PubChem check compares the full InChIKey and its connectivity block (first 14
characters; stereo and isotope layers excluded, as ECFP4 here ignores them).
ECFP4 = Morgan radius 2, 1024 bits, from the parent. Target class (idr0064 only) = the vendor
`target_class` annotation; idr0119 has no vendor class and, per A1 C4, uses ECFP4 and the
pretrained embedding only.

Outputs: data/structures/compounds_<set>.csv (compound, source, smiles_input, smiles_parent,
inchikey_parent, ...), data/structures/ecfp4_<set>.npz (compounds, bits uint8),
results/d2/target_classes_idr0064.csv, results/d2/structures_summary.json.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from drugworld import REPO, backup, idr0064, idr0119

RDLogger.DisableLog("rdApp.*")
OUT = REPO / "data" / "structures"
RES = REPO / "results" / "d2"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{}/property/SMILES/JSON"


def parent(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None, None
    m = rdMolStandardize.LargestFragmentChooser().choose(m)
    can = Chem.MolToSmiles(m)
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    return can, Chem.MolToInchiKey(m), scaf


def ecfp4(smi: str) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024)
    return np.array(list(fp), dtype=np.uint8)


def pubchem_smiles(inchikey: str) -> str | None:
    try:
        with urllib.request.urlopen(PUBCHEM.format(inchikey), timeout=30) as r:
            return json.loads(r.read())["PropertyTable"]["Properties"][0]["SMILES"]
    except Exception:
        return None


def idr0119_table() -> pd.DataFrame:
    a = idr0119.annotations()
    rows = {}
    for k in ("1", "2"):
        for d, s, ik in zip(a[f"drug{k}"], a[f"smiles{k}"], a[f"inchikey{k}"]):
            if d and d not in rows:
                rows[d] = {"compound": d, "source": "IDR annotation SMILES",
                           "smiles_input": s, "inchikey_annotation": ik}
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["compound"]))
    out = []
    for r in df.itertuples():
        can, ik, scaf = parent(r.smiles_input)
        pc = pubchem_smiles(r.inchikey_annotation)
        time.sleep(0.3)
        pc_ik = parent(pc)[1] if pc else None
        out.append({"smiles_parent": can, "inchikey_parent": ik, "scaffold": scaf,
                    "pubchem_smiles": pc, "pubchem_parent_inchikey_agrees": pc_ik == ik,
                    "pubchem_connectivity_agrees": bool(pc_ik) and pc_ik[:14] == ik[:14]})
    return pd.concat([df, pd.DataFrame(out)], axis=1)


def idr0064_table() -> pd.DataFrame:
    a = idr0064.annotations()
    a = a[a.compound.ne("") & a.control.eq("")]
    a = a.assign(CID=[int(re.search(r"(\d+)\s*$", u).group(1)) for u in a.pubchem_url])
    # one compound per well; a name used by two wells with different CAS numbers (R406 free
    # base and besylate) is keyed "name [CAS]"
    dup = a.compound.duplicated(keep=False)
    a.loc[dup, "compound"] = a.loc[dup, "compound"] + " [" + a.loc[dup, "cas"] + "]"
    assert not a.compound.duplicated().any()
    pc = pd.read_csv(OUT / "idr0064_pubchem.csv")
    df = a[["plate", "well", "compound", "cas", "CID", "target_class"]].merge(
        pc[["CID", "SMILES", "InChIKey"]], on="CID", how="left")
    df = df.rename(columns={"SMILES": "smiles_input", "InChIKey": "inchikey_pubchem"})
    df.insert(1, "source", "PubChem SMILES for the CID in the IDR annotation")
    par = [parent(s) for s in df.smiles_input]
    df["smiles_parent"] = [p[0] for p in par]
    df["inchikey_parent"] = [p[1] for p in par]
    df["scaffold"] = [p[2] for p in par]
    return df.sort_values("compound").reset_index(drop=True)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    summary = {}
    for which, table in (("idr0119", idr0119_table), ("idr0064", idr0064_table)):
        df = table()
        bad = df[df.smiles_parent.isna()]
        if len(bad):
            raise SystemExit(f"{which}: unparsable SMILES for {bad.compound.tolist()}")
        path = OUT / f"compounds_{which}.csv"
        df.to_csv(path, index=False)
        bits = np.stack([ecfp4(s) for s in df.smiles_parent])
        npz = OUT / f"ecfp4_{which}.npz"
        np.savez(npz, compounds=df.compound.to_numpy(), bits=bits)
        dup = df[df.inchikey_parent.duplicated(keep=False)]
        summary[which] = {"compounds": len(df), "scaffolds": int(df.scaffold.nunique()),
                          "parent_inchikey_duplicates": dup.compound.tolist(),
                          "ecfp4_bits_mean_on": float(bits.sum(1).mean())}
        if which == "idr0119":
            summary[which]["pubchem_agrees_full_inchikey"] = df.set_index("compound")[
                "pubchem_parent_inchikey_agrees"].to_dict()
            summary[which]["pubchem_agrees_connectivity"] = df.set_index("compound")[
                "pubchem_connectivity_agrees"].to_dict()
        else:
            tc = df[["compound", "target_class"]]
            tc.to_csv(RES / "target_classes_idr0064.csv", index=False)
            summary[which]["target_classes"] = tc.target_class.value_counts().to_dict()
            backup.backup(RES / "target_classes_idr0064.csv")
        for p in (path, npz):
            backup.backup(p)
    (RES / "structures_summary.json").write_text(json.dumps(summary, indent=1))
    backup.backup(RES / "structures_summary.json")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
