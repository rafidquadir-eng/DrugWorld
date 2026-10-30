"""idr0119 (Gross et al. 2023) file map from the IDR companion OME-XML files.

Each companion lists, per image (= well field), the single-plane TIFF for every (channel, T)
plane. T in the companion is a sequence index; the acquisition time is taken from the file name
(``..._DDdHHhMMm.tif``), so a missing frame leaves a gap on the 30-min grid instead of shifting
later frames. Files are served over HTTPS from the EBI FTP mirror of IDR.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from drugworld import REPO

FTP = "https://ftp.ebi.ac.uk/pub/databases/IDR/idr0119-gross-cellresponse"
ANN = REPO / "data" / "annotations"
COMPANIONS = ANN / "idr0119_companions"
PLATES_TSV = ANN / "idr0119-gross-cellresponse" / "screenA" / "idr0119-screenA-plates.tsv"
ANNOTATION = ANN / "idr0119-gross-cellresponse" / "screenA" / "idr0119-screenA-annotation.csv"
FRAME_MIN = 30
N_GRID = 193            # 0 .. 96 h at 30 min
CHANNEL = {"R": "nuclear", "G": "hdhb", "P": "phase"}
_NS = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
_TIME = re.compile(r"_(\d{2})d(\d{2})h(\d{2})m\.tif$")
_ROWS = "ABCDEFGHIJKLMNOP"


def plate_companions() -> dict[str, str]:
    """IDR plate name -> companion path relative to the FTP root."""
    out = {}
    for line in PLATES_TSV.read_text().splitlines():
        if line.strip():
            name, path = line.split("\t")[:2]
            out[name] = path.split("idr0119-gross-cellresponse/", 1)[1]
    return out


def parse_companion(plate: str, rel: str) -> pd.DataFrame:
    """One row per plane: plate, well, field, channel, t_index, minutes, url."""
    base = rel.rsplit("/", 1)[0]
    root = ET.parse(COMPANIONS / Path(rel).name).getroot()
    pl = root.find(f"{_NS}Plate")
    image_well = {}
    for w in pl.findall(f"{_NS}Well"):
        well = f"{_ROWS[int(w.get('Row'))]}{int(w.get('Column')) + 1}"
        for ws in w.findall(f"{_NS}WellSample"):
            image_well[ws.find(f"{_NS}ImageRef").get("ID")] = (well, int(ws.get("Index")) + 1)
    rows = []
    for im in root.findall(f"{_NS}Image"):
        well, field = image_well[im.get("ID")]
        px = im.find(f"{_NS}Pixels")
        names = [c.get("Name") for c in px.findall(f"{_NS}Channel")]
        for td in px.findall(f"{_NS}TiffData"):
            fn = td.find(f"{_NS}UUID").get("FileName")
            m = _TIME.search(fn)
            minutes = int(m[1]) * 1440 + int(m[2]) * 60 + int(m[3])
            rows.append((plate, well, field, CHANNEL[names[int(td.get("FirstC"))]],
                         int(td.get("FirstT")), minutes, f"{FTP}/{base}/{fn}"))
    return pd.DataFrame(rows, columns=["plate", "well", "field", "channel", "t_index",
                                       "minutes", "url"])


def file_map() -> pd.DataFrame:
    """All planes of all plates, with the 30-min grid slot (nearest, |offset| <= 10 min)."""
    df = pd.concat([parse_companion(p, r) for p, r in plate_companions().items()],
                   ignore_index=True)
    slot = (df.minutes / FRAME_MIN).round().astype(int)
    df["slot"] = slot.where((df.minutes - slot * FRAME_MIN).abs() <= 10, -1)
    return df


def annotations() -> pd.DataFrame:
    """Per-well conditions from the IDR library file (doses as given there, see the audit)."""
    a = pd.read_csv(ANNOTATION)
    return pd.DataFrame({
        "plate": a.Plate, "well": a.Well, "line": a["Characteristics [Cell Line]"],
        "drug1": a["Compound 1 Name"].fillna(""),
        "dose1": a["Compound 1 Concentration (microMolar)"],
        "smiles1": a["Compound 1 SMILES"].fillna(""),
        "inchikey1": a["Compound 1 InChIKey"].fillna(""),
        "drug2": a["Compound 2 Name"].fillna(""),
        "dose2": a["Compound 2 Concentration (nanoMolar)"],
        "smiles2": a["Compound 2 SMILES"].fillna(""),
        "inchikey2": a["Compound 2 InChIKey"].fillna(""),
        "control": a["Control Type"].fillna(""),
    })
