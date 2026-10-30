"""idr0064 (Goglia et al. 2020) from its OME-NGFF copy in the BioImage Archive (S-BIAD992).

Nine plate zarrs, one per IDR plate (plate name in `.zattrs`), wells at `<row>/<col>`, one
field (`0`), full resolution at `0`: shape (121, 2, 1, 512, 512) uint16 big-endian, one chunk
per (t, c), blosc-lz4. Channel 0 = Erk-KTR (405), 1 = H2B-RFP (561) as in the IDR record.
Frames every 3 min (TIFF `finterval` 180 s); imaging began 30 min after drug addition.
"""
from __future__ import annotations

import json
import re
import urllib.request
from functools import lru_cache

import numcodecs
import numpy as np
import pandas as pd

from drugworld import REPO

S3 = "https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data"
STUDY = "S-BIAD992"
ANNOTATION = (REPO / "data" / "annotations" / "idr0064-goglia-erkdynamics" / "screenA" /
              "idr0064-screenA-annotation.csv")
N_T, N_C, SHAPE = 121, 2, (512, 512)
FRAME_MIN = 3.0
ADDITION_BEFORE_FIRST_FRAME_MIN = 30
_BLOSC = numcodecs.Blosc()


@lru_cache(1)
def plate_zarrs() -> dict[str, str]:
    """IDR plate name -> zarr URL."""
    xml = urllib.request.urlopen(f"{S3}/?list-type=2&prefix={STUDY}/&delimiter=/").read()
    uuids = re.findall(rf"<Prefix>{STUDY}/([0-9a-f-]{{36}})/</Prefix>", xml.decode())
    out = {}
    for u in uuids:
        z = f"{S3}/{STUDY}/{u}/{u}.zarr"
        out[json.load(urllib.request.urlopen(f"{z}/.zattrs"))["plate"]["name"]] = z
    return out


def chunk_urls(plate: str, well: str) -> list[tuple[int, int, str]]:
    """(t, c, url) for all chunks of a well (well like 'C9')."""
    z = plate_zarrs()[plate]
    row, col = well[0], well[1:]
    return [(t, c, f"{z}/{row}/{col}/0/0/{t}/{c}/0/0/0") for t in range(N_T)
            for c in range(N_C)]


def decode(body: bytes) -> np.ndarray:
    return np.frombuffer(_BLOSC.decode(body), ">u2").reshape(SHAPE).astype(np.uint16)


def annotations() -> pd.DataFrame:
    a = pd.read_csv(ANNOTATION)
    return pd.DataFrame({
        "plate": a.Plate.astype(str), "well": a.Well, "compound": a["Compound Name"],
        "cas": a["Compound Identifier (CAS Number)"], "pubchem_url": a["Compound URL"],
        "dose_uM": a["Compound Concentration (microMolar)"],
        "target_class": a["Compound General Target Class"],
        "control": a["Control Type"].fillna("")})
