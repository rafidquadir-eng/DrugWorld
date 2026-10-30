"""Shard manifest and guard (GeneWorld `cache.check_shard_manifest`, plan section 4).

Every latent and readout shard is registered with its SHA-256 when written
(manifests/latent_shards.tsv, rows appended only). `check()` refuses, listing the paths, if a
shard on disk is not in the manifest, a manifest shard is missing, or (with hashes=True) a hash
differs; loaders call it before reading any shard.
"""
from __future__ import annotations

import fcntl
import time
from pathlib import Path

from drugworld import REPO
from drugworld.backup import sha256

MANIFEST = REPO / "manifests" / "latent_shards.tsv"
ROOTS = (REPO / "data" / "latents", REPO / "data" / "readouts")


class ShardError(RuntimeError):
    pass


def register(path: Path) -> str:
    path = Path(path).resolve()
    h = sha256(path)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        if fh.tell() == 0:
            fh.write("path\tbytes\tsha256\tutc\n")
        fh.write(f"{path.relative_to(REPO)}\t{path.stat().st_size}\t{h}\t"
                 f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    return h


def manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    rows = [ln.split("\t") for ln in MANIFEST.read_text().splitlines()[1:] if ln.strip()]
    return {r[0]: r[2] for r in rows}       # the last row of a path wins


def check(hashes: bool = False) -> None:
    man = manifest()
    disk = {str(p.relative_to(REPO)) for root in ROOTS if root.exists()
            for p in root.rglob("*.npz")}
    extra, missing = sorted(disk - set(man)), sorted(set(man) - disk)
    bad = [p for p in sorted(disk & set(man)) if hashes and sha256(REPO / p) != man[p]]
    if extra or missing or bad:
        raise ShardError(f"{len(extra)} shards not in the manifest {extra[:5]}; "
                         f"{len(missing)} missing {missing[:5]}; {len(bad)} hash "
                         f"mismatches {bad[:5]}")
