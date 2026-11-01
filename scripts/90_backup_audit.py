"""Backup audit before leaving an instance (amendment A1 section 3).

Every file under data/ (annotation git clones' .git excluded), results/, logs/ and manifests/
must exist on ~/LambdaFilesystem/drugworld_backups/ with the same SHA-256 and have a ledger row
(reports/BACKUP_LEDGER.tsv, append-only) with that hash. Files that fail either test are backed
up again (a new ledger row; old rows are kept). Then every file is re-verified against its
backup copy and the counts are written to results/backup_audit.json.

  PYTHONPATH=src python scripts/90_backup_audit.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from drugworld import REPO, backup

ROOTS = ("data", "results", "logs", "manifests")


def files():
    out = []
    for root in ROOTS:
        for p in sorted((REPO / root).rglob("*")):
            if p.is_file() and ".git" not in p.relative_to(REPO).parts \
                    and not p.name.endswith(".part"):
                out.append(p)
    return out


def ledger() -> dict[str, set[str]]:
    rows = [ln.split("\t") for ln in backup.LEDGER.read_text().splitlines()[1:] if ln.strip()]
    led: dict[str, set[str]] = {}
    for r in rows:
        led.setdefault(r[0], set()).add(r[2])
    return led


def state(p):
    rel = str(p.relative_to(REPO))
    h = backup.sha256(p)
    dest = backup.DEST_ROOT / rel
    ok_copy = dest.exists() and backup.sha256(dest) == h
    return rel, h, ok_copy


def main() -> int:
    fs = files()
    led = ledger()
    with ThreadPoolExecutor(32) as ex:
        st = list(ex.map(state, fs))
    redo = [p for p, (rel, h, ok) in zip(fs, st) if not ok or h not in led.get(rel, set())]
    print(f"{len(fs)} files; {len(redo)} need a backup or a ledger row")
    for p in redo:                      # sequential: ledger appends stay ordered
        backup.backup(p)
    # results/backup_audit.json is written last and backed up by itself below
    led = ledger()
    with ThreadPoolExecutor(32) as ex:
        st = list(ex.map(state, fs))
    bad = [rel for rel, h, ok in st if not ok or h not in led.get(rel, set())]
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "roots": ROOTS,
           "files": len(fs), "bytes": sum(p.stat().st_size for p in fs),
           "backed_up_now": [str(p.relative_to(REPO)) for p in redo], "failed": bad}
    path = REPO / "results" / "backup_audit.json"
    path.write_text(json.dumps(out, indent=1))
    backup.backup(path)
    print(f"verified {len(fs) - len(bad)} of {len(fs)}; failed {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
