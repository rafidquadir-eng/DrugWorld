# DrugWorld

A drug-conditioned latent world model of live-cell time-lapse video, designed so that a
positive result can only come from generalising through what the action is (dose, structure,
composition). The design document is `docs/DRUGWORLD_PLAN.md` (author, 2026-10-30; copied
unchanged from `rafidquadir-eng/GeneWorld` branch `drugworld-plan`, commit `dad0c40`,
SHA-256 `bb95b7b2d85042dba4d6fd87f2240870bf2071593ae943880c1cbdf6596e41ed`).

- Code shared with GeneWorld: `docs/SHARED_CODE.md`.
- Environment: `~/dwvenv`, built from `requirements-lock-geneworld.txt` plus `rdkit==2026.3.6`;
  the installed set is `requirements-lock.txt` (`pip freeze`).
- Operations: `scripts/00_os_freeze.sh` (plan section 11), log in `logs/setup/os_freeze.log`.
- Backups: `~/LambdaFilesystem/drugworld_backups/`, ledger `reports/BACKUP_LEDGER.tsv`.
- Progress: `reports/DW_PROGRESS.md`.

Sole author: Rafid Quadir. Nothing here is submitted, posted or released.
