# Code shared with GeneWorld

`src/geneworld/` is a verbatim copy of GeneWorld's `src/geneworld/` (repository
`rafidquadir-eng/GeneWorld`, branch `stage1`, commit `ed4d43fc1376ec32518dd22742b8bcc51256aacd`), taken with
`git archive` on 2026-10-30. Per-file SHA-256 values are in
`manifests/shared_code_geneworld.sha256`. It is kept unchanged so that provenance, the claims
checker, stats, cache, s3fetch, the trainer and the evaluation code can be traced to that
commit; DrugWorld code lives in `src/drugworld/` and imports from it.

Changed copies (in `src/drugworld/`, with the difference stated in the module docstring):

| module | from | difference |
|---|---|---|
| `drugworld/backup.py` | `geneworld/backup.py` | `DEST_ROOT` = `~/LambdaFilesystem/drugworld_backups`; `REPO` = this repository |

Also copied: `requirements-lock-geneworld.txt` (GeneWorld's `requirements-lock.txt` at the
same commit), `manifests/models_geneworld.yaml` (pinned backbone revisions and weight hashes),
`docs/tijepa_rules/` (rules), and `docs/geneworld_context/` (GeneWorld's
`reports/ROBUSTNESS_STAGE2.md` and `reports/STAGE2_COMPLETION_SUMMARY.md`, for context).
