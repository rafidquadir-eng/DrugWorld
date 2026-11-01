# DrugWorld status report 2 (2026-11-01, 15:10 UTC)

Branch `drugworld` of private `rafidquadir-eng/DrugWorld`. Design commit: `eefeeb0`, which is
Amendment A1 merged unchanged. As the author decided, we stay on the 8 x A100-40GB instance and do not move to a 1-GPU instance (A1 section 3), so the A1
restore/re-embed check is not needed. The earlier status is in `reports/DW_STATUS_SUMMARY.md` (`d70d5d0`)
and `reports/DW_PROGRESS.md`.

## Done

| Step | What | Commit |
|---|---|---|
| 1 | `origin/author-a1` merged unchanged (fast-forward) and pushed: the new design commit | `eefeeb0` |
| — | Backup audit: 245 files (4.70 GB) verified on `~/LambdaFilesystem/drugworld_backups/`, 126 ledger rows added | `08255fc` |
| 2 | D1 launcher for idr0119: all three channels, half size, 8 workers x 3 connections = 24 (EBI limit), this instance only, field-interleaved shards `<plate>.w<k>of8` | `f89eec4` |
| 2 | CPU work: idr0119 SMILES from IDR, agreeing with PubChem on connectivity for all 7 drugs; idr0064 PubChem SMILES by CID (429 compounds, 376 scaffolds, 10 salt/parent pairs recorded); ECFP4; target-class table; MoLFormer-XL (A1 O3) run on CPU after verifying 11 file hashes (`manifests/models.yaml`), so `scripts/run_molformer.sh` was not needed; AU565 re-dose record (A1 C2) | `37ca0c0` |
| 3 (code) | D2 code, written before G-Data and the freeze: shard loader, decoder targets, G-Data rule, idr0064 scaffold split (A1 O2 fallback: `data/annotations/goglia_supp/` is empty), selection plates, H2a model set (A1 O4), freeze script (protocol with A1 C1-C6, encodings, normalisation, vehicle start states, decoders), feasibility script | `eb3c396` |
| 5 (code) | D3 code: content-only DrugPredictor (no ID embedding), model sets h1 / h2a / line:&lt;L&gt; / h2b, all families, job runner, selection rule | `6d9260f` |
| 5 (code) | D4/D5 evaluation code: Gross et al. phase-rate ODE baseline, latent and readout baselines, baseline-strength chains, sign-flip + Holm gates, G-Plan (a)/(b), own-start and global reporting; test evaluation is locked behind `DW_D5_TEST=1` | `c8aaad1` |
| 5 (code) | Controls (static clip, context swap, action shuffle, leakage, plate-batch probe) and the 9-item D4 checklist | `e7f65a7` |
| 5 (code) | ODE fit capped at 1,000 DE generations; tested on synthetic data first, where it recovered the true 72 h cell number (0.170 fit vs 0.162 true at 3 nM, 0.016 vs 0.015 at 30 nM) | `fb9c56c` |
| — | This report, the shard manifest and the backup ledger rows up to now | this commit |

All of the code above was committed before the freeze. Nothing has been frozen, no model has been trained, and
no drug-treated well has been evaluated.

## Running now: D1 idr0119 embedding

- **Progress:** 2,316 of 3,026 fields done (77%) in 12 h, at about 190 fields per hour, with 0 fetch or processing errors. The 8 failure files are empty. 400 shards (latents plus readouts, 19 GB of latents) are in `manifests/latent_shards.tsv`.
- **Backups:** every shard matches its ledger row and its backup by SHA-256 (checked just now).
- **ETA:** about 3.7 h, which is around 19:00 UTC.
- **GPUs:** not idle. They are running the D1 encoder, which is network-bound, so utilisation is low.
- **Cost:** about $10/h for the 8 x A100 instance (please confirm the rate on the console).

## Points for the author

1. **Mixed AU565 plates** (AU00901, AU01001, AU01101: combination, single-drug and vehicle wells on the same plate).
   The study text says only that single-drug AU565 plates were re-dosed at 48 h. The 48 h
   signature is absent from these plates' vehicle wells: no missing slot 96 or 60-min gap, and no count
   step. So I plan to set the re-dose covariate to 0 for them (1 for single-drug-only plates).
   Before the freeze I will confirm this on all D1 vehicle fields.
   If the evidence is ambiguous, I will stop and ask.
2. **D3 estimate already sent:** about 390 runs. Expected about 130 GPU-h, which is about 16 h on 8 GPUs (about $170);
   worst case about 280 GPU-h, about 35 h (about $360). I will re-time it on the frozen data before D3.

## Remaining, in order

1. Finish D1. Then check the failures and the shard guard, and confirm the mixed-plate re-dose setting.
2. G-Data (`scripts/40_gdata.py`; at least 95% of wells), committed.
3. An end-to-end smoke test of freeze → short training → predict → gates → controls → checklist, in a
   scratch copy. Its results are thrown away.
4. **The D2 freeze, in one commit:** splits (seed 20261030), protocol (A1 C1-C6), decoders,
   encodings, normalisation.
5. Feasibility gates: F-Decode on training wells of both sets, F-Dose, F-Power (`scripts/42_feasibility.py`).
   If F-Dose or F-Power fails, I stop and report. A dry run on the split code gave F-Power counts of
   H1 test 46 interior + 17 extrapolation, H2a 10 and H2b 88; this is not a result.
6. Re-timed GPU-hour estimate for D3, sent to you.
7. D3: the grid on 8 GPUs, the ODE fit on CPU, and the selection, committed before any evaluation. Then
   references, H1c and the global-normalisation runs.
8. D4: validation forecasts, gates, controls, the checklist and `reports/DW_GATES_VAL_SUMMARY.md`.
   **Then I stop for your "go".**
9. After "go": D5 (test once) and D6.
