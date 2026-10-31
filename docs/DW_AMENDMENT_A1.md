# DrugWorld amendment A1 (author, 2026-10-30, before the D2 freeze)

Amends `docs/DRUGWORLD_PLAN.md`. Written after D0 (`reports/DW_STATUS_SUMMARY.md`,
`d70d5d0`) and **before any split is frozen, any model is trained, or any drug-treated well is
evaluated**. Every change below follows from the annotation audit, the throughput measurements
and the vehicle-only pilot, never from an outcome on drug-treated data. This file's commit hash
becomes `design_commit` for every later result.

## 1. Decisions on the agent's open items

| # | item | decision |
|---|---|---|
| O1 | Provisional F-Decode fails for idr0064 (Erk C/N R² 0.479 on 18 DMSO wells; target SD 0.061) | The plan defines F-Decode "on training wells". The vehicle-only pilot was a proxy with almost no target variance. **Run F-Decode as written after the D2 freeze, on idr0064 training-split wells** (with the same ridge and plate-grouped cross-validation). If it still fails, apply the plan's stated consequence: drop the idr0064 *decoded readouts* only. H2b on latents is unaffected, and the project continues. Report the provisional value beside the final one. |
| O2 | Goglia Table S1 (Erk-phenotype classes) is captcha-blocked | The author downloads it (PMC or Europe PMC) and copies it to `data/annotations/goglia_supp/`. **Fallback, fixed now:** if it is not present at the D2 freeze, the scaffold split is stratified by vendor target class only, and that is recorded. Never work around the captcha. |
| O3 | MoLFormer-XL needs `trust_remote_code` (revision `7b12d946`, code inspected) | Approved at that pinned revision, with the SHA-256 of every remote code file recorded in `manifests/models.yaml` before first use. If auto mode still blocks it, write the exact command to `scripts/run_molformer.sh` and tell the author, who runs it once. If neither is possible by the D2 freeze, drop the source and apply G-Compound's Holm correction over the remaining sources. |
| O4 | Combination component doses fall in held-out single-drug doses | Accepted: **H2a uses its own model set**, trained on all AU565 single-drug wells plus the training-pair combinations. H2a models are never evaluated on H1 sets, and H1 models never on H2a sets. The additive-composition and HSA baselines use the same single-drug wells. |
| O5 | Instance and cost | See section 3: move to 1 x A100 for D1 now. |

## 2. Design corrections from the audit

**C1. Start state: counterfactual for both datasets.** The plan's rule ("own first clip if drug
addition precedes the first frame by <= 1 h") was meant to keep the drug's effect out of the
context. The audit shows it does not:
- idr0064 drugs act 30 min before the first frame, and Erk responds within minutes;
- idr0119 clip 0 spans 0–8 h on drug.

**Primary rule (replaces section 5 "Start state"):** every forecast starts from the
**plate-matched vehicle state**, the mean first-clip latent of that plate's vehicle (DMSO)
wells, given per tile position. So the model must produce the treated trajectory from an
untreated state plus the action.
- Secondary: the well's own first clip, reported beside.
- If a plate has no usable vehicle well, use the mean of the same line's vehicle wells on the
  plates imaged in the same batch, and list it.
- Horizons are measured from drug addition (idr0119) and from the first frame (idr0064).

**C2. AU565 re-dosing at 48 h.** AU565 single-drug plates had the medium replaced (vehicle
wells included) and fresh drug added at 48 h. The combination plates are recorded as not
re-dosed; verify this from the study files and record it.
- Every model, the action-agnostic ones included, receives a per-plate protocol covariate,
  "medium change and re-dose at 48 h" (0/1). It is a covariate, not action content.
- **H2a primary horizons** are the two clips that **end at or before 48 h** (16–24 h and
  40–48 h). The 72 h horizon is secondary for H2a, since the protocols differ after 48 h.
- H1 keeps 48 h and 72 h as primary, measured as clip end times, so that the 48 h clip does
  not straddle the re-dose. State the clip definition in the protocol.

**C3. Normalisation with few vehicle wells.** idr0119 has 1–3 vehicle wells per plate and
idr0064 has 2 DMSO wells per plate, too few for per-plate SDs (the GeneWorld lesson).
Replaces section 5 "Normalisation":
- **idr0119:** per **line** and clip, the mean and SD over all of that line's vehicle fields
  (18–23 wells, about 70–90 fields per line); the SD is floored at the 5th percentile.
  Per-plate vehicle centring is a sensitivity analysis.
- **idr0064:** per plate and clip, centre on the **median of all compound wells on the plate**
  (about 50, a robust "typical" state); scale by the global SD of training wells.
  DMSO-centred normalisation is a sensitivity analysis.
- The counterfactual start (C1) is built in the same normalised space.

**C4. Drug encoding for idr0119 (7 drugs).** A 128-dimensional PCA fitted on 7 drugs is
undefined. For idr0119, the drug encoding is ECFP4 (1024 bits, as is) plus the pretrained
molecular embedding (as is). With 7 drugs it works as a fixed drug code; that is intended,
because H1 tests generalisation in **dose**, not structure. No structure-generalisation claim
is made from idr0119. The 128-dimensional PCA stays for idr0064, fitted on its training
compounds.

**C5. H1c "same dose" across lines.** The dose grids differ by line, so the
other-lines baseline uses each training line's response **interpolated in log dose** at the
held-out dose, restricted to doses inside that line's measured range. Conditions outside every
training line's range are excluded from H1c and counted.

**C6. Statistical units, confirmed.**
- H1: drug × line × dose (expected test count: about 46 interior conditions, plus the
  extrapolation set).
- H2a: combination condition (test: 10 conditions in 4 pairs, of which 7 are
  Lapatinib+Palbociclib; also report the result without that pair).
- H2b: the compound (one well each; about 86 in test).
- F-Power is evaluated on these counts at the freeze.

## 3. Operations

- **Move to 1 x A100-40GB for D1.** The idr0119 transfer is limited by EBI's per-client
  connection cap (about 24–32 connections), not by GPUs: about 25 h on 8 GPUs (about $250) or
  about 35 h on one (about $45).
  1. Before the move, back up everything and push.
  2. Restore the idr0064 shards on the new instance from `~/LambdaFilesystem/drugworld_backups/`.
  3. Verify the shard manifest.
  4. Re-embed 2 idr0064 wells and require byte identity (plan section 4).
- **Never run downloads from more than one instance or IP at a time** to get around EBI's
  per-client limit; that would bypass an access control.
- The D3 grid decision (1 vs 8 GPUs) comes back to the author with the cost of each, as planned.

## 4. Order from here

1. Merge this file and commit it unchanged.
2. On the 1-GPU instance:
   - restore and check (section 3);
   - D1 idr0119 with the chosen configuration (all three channels, half size), then G-Data;
   - meanwhile, on CPU: PubChem SMILES, ECFP4, and the target-class table; MoLFormer per O3.
3. D2 freeze (one commit):
   - splits (the drafted code, seed 20261030, plus the O2 fallback if needed);
   - the protocol with C1–C6;
   - the decoders;
   - the encodings.
4. F-Decode (training wells, both datasets), F-Dose and F-Power. Report and **stop if F-Dose or
   F-Power fails**; an F-Decode failure applies only its stated consequence.
5. The D3 GPU estimate goes to the author; then D3 and D4, then stop for "go".
