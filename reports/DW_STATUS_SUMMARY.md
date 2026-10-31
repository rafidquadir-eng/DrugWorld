# DrugWorld status summary (2026-10-31)

Plan: `docs/DRUGWORLD_PLAN.md` (unchanged copy of GeneWorld `drugworld-plan` `dad0c40`, SHA-256
`bb95b7b2…41ed`). Repository: private `rafidquadir-eng/DrugWorld`, branch `drugworld`.
Detailed log: `reports/DW_PROGRESS.md`.

**Where it stands:** step 0 is done and D0 is done except for F-Dose and F-Power, which need the
D2 split. Work is **stopped at D0**: the provisional F-Decode gate fails for idr0064 (Erk C/N
R² 0.479, threshold 0.5), and the plan says to stop and report when a feasibility gate fails.
No split has been frozen and no model has been trained. All 8 GPUs are idle with nothing
queued (about $10/h at list price; the author should confirm the rate in the console).

## 1. Done

### Step 0: setup (`b191c51`)
- **Instance:** 8 x A100-SXM4-40GB, us-east-1, 124 CPUs, 1.7 TB RAM, driver 580.178.04. The
  author chose this instance over the 1-GPU instance the plan names ("Run here, and parallelize
  anything that you can").
- **OS freeze** (`scripts/00_os_freeze.sh`, `logs/setup/os_freeze.log`): apt timers and
  unattended-upgrades disabled and masked, 24 NVIDIA/Fabric Manager packages held, needrestart
  set to list only. No upgrade and no reboot.
- **Environment:** `~/dwvenv`, built from GeneWorld's `requirements-lock.txt` plus
  `rdkit==2026.3.6`. Its `pip freeze` is saved as `requirements-lock.txt`.
- **Shared code:** `src/geneworld/` is a verbatim copy of GeneWorld `stage1` `ed4d43f`
  (`docs/SHARED_CODE.md`; hashes in `manifests/shared_code_geneworld.sha256`).

### D0.0: data search (`12f850a`, `reports/D0_DATA_SEARCH.md`)
Across IDR, the BioImage Archive, Zenodo, figshare, LINCS and the Cell Painting Gallery, 21
candidate datasets were checked. None meets all four criteria and none is unclear, so the
project proceeds with idr0119 and idr0064 as designed.

### Annotation audit (`12f850a`, `reports/D0_DATA_AUDIT.md`, `results/d0/audit.json`)
- **idr0119:**
  - 34 plates, 760 wells, 3,026 fields, 1.75 M single-plane TIFFs (1408 x 1040).
  - The cell-cycle reporter is **clover-HDHB, not FUCCI**; the other channels are
    NLS-mCherry and phase.
  - All plates have **30-min frames** over 0–96 h (the study text says 2 h for three lines; the
    files are used).
  - Doses are **nM**, although the IDR column is labelled micromolar.
  - **Drugs are added at frame 0.** AU565 single-drug wells were re-dosed at 48 h.
  - Combinations exist for AU565 only: 10 drug pairs, 32 conditions, 3 replicates each.
- **idr0064:**
  - 9 plates, 450 wells, 429 compounds at 2.5 µM.
  - **One well per compound** (no replicate wells).
  - 121 frames at 3-min intervals; **drugs are added 30 min before the first frame**.
  - Channel 0 is Erk-KTR and channel 1 is H2B, confirmed visually.
  - Structures: all 429 compounds have PubChem CIDs, with 380 Murcko scaffolds.

### Access and throughput (`12f850a`, `reports/D0_THROUGHPUT.md`)
- **idr0119** is available only from EBI over HTTPS, and EBI refuses more than about 24–32
  connections per client. At 29–38 MB/s, the 3.13 TB set takes **about 23–30 h**.
- **idr0064** comes from the BIA S3 bucket (OME-Zarr) at 39–102 MB/s: 57 GB, under 30 min.

### Pilot on vehicle wells only (`4c62fee` code and rule; `2ab4ae4` results)
- **Coverage:** idr0119 has 83 vehicle fields across all 34 plates; idr0064 has 18 DMSO wells
  across all 9 plates.
- **Encoder:** frozen V-JEPA 2 ViT-L.
- **Data kept:** latents and per-cell readouts, backed up.
- **Probe:** ridge regression scored with 5-fold cross-validation grouped by plate. The rule
  was committed before any result was computed.

| idr0119 config (channels, scale) | log count R² | G1 fraction R² | mean |
|---|---|---|---|
| phase, full size | 0.887 | 0.514 | 0.700 |
| phase, half size | 0.900 | 0.670 | 0.785 |
| all three channels, full size | 0.933 | 0.681 | 0.807 |
| **all three channels, half size (chosen)** | **0.924** | **0.797** | **0.860** |

**Provisional F-Decode on vehicle wells:**
- **idr0119 passes:** count R² 0.924 (threshold 0.8) and G1 R² 0.797 (threshold 0.5).
- **idr0064 fails: Erk C/N R² = 0.479 < 0.5.** This uses 2,016 data points from 18 wells on 9
  plates, and the target SD is only 0.061.

### D1 embedding, idr0064 (`bd6c777` script; `2ab4ae4` results)
- **Result:** all 450 wells, with 0 failures and 0 gap-filled frames.
- **Storage:** 18 files (9 latent, 9 per-cell readout; 3.3 GB), registered with SHA-256 in
  `manifests/latent_shards.tsv`. The hash check passed, and all files are backed up to
  `~/LambdaFilesystem/drugworld_backups/`.
- **idr0119** D1 has **not** been launched.

### Draft D2 split code (`bd6c777`, `src/drugworld/splits.py`; not frozen)
- **Seed:** 20261030.
- **Dose split:** per drug and cell line, the lowest dose is training and the highest is the
  extrapolation set. Three interior doses are held out (1 validation, 2 test), with as few
  adjacent held-out doses as the grid allows.
- **Lines and combinations:** the seed draws the validation cell line; combination pairs are
  held out leave-pair-out.
- **Dry run:**
  - validation line MDAMB157;
  - test pairs Paclitaxel+Palbociclib, Doxorubicin+Paclitaxel, Gemcitabine+Paclitaxel and
    Lapatinib+Palbociclib (10 conditions);
  - validation pair Gemcitabine+Lapatinib.
- **idr0064 scaffold split:** waits for Goglia Table S1 (open item 2 below).

### Other code (`src/drugworld/`)
- **Streaming:** `fetch.py` downloads in parallel within the connection limit (12 attempts,
  waits capped at 60 s); `stream.py` streams one field into memory (a field fails if a
  channel misses more than 10% of its frames); no raw images are kept.
- **Readouts:** `readouts.py` segments nuclei and computes the HDHB and Erk ratios.
- **Embedding:** `embed.py` wraps the encoder.
- **Integrity:** `shards.py` keeps the shard manifest and refuses to load unregistered,
  missing or altered files.
- **Backups:** `backup.py` copies each file to the backup folder with SHA-256 (120 ledger
  rows).

## 2. Open items for the author

1. **The F-Decode failure for idr0064.** The plan defines F-Decode "on training wells", which
   include drug-treated wells. The pilot used only DMSO wells, where Erk C/N barely varies, so
   R² is limited. Choose one:
   - (a) accept the failure: drop the idr0064 decoded readouts and the ODE comparisons that
     use them, keeping H2b on latents;
   - (b) freeze D2 first, then rerun F-Decode on the idr0064 training-split wells and use that
     result. This is the recommendation.
2. **Goglia Table S1** (Erk-phenotype classes for the stratified scaffold split) is reachable
   only from PMC, which shows a captcha. The author would need to download
   `NIHMS1572046-supplement-7.pdf` from https://pmc.ncbi.nlm.nih.gov/articles/PMC7540725/ into
   `data/annotations/goglia_supp/`.
3. **MoLFormer-XL** (Apache-2.0) needs Hugging Face `trust_remote_code`. Revision `7b12d946` was
   pinned and its code checked: torch and transformers only, weights loaded from safetensors.
   The auto-mode check blocked running it. Either allow it, or this structure source is left
   out.
4. **Combinations and the dose split.** Some combination component doses are held out of the
   single-drug training data: palbociclib 50 nM and lapatinib 50/250 nM (test), gemcitabine
   5 nM (validation). The proposal is a separate H2a model set trained on all single-drug
   wells, following the plan's "single-drug wells of every drug are always available to
   training".
5. **Instance and cost.** The full idr0119 embedding is limited by EBI's connection cap, not by
   the GPUs. It takes about 25 h here (about $10/h) or about 35 h on a 1-GPU instance (about
   $1.30/h). The plan says not to keep a multi-GPU instance through a review pause.

## 3. Next steps after the author replies
1. Launch idr0119 D1 with the chosen configuration (all three channels, half size).
2. Run G-Data.
3. Make the D2 freeze commit: splits, protocol, decoders and encodings.
4. Run F-Dose, F-Power and the confirmed F-Decode.
5. Estimate the GPU-hours for the grid and ask the author about the instance.
6. Run D3 and D4, then stop for "go".

## 4. Commits
| commit | content |
|---|---|
| `b191c51` | step 0: setup, plan copy, shared code |
| `12f850a` | D0.0 search, annotation audit, throughput |
| `4c62fee` | pilot code and probe rule (before any result) |
| `bd6c777` | D1 embedding script, shard guard, draft split code (seed fixed) |
| `2ab4ae4` | pilot results and probe, idr0064 D1 complete, fetch retries |
