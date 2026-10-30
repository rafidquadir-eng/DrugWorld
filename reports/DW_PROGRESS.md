# DrugWorld progress

## Step 0: setup (2026-10-30)

- **Instance:** 8 x A100-SXM4-40GB (Lambda, us-east-1; the GeneWorld instance), driver
  580.178.04, 124 CPUs, 1.7 TB RAM, persistent filesystem `~/LambdaFilesystem`. Plan section 11
  names 1 x A100-40GB for D0-D2; the author chose on 2026-10-30 to run DrugWorld on this
  instance and to use all of its GPUs and CPUs ("Run here, and parallelize anything that you
  can"). Approximate list cost: 8 x A100-40GB about $10/h (author to confirm from the console).
- **OS freeze** (`scripts/00_os_freeze.sh`, `logs/setup/os_freeze.log`): apt timers and
  unattended-upgrades disabled, apt services masked, 24 NVIDIA/Fabric Manager packages held
  (no CUDA system packages are installed; torch ships its own CUDA), needrestart list-only. No
  reboot, no `apt upgrade`. The freeze was first applied on this instance for GeneWorld
  (2026-10-23) and re-verified here.
- **Environment:** `~/dwvenv` (Python 3.12.3) from GeneWorld's `requirements-lock.txt` plus
  `rdkit==2026.3.6`; `requirements-lock.txt` here is the resulting `pip freeze` (identical to
  GeneWorld's except for rdkit). torch 2.11.0+cu128 sees 8 GPUs.
- **Repository:** private `rafidquadir-eng/DrugWorld`, branch `drugworld` (no `main` push).
  Plan copied unchanged (`docs/DRUGWORLD_PLAN.md`, SHA-256 `bb95b7b2…41ed`, from GeneWorld
  `dad0c40`). Shared code from GeneWorld `stage1` commit `ed4d43f` (`docs/SHARED_CODE.md`).

## Step 1, D0 (in progress)

- **D0.0 data search** (`reports/D0_DATA_SEARCH.md`): 21 candidates recorded across IDR, the
  BioImage Archive, Zenodo, figshare, LINCS and the Cell Painting Gallery; none meets all four
  criteria, none is unclear. Proceeding with idr0119 and idr0064 as designed.
- **Annotation audit** (`reports/D0_DATA_AUDIT.md`, `results/d0/audit.json`).
- **Access and throughput** (`reports/D0_THROUGHPUT.md`): idr0119 is only on EBI FTP (HTTPS,
  per-client connection limit, about 29–38 MB/s, 3.13 TB, 23–30 h for all channels); idr0064
  on the BIA S3 bucket (OME-Zarr, 39–102 MB/s, 57 GB, 10–25 min).
- **Pilot** (`scripts/10_pilot.py`, vehicle wells only, field 1): idr0119 83 fields on all 34
  plates, idr0064 18 DMSO wells on all 9 plates; latents and per-cell readouts in
  `data/pilot/` (backed up, `reports/BACKUP_LEDGER.tsv`). Fetch about 5 min per idr0119 field
  at 3 connections per worker; embedding about 42 s per field per GPU.
- **Pilot probe** (`scripts/11_pilot_probe.py`, rule committed in `4c62fee` before any result;
  `results/d0/pilot_probe.json`): ridge, GroupKFold over plates, pooled held-out R^2.

  | idr0119 config | log count R^2 | G1 fraction R^2 | mean |
  |---|---|---|---|
  | phase_L0 | 0.887 | 0.514 | 0.700 |
  | phase_L1 | 0.900 | 0.670 | 0.785 |
  | rgb_L0 | 0.933 | 0.681 | 0.807 |
  | **rgb_L1 (chosen)** | **0.924** | **0.797** | **0.860** |

  G1 threshold (GMM valley of the log HDHB nuclear/ring ratio): 0.575.
- **F-Decode, provisional (vehicle wells):** idr0119 passes (count 0.924 >= 0.8, G1 0.797 >=
  0.5). **idr0064 fails: Erk C/N R^2 = 0.479 < 0.5** (18 DMSO wells, 2,016 clip-tiles, 9
  plates; target SD 0.061). Per plan section 3.4 and the author's instruction, work stopped
  here and was reported to the author. F-Dose and F-Power were not run (they need the D2 split).
- **idr0064 D1 embedding** (run in parallel before the probe; the idr0064 configuration is fixed
  by the plan, not by the pilot): 9 plates, 450 wells, 0 failures, 0 filled frames; 18 shards
  in `manifests/latent_shards.tsv`, hash check passed, backed up. No model has been trained and
  no split has been frozen. idr0119 D1 has not been launched.
