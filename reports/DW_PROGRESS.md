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
