"""D3: train one model = model set x family x grid configuration x seed (plan section 6).

As in GeneWorld's trainer (scripts/50_train_a2.py there): loss = L1 teacher-forced one-step
error + L1 rollout error (context of 1-4 measured steps drawn per batch, rollout of 6 steps),
AdamW at lr 3e-4 with OneCycle over 12,000 steps (5% warm-up), batch 256 fields, gradient
clipping at 1.0, bf16 autocast. Every 250 steps the validation primary error is computed
(rollout from the counterfactual start state, MSE over dimensions at the model set's primary
horizon clips, mean over validation fields); the checkpoint with the lowest value is kept and
training stops 3,000 steps after the last improvement. Depth: 6 at width 512, 7 at width 768
(GeneWorld's grid). At the end the selection-only fields' error (train_sel) is also recorded.

Single writer per run (lock file); checkpoints written atomically and backed up with SHA-256.

  PYTHONPATH=src python scripts/50_train.py --set h1 --family dw_full --config w512_d0.1_wd0.05 --seed 0
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np
import torch

from drugworld import REPO, backup
from drugworld import train as TR
from drugworld.model import DrugPredictor

print = functools.partial(print, flush=True)
DEV = "cuda"
GRID = {"width": {512: 6, 768: 7}, "dropout": [0.1, 0.3], "weight_decay": [0.05, 0.2]}
LR, BATCH, MAX_STEPS, EVAL_EVERY, PATIENCE, ROLLOUT = 3e-4, 256, 12_000, 250, 3_000, 6
PRIMARY = {"h1": ["48h", "72h"], "h2a": ["24h", "48h"], "h2b": ["2h", "3h", "4h"]}
CK = REPO / "checkpoints" / os.environ.get("DW_CK_SUBDIR", "d3")
RES = REPO / "results" / "d3" / "train"


def configs():
    return [{"id": f"w{w}_d{d}_wd{wd}", "width": w, "depth": GRID["width"][w], "dropout": d,
             "weight_decay": wd} for w in GRID["width"] for d in GRID["dropout"]
            for wd in GRID["weight_decay"]]


def primary_clips(model_set, prot):
    key = "h1" if model_set.startswith("line:") else model_set
    which = TR.dataset_of(model_set)
    hz = prot["sets"][which]["clips"]["horizons"]
    return [hz[h]["clip_index"] for h in PRIMARY[key]]


def take_lock(path):
    if path.exists():
        info = json.loads(path.read_text())
        if info["host"] == socket.gethostname() and os.path.exists(f"/proc/{info['pid']}"):
            sys.exit(f"{path} is held by live PID {info['pid']}")
    path.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                                "start": time.time(),
                                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}))


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Batch:
    def __init__(self, act: TR.Actions):
        self.toks = torch.from_numpy(act.toks).to(DEV)
        self.tmask = torch.from_numpy(act.tmask).to(DEV)
        self.veh = torch.from_numpy(act.vehicle).to(DEV)
        self.red = torch.from_numpy(act.redose).to(DEV)

    def get(self, j):
        return self.toks[j], self.tmask[j], self.veh[j], self.red[j]


@torch.no_grad()
def rollout_error(model, z, bat, idx, clips, bs=256):
    """Per-field MSE at `clips` of a rollout from step 0 (z on the GPU, float16)."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i:i + bs], device=DEV)
        zz = z[j].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.rollout(zz[:, :1], *bat.get(j), max(clips)).float()
        sel = [c - 1 for c in clips]
        out.append(((pred[:, sel] - zz[:, clips]) ** 2).mean((1, 2, 3)).cpu())
    model.train()
    return torch.cat(out).numpy() if out else np.zeros(0)


def run_name(model_set, family, cfg, seed, norm, start):
    tag = "" if (norm, start) == ("primary", "vehicle") else f"__{norm}_{start}"
    return f"{model_set.replace(':', '-')}__{family.replace(':', '-')}__{cfg}__s{seed}{tag}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--norm", default="primary", choices=["primary", "global"])
    ap.add_argument("--start", default="vehicle", choices=["vehicle", "own"])
    ap.add_argument("--max-steps", type=int, default=0, help="benchmark / smoke tests only")
    args = ap.parse_args()
    prot = TR.protocol()
    cfg = next(c for c in configs() if c["id"] == args.config)
    max_steps = args.max_steps or MAX_STEPS
    name = run_name(args.set, args.family, cfg["id"], args.seed, args.norm, args.start)
    CK.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    lock = CK / f"{name}.lock"
    take_lock(lock)
    ck = CK / f"{name}.pt"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    which = TR.dataset_of(args.set)
    d = TR.load(which, args.norm, "vehicle" if args.start == "vehicle" else "own")
    role = TR.roles(d, args.set).to_numpy()
    fam = "dw_full" if args.family == "dw_line" else args.family
    act = TR.Actions(d, fam, args.seed)
    tr_idx, va_idx = np.flatnonzero(role == "fit"), np.flatnonzero(role == "val")
    sel_idx = np.flatnonzero(role == "sel")
    clips = primary_clips(args.set, prot)
    z = torch.from_numpy(d.z).to(DEV)
    T = z.shape[1]
    bat = Batch(act)
    model = DrugPredictor(z.shape[-1], act.tok_dim, TR.mode_of(fam), width=cfg["width"],
                          depth=cfg["depth"], n_steps=T, drop=cfg["dropout"]).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=max_steps, pct_start=0.05)
    fit_sha = hashlib.sha256("\n".join(
        f"{r.plate}|{r.well}|{r.field}" for r in d.fields.iloc[tr_idx].itertuples()).encode()
    ).hexdigest()
    print(f"{name}: fit fields {len(tr_idx)}, val fields {len(va_idx)}, sel {len(sel_idx)}, "
          f"params {n_par / 1e6:.1f}M, primary clips {clips}")
    best, best_step, curve, t0 = np.inf, 0, [], time.time()
    for step in range(1, max_steps + 1):
        j = torch.as_tensor(tr_idx[torch.randint(len(tr_idx), (BATCH,), generator=gen)
                                   .numpy()], device=DEV)
        zz = z[j].float()
        a = bat.get(j)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(zz[:, :-1], *a)
            loss_tf = (pred.float() - zz[:, 1:]).abs().mean()
            ctx = int(rng.integers(1, 5))
            r = min(ROLLOUT, T - ctx)
            roll = model.rollout(zz[:, :ctx], *a, r)
            loss_ro = (roll.float() - zz[:, ctx:ctx + r]).abs().mean()
            loss = loss_tf + loss_ro
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0 or step == max_steps:
            score = float(rollout_error(model, z, bat, va_idx, clips).mean())
            curve.append({"step": step, "val_primary": score, "loss_tf": float(loss_tf),
                          "loss_ro": float(loss_ro), "minutes": (time.time() - t0) / 60})
            if score < best:
                best, best_step = score, step
                atomic_save({"model": model.state_dict(), "step": step, "val_primary": score,
                             "args": vars(args), "config": cfg, "tok_dim": act.tok_dim,
                             "dim": int(z.shape[-1]), "n_steps": T, "mode": TR.mode_of(fam),
                             "permutations": act.record, "fit_fields_sha256": fit_sha,
                             "protocol_sha256": backup.sha256(TR.PROT_PATH),
                             "design_commit": prot["design_commit"]}, ck)
            if step - best_step >= PATIENCE:
                break
    sd = torch.load(ck, map_location=DEV, weights_only=False)
    model.load_state_dict(sd["model"])
    sel_err = float(rollout_error(model, z, bat, sel_idx, clips).mean()) if len(sel_idx) \
        else None
    sd["curve"], sd["stopped_at"], sd["sel_primary"] = curve, step, sel_err
    atomic_save(sd, ck)
    if not args.max_steps:
        backup.backup(ck)
    rec = {"name": name, "set": args.set, "family": args.family, "config": cfg["id"],
           "seed": args.seed, "norm": args.norm, "start": args.start,
           "best_val_primary": best, "best_step": best_step, "stopped_at": step,
           "sel_primary": sel_err, "n_params": n_par, "n_fit_fields": int(len(tr_idx)),
           "n_val_fields": int(len(va_idx)), "minutes": (time.time() - t0) / 60,
           "sec_per_step": (time.time() - t0) / step, "curve": curve,
           "permutations": act.record, "fit_fields_sha256": fit_sha,
           "checkpoint": ck.name, "checkpoint_sha256": backup.sha256(ck),
           "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
           "design_commit": prot["design_commit"]}
    out = RES if not args.max_steps else REPO / "results" / "d3" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
    if not args.max_steps:
        backup.backup(out / f"{name}.json")
    lock.unlink(missing_ok=True)
    print(f"{name}: best {best:.4f} at {best_step} (stopped {step}); "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
