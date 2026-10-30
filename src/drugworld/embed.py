"""Frozen V-JEPA 2 ViT-L embedding of multi-channel clips (plan section 4).

Same backbone, pinned revision and weights hash as GeneWorld (`geneworld.eval.vjepa2`):
input (B, 16, 3, 256, 256) in [0, 1], ImageNet mean/std, bfloat16 autocast. Per clip-tile the
last-layer tokens (8 x 16 x 16) are kept as their mean over time and space (`mean`, D = 1024)
and as a time-mean 4 x 4 spatial grid (`grid`, 16 x D), both float16 (GeneWorld pooling).

Channel mappings (plan section 3, item 3), applied to per-field percentile-scaled channels:
  (i)   ``phase``: phase contrast as grey, replicated to R, G, B;
  (ii)  ``rgb``: R = nuclear reporter (NLS-mCherry), G = cell-cycle reporter (clover-HDHB),
        B = phase contrast (the plan's FUCCI-red / FUCCI-green / phase, for this reporter);
  (iii) ``erk``: R = Erk-KTR, G = H2B-RFP, B = their mean (idr0064).
"""
from __future__ import annotations

import numpy as np
import torch

from geneworld.eval.vjepa2 import REPO_ID, REVISION

NF, TILE = 16, 256
DEV = "cuda"
MAPPINGS = {"phase": ("phase", "phase", "phase"), "rgb": ("nuclear", "hdhb", "phase"),
            "erk": ("ktr", "h2b", "mean")}


def scale(stack: np.ndarray | torch.Tensor, lo_q: float = 0.005,
          hi_q: float = 0.998) -> torch.Tensor:
    """(T, H, W) -> float in [0, 1] by the movie's 0.5th / 99.8th percentiles (GeneWorld)."""
    f = torch.as_tensor(np.asarray(stack, np.float32) if isinstance(stack, np.ndarray)
                        else stack, device=DEV).float()
    sub = f[::4, ::4, ::4].flatten()
    lo, hi = torch.quantile(sub, torch.tensor([lo_q, hi_q], device=DEV))
    return ((f - lo) / torch.clamp(hi - lo, min=1e-6)).clamp_(0, 1)


def compose(ch: dict[str, torch.Tensor], mapping: str) -> torch.Tensor:
    """Scaled channels {name: (T, H, W)} -> (T, 3, H, W) for a mapping."""
    planes = []
    for name in MAPPINGS[mapping]:
        planes.append((ch["ktr"] + ch["h2b"]) / 2 if name == "mean" else ch[name])
    return torch.stack(planes, 1)


def downsample(x: torch.Tensor, k: int) -> torch.Tensor:
    """(T, C, H, W) mean-pooled by k (pyramid level log2 k)."""
    return x if k == 1 else torch.nn.functional.avg_pool2d(x, k)


def tile_clips(x: torch.Tensor, starts: list[int], tiles: list[tuple[int, int]]) -> torch.Tensor:
    """(T, 3, H, W) -> (len(starts) * len(tiles), 16, 3, 256, 256), clip-major."""
    return torch.stack([x[s:s + NF, :, y:y + TILE, xx:xx + TILE]
                        for s in starts for (y, xx) in tiles])


class VJEPA2:
    def __init__(self):
        from transformers import VJEPA2Model
        self.m = VJEPA2Model.from_pretrained(REPO_ID, revision=REVISION, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, clips: torch.Tensor, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
        means, grids = [], []
        for i in range(0, len(clips), batch):
            v = (clips[i:i + batch].to(DEV) - self.mean) / self.std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = self.m(pixel_values_videos=v).last_hidden_state
            b, _, d = tok.shape
            g = tok.float().reshape(b, 8, 16, 16, d).mean(1).permute(0, 3, 1, 2)
            means.append(g.mean((2, 3)).half().cpu())
            grids.append(torch.nn.functional.adaptive_avg_pool2d(g, 4).flatten(2)
                         .transpose(1, 2).half().cpu())
        return torch.cat(means).numpy(), torch.cat(grids).numpy()
