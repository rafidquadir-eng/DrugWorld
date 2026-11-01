"""DrugWorld predictor (plan section 6): GeneWorld's block-causal action-conditioned transformer
(`geneworld.models.predictor`) with the action pathway replaced by content only.

Conditioning tokens (2): the action token and a protocol-covariate token.
  action = sum over the condition's (drug, dose) tokens of MLP([drug encoding, dose features])
           (DeepSets; order-invariant; at most two tokens) + vehicle * v, where v is a learned
           vector for the vehicle indicator (a content feature, not an ID). For `none` the
           action token is a constant.
  covariate = c0 + redose * c1 (A1 C2; given to every model, action-agnostic included).
No per-condition, per-drug or per-well learned embedding exists (L2). Tile tokens: input
projection + step embedding, no tile-position embedding; output = z_t + delta for step t + 1.
"""
from __future__ import annotations

import torch
from torch import nn

from geneworld.models.predictor import Block


class DrugPredictor(nn.Module):
    def __init__(self, dim: int, tok_dim: int, mode: str, width: int = 512, depth: int = 6,
                 heads: int = 8, n_steps: int = 31, drop: float = 0.1):
        super().__init__()
        self.mode, self.n_steps = mode, n_steps
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        self.step = nn.Parameter(torch.zeros(n_steps, width))
        nn.init.normal_(self.step, std=0.02)
        self.null = nn.Parameter(torch.zeros(width))
        self.tok = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, width), nn.GELU(),
                                 nn.Linear(width, width))
        self.post = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.vehicle = nn.Parameter(torch.randn(width) * 0.02)
        self.cov = nn.Parameter(torch.randn(2, width) * 0.02)
        self.blocks = nn.ModuleList(Block(width, heads, drop=drop) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def action_token(self, toks, tmask, vehicle):
        """toks (B, 2, tok_dim), tmask (B, 2) bool, vehicle (B,) float -> (B, width)."""
        b = toks.shape[0]
        if self.mode == "none":
            return self.null.expand(b, -1)
        h = (self.tok(toks) * tmask[..., None]).sum(1)
        return self.post(h) + vehicle[:, None] * self.vehicle

    def cond(self, toks, tmask, vehicle, redose):
        cov = self.cov[0] + redose[:, None] * self.cov[1]
        return torch.stack([self.action_token(toks, tmask, vehicle), cov], 1)

    def forward(self, z, toks, tmask, vehicle, redose):
        """z (B, T, K, D) -> predicted z at steps 1..T (B, T, K, D)."""
        b, t, k, _ = z.shape
        x = self.inp(z) + self.step[:t, None, :]
        x = torch.cat([self.cond(toks, tmask, vehicle, redose), x.reshape(b, t * k, -1)], 1)
        m = _mask(t, k, z.device)
        for blk in self.blocks:
            x = blk(x, m)
        return z + self.out(self.norm(x[:, 2:])).reshape(b, t, k, -1)

    def rollout(self, z0, toks, tmask, vehicle, redose, steps: int):
        z = z0
        for _ in range(steps):
            z = torch.cat([z, self(z, toks, tmask, vehicle, redose)[:, -1:]], 1)
        return z[:, z0.shape[1]:]


def _mask(t: int, k: int, device) -> torch.Tensor:
    step = torch.arange(t, device=device).repeat_interleave(k)
    m = torch.ones(2 + t * k, 2 + t * k, dtype=torch.bool, device=device)
    m[:2, 2:] = False
    m[2:, 2:] = step[:, None] >= step[None, :]
    return ~m
