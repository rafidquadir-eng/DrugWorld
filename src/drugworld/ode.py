"""Gross et al. (Nat Commun 2023) phase-specific rate model, reimplemented from the paper's
Methods (plan section 7, H2a / H3b tier-2 baseline; fitted to training conditions only).

Linear chain: G1 in 4 parts x 2 subphases, S/G2 in 4 parts x 5 subphases (28 states). Part k of
G1 has progression rate alpha_k and death rate gamma1_k; part i of S/G2 has beta_i and gamma2_i;
cells leaving the last S/G2 subphase enter G1 part 1 as two cells. Every rate follows a Hill
curve in dose, r(C) = r0 + (rmax - r0) C^k / (C^k + EC50^k); r0 (no drug) is shared by all drugs,
EC50 and k are per drug (shared by its 16 rates; the paper does not state the sharing, recorded
as our choice), rmax per drug and rate. Combinations (paper Eqs. 13-15): progression effects
scaled to inhibition (r0 - r) / r0 combine by Bliss, E_ab = E_a + E_b - E_a E_b; death-rate
increases over r0 add. Initial state: one cell, fraction f_G1 (shared parameter) in G1, spread
over subphases in proportion to dwell time (1 / rate). Solved by the matrix exponential over
0.5 h steps. Fitted to the mean (fields, replicate wells) measured G1 and S/G2 cell numbers per
frame, normalised to frame 0 and Savitzky-Golay smoothed (window 9, order 2), every second frame
(hourly), by summed squared error; optimiser: scipy differential evolution (seed 0, population 5 x parameters)
with L-BFGS-B polish (the paper used BlackBoxOptim's adaptive DE).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter

NG1, SG1, NG2, SG2 = 4, 2, 4, 5
N_STATE = NG1 * SG1 + NG2 * SG2
DT = 0.5


def generator(alpha, beta, g1, g2):
    """Rates (4 each) -> 28 x 28 matrix A with dx/dt = A x."""
    A = np.zeros((N_STATE, N_STATE))
    idx = []
    for k in range(NG1):
        idx += [("G1", k)] * SG1
    for i in range(NG2):
        idx += [("G2", i)] * SG2
    for s, (ph, part) in enumerate(idx):
        prog = alpha[part] if ph == "G1" else beta[part]
        death = g1[part] if ph == "G1" else g2[part]
        A[s, s] -= prog + death
        if s + 1 < N_STATE:
            A[s + 1, s] += prog
        else:
            A[0, s] += 2 * prog
    return A


def x0(f_g1, alpha, beta):
    w1 = np.repeat(1 / np.asarray(alpha), SG1)
    w2 = np.repeat(1 / np.asarray(beta), SG2)
    return np.concatenate([f_g1 * w1 / w1.sum(), (1 - f_g1) * w2 / w2.sum()])


def simulate(rates, f_g1, n_frames, every=1):
    """rates: (16,) alpha1-4, beta1-4, gamma1_1-4, gamma2_1-4 -> G1 and S/G2 cell numbers at
    frames 0, every, 2 * every, ... (< n_frames), shape (n, 2)."""
    a, b, g1, g2 = rates[:4], rates[4:8], rates[8:12], rates[12:]
    step = expm(generator(a, b, g1, g2) * DT * every)
    n_frames = len(range(0, n_frames, every))
    x = x0(f_g1, a, b)
    out = np.zeros((n_frames, 2))
    for t in range(n_frames):
        out[t] = x[:NG1 * SG1].sum(), x[NG1 * SG1:].sum()
        x = step @ x
    return out


class Params:
    """Flat vector: log r0 (16), logit f_G1, then per drug: log EC50, log k, 16 values u in
    [0, 1] mapped to rmax (progression: r0 * u, i.e. slowed; death: r0 + u * DEATH_MAX)."""
    DEATH_MAX = 0.2

    def __init__(self, drugs, dose_range):
        self.drugs = list(drugs)
        self.dose_range = dose_range          # drug -> (min, max) measured dose, nM

    def bounds(self):
        b = [(np.log(1e-3), np.log(5.0))] * 8 + [(np.log(1e-5), np.log(0.05))] * 8
        b += [(-3.0, 3.0)]
        for d in self.drugs:
            lo, hi = self.dose_range[d]
            b += [(np.log(lo / 10), np.log(hi * 10)), (np.log(0.2), np.log(5.0))]
            b += [(0.0, 1.0)] * 16
        return b

    def unpack(self, v):
        r0 = np.exp(v[:16])
        f = 1 / (1 + np.exp(-v[16]))
        drug = {}
        for i, d in enumerate(self.drugs):
            o = 17 + 18 * i
            u = v[o + 2:o + 18]
            rmax = np.concatenate([r0[:8] * u[:8], r0[8:] + u[8:] * self.DEATH_MAX])
            drug[d] = (np.exp(v[o]), np.exp(v[o + 1]), rmax)
        return r0, f, drug


def rates_single(r0, drug, d, c):
    if not d:
        return r0
    ec50, k, rmax = drug[d]
    h = c ** k / (c ** k + ec50 ** k)
    return r0 + (rmax - r0) * h


def rates_combo(r0, drug, comps):
    """comps: [(drug, dose), ...] -> combined rates (Bliss on progression, additive death)."""
    single = [rates_single(r0, drug, d, c) for d, c in comps]
    e = 1 - np.prod([1 - (r0[:8] - r[:8]) / r0[:8] for r in single], axis=0)
    prog = r0[:8] * (1 - e)
    death = r0[8:] + sum(r[8:] - r0[8:] for r in single)
    return np.concatenate([prog, death])


def observed(cells_per_field: list[np.ndarray], thr: float, n_frames: int) -> np.ndarray:
    """Mean over fields of (G1 count, S/G2 count) per frame, from per-cell readouts."""
    out = []
    for c in cells_per_field:
        t = c[:, 0].astype(int)
        v = c[:, 3]
        ok = np.isfinite(v)
        g1 = np.bincount(t[ok & (v > thr)], minlength=n_frames)[:n_frames]
        g2 = np.bincount(t[ok & (v <= thr)], minlength=n_frames)[:n_frames]
        out.append(np.column_stack([g1, g2]).astype(float))
    return np.mean(out, 0)


def prepare(obs: np.ndarray) -> np.ndarray:
    """Normalise to frame 0 (total) and smooth (Savitzky-Golay, window 9, order 2)."""
    tot0 = obs[0].sum()
    return savgol_filter(obs / tot0, 9, 2, axis=0)


class Loss:
    """Summed squared error over conditions (module level, so worker processes can pickle it)."""

    def __init__(self, P, data, keys, sub, n_frames):
        self.P, self.data, self.keys, self.sub, self.n = P, data, keys, sub, n_frames

    def __call__(self, v):
        r0, f, drug = self.P.unpack(v)
        s = 0.0
        for d, c in self.keys:
            r = rates_single(r0, drug, d, c)
            if np.any(r <= 0):
                return 1e6
            s += float(((simulate(r, f, self.n, 2) - self.data[(d, c)]) ** 2).sum())
        return s


def fit(conditions: dict, drugs, dose_range, n_frames, workers=64, maxiter=2000, seed=0,
        popsize=5):
    """conditions: {(drug, dose) or ('', 0): prepared (n_frames, 2) array}."""
    P = Params(drugs, dose_range)
    keys = sorted(conditions)
    sub = np.arange(0, n_frames, 2)
    loss = Loss(P, {k: conditions[k][sub] for k in keys}, keys, sub, n_frames)
    res = differential_evolution(loss, P.bounds(), seed=seed, maxiter=maxiter, tol=1e-8,
                                 workers=workers, updating="deferred", polish=True,
                                 popsize=popsize)
    return P, res


def predict(P: Params, v, comps, n_frames):
    """comps: [] (vehicle), [(d, c)] or [(d1, c1), (d2, c2)] -> (n_frames, 2) relative."""
    r0, f, drug = P.unpack(v)
    r = r0 if not comps else rates_single(r0, drug, *comps[0]) if len(comps) == 1 else \
        rates_combo(r0, drug, comps)
    return simulate(r, f, n_frames)
