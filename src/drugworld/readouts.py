"""Image readouts for the probes and decoders (plan section 3, item 3): simple thresholding.

idr0119 (per frame): nuclei from the NLS-mCherry channel (background-subtracted, Otsu mask,
seeds = local maxima, watershed); cell count = seeds; cell-cycle phase from clover-HDHB as in
Gross et al.: mean reporter intensity in the nucleus over the mean in a 5-pixel ring around it
(ring pixels of other nuclei excluded); a cell is G1 when the ratio is above a threshold. The
paper set its threshold by hand; here it is fixed once from vehicle wells of the pilot (the
valley of a two-component Gaussian mixture of log ratios) and recorded.

idr0064 (per frame): nuclei from H2B-RFP, Erk activity = cytoplasmic (5-pixel ring) over
nuclear Erk-KTR intensity (C/N ratio), the well-averaged readout of Goglia et al.

Readouts are returned per tile of a fixed tile grid (same grid as the embedding).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed

RING = 5


def nuclei(img: np.ndarray, sigma: float = 1.0, bg_sigma: float = 25.0,
           min_distance: int = 4) -> np.ndarray:
    """Label image of nuclei from a nuclear-marker frame."""
    x = img.astype(np.float32)
    x = gaussian(x, sigma, preserve_range=True) - gaussian(x, bg_sigma, preserve_range=True)
    mask = x > threshold_otsu(x)
    mask = ndi.binary_opening(mask, iterations=1)
    pk = peak_local_max(x, min_distance=min_distance, labels=ndi.label(mask)[0],
                        exclude_border=False)
    markers = np.zeros(x.shape, np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    return watershed(-x, markers, mask=mask)


def ring_ratio(labels: np.ndarray, reporter: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray]:
    """Per nucleus: (centroid y, centroid x, nuclear mean / ring mean) of `reporter`."""
    n = int(labels.max())
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    idx = np.arange(1, n + 1)
    rep = reporter.astype(np.float32)
    nuc = ndi.mean(rep, labels, idx)
    grown = ndi.grey_dilation(labels, size=2 * RING + 1)
    ring = np.where(labels == 0, grown, 0)
    cyt = ndi.mean(rep, ring, idx)
    cy, cx = np.array(ndi.center_of_mass(np.ones_like(labels), labels, idx)).T
    with np.errstate(divide="ignore", invalid="ignore"):
        r = nuc / cyt
    return cy, cx, r


def per_tile(cy: np.ndarray, cx: np.ndarray, values: dict[str, np.ndarray],
             tiles: list[tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
    """Counts and means of per-cell values within each tile (y0, x0, h, w)."""
    out = {"count": np.zeros(len(tiles))}
    out.update({k: np.full(len(tiles), np.nan) for k in values})
    for i, (y0, x0, h, w) in enumerate(tiles):
        m = (cy >= y0) & (cy < y0 + h) & (cx >= x0) & (cx < x0 + w)
        out["count"][i] = m.sum()
        for k, v in values.items():
            vv = v[m]
            vv = vv[np.isfinite(vv)]
            if len(vv):
                out[k][i] = vv.mean()
    return out


def idr0119_frame(red: np.ndarray, green: np.ndarray, tiles, g1_threshold: float | None):
    """Per tile: count, mean log N/C ratio, G1 fraction (if a threshold is given)."""
    lab = nuclei(red)
    cy, cx, r = ring_ratio(lab, green)
    lr = np.log(r)
    vals = {"log_nc": lr}
    if g1_threshold is not None:
        vals["g1"] = (lr > g1_threshold).astype(np.float32)
    return per_tile(cy, cx, vals, tiles), lr[np.isfinite(lr)]


def idr0064_frame(h2b: np.ndarray, ktr: np.ndarray, tiles):
    """Per tile: count and mean Erk C/N ratio (ring over nucleus)."""
    lab = nuclei(h2b, min_distance=3)
    cy, cx, r = ring_ratio(lab, ktr)          # nucleus / ring
    with np.errstate(divide="ignore"):
        cn = 1.0 / r
    return per_tile(cy, cx, {"erk_cn": cn}, tiles)


def gmm_valley(x: np.ndarray, seed: int = 0) -> float:
    """Threshold between the two components of a 1-D Gaussian mixture (density minimum)."""
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, random_state=seed).fit(x.reshape(-1, 1))
    lo, hi = np.sort(g.means_.ravel())
    grid = np.linspace(lo, hi, 2001)
    dens = np.exp(g.score_samples(grid.reshape(-1, 1)))
    return float(grid[np.argmin(dens)])
