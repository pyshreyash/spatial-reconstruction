"""Shared 1-D estimators: histogram peaks and robust offsets with block-bootstrap sigma."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Floor on any plane-offset sigma: LiDAR range bias and residual pose error not captured by chunk spread.
SIGMA_SYS = 0.003


@dataclass
class Offset:
    value: float
    sigma: float
    n_points: int
    n_chunks: int


def histogram_peaks(
    values: np.ndarray, res: float, min_frac: float, min_count: float, min_sep: float
) -> np.ndarray:
    if len(values) == 0:
        return np.empty(0)
    edges = np.arange(values.min() - 2 * res, values.max() + 3 * res, res)
    h, edges = np.histogram(values, edges)
    hs = gaussian_filter1d(h.astype(float), 1.0)
    height = max(min_count, min_frac * hs.max())
    idx, _ = find_peaks(hs, height=height, distance=max(1, int(round(min_sep / res))))
    return (edges[idx] + edges[idx + 1]) / 2


def robust_offset(values: np.ndarray, chunks: np.ndarray, sigma_sys: float = SIGMA_SYS) -> Offset:
    """Trimmed mean; sigma from the spread of per-chunk means (points within a chunk share pose error)."""
    m = float(np.median(values))
    keep = np.ones(len(values), bool)
    for _ in range(3):
        r = values - m
        mad = 1.4826 * np.median(np.abs(r)) + 1e-4
        keep = np.abs(r) < 2.5 * mad
        m = float(values[keep].mean())
    v, ch = values[keep], chunks[keep]
    _, inv, cnt = np.unique(ch, return_inverse=True, return_counts=True)
    means = np.bincount(inv, v) / cnt
    k = len(means)
    se = float(means.std(ddof=1) / np.sqrt(k)) if k > 1 else 0.05
    return Offset(m, float(np.hypot(se, sigma_sys)), int(keep.sum()), k)
