"""Step 4: floor and ceiling planes (METHOD.md §4). Points must be in the gravity-aligned frame."""
from __future__ import annotations

import numpy as np

from roomscan.geometry.stats import Offset, histogram_peaks, robust_offset

MIN_CEILING_POINTS = 500
MIN_CEILING_ABOVE_FLOOR = 1.8


def floor_plane(pts: np.ndarray, nrm: np.ndarray, chunks: np.ndarray) -> Offset:
    up = nrm[:, 1] > 0.95
    y = pts[up, 1]
    peaks = histogram_peaks(y, res=0.01, min_frac=0.2, min_count=50, min_sep=0.1)
    if len(peaks) == 0:
        raise RuntimeError("No floor plane found")
    inl = np.abs(y - peaks.min()) < 0.03
    return robust_offset(y[inl], chunks[up][inl])


def ceiling_plane(pts: np.ndarray, nrm: np.ndarray, chunks: np.ndarray, floor_y: float) -> Offset | None:
    """Highest well-supported downward-facing plane; None if the ceiling was not scanned."""
    down = (nrm[:, 1] < -0.95) & (pts[:, 1] > floor_y + MIN_CEILING_ABOVE_FLOOR)
    if down.sum() < MIN_CEILING_POINTS:
        return None
    y = pts[down, 1]
    peaks = histogram_peaks(y, res=0.01, min_frac=0.2, min_count=MIN_CEILING_POINTS / 10, min_sep=0.1)
    if len(peaks) == 0:
        return None
    inl = np.abs(y - peaks.max()) < 0.03
    if inl.sum() < MIN_CEILING_POINTS:
        return None
    return robust_offset(y[inl], chunks[down][inl])
