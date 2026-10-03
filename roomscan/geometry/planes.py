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


def room_ceilings(
    pts: np.ndarray, nrm: np.ndarray, chunks: np.ndarray, floor_y: float,
    xs: np.ndarray, zs: np.ndarray, lab: np.ndarray,
) -> dict[int, Offset]:
    """Ceiling plane per room label, from downward-facing points above that room's cells."""
    down = (nrm[:, 1] < -0.95) & (pts[:, 1] > floor_y + MIN_CEILING_ABOVE_FLOOR)
    p, ch = pts[down], chunks[down]
    i = np.searchsorted(xs, p[:, 0]) - 1
    j = np.searchsorted(zs, p[:, 2]) - 1
    ok = (i >= 0) & (i < lab.shape[0]) & (j >= 0) & (j < lab.shape[1])
    room = np.zeros(len(p), int)
    room[ok] = lab[i[ok], j[ok]]
    out = {}
    for k in range(1, lab.max() + 1):
        sel = room == k
        if sel.sum() < MIN_CEILING_POINTS / 2:
            continue
        y = p[sel, 1]
        peaks = histogram_peaks(y, res=0.01, min_frac=0.2, min_count=MIN_CEILING_POINTS / 20, min_sep=0.1)
        if len(peaks) == 0:
            continue
        inl = np.abs(y - peaks.max()) < 0.03
        if inl.sum() >= MIN_CEILING_POINTS / 4:
            out[k] = robust_offset(y[inl], ch[sel][inl])
    return out
