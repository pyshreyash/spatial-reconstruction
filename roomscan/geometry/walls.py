"""Step 6: axis-aligned wall planes in the Manhattan frame (METHOD.md §6)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.geometry.stats import histogram_peaks, robust_offset

AXES = {"x": 0, "z": 2}
MIN_VERTICAL_EXTENT = 0.8  # rejects furniture sides
MIN_LENGTH = 0.3
INLIER_BAND = 0.04


@dataclass
class WallPlane:
    axis: str           # "x": plane x = offset (wall runs along z); "z": plane z = offset
    facing: int         # +1 if the normal (room side) points to +axis
    offset: float
    sigma: float
    n_points: int
    span: tuple[float, float]  # extent along the wall
    along: np.ndarray   # along-wall coordinates of high inliers (subsampled), for coverage


def detect_walls(
    pts: np.ndarray, nrm: np.ndarray, chunks: np.ndarray, y_lo: float, y_hi: float, y_high: float
) -> list[WallPlane]:
    """Wall planes between y_lo and y_hi; coverage (`along`) only from points above y_high,
    so beds and desks that line up with a wall plane cannot fake a wall."""
    band = (pts[:, 1] > y_lo) & (pts[:, 1] < y_hi)
    rng = np.random.default_rng(0)
    walls: list[WallPlane] = []
    for axis, ai in AXES.items():
        other = 2 if ai == 0 else 0
        for facing in (1, -1):
            m = band & (nrm[:, ai] * facing > 0.9)
            c = pts[m, ai]
            for peak in histogram_peaks(c, res=0.01, min_frac=0.02, min_count=100, min_sep=0.08):
                inl = np.abs(c - peak) < INLIER_BAND
                y = pts[m, 1][inl]
                if np.percentile(y, 97) - np.percentile(y, 3) < MIN_VERTICAL_EXTENT:
                    continue
                along = pts[m, other][inl]
                lo, hi = np.percentile(along, [1, 99])
                if hi - lo < MIN_LENGTH:
                    continue
                off = robust_offset(c[inl], chunks[m][inl])
                along = along[y > y_high]
                if len(along) > 20000:
                    along = rng.choice(along, 20000, replace=False)
                walls.append(WallPlane(axis, facing, off.value, off.sigma, off.n_points, (lo, hi), along))
    return walls
