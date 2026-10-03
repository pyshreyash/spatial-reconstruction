"""Step 6: axis-aligned wall planes in the Manhattan frame (METHOD.md §6)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.geometry.stats import histogram_peaks, robust_offset

AXES = {"x": 0, "z": 2}
MIN_VERTICAL_EXTENT = 0.8  # rejects furniture sides
MIN_LENGTH = 0.3
INLIER_BAND = 0.04
SNAP_TOL = 0.08  # same-facing faces closer than this are one wall (skirting, frames, drift)
OBJ_MAX_LEN = 2.5       # fridges, wardrobes, cupboards
OBJ_DEPTH = (0.2, 1.0)  # distance of the face in front of the wall behind it
OBJ_CLEARANCE = 0.1     # wall evidence must exist this far above the object's top
OBJ_MIN_EVIDENCE = 30


@dataclass
class WallPlane:
    axis: str           # "x": plane x = offset (wall runs along z); "z": plane z = offset
    facing: int         # +1 if the normal (room side) points to +axis
    offset: float
    sigma: float
    n_points: int
    span: tuple[float, float]  # extent along the wall
    along: np.ndarray   # along-wall coordinates of high inliers (subsampled), for coverage
    top: float = np.inf      # y of highest evidence (97th pct)
    bottom: float = -np.inf  # y of lowest evidence (3rd pct)
    samples: np.ndarray | None = None  # (k,2) along-wall coordinate, y of inliers (subsampled)


def detect_walls(
    pts: np.ndarray, nrm: np.ndarray, chunks: np.ndarray, y_lo: float, y_hi: float,
    y_high: float, y_head: float = np.inf,
) -> list[WallPlane]:
    """Wall planes between y_lo and y_hi. Coverage (`along`) only from points in (y_high, y_head):
    above furniture so a bed cannot fake a wall, below door heads so a lintel cannot close a door."""
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
                bottom, top = np.percentile(y, [3, 97])
                if top - bottom < MIN_VERTICAL_EXTENT:
                    continue
                along = pts[m, other][inl]
                lo, hi = np.percentile(along, [1, 99])
                if hi - lo < MIN_LENGTH:
                    continue
                off = robust_offset(c[inl], chunks[m][inl])
                samples = np.stack([along, y], axis=1)
                if len(samples) > 20000:
                    samples = samples[rng.choice(len(samples), 20000, replace=False)]
                along = along[(y > y_high) & (y < y_head)]
                if len(along) > 20000:
                    along = rng.choice(along, 20000, replace=False)
                walls.append(WallPlane(axis, facing, off.value, off.sigma, off.n_points, (lo, hi), along,
                                       float(top), float(bottom), samples))
    return walls


def consolidate(walls: list[WallPlane]) -> list[WallPlane]:
    """Fold weaker same-facing faces within SNAP_TOL into the strongest one."""
    kept: list[WallPlane] = []
    for w in sorted(walls, key=lambda w: -w.n_points):
        host = next((k for k in kept if k.axis == w.axis and k.facing == w.facing
                     and abs(k.offset - w.offset) < SNAP_TOL), None)
        if host is None:
            kept.append(WallPlane(**vars(w)))
            continue
        host.along = np.concatenate([host.along, w.along])
        if host.samples is not None and w.samples is not None:
            host.samples = np.concatenate([host.samples, w.samples])
        host.span = (min(host.span[0], w.span[0]), max(host.span[1], w.span[1]))
        host.n_points += w.n_points
        host.top, host.bottom = max(host.top, w.top), min(host.bottom, w.bottom)
    return kept


def split_structural(walls: list[WallPlane]) -> tuple[list[WallPlane], list[WallPlane]]:
    """(walls, objects). A short face is furniture when the wall right behind it is seen
    continuing above its top: it stops below the ceiling and does not bound the room."""
    objects = []
    for f in walls:
        length = f.span[1] - f.span[0]
        if length > OBJ_MAX_LEN:
            continue
        for w in walls:
            if w is f or w.axis != f.axis or w.facing != f.facing or w.samples is None:
                continue
            depth = (f.offset - w.offset) * f.facing
            overlap = min(f.span[1], w.span[1]) - max(f.span[0], w.span[0])
            if not (OBJ_DEPTH[0] <= depth <= OBJ_DEPTH[1]) or overlap < 0.5 * length:
                continue
            s = w.samples
            above = (s[:, 0] > f.span[0]) & (s[:, 0] < f.span[1]) & (s[:, 1] > f.top + OBJ_CLEARANCE)
            if above.sum() >= OBJ_MIN_EVIDENCE:
                objects.append(f)
                break
    return [w for w in walls if all(w is not o for o in objects)], objects
