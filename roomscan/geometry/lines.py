"""Video tier: structural lines triangulated from ARKit-posed keyframes (METHOD_VIDEO.md §V8).

A 3-D line along Manhattan axis a, seen as an image segment, lies in the plane through the camera centre and
the segment. That plane contains the direction e_a, so projected along a it becomes a 2-D line through the
camera in the plane of the other two axes (b, c). Lines from many keyframes cross at the 3-D line's (b, c)
position: wall corners and door jambs for vertical lines, wall-floor/ceiling junctions for horizontal ones.
Positions are voted on a grid (no frame-to-frame matching), then refined by weighted least squares."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

AXES = "xyz"
MIN_SEG_PX = 20            # at WORK_WIDTH
MAX_SEGS = 200             # longest segments kept per keyframe
AXIS_TOL = np.sin(np.radians(1.5))  # |n . e_a| for a segment to be along axis a
AMBIGUOUS = 0.1            # segment plane nearly contains two axes: direction undecidable
RANGE_REL = 0.2            # +- relative range along the ray around the helper depth
VOXEL = 0.05               # m, occupancy grid of the depth points for ray casting
VOXEL_MIN_POINTS = 3
RAY_RANGE = (0.3, 6.0)     # m
RAY_STEP = 0.025           # m
GRID = 0.02                # m, voting cell
SAMPLE = 0.01              # m, step along each vote ray
PEAK_WINDOW = 7            # cells
INLIER = 0.04              # m, perpendicular distance of an observation from the refined position
MIN_FRAMES = 4
MIN_SPREAD = np.radians(10.0)  # spread of viewing directions; below it the position is ill-defined
CHUNK_FRAMES = 60
SIGMA_FLOOR = 0.005
MERGE = 0.05               # m, refined positions closer than this are one line
MIN_EXTENT = {"y": 1.0, "x": 0.5, "z": 0.5}  # m along the line: shorter vertical edges are furniture


@dataclass
class Line3D:
    axis: str              # direction of the line
    pos: np.ndarray        # (2,) coordinates along the other two axes, in AXES order
    sigma: np.ndarray      # (2,)
    extent: tuple[float, float]  # along the line's own axis
    n_obs: int
    n_frames: int


def detect_segments(gray: np.ndarray) -> np.ndarray:
    """(n, 4) segments x1, y1, x2, y2, longest first."""
    if "lsd" not in _CV:
        _CV["lsd"] = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    seg = _CV["lsd"].detect(gray)[0]
    if seg is None:
        return np.empty((0, 4), np.float32)
    seg = seg.reshape(-1, 4)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    order = np.argsort(-length)
    keep = order[length[order] >= MIN_SEG_PX][:MAX_SEGS]
    return seg[keep].astype(np.float32)


_CV: dict = {}


def observations(segs: dict[int, tuple], R: np.ndarray, C: np.ndarray, K: dict[int, np.ndarray]):
    """World-frame observation arrays from per-keyframe segments (x1,y1,x2,y2) and helper depth z at the
    midpoint (nan if unknown). Returns frame, centre, endpoint rays (unit), mid ray (unit), distance."""
    rows = []
    for k, (seg, z) in segs.items():
        if not len(seg):
            continue
        Ki = np.linalg.inv(K[k])
        p1 = np.c_[seg[:, :2], np.ones(len(seg))] @ Ki.T
        p2 = np.c_[seg[:, 2:], np.ones(len(seg))] @ Ki.T
        pm = 0.5 * (p1 + p2)
        dist = z * np.linalg.norm(pm, axis=1)  # depth along the optical axis -> distance along the ray
        unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)  # noqa: E731
        w1, w2, wm = unit(p1) @ R[k].T, unit(p2) @ R[k].T, unit(pm) @ R[k].T
        length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        rows.append((np.full(len(seg), k), np.tile(C[k], (len(seg), 1)), w1, w2, wm, dist, length))
    cols = list(zip(*rows))
    return [np.concatenate(c) for c in cols]


def occupancy(pts: np.ndarray, res: float = VOXEL):
    """Voxel grid of the depth-helper points (any frame): (origin, res, bool grid)."""
    origin = pts.min(0) - res
    idx = ((pts - origin) / res).astype(np.int64)
    shape = idx.max(0) + 2
    count = np.zeros(shape, np.int32)
    np.add.at(count, tuple(idx.T), 1)
    return origin, res, count >= VOXEL_MIN_POINTS


def ray_depth(o: np.ndarray, d: np.ndarray, occ) -> np.ndarray:
    """Distance along unit rays to the first occupied voxel (nan if none)."""
    origin, res, grid = occ
    s = np.arange(*RAY_RANGE, RAY_STEP)
    out = np.full(len(o), np.nan)
    for i0 in range(0, len(o), 5000):
        oo, dd = o[i0:i0 + 5000], d[i0:i0 + 5000]
        q = ((oo[:, None, :] + s[None, :, None] * dd[:, None, :] - origin) / res).astype(np.int64)
        inside = np.all((q >= 0) & (q < np.array(grid.shape)), axis=2)
        q = np.where(inside[..., None], q, 0)
        hit = inside & grid[q[..., 0], q[..., 1], q[..., 2]]
        first = hit.argmax(1)
        any_hit = hit.any(1)
        out[i0:i0 + 5000][any_hit] = s[first[any_hit]]
    return out


def triangulate(obs, Rm: np.ndarray, occ=None) -> list[Line3D]:
    """Vote and refine Manhattan lines. `Rm` rotates world into the Manhattan frame; `occ` (occupancy of
    the depth points, Manhattan frame) supplies a range gate where the keyframe had no depth of its own."""
    frame, C, w1, w2, wm, dist, length = obs
    C, w1, w2, wm = C @ Rm.T, w1 @ Rm.T, w2 @ Rm.T, wm @ Rm.T
    dist = dist.copy()
    if occ is not None:
        miss = ~np.isfinite(dist)
        dist[miss] = ray_depth(C[miss], wm[miss], occ)
    n = np.cross(w1, w2)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    out: list[Line3D] = []
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        sel = (np.abs(n[:, a]) < AXIS_TOL) & (np.abs(n[:, b]) > AMBIGUOUS) & (np.abs(n[:, c]) > AMBIGUOUS) \
            & np.isfinite(dist)
        if sel.sum() < MIN_FRAMES:
            continue
        m = n[sel][:, [b, c]]
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        t = np.c_[-m[:, 1], m[:, 0]]
        o = C[sel][:, [b, c]]
        r = wm[sel][:, [b, c]]
        t[(t * r).sum(1) < 0] *= -1  # the line is in front of the camera
        s_mid = dist[sel] * np.maximum(np.linalg.norm(r, axis=1), 1e-6)  # helper-depth point, along t
        lo, hi = s_mid * (1 - RANGE_REL), s_mid * (1 + RANGE_REL)
        wt = np.minimum(1.0, length[sel] / 60.0)
        peaks, origin, shape = _vote(o, t, lo, hi, wt)
        lines = []
        for q0 in peaks:
            ln = _refine(q0, o, m, t, lo, hi, wt, frame[sel], w1[sel], w2[sel], C[sel], a, b, c)
            if ln is not None:
                lines.append(ln)
        lines.sort(key=lambda l: -l.n_obs)
        kept: list[Line3D] = []
        for ln in lines:
            if all(np.linalg.norm(ln.pos - k.pos) > MERGE for k in kept):
                kept.append(ln)
        out += [ln for ln in kept if ln.extent[1] - ln.extent[0] >= MIN_EXTENT[AXES[a]]]
    return out


def _vote(o, t, lo, hi, wt):
    ends = np.concatenate([o + lo[:, None] * t, o + hi[:, None] * t])
    origin = np.percentile(ends, 1, axis=0) - 0.5
    top = np.percentile(ends, 99, axis=0) + 0.5
    shape = np.ceil((top - origin) / GRID).astype(int) + 1
    acc = np.zeros(shape)
    for i in range(len(o)):
        s = np.arange(lo[i], hi[i], SAMPLE)
        q = ((o[i] + s[:, None] * t[i] - origin) / GRID).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < shape[0]) & (q[:, 1] < shape[1])
        q = np.unique(q[ok], axis=0)  # one vote per cell per observation
        acc[q[:, 0], q[:, 1]] += wt[i]
    sm = gaussian_filter(acc, 1.0)
    is_peak = (sm == maximum_filter(sm, PEAK_WINDOW)) & (sm >= 3.0)
    idx = np.argwhere(is_peak)
    idx = idx[np.argsort(-sm[is_peak])][:500]
    return origin + (idx + 0.5) * GRID, origin, shape


def _refine(q0, o, m, t, lo, hi, wt, frame, w1, w2, C, a, b, c) -> Line3D | None:
    q = q0.copy()
    inl = None
    for _ in range(4):
        res = ((q - o) * m).sum(1)
        s = ((q - o) * t).sum(1)
        inl = (np.abs(res) < (2 * INLIER if inl is None else INLIER)) & (s > lo * 0.9) & (s < hi * 1.1)
        if inl.sum() < MIN_FRAMES:
            return None
        A = m[inl] * np.sqrt(wt[inl])[:, None]
        y = (m[inl] * o[inl]).sum(1) * np.sqrt(wt[inl])
        q = np.linalg.lstsq(A, y, rcond=None)[0]
    frames = frame[inl]
    if len(np.unique(frames)) < MIN_FRAMES:
        return None
    ang = np.arctan2(m[inl, 1], m[inl, 0]) % np.pi
    spread = np.ptp(np.unwrap(2 * ang) / 2)
    if spread < MIN_SPREAD:
        return None
    res = ((q - o[inl]) * m[inl]).sum(1)
    W = wt[inl]
    N = (m[inl] * W[:, None]).T @ m[inl]
    var_r = float((W * res**2).sum() / max(W.sum() - 2, 1e-6))
    n_chunks = len(np.unique(frames // CHUNK_FRAMES))
    cov = var_r * np.linalg.inv(N) * (inl.sum() / n_chunks)  # observations within a chunk share pose error
    sigma = np.sqrt(np.maximum(np.diag(cov), 0)) + SIGMA_FLOOR
    along = []
    for w in (w1[inl], w2[inl]):  # where the endpoint rays pass the line, along axis a
        wbc = w[:, [b, c]]
        lam = ((q - C[inl][:, [b, c]]) * wbc).sum(1) / np.maximum((wbc**2).sum(1), 1e-9)
        along.append(C[inl][:, a] + lam * w[:, a])
    along = np.concatenate(along)
    return Line3D(AXES[a], q, sigma, (float(np.percentile(along, 5)), float(np.percentile(along, 95))),
                  int(inl.sum()), int(len(np.unique(frames))))


# Wall refinement: depth proposes wall planes, lines measure them.
FRONT = 0.20               # m, search in front of a depth wall (room side): depth error, not furniture
BEHIND = 0.10              # m, search behind it: less than a partition's thickness
JUNCTION = 0.15            # m, a horizontal line this close to floor or ceiling height is a wall junction
CLUSTER = 0.04             # m, line positions closer than this measure the same surface
LINE_WALL_SIGMA_FLOOR = 0.01   # ARKit pose error and drift


def refine_walls(walls, lines: list[Line3D], floor_y: float, ceil_y: float | None) -> tuple[list, dict]:
    """Move each depth wall onto the deepest cluster (>= 2 lines) of structural line evidence near it:
    vertical edges (corners, jambs) and horizontal wall-floor / wall-ceiling junctions. Furniture edges
    stand in front of walls, so the deepest cluster wins. Returns walls and counts."""
    levels = [floor_y] + ([ceil_y] if ceil_y is not None else [])
    n_line = 0
    shifts = []
    for w in walls:
        ai = 0 if w.axis == "x" else 2          # plane coordinate
        ri = 2 if w.axis == "x" else 0          # coordinate along the wall
        vals, sig = [], []
        for ln in lines:
            if ln.axis == "y":
                coord = {0: ln.pos[0], 2: ln.pos[1]}
                if w.span[0] - 0.3 <= coord[ri] <= w.span[1] + 0.3:
                    vals.append(coord[ai])
                    sig.append(ln.sigma[0 if ai == 0 else 1])
            elif AXES.index(ln.axis) == ri:     # horizontal line running along the wall
                other = [i for i in range(3) if i != ri]
                j = other.index(ai)
                y = ln.pos[other.index(1)]
                lo, hi = ln.extent
                if min(abs(y - lv) for lv in levels) < JUNCTION and min(hi, w.span[1]) - max(lo, w.span[0]) > 0.3:
                    vals.append(ln.pos[j])
                    sig.append(ln.sigma[j])
        vals, sig = np.array(vals), np.array(sig)
        depth_in_front = (vals - w.offset) * w.facing if len(vals) else np.zeros(0)
        near = (depth_in_front > -BEHIND) & (depth_in_front < FRONT)
        chosen = None
        if near.sum() >= 2:
            v, s = vals[near], sig[near]
            order = np.argsort(v * w.facing)    # deepest first
            v, s = v[order], s[order]
            gaps = np.abs(np.diff(v)) > CLUSTER
            groups = [g for g in np.split(np.arange(len(v)), np.nonzero(gaps)[0] + 1) if len(g) >= 2]
            if groups:
                chosen = (v[groups[0]], s[groups[0]])
        if chosen is None:
            continue
        v, s = chosen
        inv = 1 / s**2
        off = float((v * inv).sum() / inv.sum())
        sigma = max(1 / np.sqrt(inv.sum()), float(np.std(v) / np.sqrt(len(v))), LINE_WALL_SIGMA_FLOOR)
        shifts.append(off - w.offset)
        w.offset, w.sigma = off, float(sigma)
        w.measured_by = "lines"
        n_line += 1
    info = {"walls": len(walls), "walls_measured_by_lines": n_line,
            "line_shift_median_m": round(float(np.median(np.abs(shifts))), 4) if shifts else None}
    return walls, info
