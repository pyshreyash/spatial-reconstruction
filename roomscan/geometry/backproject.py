"""Step 1: depth pixels -> world points with normals (METHOD.md §1)."""
from __future__ import annotations

import numpy as np

from roomscan.io.stray import RGB_WIDTH, StrayCapture


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def frame_points(
    cap: StrayCapture,
    i: int,
    pixel_stride: int = 2,
    min_conf: int = 2,
    z_range: tuple[float, float] = (0.2, 4.0),
) -> tuple[np.ndarray, np.ndarray]:
    """World points and unit normals (pointing towards the camera) for sample i."""
    d = cap.depth(i)
    ok = cap.confidence(i) >= min_conf
    h, w = d.shape
    fx, fy, cx, cy = cap.intrinsics[i] * (w / RGB_WIDTH)

    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    P = np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d], axis=-1)
    return organised_points(P, ok, quat_to_rot(cap.quats[i]), cap.positions[i], pixel_stride, z_range)


def organised_points(
    P: np.ndarray,
    ok: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pixel_stride: int,
    z_range: tuple[float, float],
    step: int = 1,
    P_normals: np.ndarray | None = None,
    extra: np.ndarray | None = None,
):
    """Organised camera-frame point map (H,W,3) -> world points and normals (and `extra` (H,W) per point).
    Normals from central differences `step` pixels apart, on `P_normals` (e.g. a smoothed map) if given."""
    d = P[..., 2]
    Q = P if P_normals is None else P_normals
    k = step
    du = np.zeros_like(Q)
    dv = np.zeros_like(Q)
    du[:, k:-k] = Q[:, 2 * k:] - Q[:, :-2 * k]
    dv[k:-k] = Q[2 * k:] - Q[:-2 * k]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1)
    n /= np.maximum(norm, 1e-12)[..., None]
    n[(n * P).sum(-1) > 0] *= -1

    nb_ok = ok.copy()
    nb_ok[:, k:-k] &= ok[:, 2 * k:] & ok[:, :-2 * k]
    nb_ok[k:-k] &= ok[2 * k:] & ok[:-2 * k]
    nb_ok[:, :k] = nb_ok[:, -k:] = False
    nb_ok[:k, :] = nb_ok[-k:, :] = False

    jump = np.zeros_like(d)
    jump[:, k:-k] = np.abs(d[:, 2 * k:] - d[:, :-2 * k])
    jump[k:-k] = np.maximum(jump[k:-k], np.abs(d[2 * k:] - d[:-2 * k]))
    smooth = jump < (0.05 * d + 0.01) * k

    valid = nb_ok & smooth & (d > z_range[0]) & (d < z_range[1]) & (norm > 0)
    s = slice(None, None, pixel_stride)
    sel = valid[s, s]
    Pc = P[s, s][sel]
    Nc = n[s, s][sel]
    out = (Pc @ R.T + t).astype(np.float32), (Nc @ R.T).astype(np.float32)
    return out if extra is None else (*out, extra[s, s][sel])


def collect_points(
    cap: StrayCapture, frame_stride: int = 5, pixel_stride: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse sampled frames into one cloud. Returns points, normals, sample index per point."""
    pts, nrm, idx = [], [], []
    for i in range(0, cap.n_frames, frame_stride):
        p, n = frame_points(cap, i, pixel_stride)
        pts.append(p)
        nrm.append(n)
        idx.append(np.full(len(p), i, dtype=np.int32))
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(idx)
