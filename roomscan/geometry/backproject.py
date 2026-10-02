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

    # Normals from central differences on the organised depth grid.
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1)
    n /= np.maximum(norm, 1e-12)[..., None]
    n[(n * P).sum(-1) > 0] *= -1

    nb_ok = ok.copy()
    nb_ok[:, 1:-1] &= ok[:, 2:] & ok[:, :-2]
    nb_ok[1:-1] &= ok[2:] & ok[:-2]
    nb_ok[:, [0, -1]] = False
    nb_ok[[0, -1], :] = False

    jump = np.zeros_like(d)
    jump[:, 1:-1] = np.abs(d[:, 2:] - d[:, :-2])
    jump[1:-1] = np.maximum(jump[1:-1], np.abs(d[2:] - d[:-2]))
    smooth = jump < 0.05 * d + 0.01

    valid = nb_ok & smooth & (d > z_range[0]) & (d < z_range[1]) & (norm > 0)
    s = slice(None, None, pixel_stride)
    sel = valid[s, s]
    Pc = P[s, s][sel]
    Nc = n[s, s][sel]

    R = quat_to_rot(cap.quats[i])
    return (Pc @ R.T + cap.positions[i]).astype(np.float32), (Nc @ R.T).astype(np.float32)


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
