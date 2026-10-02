"""Raw fused scan export: voxel-averaged point cloud as binary PLY (for reference/inspection)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Mean of the points falling in each voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    out = np.stack([np.bincount(inv, points[:, k]) for k in range(3)], axis=1)
    return out / counts[:, None]


def height_colours(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((y - lo) / max(hi - lo, 1e-6), 0, 1)
    return (colormaps["turbo"](t)[:, :3] * 255).astype(np.uint8)


def write_ply(path: Path, points: np.ndarray, colours: np.ndarray) -> None:
    data = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = points.T
    data["red"], data["green"], data["blue"] = colours.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def export_raw_scan(path: Path, points: np.ndarray, top: float, voxel: float = 0.02) -> int:
    """Points in the plan frame (floor at y = 0), coloured by height. Returns the point count."""
    pts = voxel_downsample(points.astype(np.float64), voxel)
    write_ply(path, pts.astype(np.float32), height_colours(pts[:, 1], 0.0, top))
    return len(pts)
