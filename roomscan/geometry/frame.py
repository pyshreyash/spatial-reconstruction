"""Step 3: gravity + Manhattan frame (METHOD.md §3)."""
from __future__ import annotations

import numpy as np


def manhattan_yaw(normals: np.ndarray) -> float:
    """Dominant wall direction modulo 90 deg: theta = 1/4 atan2(sum w sin4phi, sum w cos4phi)."""
    horiz = np.abs(normals[:, 1]) < 0.2
    n = normals[horiz].astype(np.float64)
    phi = np.arctan2(n[:, 2], n[:, 0])
    w = np.hypot(n[:, 0], n[:, 2])
    return float(0.25 * np.arctan2((w * np.sin(4 * phi)).sum(), (w * np.cos(4 * phi)).sum()))


def yaw_rotation(theta: float) -> np.ndarray:
    """Rotation about +y that maps a horizontal direction at angle phi to phi - theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
