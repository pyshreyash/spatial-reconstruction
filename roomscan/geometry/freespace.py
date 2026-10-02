"""Step 5: 2-D free-space map by ray carving (METHOD.md §5)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RAY_CHUNK = 10000


@dataclass
class FreeSpaceGrid:
    x0: float
    z0: float
    res: float
    free: np.ndarray  # (nx, nz) ray pass-through counts
    occ: np.ndarray   # (nx, nz) endpoint counts

    @property
    def extent(self) -> tuple[float, float, float, float]:
        nx, nz = self.free.shape
        return self.x0, self.x0 + nx * self.res, self.z0, self.z0 + nz * self.res

    def free_mask(self) -> np.ndarray:
        return (self.free >= 2) & (self.free > 3 * self.occ)

    def index(self, x: float, z: float) -> tuple[int, int]:
        nx, nz = self.free.shape
        ix = int(np.clip(np.floor((x - self.x0) / self.res), 0, nx))
        iz = int(np.clip(np.floor((z - self.z0) / self.res), 0, nz))
        return ix, iz


def carve(
    origins: np.ndarray,
    ends: np.ndarray,
    occupied: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float = 0.02,
) -> FreeSpaceGrid:
    """origins/ends: (M,2) ray start (camera) and end (surface point) in (x, z).

    A 3-D ray passing over a 2-D cell at any height proves that column is not a full-height wall,
    so all rays carve; only vertical-surface points (`occupied`, (K,2)) count as occupancy.
    """
    x0, x1, z0, z1 = bounds
    nx, nz = int(np.ceil((x1 - x0) / res)), int(np.ceil((z1 - z0) / res))
    free = np.zeros(nx * nz, np.int64)
    occ = np.zeros(nx * nz, np.int64)

    def flat(xz: np.ndarray) -> np.ndarray:
        ix = np.floor((xz[..., 0] - x0) / res).astype(np.int64)
        iz = np.floor((xz[..., 1] - z0) / res).astype(np.int64)
        inside = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
        return np.where(inside, ix * nz + iz, -1)

    e = flat(occupied)
    occ += np.bincount(e[e >= 0], minlength=nx * nz)

    for s in range(0, len(origins), RAY_CHUNK):
        o, en = origins[s:s + RAY_CHUNK], ends[s:s + RAY_CHUNK]
        d = en - o
        length = np.linalg.norm(d, axis=1)
        steps = np.arange(int(np.ceil(length.max() / res)) + 1) * res
        t = steps[None, :]
        valid = t < (length[:, None] - 1.5 * res)
        samples = o[:, None, :] + (d / np.maximum(length, 1e-9)[:, None])[:, None, :] * t[..., None]
        f = flat(samples)[valid]
        free += np.bincount(f[f >= 0], minlength=nx * nz)

    return FreeSpaceGrid(x0, z0, res, free.reshape(nx, nz), occ.reshape(nx, nz))
