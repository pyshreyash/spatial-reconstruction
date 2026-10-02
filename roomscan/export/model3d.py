"""Lightweight 3-D plan: extruded walls with door cut-outs plus per-room floor slabs."""
from __future__ import annotations

import numpy as np

from roomscan.export.gltf import Primitive
from roomscan.geometry.polygon import CellComplex
from roomscan.geometry.rooms import Opening, Room

DOOR_HEIGHT = 2.03  # assumed until opening heights are measured
WALL_RGBA = (0.88, 0.87, 0.84, 1.0)
PALETTE = [
    (0.55, 0.71, 0.89), (0.95, 0.70, 0.45), (0.60, 0.83, 0.60), (0.85, 0.60, 0.80),
    (0.95, 0.88, 0.50), (0.60, 0.80, 0.85), (0.80, 0.70, 0.60), (0.75, 0.75, 0.95),
]


class _Mesh:
    def __init__(self) -> None:
        self.pos: list[np.ndarray] = []
        self.tri: list[np.ndarray] = []
        self.n = 0

    def quad(self, corners: np.ndarray) -> None:
        self.pos.append(corners)
        self.tri.append(np.array([[0, 1, 2], [0, 2, 3]]) + self.n)
        self.n += 4

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.pos:
            return np.zeros((0, 3)), np.zeros((0, 3), int)
        return np.concatenate(self.pos), np.concatenate(self.tri)


def _point(axis: str, offset: float, s: float) -> tuple[float, float]:
    return (offset, s) if axis == "x" else (s, offset)


def _wall_quad(mesh: _Mesh, axis: str, offset: float, s0: float, s1: float, y0: float, y1: float) -> None:
    (ax, az), (bx, bz) = _point(axis, offset, s0), _point(axis, offset, s1)
    mesh.quad(np.array([[ax, y0, az], [bx, y0, bz], [bx, y1, bz], [ax, y1, az]]))


def build_model(
    cx: CellComplex, lab: np.ndarray, rooms: list[Room], openings: list[Opening], height: float
) -> list[Primitive]:
    xs, zs = cx.xs, cx.zs
    prims = []
    for k, room in enumerate(rooms, start=1):
        floor = _Mesh()
        for i, j in zip(*np.nonzero(lab == k)):
            floor.quad(np.array([[xs[i], 0, zs[j]], [xs[i + 1], 0, zs[j]],
                                 [xs[i + 1], 0, zs[j + 1]], [xs[i], 0, zs[j + 1]]]))
        p, t = floor.arrays()
        prims.append(Primitive(f"{room.id} floor", p, t, (*PALETTE[(k - 1) % len(PALETTE)], 1.0)))

    walls = _Mesh()
    for room in rooms:
        for w in room.walls:
            axis, idx = w.line
            offset = float((cx.xl if axis == "x" else cx.zl)[idx].offset)
            lo, hi = w.along
            cuts = sorted(
                (max(o.start, lo), min(o.end, hi), o.kind) for o in openings
                if o.axis == axis and abs(o.offset - offset) < 1e-6 and o.start < hi and o.end > lo
            )
            s = lo
            for c0, c1, kind in cuts:
                if c0 > s:
                    _wall_quad(walls, axis, offset, s, c0, 0.0, height)
                if kind == "door" and height > DOOR_HEIGHT:
                    _wall_quad(walls, axis, offset, c0, c1, DOOR_HEIGHT, height)
                s = max(s, c1)
            if hi > s:
                _wall_quad(walls, axis, offset, s, hi, 0.0, height)
    p, t = walls.arrays()
    prims.append(Primitive("walls", p, t, WALL_RGBA))
    return prims
