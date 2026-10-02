"""Step 7: cell complex from wall lines, inside/outside labelled by min-cut (METHOD.md §7)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from roomscan.geometry.freespace import FreeSpaceGrid
from roomscan.geometry.walls import WallPlane

MERGE_TOL = 0.02
CAPACITY_SCALE = 1e4  # max-flow needs integer capacities
COVER_RES = 0.02


@dataclass
class Line:
    axis: str
    offset: float
    sigma: float
    along: np.ndarray

    def covered(self, a: float, b: float, res: float = COVER_RES) -> np.ndarray:
        """Per-bin wall evidence along [a, b]."""
        a, b = min(a, b), max(a, b)
        n = max(1, int(round((b - a) / res)))
        h, _ = np.histogram(self.along, bins=np.linspace(a, b, n + 1))
        return h >= 2

    def coverage(self, a: float, b: float) -> float:
        return float(self.covered(a, b).mean())


@dataclass
class CellComplex:
    xl: list[Line]
    zl: list[Line]
    inside: np.ndarray  # (ncx, ncz) bool

    @property
    def xs(self) -> np.ndarray:
        return np.array([l.offset for l in self.xl])

    @property
    def zs(self) -> np.ndarray:
        return np.array([l.offset for l in self.zl])

    @property
    def area(self) -> np.ndarray:
        return np.diff(self.xs)[:, None] * np.diff(self.zs)[None, :]


def build_lines(walls: list[WallPlane], axis: str) -> list[Line]:
    ws = sorted((w for w in walls if w.axis == axis), key=lambda w: w.offset)
    groups: list[list[WallPlane]] = []
    for w in ws:
        if groups and w.offset - groups[-1][-1].offset < MERGE_TOL:
            groups[-1].append(w)
        else:
            groups.append([w])
    lines = []
    for g in groups:
        inv = np.array([1 / w.sigma**2 for w in g])
        off = float(np.dot(inv, [w.offset for w in g]) / inv.sum())
        lines.append(Line(axis, off, float(1 / np.sqrt(inv.sum())), np.concatenate([w.along for w in g])))
    return lines


def _mincut(unary_in: np.ndarray, unary_out: np.ndarray, pairs: list[tuple[int, int, float]]) -> np.ndarray:
    """Exact binary labelling minimising sum unary + sum pairwise[label differs]. True = inside."""
    n = len(unary_in)
    src, snk = n, n + 1
    C = np.zeros((n + 2, n + 2), np.int64)
    C[src, :n] = np.round(unary_out * CAPACITY_SCALE)
    C[:n, snk] = np.round(unary_in * CAPACITY_SCALE)
    for i, j, w in pairs:
        c = int(round(w * CAPACITY_SCALE))
        C[i, j] += c
        C[j, i] += c
    flow = maximum_flow(csr_matrix(C.astype(np.int32)), src, snk).flow.toarray()
    net = flow if np.array_equal(flow, -flow.T) else flow - flow.T
    residual = csr_matrix(np.maximum(C - net, 0))
    reach = np.zeros(n + 2, bool)
    reach[breadth_first_order(residual, src, directed=True, return_predecessors=False)] = True
    return reach[:n]


def _trace_loops(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Boundary loops of a cell mask as vertex-index sequences, counter-clockwise in (x, z)."""
    nx, nz = mask.shape

    def inside(i: int, j: int) -> bool:
        return 0 <= i < nx and 0 <= j < nz and bool(mask[i, j])

    nxt: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i, j in zip(*np.nonzero(mask)):
        if not inside(i, j - 1):
            nxt[(i, j)].append((i + 1, j))
        if not inside(i + 1, j):
            nxt[(i + 1, j)].append((i + 1, j + 1))
        if not inside(i, j + 1):
            nxt[(i + 1, j + 1)].append((i, j + 1))
        if not inside(i - 1, j):
            nxt[(i, j + 1)].append((i, j))

    loops = []
    while any(nxt.values()):
        start = next(k for k, v in nxt.items() if v)
        loop, v = [start], nxt[start].pop()
        while v != start:
            loop.append(v)
            v = nxt[v].pop()
        loops.append(loop)

    simplified = []
    for loop in loops:
        keep = [
            v for k, v in enumerate(loop)
            if not (loop[k - 1][0] == v[0] == loop[(k + 1) % len(loop)][0]
                    or loop[k - 1][1] == v[1] == loop[(k + 1) % len(loop)][1])
        ]
        simplified.append(keep)
    return simplified


def shoelace(pts: list[tuple[float, float]]) -> float:
    p = np.asarray(pts)
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def outer_loop(mask: np.ndarray, xs: np.ndarray, zs: np.ndarray) -> list[tuple[int, int]]:
    loops = _trace_loops(mask)
    return max(loops, key=lambda lp: abs(shoelace([(xs[a], zs[b]) for a, b in lp])))


def build_complex(walls: list[WallPlane], grid: FreeSpaceGrid, lam: float = 0.5) -> CellComplex | None:
    xl, zl = build_lines(walls, "x"), build_lines(walls, "z")
    if len(xl) < 2 or len(zl) < 2:
        return None
    xs = np.array([l.offset for l in xl])
    zs = np.array([l.offset for l in zl])
    ncx, ncz = len(xs) - 1, len(zs) - 1
    dx, dz = np.diff(xs), np.diff(zs)

    free = grid.free_mask()
    frac = np.zeros((ncx, ncz))
    for i in range(ncx):
        for j in range(ncz):
            ix0, iz0 = grid.index(xs[i], zs[j])
            ix1, iz1 = grid.index(xs[i + 1], zs[j + 1])
            block = free[ix0:ix1, iz0:iz1]
            frac[i, j] = block.mean() if block.size else 0.0

    area = dx[:, None] * dz[None, :]
    u_in = area * (1 - frac)
    u_out = area * frac

    def edge_cost(line: Line, a: float, b: float) -> float:
        return lam * (b - a) * (1 - line.coverage(a, b))

    # Boundary of the arrangement: an inside cell there must be closed by that outer line.
    for j in range(ncz):
        u_in[0, j] += edge_cost(xl[0], zs[j], zs[j + 1])
        u_in[-1, j] += edge_cost(xl[-1], zs[j], zs[j + 1])
    for i in range(ncx):
        u_in[i, 0] += edge_cost(zl[0], xs[i], xs[i + 1])
        u_in[i, -1] += edge_cost(zl[-1], xs[i], xs[i + 1])

    cid = np.arange(ncx * ncz).reshape(ncx, ncz)
    pairs = []
    for i in range(ncx):
        for j in range(ncz):
            if i + 1 < ncx:
                pairs.append((cid[i, j], cid[i + 1, j], edge_cost(xl[i + 1], zs[j], zs[j + 1])))
            if j + 1 < ncz:
                pairs.append((cid[i, j], cid[i, j + 1], edge_cost(zl[j + 1], xs[i], xs[i + 1])))

    inside = _mincut(u_in.ravel(), u_out.ravel(), pairs).reshape(ncx, ncz)
    return CellComplex(xl, zl, inside)
