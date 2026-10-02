"""Step 9a: split the inside region into rooms at walls and doorways; find openings between rooms.

Each wall line gets a profile of where it is *closed*: high-band wall evidence, plus gaps of at most
door width that are bounded on both sides by wall or by a perpendicular wall (corner). Inside cells
are connected only across edges that are mostly open; connected components are rooms.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from roomscan.geometry.polygon import COVER_RES, CellComplex, outer_loop

MIN_ROOM_AREA = 1.0
MIN_DOOR_WIDTH = 0.5
MAX_DOOR_WIDTH = 1.0   # interior door clear widths; wider gaps are open plan
MIN_PARTITION = 0.6    # metres of high-band wall evidence before a line can carry a doorway
CORNER_TOL = 0.15
MAX_WALL_THICKNESS = 0.35


@dataclass
class WallEdge:
    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    sigma: float
    coverage: float
    line: tuple[str, int]  # ("x", i): on plane x = xs[i]
    along: tuple[float, float]


@dataclass
class Room:
    id: str
    polygon: list[tuple[float, float]]
    walls: list[WallEdge]
    area: float
    area_sigma: float
    label_xy: tuple[float, float]
    cells: np.ndarray = field(repr=False)


@dataclass
class Opening:
    id: str
    kind: str              # "door"
    rooms: tuple[str, str]
    axis: str
    offset: float
    start: float           # along-wall interval
    end: float
    walls: list[str]

    @property
    def width(self) -> float:
        return self.end - self.start

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.axis == "x":
            return (self.offset, self.start), (self.offset, self.end)
        return (self.start, self.offset), (self.end, self.offset)


@dataclass
class Adjacency:
    rooms: tuple[str, str]
    via: str               # "door" | "wall"


@dataclass
class LineProfile:
    lo: float
    res: float
    closed: np.ndarray
    doors: list[tuple[float, float]]

    def closed_fraction(self, a: float, b: float) -> float:
        i0 = int(np.floor((min(a, b) - self.lo) / self.res + 1e-6))
        i1 = int(np.ceil((max(a, b) - self.lo) / self.res - 1e-6))
        seg = self.closed[max(i0, 0):max(i1, i0 + 1)]
        return float(seg.mean()) if seg.size else 0.0


def line_profiles(cx: CellComplex) -> dict[tuple[str, int], LineProfile]:
    out = {}
    for axis, lines, perp, ext in (("x", cx.xl, cx.zl, cx.zs), ("z", cx.zl, cx.xl, cx.xs)):
        lo, hi = float(ext[0]), float(ext[-1])
        n = max(1, int(round((hi - lo) / COVER_RES)))
        res = (hi - lo) / n
        edges = lo + np.arange(n + 1) * res
        for idx, line in enumerate(lines):
            covered = np.histogram(line.along, edges)[0] >= 2
            solid = covered.copy()
            for p in perp:
                if p.covered(line.offset - CORNER_TOL, line.offset + CORNER_TOL).any():
                    solid[int(np.clip((p.offset - lo) / res, 0, n - 1))] = True
            closed, doors = solid.copy(), []
            if covered.sum() * res >= MIN_PARTITION:
                runs, m = ndimage.label(~solid)
                for r in range(1, m + 1):
                    ix = np.nonzero(runs == r)[0]
                    width = len(ix) * res
                    if ix[0] == 0 or ix[-1] == n - 1 or width > MAX_DOOR_WIDTH:
                        continue
                    closed[ix] = True
                    if width >= MIN_DOOR_WIDTH:
                        doors.append((lo + ix[0] * res, lo + (ix[-1] + 1) * res))
            out[(axis, idx)] = LineProfile(lo, res, closed, doors)
    return out


def _neighbours(lab: np.ndarray):
    """Yield (labelA, labelB, axis, line_index, cell_i, cell_j) for each interior cell edge."""
    ncx, ncz = lab.shape
    for i in range(ncx):
        for j in range(ncz):
            if i + 1 < ncx:
                yield lab[i, j], lab[i + 1, j], "x", i + 1, i, j
            if j + 1 < ncz:
                yield lab[i, j], lab[i, j + 1], "z", j + 1, i, j


def segment_cells(cx: CellComplex, prof: dict[tuple[str, int], LineProfile]) -> np.ndarray:
    """Room label per cell (0 = outside), labels ordered by decreasing area."""
    xs, zs, inside = cx.xs, cx.zs, cx.inside
    cid = np.arange(inside.size).reshape(inside.shape)
    rows, cols = [], []
    for a, b, axis, idx, i, j in _neighbours(cid):
        if not (inside.flat[a] and inside.flat[b]):
            continue
        span = (zs[j], zs[j + 1]) if axis == "x" else (xs[i], xs[i + 1])
        if prof[(axis, idx)].closed_fraction(*span) < 0.5:
            rows.append(a)
            cols.append(b)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(inside.size, inside.size))
    _, comp = connected_components(graph, directed=False)
    lab = np.where(inside, comp.reshape(inside.shape) + 1, 0)

    area = cx.area
    while True:
        tiny = [k for k in np.unique(lab) if k > 0 and area[lab == k].sum() < MIN_ROOM_AREA]
        if not tiny:
            break
        k = tiny[0]
        shared: dict[int, float] = defaultdict(float)
        for a, b, axis, _, i, j in _neighbours(lab):
            if k in (a, b) and a != b and a > 0 and b > 0:
                shared[b if a == k else a] += np.diff(zs)[j] if axis == "x" else np.diff(xs)[i]
        lab[lab == k] = max(shared, key=shared.get) if shared else 0

    order = sorted((k for k in np.unique(lab) if k > 0), key=lambda k: -area[lab == k].sum())
    out = np.zeros_like(lab)
    for new, k in enumerate(order, start=1):
        out[lab == k] = new
    return out


def build_rooms(cx: CellComplex, lab: np.ndarray) -> list[Room]:
    xs, zs, area = cx.xs, cx.zs, cx.area
    rooms = []
    for k in range(1, lab.max() + 1):
        mask = lab == k
        rid = f"R{k}"
        loop = outer_loop(mask, xs, zs)
        edges, on_line = [], defaultdict(float)
        for n, (a1, b1) in enumerate(loop):
            a2, b2 = loop[(n + 1) % len(loop)]
            if b1 == b2:
                line, key, along = cx.zl[b1], ("z", b1), (xs[a1], xs[a2])
                sigma = float(np.hypot(cx.xl[a1].sigma, cx.xl[a2].sigma))
            else:
                line, key, along = cx.xl[a1], ("x", a1), (zs[b1], zs[b2])
                sigma = float(np.hypot(cx.zl[b1].sigma, cx.zl[b2].sigma))
            length = abs(along[1] - along[0])
            on_line[key] += length
            edges.append(WallEdge(
                f"{rid}.W{n + 1}", (float(xs[a1]), float(zs[b1])), (float(xs[a2]), float(zs[b2])),
                float(length), sigma, line.coverage(*along), key, (min(along), max(along)),
            ))
        area_sigma = float(np.sqrt(sum(
            (length * (cx.xl if ax == "x" else cx.zl)[idx].sigma) ** 2 for (ax, idx), length in on_line.items()
        )))
        bi, bj = np.unravel_index(np.argmax(np.where(mask, area, -1)), mask.shape)
        label_xy = (float(xs[bi] + xs[bi + 1]) / 2, float(zs[bj] + zs[bj + 1]) / 2)
        rooms.append(Room(rid, [(float(xs[a]), float(zs[b])) for a, b in loop], edges,
                          float(area[mask].sum()), area_sigma, label_xy, mask))
    return rooms


def find_openings(
    cx: CellComplex, lab: np.ndarray, rooms: list[Room], prof: dict[tuple[str, int], LineProfile]
) -> tuple[list[Opening], list[Adjacency]]:
    xs, zs = cx.xs, cx.zs
    ncx, ncz = lab.shape
    via: dict[tuple[int, int], str] = {}
    for a, b, *_ in _neighbours(lab):
        if a > 0 and b > 0 and a != b:
            via[(min(a, b), max(a, b))] = "wall"

    # Rooms on either side of a thin outside strip share a wall with thickness.
    dx, dz = np.diff(xs), np.diff(zs)
    for i in range(1, ncx - 1):
        for j in range(ncz):
            if lab[i, j] == 0 and dx[i] < MAX_WALL_THICKNESS and 0 < lab[i - 1, j] != lab[i + 1, j] > 0:
                via.setdefault((min(lab[i - 1, j], lab[i + 1, j]), max(lab[i - 1, j], lab[i + 1, j])), "wall")
    for i in range(ncx):
        for j in range(1, ncz - 1):
            if lab[i, j] == 0 and dz[j] < MAX_WALL_THICKNESS and 0 < lab[i, j - 1] != lab[i, j + 1] > 0:
                via.setdefault((min(lab[i, j - 1], lab[i, j + 1]), max(lab[i, j - 1], lab[i, j + 1])), "wall")

    by_id = {r.id: r for r in rooms}
    openings: list[Opening] = []
    for (axis, idx), p in sorted(prof.items()):
        line = (cx.xl if axis == "x" else cx.zl)[idx]
        for start, end in p.doors:
            mid = (start + end) / 2
            if axis == "x":
                j = int(np.searchsorted(zs, mid)) - 1
                sides = [lab[i, j] for i in (idx - 1, idx) if 0 <= i < ncx and 0 <= j < ncz]
            else:
                i = int(np.searchsorted(xs, mid)) - 1
                sides = [lab[i, j] for j in (idx - 1, idx) if 0 <= i < ncx and 0 <= j < ncz]
            if len(sides) != 2 or 0 in sides or sides[0] == sides[1]:
                continue
            a, b = sorted(int(s) for s in sides)
            ra, rb = f"R{a}", f"R{b}"
            host = [w.id for r in (ra, rb) for w in by_id[r].walls
                    if w.line == (axis, idx) and min(end, w.along[1]) - max(start, w.along[0]) > 0.5 * (end - start)]
            openings.append(Opening(f"O{len(openings) + 1}", "door", (ra, rb), axis, float(line.offset),
                                    float(start), float(end), host))
            via[(a, b)] = "door"
    adjacency = [Adjacency((f"R{a}", f"R{b}"), v) for (a, b), v in sorted(via.items())]
    return openings, adjacency
