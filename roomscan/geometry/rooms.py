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
PATH_SEARCH = 0.3      # how far from a path crossing to look for the wall gap
MIN_PATH_DOOR = 0.4    # narrower gaps at a crossing are pose jitter through a jamb
MAX_PATH_DOOR = 1.6


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
    kind: str              # "door" | "open"
    rooms: tuple[str, str]
    axis: str
    offset: float
    start: float           # along-wall interval
    end: float
    walls: list[str]
    source: str = "wall_gap"            # "wall_gap" | "camera_path"
    other_offset: float | None = None   # far face when the opening passes through a thick wall

    @property
    def width(self) -> float:
        return self.end - self.start

    @property
    def offsets(self) -> list[float]:
        return [self.offset] + ([self.other_offset] if self.other_offset is not None else [])

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.axis == "x":
            return (self.offset, self.start), (self.offset, self.end)
        return (self.start, self.offset), (self.end, self.offset)


@dataclass
class Adjacency:
    rooms: tuple[str, str]
    via: str               # "door" | "open" | "wall"
    thickness: float | None = None  # measured wall thickness between the rooms


@dataclass
class LineProfile:
    lo: float
    res: float
    closed: np.ndarray
    doors: list[tuple[float, float]]
    covered: np.ndarray

    def bin(self, t: float) -> int:
        return int(np.clip(np.floor((t - self.lo) / self.res), 0, len(self.closed) - 1))

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
            evidence = line.trusted if line.trusted is not None else line.along
            covered = np.histogram(evidence, edges)[0] >= 2
            solid = covered.copy()
            for p in perp:
                pe = p.trusted if p.trusted is not None else p.along
                a, b = line.offset - CORNER_TOL, line.offset + CORNER_TOL
                if (np.histogram(pe, np.linspace(a, b, max(1, int(round((b - a) / COVER_RES))) + 1))[0] >= 2).any():
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
            out[(axis, idx)] = LineProfile(lo, res, closed, doors, covered)
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


def _across(cx: CellComplex, lab: np.ndarray, axis: str, idx: int, t: float, side: int) -> tuple[int, int | None]:
    """Room on one side (-1/+1) of line idx at along-coordinate t, hopping over a thin wall strip.
    Returns (label, index of the wall's far face line or None)."""
    ncx, ncz = lab.shape
    if axis == "x":
        j = int(np.searchsorted(cx.zs, t)) - 1
        i = idx - 1 if side < 0 else idx
        if not (0 <= j < ncz and 0 <= i < ncx):
            return 0, None
        if lab[i, j] == 0 and np.diff(cx.xs)[i] < MAX_WALL_THICKNESS:
            i2 = i + side
            return (int(lab[i2, j]), i if side < 0 else i + 1) if 0 <= i2 < ncx else (0, None)
        return int(lab[i, j]), None
    i = int(np.searchsorted(cx.xs, t)) - 1
    j = idx - 1 if side < 0 else idx
    if not (0 <= i < ncx and 0 <= j < ncz):
        return 0, None
    if lab[i, j] == 0 and np.diff(cx.zs)[j] < MAX_WALL_THICKNESS:
        j2 = j + side
        return (int(lab[i, j2]), j if side < 0 else j + 1) if 0 <= j2 < ncz else (0, None)
    return int(lab[i, j]), None


def _path_gaps(
    cx: CellComplex, lab: np.ndarray, prof: dict[tuple[str, int], LineProfile], path: np.ndarray
) -> list[tuple[str, int, float, float]]:
    """Wall gaps the camera walked through: (axis, line index, start, end)."""
    out = []
    for axis, lines, ai in (("x", cx.xl, 0), ("z", cx.zl, 1)):
        for idx, line in enumerate(lines):
            d = path[:, ai] - line.offset
            for k in np.nonzero(d[:-1] * d[1:] < 0)[0]:
                w = d[k] / (d[k] - d[k + 1])
                t = float(path[k, 1 - ai] + w * (path[k + 1, 1 - ai] - path[k, 1 - ai]))
                a, _ = _across(cx, lab, axis, idx, t, -1)
                b, _ = _across(cx, lab, axis, idx, t, +1)
                if a == 0 or b == 0 or a == b:
                    continue
                p = prof[(axis, idx)]
                free = ~p.covered
                c = p.bin(t)
                if not free[c]:
                    near = [c + o for o in range(-int(PATH_SEARCH / p.res), int(PATH_SEARCH / p.res) + 1)
                            if 0 <= c + o < len(free) and free[c + o]]
                    if not near:
                        continue
                    c = min(near, key=lambda q: abs(q - c))
                lo_b = hi_b = c
                while lo_b > 0 and free[lo_b - 1]:
                    lo_b -= 1
                while hi_b < len(free) - 1 and free[hi_b + 1]:
                    hi_b += 1
                start, end = p.lo + lo_b * p.res, p.lo + (hi_b + 1) * p.res
                # Keep the gap within the stretch where these same two rooms face each other.
                bounds = cx.zs if axis == "x" else cx.xs
                j = int(np.clip(np.searchsorted(bounds, t) - 1, 0, len(bounds) - 2))
                j_lo = j_hi = j
                pair = (a, b)
                while j_lo > 0 and (_across(cx, lab, axis, idx, (bounds[j_lo - 1] + bounds[j_lo]) / 2, -1)[0],
                                    _across(cx, lab, axis, idx, (bounds[j_lo - 1] + bounds[j_lo]) / 2, 1)[0]) == pair:
                    j_lo -= 1
                while j_hi < len(bounds) - 2 and (_across(cx, lab, axis, idx, (bounds[j_hi + 1] + bounds[j_hi + 2]) / 2, -1)[0],
                                                  _across(cx, lab, axis, idx, (bounds[j_hi + 1] + bounds[j_hi + 2]) / 2, 1)[0]) == pair:
                    j_hi += 1
                start, end = max(start, bounds[j_lo]), min(end, bounds[j_hi + 1])
                if MIN_PATH_DOOR <= end - start <= MAX_PATH_DOOR:
                    out.append((axis, idx, float(start), float(end)))
    return out


def wall_cells(cx: CellComplex, lab: np.ndarray) -> np.ndarray:
    """Outside cells that are the body of a wall between rooms (thin strip with rooms both sides)."""
    dx, dz = np.diff(cx.xs), np.diff(cx.zs)
    ncx, ncz = lab.shape
    out = np.zeros(lab.shape, bool)
    for i in range(ncx):
        for j in range(ncz):
            if lab[i, j]:
                continue
            if dx[i] < MAX_WALL_THICKNESS and 0 < i < ncx - 1 and lab[i - 1, j] and lab[i + 1, j]:
                out[i, j] = True
            if dz[j] < MAX_WALL_THICKNESS and 0 < j < ncz - 1 and lab[i, j - 1] and lab[i, j + 1]:
                out[i, j] = True
    # Junctions where wall strips meet.
    junction = np.zeros_like(out)
    for i in range(ncx):
        for j in range(ncz):
            if lab[i, j] or out[i, j] or dx[i] >= MAX_WALL_THICKNESS or dz[j] >= MAX_WALL_THICKNESS:
                continue
            nb = sum(out[a, b] for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                     if 0 <= a < ncx and 0 <= b < ncz)
            junction[i, j] = nb >= 2
    return out | junction


def find_openings(
    cx: CellComplex,
    lab: np.ndarray,
    rooms: list[Room],
    prof: dict[tuple[str, int], LineProfile],
    path: np.ndarray | None = None,
) -> tuple[list[Opening], list[Adjacency]]:
    xs, zs = cx.xs, cx.zs
    ncx, ncz = lab.shape
    via: dict[tuple[int, int], str] = {}
    for a, b, *_ in _neighbours(lab):
        if a > 0 and b > 0 and a != b:
            via[(min(a, b), max(a, b))] = "wall"

    # Rooms on either side of a thin outside strip share a wall with thickness.
    thick: dict[tuple[int, int], list[float]] = defaultdict(list)
    dx, dz = np.diff(xs), np.diff(zs)
    for i in range(1, ncx - 1):
        for j in range(ncz):
            if lab[i, j] == 0 and dx[i] < MAX_WALL_THICKNESS and 0 < lab[i - 1, j] != lab[i + 1, j] > 0:
                thick[(min(lab[i - 1, j], lab[i + 1, j]), max(lab[i - 1, j], lab[i + 1, j]))].append(dx[i])
    for i in range(ncx):
        for j in range(1, ncz - 1):
            if lab[i, j] == 0 and dz[j] < MAX_WALL_THICKNESS and 0 < lab[i, j - 1] != lab[i, j + 1] > 0:
                thick[(min(lab[i, j - 1], lab[i, j + 1]), max(lab[i, j - 1], lab[i, j + 1]))].append(dz[j])
    for pair in thick:
        via.setdefault(pair, "wall")

    candidates = [(axis, idx, s, e, "wall_gap") for (axis, idx), p in sorted(prof.items()) for s, e in p.doors]
    if path is not None and len(path) > 1:
        candidates += [(axis, idx, s, e, "camera_path") for axis, idx, s, e in _path_gaps(cx, lab, prof, path)]

    by_id = {r.id: r for r in rooms}
    openings: list[Opening] = []
    for axis, idx, start, end, source in candidates:
        mid = (start + end) / 2
        a, far_a = _across(cx, lab, axis, idx, mid, -1)
        b, far_b = _across(cx, lab, axis, idx, mid, +1)
        if a == 0 or b == 0 or a == b:
            continue
        lines = cx.xl if axis == "x" else cx.zl
        far = far_a if far_a is not None else far_b
        offset = float(lines[idx].offset)
        other = float(lines[far].offset) if far is not None else None
        ra, rb = f"R{min(a, b)}", f"R{max(a, b)}"
        dup = next((o for o in openings if o.rooms == (ra, rb) and o.axis == axis
                    and min(abs(f - offset) for f in o.offsets) <= MAX_WALL_THICKNESS
                    and min(o.end, end) > max(o.start, start)), None)
        if dup is not None:
            continue
        host = [w.id for r in (ra, rb) for w in by_id[r].walls
                if w.line[0] == axis and abs((lines[w.line[1]].offset) - offset) <= MAX_WALL_THICKNESS + 1e-6
                and min(end, w.along[1]) - max(start, w.along[0]) > 0.5 * (end - start)]
        kind = "door" if end - start <= MAX_DOOR_WIDTH + 0.1 else "open"
        openings.append(Opening(f"O{len(openings) + 1}", kind, (ra, rb), axis, offset, float(start), float(end),
                                host, source, other))
        pair = (min(a, b), max(a, b))
        via[pair] = "door" if kind == "door" or via.get(pair) == "door" else "open"

    adjacency = [Adjacency((f"R{a}", f"R{b}"), v, float(np.median(thick[(a, b)])) if (a, b) in thick else None)
                 for (a, b), v in sorted(via.items())]
    return openings, adjacency
