"""Top-down dimensioned plan rendering."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roomscan.export.model3d import PALETTE  # noqa: E402
from roomscan.geometry.freespace import FreeSpaceGrid  # noqa: E402
from roomscan.geometry.polygon import CellComplex, _trace_loops  # noqa: E402
from roomscan.geometry.rooms import Opening, Room  # noqa: E402
from roomscan.geometry.walls import WallPlane  # noqa: E402

Z90 = 1.645
WALL_FILL = "#4a4a4a"


def _segment(ax, axis: str, offset: float, a: float, b: float, **kw) -> None:
    if axis == "x":
        ax.plot([offset, offset], [a, b], **kw)
    else:
        ax.plot([a, b], [offset, offset], **kw)


def render_plan(
    rooms: list[Room],
    openings: list[Opening],
    objects: list[WallPlane],
    cx: CellComplex,
    lab: np.ndarray,
    wall_mask: np.ndarray,
    heights: dict[str, float | None],
    floor_y: float,
    grid: FreeSpaceGrid,
    trajectory: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    xs, zs = cx.xs, cx.zs
    ncx, ncz = lab.shape
    fig, ax = plt.subplots(figsize=(11, 11))
    free = np.ma.masked_where(~grid.free_mask().T, grid.free_mask().T)
    ax.imshow(free, origin="lower", extent=grid.extent, cmap="Greys", vmin=0, vmax=6, alpha=0.3)
    ax.plot(trajectory[:, 0], trajectory[:, 1], ":", color="tab:orange", lw=0.8, alpha=0.7, label="camera path")

    for k, room in enumerate(rooms):
        poly = np.array(room.polygon + room.polygon[:1])
        ax.fill(poly[:, 0], poly[:, 1], color=PALETTE[k % len(PALETTE)], alpha=0.45, lw=0)

    # Wall bodies between rooms; their open ends (the thickness) dotted.
    for i, j in zip(*np.nonzero(wall_mask)):
        ax.fill([xs[i], xs[i + 1], xs[i + 1], xs[i]], [zs[j], zs[j], zs[j + 1], zs[j + 1]], color=WALL_FILL, lw=0)
        for (ni, nj), axis, off, a, b in (
            ((i - 1, j), "x", xs[i], zs[j], zs[j + 1]), ((i + 1, j), "x", xs[i + 1], zs[j], zs[j + 1]),
            ((i, j - 1), "z", zs[j], xs[i], xs[i + 1]), ((i, j + 1), "z", zs[j + 1], xs[i], xs[i + 1]),
        ):
            inside = 0 <= ni < ncx and 0 <= nj < ncz and (lab[ni, nj] > 0 or wall_mask[ni, nj])
            if not inside:
                _segment(ax, axis, off, a, b, color="black", lw=1.0, ls=(0, (1.5, 1.5)))

    for room in rooms:
        lx, lz = room.label_xy
        h = heights.get(room.id)
        ax.text(lx, lz, f"{room.id}\n{room.area:.2f} ± {Z90 * room.area_sigma:.2f} m²"
                + (f"\nh {h:.2f} m" if h else ""), ha="center", va="center", fontsize=9, weight="bold")
        for w in room.walls:
            (x1, z1), (x2, z2) = w.start, w.end
            ax.plot([x1, x2], [z1, z2], color="black" if w.coverage >= 0.3 else "tab:red", lw=1.2)
            mx, mz = (x1 + x2) / 2, (z1 + z2) / 2
            # Label offset towards the room interior (left of a CCW edge in x-z).
            nx, nz = -(z2 - z1), (x2 - x1)
            norm = np.hypot(nx, nz) or 1.0
            if w.length >= 0.25:
                ax.text(mx + 0.18 * nx / norm, mz + 0.18 * nz / norm,
                        f"{w.id.split('.')[1]} {w.length:.2f}±{Z90 * w.sigma:.2f}", ha="center", va="center",
                        fontsize=5.5, rotation=0 if z1 == z2 else 90)

    # Overall perimeter of rooms + wall bodies.
    for loop in _trace_loops((lab > 0) | wall_mask):
        p = np.array([(xs[a], zs[b]) for a, b in loop + loop[:1]])
        ax.plot(p[:, 0], p[:, 1], color="black", lw=3.2, solid_joinstyle="miter")

    for o in openings:
        offs = o.offsets if o.other_offset is not None else [o.offset - 0.05, o.offset + 0.05]
        lo, hi = min(offs), max(offs)
        if o.axis == "x":
            ax.fill([lo, hi, hi, lo], [o.start, o.start, o.end, o.end], color="white", lw=0, zorder=3)
        else:
            ax.fill([o.start, o.end, o.end, o.start], [lo, lo, hi, hi], color="white", lw=0, zorder=3)
        for f in offs:
            _segment(ax, o.axis, f, o.start, o.end, color="tab:green", lw=2.5, zorder=4, solid_capstyle="butt")
        (x1, z1), (x2, z2) = o.endpoints()
        ax.text((x1 + x2) / 2, (z1 + z2) / 2, f"{o.id} {o.width:.2f}", fontsize=6.5, color="darkgreen",
                ha="center", va="center", weight="bold", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))

    for n, f in enumerate(objects, start=1):
        _segment(ax, f.axis, f.offset, *f.span, color="saddlebrown", lw=1.5, ls="--")
        mid = (f.span[0] + f.span[1]) / 2
        tx, tz = (f.offset, mid) if f.axis == "x" else (mid, f.offset)
        ax.text(tx, tz, f"obj{n} {f.top - floor_y:.1f}m", fontsize=5.5, color="saddlebrown", ha="center")

    ax.set_aspect("equal")
    # Right-handed, y-up frame seen from above: +z points down the page, else the plan is mirrored.
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
