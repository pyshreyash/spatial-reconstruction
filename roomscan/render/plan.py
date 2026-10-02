"""Top-down dimensioned plan rendering."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roomscan.export.model3d import PALETTE  # noqa: E402
from roomscan.geometry.freespace import FreeSpaceGrid  # noqa: E402
from roomscan.geometry.rooms import Opening, Room  # noqa: E402
from roomscan.geometry.walls import WallPlane  # noqa: E402

Z90 = 1.645


def render_plan(
    rooms: list[Room],
    openings: list[Opening],
    walls: list[WallPlane],
    grid: FreeSpaceGrid,
    trajectory: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    free = np.ma.masked_where(~grid.free_mask().T, grid.free_mask().T)
    ax.imshow(free, origin="lower", extent=grid.extent, cmap="Greys", vmin=0, vmax=4, alpha=0.4)
    for w in walls:
        a, b = w.span
        if w.axis == "x":
            ax.plot([w.offset, w.offset], [a, b], color="tab:gray", lw=0.6)
        else:
            ax.plot([a, b], [w.offset, w.offset], color="tab:gray", lw=0.6)
    ax.plot(trajectory[:, 0], trajectory[:, 1], ":", color="tab:orange", lw=1, label="camera path")

    for k, room in enumerate(rooms):
        poly = np.array(room.polygon + room.polygon[:1])
        ax.fill(poly[:, 0], poly[:, 1], color=PALETTE[k % len(PALETTE)], alpha=0.45)
        lx, lz = room.label_xy
        ax.text(lx, lz, f"{room.id}\n{room.area:.2f} ± {Z90 * room.area_sigma:.2f} m²",
                ha="center", va="center", fontsize=10, weight="bold")
        for w in room.walls:
            (x1, z1), (x2, z2) = w.start, w.end
            ax.plot([x1, x2], [z1, z2], color="black" if w.coverage >= 0.3 else "tab:red", lw=2.2)
            mx, mz = (x1 + x2) / 2, (z1 + z2) / 2
            # Label offset towards the room interior (left of a CCW edge in x-z).
            nx, nz = -(z2 - z1), (x2 - x1)
            norm = np.hypot(nx, nz) or 1.0
            ax.text(mx + 0.2 * nx / norm, mz + 0.2 * nz / norm,
                    f"{w.id.split('.')[1]} {w.length:.2f}±{Z90 * w.sigma:.2f}", ha="center", va="center",
                    fontsize=6, rotation=0 if z1 == z2 else 90)

    for o in openings:
        (x1, z1), (x2, z2) = o.endpoints()
        ax.plot([x1, x2], [z1, z2], color="tab:green", lw=5, solid_capstyle="butt")
        ax.text((x1 + x2) / 2, (z1 + z2) / 2, f"{o.id} {o.kind}\n{o.width:.2f} m", fontsize=7,
                color="darkgreen", ha="center", va="center", weight="bold")

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
