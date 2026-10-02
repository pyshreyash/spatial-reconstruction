"""Debug view: wall-band vertical points + every detected wall plane with facing."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.geometry.backproject import collect_points  # noqa: E402
from roomscan.geometry.frame import manhattan_yaw, yaw_rotation  # noqa: E402
from roomscan.geometry.planes import floor_plane  # noqa: E402
from roomscan.geometry.walls import detect_walls  # noqa: E402
from roomscan.io.stray import load_stray  # noqa: E402

cap = load_stray(Path(sys.argv[1]))
pts, nrm, fidx = collect_points(cap, 5, 2)
R = yaw_rotation(manhattan_yaw(nrm))
pts, nrm = pts @ R.T, nrm @ R.T
cams = cap.positions @ R.T
chunks = fidx // 60
floor = floor_plane(pts, nrm, chunks)
walls = detect_walls(pts, nrm, chunks, floor.value + 0.3, floor.value + 2.2, floor.value + 1.2)

band = (pts[:, 1] > floor.value + 0.3) & (pts[:, 1] < floor.value + 2.2) & (np.abs(nrm[:, 1]) < 0.3)
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist2d(pts[band, 0], pts[band, 2], bins=400, cmap="Greys", cmin=1, vmax=200)
ax.plot(cams[:, 0], cams[:, 2], color="tab:orange", lw=1)
for w in walls:
    a, b = w.span
    col = "tab:blue" if w.facing > 0 else "tab:red"
    if w.axis == "x":
        ax.plot([w.offset] * 2, [a, b], color=col, lw=1.5)
        ax.text(w.offset, b, f"{w.offset:.2f}", fontsize=6, color=col)
    else:
        ax.plot([a, b], [w.offset] * 2, color=col, lw=1.5)
        ax.text(b, w.offset, f"{w.offset:.2f}", fontsize=6, color=col)
    print(f"{w.axis} facing {w.facing:+d} offset {w.offset:+.3f} span {a:+.2f}..{b:+.2f} n={w.n_points}")
ax.set_aspect("equal")
fig.savefig(sys.argv[2], dpi=130)
