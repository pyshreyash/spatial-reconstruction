"""Validation only: video wall planes (depth-only vs line-refined) against LiDAR wall planes, same capture.

    python scripts/eval_video_walls.py scan3

Needs the LiDAR cache (out/<capture>/.points_f5_p2.npz) and the video cache (out/<capture>_video/...)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.geometry.frame import manhattan_yaw, yaw_rotation  # noqa: E402
from roomscan.geometry.lines import observations, occupancy, refine_walls, triangulate  # noqa: E402
from roomscan.geometry.planes import ceiling_plane, floor_plane  # noqa: E402
from roomscan.geometry.semantics import CEILING, FLOOR, FURNITURE  # noqa: E402
from roomscan.geometry.stereo import WORK_WIDTH, Poses  # noqa: E402
from roomscan.geometry.walls import consolidate, detect_walls, split_structural  # noqa: E402
from roomscan.io.stray import RGB_WIDTH, load_posed_video  # noqa: E402
from roomscan.lidar import (CEILING_MARGIN, CHUNK_FRAMES, DEFAULT_TOP, DOOR_HEAD, WALL_BAND_BOTTOM,  # noqa: E402
                            WALL_HIGH_BAND)


def walls_of(pts, nrm, fidx, R, labels=None):
    pts, nrm = pts @ R.T, nrm @ R.T
    ch = fidx // CHUNK_FRAMES
    fl = cl = wl = np.ones(len(pts), bool)
    if labels is not None:
        fl, cl, wl = labels == FLOOR, labels == CEILING, labels != FURNITURE
    floor = floor_plane(pts[fl], nrm[fl], ch[fl])
    ceil = ceiling_plane(pts[cl], nrm[cl], ch[cl], floor.value)
    top = ceil.value - CEILING_MARGIN if ceil else floor.value + DEFAULT_TOP
    w = consolidate(detect_walls(pts[wl], nrm[wl], ch[wl], floor.value + WALL_BAND_BOTTOM, top,
                                 floor.value + WALL_HIGH_BAND, floor.value + DOOR_HEAD))
    return split_structural(w)[0], floor.value, (ceil.value if ceil else None)


def main(capture: str) -> None:
    lc = np.load(f"out/{capture}/.points_f5_p2.npz")
    RL = yaw_rotation(manhattan_yaw(lc["nrm"]))
    ref, _, _ = walls_of(lc["pts"], lc["nrm"], lc["fidx"], RL)

    vc = np.load(sorted(Path(f"out/{capture}_video").glob(".video_points_mono*.npz"))[-1])
    stats = json.loads(str(vc["stats"]))
    labels = vc["labels"] if "labels" in vc.files else None
    depth, floor_y, ceil_y = walls_of(vc["pts"], vc["nrm"], vc["fidx"], RL, labels)
    vid = load_posed_video(Path(capture))
    p = Poses.from_video(vid).shifted(stats["time_offset_s"])
    ok = vc["seg_ok"] if "seg_ok" in vc.files else np.ones(len(vc["seg"]), bool)
    segs = {int(k): (vc["seg"][(vc["seg_k"] == k) & ok], vc["seg_z"][(vc["seg_k"] == k) & ok])
            for k in np.unique(vc["seg_k"])}
    obs = observations(segs, p.R, p.C, {k: p.K(k, WORK_WIDTH / RGB_WIDTH) for k in segs})
    lines = triangulate(obs, RL, occupancy(vc["pts"] @ RL.T))
    refined, info = refine_walls(copy.deepcopy(depth), lines, floor_y, ceil_y)
    print(f"{capture}: {len(ref)} LiDAR walls, {len(depth)} video walls; {info}")

    rows = []
    for d, r in zip(depth, refined):
        cands = [w for w in ref if w.axis == d.axis and w.facing == d.facing and abs(w.offset - d.offset) < 0.3
                 and min(w.span[1], d.span[1]) - max(w.span[0], d.span[0]) > 0.3]
        if not cands:
            continue
        g = min(cands, key=lambda w: abs(w.offset - d.offset))
        rows.append((d.axis, d.facing, g.offset, d.offset - g.offset, r.offset - g.offset, r.sigma,
                     r.offset != d.offset))
    print(f"\naxis facing  LiDAR     depth err  refined err  sigma   by lines")
    for a, f, g, e0, e1, s, moved in sorted(rows, key=lambda r: (r[0], r[2])):
        print(f"{a}    {f:+d}    {g:+7.3f}   {e0 * 100:+7.1f} cm  {e1 * 100:+7.1f} cm   {s * 100:5.1f}   {'yes' if moved else ''}")
    e0 = np.array([abs(r[3]) for r in rows])
    e1 = np.array([abs(r[4]) for r in rows])
    z = np.array([abs(r[4]) / r[5] for r in rows])
    mv = np.array([r[6] for r in rows])
    print(f"\nmatched {len(rows)}: median |err| depth {np.median(e0) * 100:.1f} cm -> refined {np.median(e1) * 100:.1f} cm")
    if mv.any():
        print(f"  line-measured walls ({mv.sum()}): depth {np.median(e0[mv]) * 100:.1f} cm -> lines "
              f"{np.median(e1[mv]) * 100:.1f} cm")
    print(f"  LiDAR offset inside refined ci90: {np.mean(z <= 1.645):.0%}")
    for tag, sel in (("line-measured", mv), ("depth-only", ~mv)):
        if sel.any():
            print(f"  {tag}: p90 |err| {np.percentile(e1[sel], 90) * 100:.1f} cm -> sigma for 90 % coverage "
                  f"{np.percentile(e1[sel], 90) / 1.645 * 100:.1f} cm (n={sel.sum()})")


if __name__ == "__main__":
    main(sys.argv[1])
