"""LiDAR tier pipeline: Stray Scanner capture -> plan JSON, rendered plan and 3-D model."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from roomscan import __version__
from roomscan.export.gltf import write_glb
from roomscan.export.model3d import DOOR_HEIGHT, build_model
from roomscan.export.ply import export_raw_scan
from roomscan.geometry.backproject import collect_points
from roomscan.geometry.frame import manhattan_yaw, yaw_rotation
from roomscan.geometry.freespace import carve
from roomscan.geometry.planes import ceiling_plane, floor_plane, room_ceilings
from roomscan.geometry.polygon import build_complex
from roomscan.geometry.rooms import build_rooms, find_openings, line_profiles, segment_cells, wall_cells
from roomscan.geometry.walls import consolidate, detect_walls, split_structural
from roomscan.io.stray import load_stray
from roomscan.render.plan import Z90, render_plan

CHUNK_FRAMES = 60  # block-bootstrap unit, ~1.3 s at Stray's frame rate
WALL_BAND_BOTTOM = 0.3  # above floor: skips skirting boards
WALL_HIGH_BAND = 1.2  # above floor: wall evidence above typical furniture
DOOR_HEAD = 1.95  # above floor: wall evidence below door heads (lintels would close doorways)
CEILING_MARGIN = 0.15
DEFAULT_TOP = 2.2  # above floor, used when the ceiling was not scanned
ASSUMED_CEILING = 2.4  # only for the 3-D model when the ceiling was not scanned
MAX_RAYS = 400_000


def measure(value: float, sigma: float) -> dict:
    return {
        "value": round(value, 4),
        "sigma": round(sigma, 4),
        "ci90": [round(value - Z90 * sigma, 4), round(value + Z90 * sigma, 4)],
    }


def run_lidar(
    capture: Path, out_dir: Path, frame_stride: int = 5, pixel_stride: int = 2, raw_scan: bool = True
) -> dict:
    t0 = time.perf_counter()
    cap = load_stray(capture)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f".points_f{frame_stride}_p{pixel_stride}.npz"
    if cache.is_file():
        c = np.load(cache)
        pts, nrm, fidx = c["pts"], c["nrm"], c["fidx"]
    else:
        pts, nrm, fidx = collect_points(cap, frame_stride, pixel_stride)
        np.savez(cache, pts=pts, nrm=nrm, fidx=fidx)
    chunks = fidx // CHUNK_FRAMES

    theta = manhattan_yaw(nrm)
    R = yaw_rotation(theta)
    pts = (pts @ R.T).astype(np.float32)
    nrm = (nrm @ R.T).astype(np.float32)
    cams = cap.positions @ R.T

    warnings: list[str] = []
    floor = floor_plane(pts, nrm, chunks)
    ceil = ceiling_plane(pts, nrm, chunks, floor.value)
    if ceil is None:
        warnings.append("Ceiling not observed; ceiling height not reported. Sweep the phone up to the ceiling.")
    top = ceil.value - CEILING_MARGIN if ceil else floor.value + DEFAULT_TOP

    walls = consolidate(
        detect_walls(pts, nrm, chunks, floor.value + WALL_BAND_BOTTOM, top,
                     floor.value + WALL_HIGH_BAND, floor.value + DOOR_HEAD)
    )
    walls, objects = split_structural(walls)

    rng = np.random.default_rng(0)
    ray_idx = np.arange(len(pts))
    if len(ray_idx) > MAX_RAYS:
        ray_idx = np.sort(rng.choice(ray_idx, MAX_RAYS, replace=False))
    wall_band = (pts[:, 1] > floor.value + WALL_BAND_BOTTOM) & (pts[:, 1] < top)
    vertical = wall_band & (np.abs(nrm[:, 1]) < 0.3)
    lo = np.percentile(pts[wall_band][:, [0, 2]], 0.5, axis=0) - 0.5
    hi = np.percentile(pts[wall_band][:, [0, 2]], 99.5, axis=0) + 0.5
    grid = carve(
        cams[fidx[ray_idx]][:, [0, 2]],
        pts[ray_idx][:, [0, 2]],
        pts[vertical][:, [0, 2]],
        (lo[0], hi[0], lo[1], hi[1]),
    )

    cx = build_complex(walls, grid)
    if cx is None:
        raise RuntimeError("Too few walls detected to form a room")
    prof = line_profiles(cx)
    lab = segment_cells(cx, prof)
    rooms = build_rooms(cx, lab)
    openings, adjacency = find_openings(cx, lab, rooms, prof, cams[:, [0, 2]])
    wall_mask = wall_cells(cx, lab)
    if not rooms:
        warnings.append("No closed room could be formed from the detected walls.")

    global_height = None
    if ceil is not None:
        global_height = measure(ceil.value - floor.value, float(np.hypot(ceil.sigma, floor.sigma)))
    per_room = room_ceilings(pts, nrm, chunks, floor.value, cx.xs, cx.zs, lab)
    heights: dict[str, dict | None] = {}
    for k, room in enumerate(rooms, start=1):
        if k in per_room:
            c = per_room[k]
            heights[room.id] = {**measure(c.value - floor.value, float(np.hypot(c.sigma, floor.sigma))),
                                "source": "room"}
        elif global_height is not None:
            heights[room.id] = {**global_height, "source": "property"}
        else:
            heights[room.id] = None
    height = global_height

    result = {
        "schema_version": "0.3",
        "pipeline_version": __version__,
        "tier": "lidar",
        "calibrated": False,
        "capture": {
            "path": str(cap.root),
            "frames": cap.n_frames,
            "duration_s": round(cap.duration_s, 2),
            "frames_used": int(len(np.unique(fidx))),
        },
        "frame": {"manhattan_yaw_deg": round(float(np.degrees(theta)), 3), "floor_y": round(floor.value, 4)},
        "rooms": [
            {
                "id": room.id,
                "polygon_m": [[round(x, 4), round(z, 4)] for x, z in room.polygon],
                "floor_area_m2": measure(room.area, room.area_sigma),
                "ceiling_height_m": heights[room.id],
                "walls": [
                    {
                        "id": w.id,
                        "start_m": [round(w.start[0], 4), round(w.start[1], 4)],
                        "end_m": [round(w.end[0], 4), round(w.end[1], 4)],
                        "length_m": measure(w.length, w.sigma),
                        "observed_fraction": round(w.coverage, 3),
                    }
                    for w in room.walls
                ],
            }
            for room in rooms
        ],
        "openings": [
            {
                "id": o.id,
                "kind": o.kind,
                "connects": list(o.rooms),
                "walls": o.walls,
                "start_m": [round(v, 4) for v in o.endpoints()[0]],
                "end_m": [round(v, 4) for v in o.endpoints()[1]],
                "width_m": round(o.width, 3),
                "width_method": "coverage gap, 2 cm bins (jamb refinement pending)",
                "detected_by": o.source,
                "wall_thickness_m": round(abs(o.other_offset - o.offset), 3) if o.other_offset is not None else None,
            }
            for o in openings
        ],
        "adjacency": [
            {"rooms": list(a.rooms), "via": a.via,
             "wall_thickness_m": round(a.thickness, 3) if a.thickness is not None else None}
            for a in adjacency
        ],
        "objects": [
            {
                "id": f"obj{n}",
                "kind": "furniture_face",
                "start_m": [round(v, 4) for v in ((f.offset, f.span[0]) if f.axis == "x" else (f.span[0], f.offset))],
                "end_m": [round(v, 4) for v in ((f.offset, f.span[1]) if f.axis == "x" else (f.span[1], f.offset))],
                "height_m": round(f.top - floor.value, 3),
                "rule": "wall visible above this face",
            }
            for n, f in enumerate(objects, start=1)
        ],
        "warnings": warnings,
        "outputs": {"plan_png": "plan.png", "model_glb": "plan3d.glb"},
        "timing_s": None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{cap.root.name} | LiDAR | " + (
        f"ceiling {height['value']:.3f} m" if height else "ceiling not observed"
    ) + " | intervals: 90%, uncalibrated"
    render_plan(rooms, openings, objects, cx, lab, wall_mask,
                {r: (h["value"] if h else None) for r, h in heights.items()}, floor.value,
                grid, cams[:, [0, 2]], out_dir / "plan.png", title)
    model_height = height["value"] if height else ASSUMED_CEILING
    write_glb(out_dir / "plan3d.glb",
              build_model(cx, lab, rooms, openings, model_height, wall_mask, objects, floor.value))
    result["model_3d"] = {
        "wall_height_m": model_height,
        "wall_height_source": "measured" if height else "assumed",
        "door_height_m": DOOR_HEIGHT,
        "door_height_source": "assumed",
    }
    if raw_scan:
        # Same frame as plan3d.glb (floor at y = 0) so the two overlay in a viewer.
        n = export_raw_scan(out_dir / "scan_raw.ply", pts - np.array([0.0, floor.value, 0.0]), model_height)
        result["outputs"]["raw_scan_ply"] = "scan_raw.ply"
        result["outputs"]["raw_scan_points"] = n
    result["timing_s"] = round(time.perf_counter() - t0, 2)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
