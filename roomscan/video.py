"""Video tier pipeline: RGB video + ARKit poses/intrinsics (no depth) -> same outputs as the LiDAR tier.

Depth (learned shape, ARKit scale) proposes wall planes; structural lines triangulated from the ARKit poses
re-measure them (METHOD_VIDEO.md)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from roomscan.geometry.lines import observations, occupancy, refine_walls, triangulate
from roomscan.geometry.monodepth import MODEL_FILE, load_mono
from roomscan.geometry.semantics import MODEL_FILE as SEG_MODEL_FILE
from roomscan.geometry.semantics import load_segmenter
from roomscan.geometry.stereo import KEY_STRIDE, WORK_WIDTH, Poses, collect_video_points
from roomscan.io.stray import RGB_WIDTH, load_posed_video
from roomscan.lidar import reconstruct

MODEL_LICENSE = "Apache-2.0"
SEG_MODEL = "SegFormer-B2 ADE20K (NVIDIA non-commercial licence; ADE20K research use)"

# ARKit (VIO) scale error; per-wall position errors are carried by each wall's own sigma.
VIDEO_REL_SIGMA = 0.01
# Per-wall plane sigma floors: 90 % of video wall offsets were within 1.645 * sigma of LiDAR on scan3
# (scripts/eval_video_walls.py, semantic labels on): 11.4 cm for line-measured walls, 16.5 cm for depth-only.
# Fitted on dev data against LiDAR, not tape: provisional.
LINE_WALL_SIGMA = 0.115
DEPTH_WALL_SIGMA = 0.165
# One keyframe's wrong depth (194 of 222 evidence points) cut the scan3 hall in two. Room separation only
# trusts wall stretches seen from >= 2 keyframes with none supplying > 70 % (walls.py MAX_VIEW_SHARE).
MIN_WALL_VIEWS = 2
INPUTS_USED = ["rgb.mp4", "odometry.csv (ARKit poses + intrinsics)", "camera_matrix.csv"]


def run_video(capture: Path, out_dir: Path, pixel_stride: int = 4, raw_scan: bool = True, use_mono: bool = True,
              use_lines: bool = True, use_semantics: bool = True) -> dict:
    t0 = time.perf_counter()
    vid = load_posed_video(capture)
    mono = load_mono() if use_mono else None
    if use_mono and mono is None:
        raise SystemExit(f"Depth model not found: run `python scripts/fetch_models.py` (expects {MODEL_FILE})")
    segmenter = load_segmenter() if use_semantics and mono else None
    if use_semantics and mono and segmenter is None:
        raise SystemExit(f"Segmentation model not found: run `python scripts/fetch_models.py` (expects {SEG_MODEL_FILE})")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("mono" if mono else "stereo") + ("_sem" if segmenter else "")
    cache = out_dir / f".video_points_{tag}_k{KEY_STRIDE}_p{pixel_stride}.npz"
    if cache.is_file():
        c = np.load(cache)
        pts, nrm, fidx = c["pts"], c["nrm"], c["fidx"]
        stats = json.loads(str(c["stats"]))
        seg_k, seg, seg_z, seg_ok = c["seg_k"], c["seg"], c["seg_z"], c["seg_ok"]
        segments = {int(k): (seg[seg_k == k], seg_z[seg_k == k], seg_ok[seg_k == k]) for k in np.unique(seg_k)}
        labels = c["labels"] if "labels" in c.files else None
        poses = Poses.from_video(vid).shifted(stats["time_offset_s"])
    else:
        pts, nrm, fidx, poses, stats, _, segs, labels = collect_video_points(
            vid, mono=mono, pixel_stride=pixel_stride, segmenter=segmenter)
        segments = {k: tuple(v) for k, v in segs.items()}
        seg_k = np.concatenate([np.full(len(v[0]), k) for k, v in segments.items()])
        arrays = {"seg_k": seg_k, "seg": np.concatenate([v[0] for v in segments.values()]),
                  "seg_z": np.concatenate([v[1] for v in segments.values()]),
                  "seg_ok": np.concatenate([v[2] for v in segments.values()])}
        if labels is not None:
            arrays["labels"] = labels
        np.savez(cache, pts=pts, nrm=nrm, fidx=fidx, stats=json.dumps(stats), **arrays)

    scale = WORK_WIDTH / RGB_WIDTH
    use_lines = use_lines and labels is not None  # unlabelled lines (furniture edges) made walls worse
    structural = {k: (s[ok], z[ok]) for k, (s, z, ok) in segments.items()}
    obs = observations(structural, poses.R, poses.C, {k: poses.K(k, scale) for k in structural}) if use_lines else None

    def refiner(walls, Rm, floor_y, ceil_y):
        info = {"lines": use_lines}
        if use_lines:
            lines = triangulate(obs, Rm, occupancy(pts @ Rm.T))
            walls, info = refine_walls(walls, lines, floor_y, ceil_y)
            info["lines_3d"] = {a: sum(ln.axis == a for ln in lines) for a in "xyz"}
        for w in walls:
            floor = LINE_WALL_SIGMA if getattr(w, "measured_by", "") == "lines" else DEPTH_WALL_SIGMA
            w.sigma = float(max(w.sigma, floor))
        info["wall_sigma_floor_m"] = {"lines": LINE_WALL_SIGMA, "depth": DEPTH_WALL_SIGMA}
        return walls, info

    info = {"path": str(vid.root), "frames": vid.n_frames, "duration_s": round(vid.duration_s, 2),
            "inputs_used": INPUTS_USED}
    stereo = {k: v for k, v in stats.items() if k != "time_offset_curve"}
    stereo["points"] = int(len(pts))
    method = (f"Depth Anything V2 Small relative depth ({MODEL_LICENSE}), made metric per keyframe by fitting "
              "1/z = s*d + t to stereo + triangulated anchors from ARKit poses" if mono else
              "two-view rectified SGBM on ARKit-posed keyframes, two-pair agreement")
    if use_lines:
        method += "; wall planes re-measured from triangulated structural lines"
    if labels is not None:
        method += f"; {SEG_MODEL} labels select floor, ceiling, wall points and structural lines"
    extra = {"video": {"method": method, "relative_sigma_floor": VIDEO_REL_SIGMA, **stereo}}
    return reconstruct("video", info, pts, nrm, fidx, poses.C, out_dir, raw_scan, t0,
                       rel_sigma=VIDEO_REL_SIGMA, extra=extra, wall_refiner=refiner, labels=labels,
                       min_wall_views=MIN_WALL_VIEWS)
