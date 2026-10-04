"""Validation only: compare video-tier depth against the LiDAR depth of the same Stray frames.

    python scripts/eval_video_depth.py single_room [mono|stereo]

The pipeline never reads depth/ for the video tier; this script does, to measure the error."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.geometry.monodepth import load_mono  # noqa: E402
from roomscan.geometry.stereo import KEY_STRIDE, collect_video_points  # noqa: E402
from roomscan.io.stray import load_posed_video, load_stray  # noqa: E402


def main(capture: str, source: str = "mono", every: int = 3) -> None:
    vid = load_posed_video(Path(capture))
    lidar = load_stray(Path(capture))
    assert lidar.n_frames == vid.n_frames, "depth frames must map 1:1 to video frames"
    keep = {i for i in range(vid.n_frames) if (i // KEY_STRIDE) % every == 0}
    mono = load_mono() if source == "mono" else None
    t0 = time.perf_counter()
    pts, _, _, _, stats, maps, _, _ = collect_video_points(vid, mono=mono, keep_maps=keep)
    print(f"{capture} [{stats['depth_source']}]: time offset {stats['time_offset_s'] * 1000:+.1f} ms; "
          f"{stats['keyframes']} keyframes, {stats['keyframes_fused']} stereo-fused; pairs aligned "
          f"{stats['pairs_aligned']}/{stats['pairs_tried']}; {len(pts):,} points, {time.perf_counter() - t0:.0f} s")
    if mono:
        print(f"  mono keyframes {stats['keyframes_mono']} ({stats['keyframes_mono_direct']} direct), anchors median "
              f"{stats['anchors_median']}, fit inliers {stats['fit_inlier_frac_median']:.0%}, "
              f"fit residual {stats['fit_residual_median']:.2%}")

    rel, zl_all, n_lidar = [], [], 0
    for k, (zv, okv) in sorted(maps.items()):
        zl = lidar.depth(k)
        okl = lidar.confidence(k) >= 2
        h, w = zl.shape
        H, W = zv.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        U = np.clip(np.round((u + 0.5) * W / w - 0.5).astype(int), 0, W - 1)
        V = np.clip(np.round((v + 0.5) * H / h - 0.5).astype(int), 0, H - 1)
        in_range = okl & (zl > 0.5) & (zl < 4.0)
        m = in_range & okv[V, U]
        n_lidar += int(in_range.sum())
        rel.append((zv[V, U][m] - zl[m]) / zl[m])
        zl_all.append(zl[m])
    rel, zl_all = np.concatenate(rel), np.concatenate(zl_all)
    a = np.abs(rel)
    print(f"compared {len(maps)} keyframes, {len(rel):,} pixels, coverage of LiDAR pixels {len(rel) / n_lidar:.1%}")
    print(f"  bias (median rel) {np.median(rel):+.2%}   median |rel| {np.median(a):.2%}   "
          f"p90 |rel| {np.percentile(a, 90):.2%}   within 5 %: {np.mean(a < 0.05):.1%}")
    for lo, hi in [(0.5, 1.5), (1.5, 2.5), (2.5, 4.0)]:
        s = (zl_all >= lo) & (zl_all < hi)
        if s.any():
            print(f"  {lo:.1f}-{hi:.1f} m: n={s.sum():>8,}  bias {np.median(rel[s]):+.2%}  "
                  f"median |rel| {np.median(a[s]):.2%}  p90 {np.percentile(a[s], 90):.2%}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
