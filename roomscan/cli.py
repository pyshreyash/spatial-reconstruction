"""Command line: one command per capture."""
from __future__ import annotations

import argparse
from pathlib import Path

from roomscan.io.stray import find_stray_root, find_video_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="roomscan")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Process one capture folder")
    run.add_argument("capture", type=Path)
    run.add_argument("--out", type=Path, default=None, help="Output folder (default: out/<capture name>)")
    run.add_argument("--frame-stride", type=int, default=5)
    run.add_argument("--pixel-stride", type=int, default=2)
    run.add_argument("--no-raw-scan", action="store_true", help="Skip the raw point-cloud export (scan_raw.ply)")
    run.add_argument("--tier", choices=["auto", "lidar", "video"], default="auto",
                     help="video: use only rgb.mp4 + ARKit poses, ignoring any depth in the capture")
    run.add_argument("--video-depth", choices=["mono", "stereo"], default="mono",
                     help="video tier: learned relative depth scaled by ARKit anchors, or classical stereo only")
    run.add_argument("--no-lines", action="store_true",
                     help="video tier ablation: keep depth-only wall planes (no structural line re-measurement)")
    run.add_argument("--no-semantics", action="store_true",
                     help="video tier ablation: no SegFormer labels for points and lines")
    args = parser.parse_args(argv)

    out = args.out or Path("out") / args.capture.name
    tier = args.tier
    if tier == "auto":
        tier = "lidar" if find_stray_root(args.capture) else "video" if find_video_root(args.capture) else None
    if tier == "lidar" and find_stray_root(args.capture) is not None:
        from roomscan.lidar import run_lidar

        result = run_lidar(args.capture, out, args.frame_stride, args.pixel_stride, not args.no_raw_scan)
    elif tier == "video" and find_video_root(args.capture) is not None:
        from roomscan.video import run_video

        result = run_video(args.capture, out, raw_scan=not args.no_raw_scan, use_mono=args.video_depth == "mono",
                           use_lines=not args.no_lines, use_semantics=not args.no_semantics)
    else:
        raise SystemExit(f"Unrecognised capture at {args.capture} for tier '{args.tier}'")

    for room in result["rooms"]:
        area = room["floor_area_m2"]
        print(f"{room['id']}: area {area['value']:.2f} m2, {len(room['walls'])} walls")
    for w in result["warnings"]:
        print(f"WARNING: {w}")
    print(f"Wrote {out / 'result.json'} and {out / 'plan.png'} in {result['timing_s']} s")
