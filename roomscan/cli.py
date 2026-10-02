"""Command line: one command per capture."""
from __future__ import annotations

import argparse
from pathlib import Path

from roomscan.io.stray import find_stray_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="roomscan")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Process one capture folder")
    run.add_argument("capture", type=Path)
    run.add_argument("--out", type=Path, default=None, help="Output folder (default: out/<capture name>)")
    run.add_argument("--frame-stride", type=int, default=5)
    run.add_argument("--pixel-stride", type=int, default=2)
    run.add_argument("--no-raw-scan", action="store_true", help="Skip the raw point-cloud export (scan_raw.ply)")
    args = parser.parse_args(argv)

    out = args.out or Path("out") / args.capture.name
    if find_stray_root(args.capture) is not None:
        from roomscan.lidar import run_lidar

        result = run_lidar(args.capture, out, args.frame_stride, args.pixel_stride, not args.no_raw_scan)
    else:
        raise SystemExit(f"Unrecognised capture at {args.capture}: only LiDAR (Stray Scanner) is supported so far")

    for room in result["rooms"]:
        area = room["floor_area_m2"]
        print(f"{room['id']}: area {area['value']:.2f} m2, {len(room['walls'])} walls")
    for w in result["warnings"]:
        print(f"WARNING: {w}")
    print(f"Wrote {out / 'result.json'} and {out / 'plan.png'} in {result['timing_s']} s")
