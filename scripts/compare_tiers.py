"""Compare two result.json files of the same capture (e.g. video tier vs LiDAR tier).

    python scripts/compare_tiers.py out/scan3 out/scan3_video

The test plan is rotated into the reference Manhattan frame (both share the ARKit world origin).
Rooms are matched by polygon overlap; walls by axis, offset (< 0.3 m) and span overlap."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

RES = 0.02


def load(d: str) -> dict:
    return json.loads((Path(d) / "result.json").read_text(encoding="utf-8"))


def to_ref(xy: np.ndarray, dtheta: float) -> np.ndarray:
    c, s = np.cos(dtheta), np.sin(dtheta)
    return np.c_[c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]]


def walls(result: dict, dtheta: float = 0.0) -> list[dict]:
    out = []
    for room in result["rooms"]:
        for w in room["walls"]:
            a, b = to_ref(np.array([w["start_m"], w["end_m"]], float), dtheta)
            axis = "x" if abs(b[0] - a[0]) < abs(b[1] - a[1]) else "z"  # "x": plane x = const
            off = (a[0] + b[0]) / 2 if axis == "x" else (a[1] + b[1]) / 2
            span = sorted([a[1], b[1]] if axis == "x" else [a[0], b[0]])
            out.append({"room": room["id"], "id": w["id"], "axis": axis, "offset": off, "span": span,
                        "length": w["length_m"]["value"], "sigma": w["length_m"]["sigma"]})
    return out


def main(ref_dir: str, test_dir: str) -> None:
    ref, test = load(ref_dir), load(test_dir)
    d = np.radians(ref["frame"]["manhattan_yaw_deg"] - test["frame"]["manhattan_yaw_deg"])
    d = (d + np.pi / 4) % (np.pi / 2) - np.pi / 4  # Manhattan yaw is defined modulo 90 deg
    polys_r = {r["id"]: np.array(r["polygon_m"], float) for r in ref["rooms"]}
    polys_t = {r["id"]: to_ref(np.array(r["polygon_m"], float), d) for r in test["rooms"]}
    allp = np.concatenate(list(polys_r.values()) + list(polys_t.values()))
    lo, hi = allp.min(0) - 0.1, allp.max(0) + 0.1
    gx, gz = np.meshgrid(np.arange(lo[0], hi[0], RES), np.arange(lo[1], hi[1], RES))
    grid = np.c_[gx.ravel(), gz.ravel()]
    mask = lambda p: MplPath(p).contains_points(grid)  # noqa: E731
    mr = {k: mask(p) for k, p in polys_r.items()}
    mt = {k: mask(p) for k, p in polys_t.items()}
    cell = RES * RES

    print(f"{ref_dir} ({ref['tier']}) vs {test_dir} ({test['tier']}): yaw difference "
          f"{np.degrees(d):+.2f} deg, floor_y {ref['frame']['floor_y']:+.3f} vs {test['frame']['floor_y']:+.3f}")
    ur, ut = np.any(list(mr.values()), 0), np.any(list(mt.values()), 0)
    fr, ft = ur.sum() * cell, ut.sum() * cell
    print(f"footprint: {fr:.2f} vs {ft:.2f} m2 ({(ft - fr) / fr:+.1%}), IoU {(ur & ut).sum() / (ur | ut).sum():.2f}")
    print(f"rooms: {len(mr)} vs {len(mt)}, openings: {len(ref['openings'])} vs {len(test['openings'])}")

    test_rooms = {r["id"]: r for r in test["rooms"]}
    print("\nroom   ref m2   match   test m2 (+-1 sigma)   err     IoU   ceiling ref / test")
    for r in ref["rooms"]:
        best = max(mt, key=lambda k: (mr[r["id"]] & mt[k]).sum(), default=None)
        inter = (mr[r["id"]] & mt[best]).sum() if best else 0
        if not inter:
            print(f"{r['id']:<6} {r['floor_area_m2']['value']:7.2f}   (none)")
            continue
        t = test_rooms[best]
        ar, at, sa = r["floor_area_m2"]["value"], t["floor_area_m2"]["value"], t["floor_area_m2"]["sigma"]
        iou = inter / (mr[r["id"]] | mt[best]).sum()
        hr, ht = r["ceiling_height_m"], t["ceiling_height_m"]
        print(f"{r['id']:<6} {ar:7.2f}   {best:<6}  {at:7.2f} (+-{sa:.2f})      {(at - ar) / ar:+6.1%}  {iou:.2f}"
              f"   {hr['value'] if hr else '-'} / {ht['value'] if ht else '-'}")

    wr, wt = walls(ref), walls(test, d)
    rows = []
    for a in wr:
        cands = [b for b in wt if b["axis"] == a["axis"] and abs(b["offset"] - a["offset"]) < 0.3
                 and min(a["span"][1], b["span"][1]) - max(a["span"][0], b["span"][0]) > 0.5 * a["length"]]
        if cands:
            b = min(cands, key=lambda b: abs(b["offset"] - a["offset"]) + abs(b["length"] - a["length"]))
            rows.append((a, b))
    print(f"\nwalls matched: {len(rows)} of {len(wr)} reference walls (>= 0.5 m: "
          f"{sum(a['length'] >= 0.5 for a, _ in rows)} of {sum(a['length'] >= 0.5 for a in wr)})")
    if rows:
        off = np.array([b["offset"] - a["offset"] for a, b in rows])
        dl = np.array([b["length"] - a["length"] for a, b in rows])
        rel = np.array([(b["length"] - a["length"]) / a["length"] for a, b in rows if a["length"] >= 0.5])
        z = np.array([(b["length"] - a["length"]) / b["sigma"] for a, b in rows])
        print(f"  plane offset error: median |d| {np.median(np.abs(off)) * 100:.1f} cm, p90 "
              f"{np.percentile(np.abs(off), 90) * 100:.1f} cm")
        print(f"  length error: median |d| {np.median(np.abs(dl)) * 100:.1f} cm; walls >= 0.5 m: median |rel| "
              f"{np.median(np.abs(rel)):.1%}, within 3 %: {np.mean(np.abs(rel) <= 0.03):.0%}, within 8 %: "
              f"{np.mean(np.abs(rel) <= 0.08):.0%}")
        print(f"  reference inside test ci90: {np.mean(np.abs(z) <= 1.645):.0%} (nominal 90 %, uncalibrated)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
