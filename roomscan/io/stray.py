"""Loader for Stray Scanner exports (rgb.mp4, depth/, confidence/, odometry.csv, camera_matrix.csv)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Stray Scanner records RGB at 1920x1440 and its intrinsics refer to that resolution.
RGB_WIDTH = 1920


@dataclass
class StrayCapture:
    root: Path
    frame_ids: np.ndarray   # (N,) file index of depth/confidence PNGs
    timestamps: np.ndarray  # (N,) seconds
    positions: np.ndarray   # (N,3) camera centre in ARKit world (+y up)
    quats: np.ndarray       # (N,4) camera-to-world rotation, xyzw
    intrinsics: np.ndarray  # (N,4) fx, fy, cx, cy at RGB resolution

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres for sample i."""
        path = self.root / "depth" / f"{int(self.frame_ids[i]):06d}.png"
        return np.asarray(Image.open(path), dtype=np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        path = self.root / "confidence" / f"{int(self.frame_ids[i]):06d}.png"
        return np.asarray(Image.open(path))


@dataclass
class PosedVideo:
    """Video tier input: RGB video plus ARKit poses and intrinsics. Row i of the arrays is video frame i."""
    root: Path
    video: Path
    timestamps: np.ndarray  # (N,) seconds
    positions: np.ndarray   # (N,3) camera centre in ARKit world (+y up)
    quats: np.ndarray       # (N,4) camera-to-world rotation, xyzw (OpenCV camera axes)
    intrinsics: np.ndarray  # (N,4) fx, fy, cx, cy at RGB resolution

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])


def is_stray(path: Path) -> bool:
    return (path / "odometry.csv").is_file() and (path / "depth").is_dir()


def is_posed_video(path: Path) -> bool:
    return (path / "odometry.csv").is_file() and (path / "rgb.mp4").is_file()


def _find_root(path: Path, test) -> Path | None:
    path = Path(path)
    if test(path):
        return path
    if path.is_dir():
        subs = [p for p in path.iterdir() if p.is_dir() and test(p)]
        if len(subs) == 1:
            return subs[0]
    return None


def find_stray_root(path: Path) -> Path | None:
    """Accept either the capture folder itself or a parent containing exactly one capture."""
    return _find_root(path, is_stray)


def find_video_root(path: Path) -> Path | None:
    return _find_root(path, is_posed_video)


def _read_odometry(root: Path) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """(rows, column index, per-row intrinsics) from odometry.csv and camera_matrix.csv."""
    with open(root / "odometry.csv", encoding="utf-8") as f:
        header = [h.strip() for h in f.readline().split(",")]
    data = np.atleast_2d(np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1))
    col = {name: i for i, name in enumerate(header)}

    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    default_intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    if all(n in col for n in ("fx", "fy", "cx", "cy")):
        intr = data[:, [col[n] for n in ("fx", "fy", "cx", "cy")]]
        bad = ~np.isfinite(intr).all(axis=1)
        intr[bad] = default_intr
    else:
        intr = np.tile(default_intr, (len(data), 1))
    return data, col, intr


def load_posed_video(path: Path) -> PosedVideo:
    """Reads only rgb.mp4, odometry.csv and camera_matrix.csv: depth/, confidence/ and imu.csv are ignored."""
    root = find_video_root(path)
    if root is None:
        raise ValueError(f"No rgb.mp4 + odometry.csv capture found at {path}")
    data, col, intr = _read_odometry(root)

    def cols(*names: str) -> np.ndarray:
        return data[:, [col[n] for n in names]]

    return PosedVideo(
        root=root,
        video=root / "rgb.mp4",
        timestamps=cols("timestamp")[:, 0],
        positions=cols("x", "y", "z"),
        quats=cols("qx", "qy", "qz", "qw"),
        intrinsics=intr,
    )


def load_stray(path: Path) -> StrayCapture:
    root = find_stray_root(path)
    if root is None:
        raise ValueError(f"No Stray Scanner capture found at {path}")
    data, col, intr = _read_odometry(root)

    def cols(*names: str) -> np.ndarray:
        return data[:, [col[n] for n in names]]

    frame_ids = cols("frame")[:, 0].astype(int)
    exists = np.array([(root / "depth" / f"{i:06d}.png").is_file() for i in frame_ids])
    data, intr, frame_ids = data[exists], intr[exists], frame_ids[exists]

    return StrayCapture(
        root=root,
        frame_ids=frame_ids,
        timestamps=cols("timestamp")[:, 0],
        positions=cols("x", "y", "z"),
        quats=cols("qx", "qy", "qz", "qw"),
        intrinsics=intr,
    )
