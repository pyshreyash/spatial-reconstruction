"""Video tier: semantic labels from SegFormer-B2 (ADE20K, ONNX). Labels only select which points and lines may
measure walls, floor and ceiling; they never set a distance (METHOD_VIDEO.md §V9).

Licence: NVIDIA SegFormer weights (non-commercial research licence) trained on ADE20K; disclosed."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from roomscan.geometry.monodepth import MEAN, STD, sha256

MODEL_FILE = "segformer-b2-finetuned-ade-512-512/onnx/model.onnx"
MODEL_SHA256 = "819c15e6af8c4de3359c1de7ab0a17d0dde495df1d16f8908a7163f8038e0fa0"
MODEL_URL = "https://huggingface.co/Xenova/segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model.onnx"
INPUT_SIZE = (384, 288)    # ~0.19 s per frame on CPU; output logits at 1/4
LABEL_SIZE = (160, 120)    # label map: 4 WORK_WIDTH pixels per label pixel

OTHER, WALL, FLOOR, CEILING, DOOR, WINDOW, FURNITURE = range(7)
# ADE20K ids. Flat things hung on walls (painting, mirror, poster, board, clock) lie on the wall plane.
GROUPS = {WALL: (0, 22, 27, 42, 100, 144, 148), FLOOR: (3, 28), CEILING: (5,), DOOR: (14, 58), WINDOW: (8,),
          FURNITURE: (7, 10, 12, 15, 17, 19, 23, 24, 30, 31, 33, 35, 36, 37, 39, 41, 44, 45, 47, 50, 57, 62, 64,
                      65, 69, 70, 71, 73, 74, 75, 89, 97, 99, 107, 110, 118, 124, 129, 131, 141, 143)}
WALLISH = (WALL, DOOR, WINDOW)
SURFACES = (WALL, DOOR, WINDOW, FLOOR, CEILING)

ROOT = Path(__file__).resolve().parents[2]
_LUT = np.zeros(256, np.uint8)
for _g, _ids in GROUPS.items():
    _LUT[list(_ids)] = _g


def find_model() -> Path | None:
    for base in (ROOT / "models", ROOT):
        if (base / MODEL_FILE).is_file():
            return base / MODEL_FILE
    return None


class Segmenter:
    def __init__(self, path: Path):
        import onnxruntime as ort

        if sha256(path) != MODEL_SHA256:
            raise ValueError(f"{path} does not match the pinned SegFormer-B2 ADE20K weights (SHA-256)")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def predict(self, bgr: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
        """Group label map (LABEL_SIZE, uint8). `up`: world-up direction in image coordinates (u right,
        v down); the frame is rotated upright by quarter turns before inference and the labels rotated back."""
        k = upright_turns(up) if up is not None else 0
        img = np.ascontiguousarray(np.rot90(bgr, k))
        size = INPUT_SIZE if k % 2 == 0 else INPUT_SIZE[::-1]
        out = LABEL_SIZE if k % 2 == 0 else LABEL_SIZE[::-1]
        rgb = cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {"pixel_values": x})[0][0]
        c, h, w = logits.shape
        up_l = np.einsum("Hh,chw,Ww->HWc", _interp(out[1], h), logits, _interp(out[0], w), optimize=True)
        return np.ascontiguousarray(np.rot90(_LUT[up_l.argmax(2).astype(np.uint8)], -k))


def upright_turns(up: np.ndarray) -> int:
    """Quarter turns (np.rot90, counter-clockwise) that bring image direction `up` to the top."""
    ux, uy = up[0], up[1]
    if -uy >= abs(ux):
        return 0
    if ux >= abs(uy):
        return 1
    if uy >= abs(ux):
        return 2
    return 3


def _interp(n_out: int, n_in: int) -> np.ndarray:
    """(n_out, n_in) linear interpolation matrix, pixel centres aligned."""
    x = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
    i0 = np.clip(np.floor(x).astype(int), 0, n_in - 1)
    i1 = np.clip(i0 + 1, 0, n_in - 1)
    f = np.clip(x - i0, 0, 1)
    M = np.zeros((n_out, n_in), np.float32)
    M[np.arange(n_out), i0] += 1 - f
    M[np.arange(n_out), i1] += f
    return M


def load_segmenter() -> Segmenter | None:
    path = find_model()
    return Segmenter(path) if path else None


def structural(seg: np.ndarray, labels: np.ndarray, work_width: int, offset: float = 2.0) -> np.ndarray:
    """Segments (x1,y1,x2,y2 at work_width) that separate two surfaces, at least one a wall/door/window:
    wall-floor and wall-ceiling junctions, wall corners, door and window frames. Furniture edges and
    floor/ceiling texture are rejected."""
    if not len(seg):
        return np.zeros(0, bool)
    s = labels.shape[1] / work_width
    a, b = seg[:, :2] * s, seg[:, 2:] * s
    d = b - a
    nrm = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    h, w = labels.shape
    sides = []
    for sign in (1, -1):
        votes = []
        for f in (0.25, 0.5, 0.75):
            q = a + f * d + sign * offset * nrm
            u = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
            v = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
            votes.append(labels[v, u])
        votes = np.stack(votes, 1)
        major = np.array([np.bincount(r, minlength=7).argmax() for r in votes])
        sides.append(major)
    l1, l2 = sides
    surf = np.isin(l1, SURFACES) & np.isin(l2, SURFACES)
    wallish = np.isin(l1, WALLISH) | np.isin(l2, WALLISH)
    return surf & wallish
