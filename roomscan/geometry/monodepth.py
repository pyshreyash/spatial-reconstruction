"""Video tier: dense relative depth from Depth Anything V2 Small (ONNX, Apache-2.0), made metric per keyframe
from ARKit-scaled anchors (stereo + triangulated features). The model supplies shape only, never scale."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

MODEL_FILE = "depth-anything-v2-small/onnx/model.onnx"
MODEL_SHA256 = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"
MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
INPUT_SIZE = (504, 378)    # multiple of 14, 4:3 like the Stray RGB
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FIT_TOL = 0.05             # relative depth residual for an anchor to count as inlier
MIN_ANCHORS = 30
MAX_FIT_RESIDUAL = 0.04    # median relative residual of inliers
MIN_INLIER_FRAC = 0.6
EXTRAPOLATE = 1.3          # trust depth up to this factor beyond the 95th-percentile anchor depth

ROOT = Path(__file__).resolve().parents[2]


def find_model() -> Path | None:
    for base in (ROOT / "models", ROOT):
        if (base / MODEL_FILE).is_file():
            return base / MODEL_FILE
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class MonoDepth:
    def __init__(self, path: Path):
        import onnxruntime as ort

        if sha256(path) != MODEL_SHA256:
            raise ValueError(f"{path} does not match the pinned Depth Anything V2 Small weights (SHA-256)")
        self.path = path
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """Relative inverse depth (affine-invariant) at INPUT_SIZE resolution."""
        rgb = cv2.cvtColor(cv2.resize(bgr, INPUT_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        return self.session.run(None, {self.input: x})[0][0]


def load_mono() -> MonoDepth | None:
    path = find_model()
    return MonoDepth(path) if path else None


def sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return map_coordinates(img, [uv[:, 1], uv[:, 0]], order=1, mode="nearest")


def fit_inverse_affine(d: np.ndarray, z: np.ndarray, seed: int = 0) -> tuple[float, float, float, float] | None:
    """Fit 1/z = s*d + t (RANSAC on 2 anchors, then least squares on inliers).
    Returns s, t, inlier fraction, median relative residual of inliers; None if the fit is unsupported."""
    if len(d) < MIN_ANCHORS:
        return None
    rng = np.random.default_rng(seed)
    y = 1.0 / z
    best, best_n = None, 0
    for _ in range(200):
        i, j = rng.choice(len(d), 2, replace=False)
        if abs(d[i] - d[j]) < 1e-6:
            continue
        s = (y[i] - y[j]) / (d[i] - d[j])
        t = y[i] - s * d[i]
        if s <= 0:
            continue
        pred = s * d + t
        inl = (pred > 0) & (np.abs(1.0 / np.maximum(pred, 1e-6) - z) < FIT_TOL * z)
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None or best_n < MIN_ANCHORS:
        return None
    A = np.c_[d[best], np.ones(best_n)]
    s, t = np.linalg.lstsq(A * z[best, None], y[best] * z[best], rcond=None)[0]  # relative-error weighting
    if s <= 0:
        return None
    pred = s * d + t
    ok = pred > 0
    rel = np.full(len(d), np.inf)
    rel[ok] = np.abs(1.0 / pred[ok] - z[ok]) / z[ok]
    inl = rel < FIT_TOL
    if inl.sum() < MIN_ANCHORS:
        return None
    return float(s), float(t), float(inl.mean()), float(np.median(rel[inl]))


def metric_depth(d: np.ndarray, s: float, t: float, z_anchor: np.ndarray, z_floor: float = 0.3):
    """Metric depth map and validity: only where the affine fit is positive and within the anchor range."""
    inv = s * d + t
    z_max = min(4.0, EXTRAPOLATE * float(np.percentile(z_anchor, 95)))
    Z = np.where(inv > 0, 1.0 / np.maximum(inv, 1e-6), 0.0).astype(np.float32)
    return Z, (Z > z_floor) & (Z < z_max)
