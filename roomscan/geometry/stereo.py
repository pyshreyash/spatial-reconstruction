"""Video tier: dense depth from ARKit-posed RGB frames by rectified two-view stereo (METHOD_VIDEO.md).

1. Pose-to-frame time offset is estimated from the video itself (epipolar error of feature matches).
2. Each keyframe is matched against two partner frames. ARKit relative poses leave ~1 px vertical
   misalignment after rectification, too much for block matching, so feature matches fix the vertical
   direction only; horizontal disparity (hence metric scale) stays ARKit's, because a two-view essential
   matrix cannot resolve it at these baselines.
3. A pixel is kept only when both pairs agree on its depth, rejecting most false matches."""
from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from roomscan.geometry.backproject import organised_points, quat_to_rot
from roomscan.geometry.lines import detect_segments
from roomscan.io.stray import RGB_WIDTH, PosedVideo

WORK_WIDTH = 640           # stereo resolution (1/3 of Stray RGB)
KEY_STRIDE = 10            # one keyframe per 10 frames (~6 per second at 60 Hz)
PARTNER_WINDOW = 90        # frames searched on each side for partners
BASELINE = (0.10, 0.45)    # m
TARGET_BASELINE = 0.25     # m: ~1 px ARKit yaw noise is ~2.5 % of depth at 3 m
MAX_AXIS_ANGLE = 20.0      # deg between optical axes; larger changes break block matching
MAX_FORWARD = 0.7          # |cos| between baseline and optical axis; beyond this the epipole is in view
NUM_DISP = 160
AGREE_REL, AGREE_ABS = 0.03, 0.01  # partner depths must agree within 3 % + 1 cm
Z_RANGE = (0.8, 4.0)       # nearer points exceed NUM_DISP at the target baseline
MIN_MATCHES = 20           # feature matches needed to fix vertical alignment
MAX_VERTICAL_RESIDUAL = 0.6  # px, median after the fix
NORMAL_SMOOTH = 9          # px box filter on the point map before normals
NORMAL_STEP = 4            # px central-difference half-width for normals
OFFSET_SEARCH = 0.05       # s, +- range for the pose-to-frame time offset
OFFSET_FRAMES = 3000       # frames decoded to estimate it
ANCHOR_WINDOW = 120        # frames searched on each side for triangulation neighbours
ANCHOR_BASELINE = (0.08, 0.8)
ANCHOR_MAX_ANGLE = 35.0    # deg; triangulation needs overlap, not rectifiable geometry
ANCHOR_REPROJ = 1.5        # px at WORK_WIDTH
ANCHOR_MIN_PARALLAX = 2.0  # deg between the two rays
MONO_PIXEL_STRIDE = 4      # at the model's 504x378 output
MONO_Z_RANGE = (0.3, 4.0)  # same as LiDAR
PROPAGATE_WINDOW = 900     # frames (~15 s) within which keyframes share anchors
DENSE_ANCHOR_SUBSAMPLE = 4  # of a directly anchored keyframe's points, lent to other keyframes
PROPAGATED_MIN_INLIERS = 0.5  # borrowed anchors include points hidden from this view
MONO_EVERY = 1             # every 2nd keyframe halves runtime but merged rooms on scan3 (footprint -5.4 -> -8.9 %)


@dataclass
class Poses:
    t: np.ndarray      # (N,) s
    R: np.ndarray      # (N,3,3) camera-to-world, OpenCV camera axes
    C: np.ndarray      # (N,3) camera centres
    intr: np.ndarray   # (N,4) fx, fy, cx, cy at RGB resolution

    @classmethod
    def from_video(cls, vid: PosedVideo) -> "Poses":
        R = np.stack([quat_to_rot(q) for q in vid.quats])
        return cls(vid.timestamps, R, vid.positions.astype(float), vid.intrinsics)

    def shifted(self, delay: float) -> "Poses":
        """Poses evaluated at t + delay (frame i was exposed `delay` s after its pose timestamp)."""
        if delay == 0 or np.any(np.diff(self.t) <= 0):
            return self
        tt = np.clip(self.t + delay, self.t[0], self.t[-1])
        R = Slerp(self.t, Rotation.from_matrix(self.R))(tt).as_matrix()
        C = np.stack([np.interp(tt, self.t, self.C[:, c]) for c in range(3)], 1)
        return Poses(self.t, R, C, self.intr)

    def K(self, i: int, scale: float) -> np.ndarray:
        fx, fy, cx, cy = self.intr[i]
        return np.array([[fx * scale, 0, (cx + 0.5) * scale - 0.5],
                         [0, fy * scale, (cy + 0.5) * scale - 0.5],
                         [0, 0, 1.0]])


def motion_blur_score(p: Poses) -> np.ndarray:
    """Angular speed (rad/s) + linear speed / 2 m: a pose-only proxy for motion blur."""
    n = len(p.t)
    prev, nxt = np.clip(np.arange(n) - 1, 0, n - 1), np.clip(np.arange(n) + 1, 0, n - 1)
    dt = np.maximum(p.t[nxt] - p.t[prev], 1e-3)
    rel = np.einsum("nji,njk->nik", p.R[prev], p.R[nxt])
    ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
    lin = np.linalg.norm(p.C[nxt] - p.C[prev], axis=1)
    return (ang + lin / 2.0) / dt


def select_keyframes(blur: np.ndarray, stride: int = KEY_STRIDE) -> np.ndarray:
    """Least-blurred frame in each window of `stride` frames."""
    return np.array([s + int(np.argmin(blur[s:s + stride])) for s in range(0, len(blur), stride)])


def select_partners(p: Poses, blur: np.ndarray, key: int, n: int = 4) -> list[int]:
    """Up to n partner candidates with a usable baseline, best first; the second differs in side or
    direction from the first, so the two depth maps fail independently."""
    za = p.R[key][:, 2]
    med_blur = np.median(blur) + 1e-6
    cands = []
    for j in range(max(0, key - PARTNER_WINDOW), min(len(p.t), key + PARTNER_WINDOW + 1)):
        if j == key:
            continue
        base = p.C[j] - p.C[key]
        b = float(np.linalg.norm(base))
        if not BASELINE[0] <= b <= BASELINE[1]:
            continue
        zb = p.R[j][:, 2]
        axis_angle = np.degrees(np.arccos(np.clip(za @ zb, -1, 1)))
        x = base / b
        if axis_angle > MAX_AXIS_ANGLE or abs(x @ za) > MAX_FORWARD or abs(x @ zb) > MAX_FORWARD:
            continue
        score = abs(b - TARGET_BASELINE) / TARGET_BASELINE + axis_angle / MAX_AXIS_ANGLE \
            + 0.5 * blur[j] / med_blur
        cands.append((score, j, x))
    cands.sort(key=lambda c: c[0])
    if not cands:
        return []
    _, first, x1 = cands[0]
    diverse = [j for _, j, x in cands[1:] if (j - key) * (first - key) < 0 or abs(x @ x1) < 0.87]
    rest = [j for _, j, _ in cands[1:] if j not in diverse and abs(j - first) > 3]
    return ([first] + diverse[:n - 1] + rest)[:n]


def plan_pairs(p: Poses, key_stride: int = KEY_STRIDE) -> tuple[dict[int, list[int]], int]:
    blur = motion_blur_score(p)
    keys = select_keyframes(blur, key_stride)
    pairs = {int(k): select_partners(p, blur, int(k)) for k in keys}
    return {k: c for k, c in pairs.items() if len(c) >= 2}, len(keys)


def select_anchor_neighbours(p: Poses, blur: np.ndarray, key: int, n: int = 4) -> list[int]:
    """Frames for triangulating metric anchors: any baseline direction, spread in time."""
    za = p.R[key][:, 2]
    med_blur = np.median(blur) + 1e-6
    cands = []
    for j in range(max(0, key - ANCHOR_WINDOW), min(len(p.t), key + ANCHOR_WINDOW + 1)):
        b = float(np.linalg.norm(p.C[j] - p.C[key]))
        if not ANCHOR_BASELINE[0] <= b <= ANCHOR_BASELINE[1]:
            continue
        angle = np.degrees(np.arccos(np.clip(za @ p.R[j][:, 2], -1, 1)))
        if angle > ANCHOR_MAX_ANGLE:
            continue
        cands.append((angle / ANCHOR_MAX_ANGLE + blur[j] / med_blur - min(b, 0.4) / 0.4, j))
    chosen: list[int] = []
    for _, j in sorted(cands):
        if all(abs(j - c) > 10 for c in chosen):
            chosen.append(j)
            if len(chosen) == n:
                break
    return chosen


def triangulate_anchors(p: Poses, a: int, b: int, fa, fb, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixels in frame a (WORK_WIDTH coordinates) and their metric depth from matches triangulated
    with the ARKit poses of a and b."""
    pa, pb = match(fa, fb)
    if len(pa) < 8:
        return np.empty((0, 2)), np.empty(0)
    Ka, Kb = p.K(a, scale), p.K(b, scale)
    Pa = Ka @ np.c_[p.R[a].T, -p.R[a].T @ p.C[a]]
    Pb = Kb @ np.c_[p.R[b].T, -p.R[b].T @ p.C[b]]
    Xh = cv2.triangulatePoints(Pa, Pb, pa.T, pb.T)
    X = (Xh[:3] / Xh[3]).T
    xa, xb = (X - p.C[a]) @ p.R[a], (X - p.C[b]) @ p.R[b]
    ok = (xa[:, 2] > 0.3) & (xb[:, 2] > 0.3)
    ea = np.linalg.norm(xa[:, :2] / np.maximum(xa[:, 2:], 1e-6) * Ka[0, 0] + Ka[:2, 2] - pa, axis=1)
    eb = np.linalg.norm(xb[:, :2] / np.maximum(xb[:, 2:], 1e-6) * Kb[0, 0] + Kb[:2, 2] - pb, axis=1)
    ra, rb = X - p.C[a], X - p.C[b]
    cos = (ra * rb).sum(1) / np.maximum(np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1), 1e-9)
    parallax = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ok &= (ea < ANCHOR_REPROJ) & (eb < ANCHOR_REPROJ) & (parallax > ANCHOR_MIN_PARALLAX)
    return pa[ok], xa[ok, 2]


def rectify(Ka, Ra, Ca, Kb, Rb, Cb, size: int):
    """Rectifying homographies with the baseline as the new x axis, so frame a is always the left image.
    Returns Ha, Hb, Rn (world -> rectified rotation), Kn, baseline."""
    x = Cb - Ca
    b = float(np.linalg.norm(x))
    x = x / b
    y = np.cross(Ra[:, 2] + Rb[:, 2], x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    Rn = np.stack([x, y, z])
    f = 0.5 * (Ka[0, 0] + Kb[0, 0])
    Kn = np.array([[f, 0, size / 2 - 0.5], [0, f, size / 2 - 0.5], [0, 0, 1.0]])
    Ha = Kn @ Rn @ Ra @ np.linalg.inv(Ka)
    Hb = Kn @ Rn @ Rb @ np.linalg.inv(Kb)
    return Ha, Hb, Rn, Kn, b


_CV: dict = {}


def _sgbm():
    if "sgbm" not in _CV:
        _CV["sgbm"] = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=NUM_DISP, blockSize=5, P1=8 * 25, P2=32 * 25,
            disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return _CV["sgbm"]


def features(img: np.ndarray):
    if "sift" not in _CV:
        _CV["sift"] = cv2.SIFT_create(2000)
    kp, desc = _CV["sift"].detectAndCompute(img, None)
    return np.float64([k.pt for k in kp]).reshape(-1, 2), desc


def match(fa, fb) -> tuple[np.ndarray, np.ndarray]:
    if "bf" not in _CV:
        _CV["bf"] = cv2.BFMatcher()
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    m = [x for x, y in _CV["bf"].knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
    return pa[[x.queryIdx for x in m]], pb[[x.trainIdx for x in m]]


def epipolar_error(p: Poses, a: int, b: int, pa: np.ndarray, pb: np.ndarray, scale: float) -> float:
    """Median distance (px) of matches in b from the epipolar lines predicted by the poses."""
    Ka, Kb = p.K(a, scale), p.K(b, scale)
    Rab = p.R[b].T @ p.R[a]
    t = p.R[b].T @ (p.C[a] - p.C[b])
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(Kb).T @ tx @ Rab @ np.linalg.inv(Ka)
    lines = np.c_[pa, np.ones(len(pa))] @ F.T
    d = np.abs((np.c_[pb, np.ones(len(pb))] * lines).sum(1)) / np.hypot(lines[:, 0], lines[:, 1])
    return float(np.median(d))


def _open(vid: PosedVideo):
    cap = cv2.VideoCapture(str(vid.video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    if abs(width - RGB_WIDTH) > 1:
        raise ValueError(f"Expected {RGB_WIDTH}-px wide video (intrinsics refer to it), got {width}")
    scale = WORK_WIDTH / RGB_WIDTH
    return cap, scale, (WORK_WIDTH, int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)))


def _gray(frame: np.ndarray, size) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)


def estimate_time_offset(vid: PosedVideo, p: Poses, n_keys: int = 80) -> tuple[float, list]:
    """Delay (s) of each video frame relative to its pose timestamp that minimises the epipolar error of
    feature matches. Uses the video only. Returns the delay and the (delay, error) curve."""
    pairs, _ = plan_pairs(p)
    keys = [k for k in sorted(pairs) if max(pairs[k][:2]) < OFFSET_FRAMES]
    keys = keys[:: max(1, len(keys) // n_keys)][:n_keys]
    need = {f for k in keys for f in (k, *pairs[k][:2])}
    if not need:
        return 0.0, []
    cap, scale, size = _open(vid)
    feats = {}
    for i in range(max(need) + 1):
        if i in need:
            ok, frame = cap.read()
            if not ok:
                break
            feats[i] = features(_gray(frame, size))
        elif not cap.grab():
            break
    cap.release()
    obs = []
    for k in keys:
        for j in pairs[k][:2]:
            if k in feats and j in feats:
                pa, pb = match(feats[k], feats[j])
                if len(pa) >= 30:
                    obs.append((k, j, pa, pb))
    if len(obs) < 10:
        return 0.0, []
    grid = np.arange(-OFFSET_SEARCH, OFFSET_SEARCH + 1e-9, 0.004)
    curve = []
    for d in grid:
        q = p.shifted(float(d))
        curve.append(float(np.median([epipolar_error(q, a, b, pa, pb, scale) for a, b, pa, pb in obs])))
    i = int(np.argmin(curve))
    delay = float(grid[i])
    if 0 < i < len(grid) - 1:  # parabola through the minimum and its neighbours
        y0, y1, y2 = curve[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if den > 0:
            delay += 0.5 * (y0 - y2) / den * (grid[1] - grid[0])
    return delay, [(round(float(d), 3), round(c, 3)) for d, c in zip(grid, curve)]


def vertical_fix(ra: np.ndarray, rb: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Affine on rectified image b that moves rows only: v' = v + c0 + c1 u + c2 v, fitted so matched
    features share rows. Returns the 2x3 warp and the median residual, or None if unsupported."""
    if len(ra) < MIN_MATCHES:
        return None
    A = np.c_[np.ones(len(rb)), rb]
    dv = ra[:, 1] - rb[:, 1]
    keep = np.abs(dv - np.median(dv)) < 3.0
    for _ in range(3):
        if keep.sum() < MIN_MATCHES:
            return None
        c, *_ = np.linalg.lstsq(A[keep], dv[keep], rcond=None)
        res = np.abs(A @ c - dv)
        keep = res < max(0.75, 2.5 * 1.4826 * np.median(res[keep]))
    if keep.sum() < MIN_MATCHES:
        return None
    return np.array([[1.0, 0, 0], [c[1], 1 + c[2], c[0]]]), float(np.median(res[keep]))


def pair_points(img_a, img_b, Ka, Ra, Ca, Kb, Rb, Cb, fa, fb) -> tuple[np.ndarray, np.ndarray] | None:
    """Point map (H,W,3) in camera-a coordinates on frame a's own pixel grid, and its validity mask.
    None when the pair cannot be aligned to sub-pixel rows."""
    h, w = img_a.shape
    size = max(h, w)
    Ha, Hb, Rn, Kn, b = rectify(Ka, Ra, Ca, Kb, Rb, Cb, size)
    pa, pb = match(fa, fb)
    if len(pa) < MIN_MATCHES:
        return None
    fix = vertical_fix(cv2.perspectiveTransform(pa[None], Ha)[0], cv2.perspectiveTransform(pb[None], Hb)[0])
    if fix is None or fix[1] > MAX_VERTICAL_RESIDUAL:
        return None
    Hb = np.vstack([fix[0], [0, 0, 1]]) @ Hb
    ra = cv2.warpPerspective(img_a, Ha, (size, size), flags=cv2.INTER_LINEAR)
    rb = cv2.warpPerspective(img_b, Hb, (size, size), flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(np.ones_like(img_a), Ha, (size, size), flags=cv2.INTER_NEAREST)
    disp = _sgbm().compute(ra, rb).astype(np.float32) / 16.0
    disp[(disp <= 0) | (cover == 0)] = np.nan

    # Sample the rectified disparity at every original pixel of frame a.
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pr = np.stack([u, v, np.ones_like(u)], -1) @ Ha.T
    ur, vr = (pr[..., 0] / pr[..., 2]).astype(np.float32), (pr[..., 1] / pr[..., 2]).astype(np.float32)
    d = cv2.remap(np.nan_to_num(disp), ur, vr, cv2.INTER_LINEAR)
    g = cv2.remap(np.isfinite(disp).astype(np.float32), ur, vr, cv2.INTER_LINEAR)
    ok = (g > 0.999) & (d > 0)

    f = Kn[0, 0]
    Z = np.where(ok, f * b / np.maximum(d, 1e-6), 0.0)
    Xr = np.stack([(ur - Kn[0, 2]) / f * Z, (vr - Kn[1, 2]) / f * Z, Z], -1)
    P = Xr @ (Rn @ Ra)  # rectified -> camera a: (Rn Ra)^T applied to row vectors
    ok &= P[..., 2] > 0
    return P.astype(np.float32), ok


def fuse_pair_maps(P1, o1, P2, o2) -> tuple[np.ndarray, np.ndarray]:
    """Keep pixels where both pair depths agree; average them."""
    z1, z2 = P1[..., 2], P2[..., 2]
    ok = o1 & o2 & (np.abs(z1 - z2) < AGREE_REL * 0.5 * (z1 + z2) + AGREE_ABS)
    return np.where(ok[..., None], 0.5 * (P1 + P2), 0.0).astype(np.float32), ok


def smoothed(P: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Normalised box filter of the point map over valid pixels (for normals only)."""
    k = (NORMAL_SMOOTH, NORMAL_SMOOTH)
    m = ok.astype(np.float32)
    den = cv2.boxFilter(m, -1, k, normalize=False)
    num = np.stack([cv2.boxFilter(P[..., c] * m, -1, k, normalize=False) for c in range(3)], -1)
    return num / np.maximum(den, 1e-6)[..., None]


def _anchors_from_map(P: np.ndarray, ok: np.ndarray, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.nonzero(ok[::step, ::step])
    v, u = v * step, u * step
    return np.c_[u, v].astype(np.float64), P[v, u, 2].astype(np.float64)


def collect_video_points(
    vid: PosedVideo, mono=None, key_stride: int = KEY_STRIDE, pixel_stride: int = 4,
    keep_maps: set[int] | None = None, segmenter=None,
):
    """Estimate the time offset, then stream the video once. Per keyframe: stereo depth from two partners
    and metric anchors (stereo pixels + features triangulated with ARKit poses). With `mono`, the dense
    relative depth of the model is fitted to the anchors (1/z = s*d + t) and replaces the stereo points.
    With `segmenter` (needs `mono`), each point gets a semantic label and each line segment a structural flag.
    Returns points, normals, keyframe index per point, corrected poses, stats, depth maps for `keep_maps`,
    segments {k: [seg, helper depth, structural]} and point labels (or None)."""
    from roomscan.geometry.monodepth import INPUT_SIZE
    from roomscan.geometry.semantics import structural

    raw = Poses.from_video(vid)
    delay, curve = estimate_time_offset(vid, raw)
    p = raw.shifted(delay)
    blur = motion_blur_score(p)
    keys = [int(k) for k in select_keyframes(blur, key_stride)]
    stereo = {k: c if len(c) >= 2 else [] for k in keys for c in [select_partners(p, blur, k)]}
    anchor_nb = {k: select_anchor_neighbours(p, blur, k) if mono else [] for k in keys}
    mono_keys = set(keys[::MONO_EVERY]) if mono else set()
    ready = {k: max([k, *stereo[k], *anchor_nb[k]]) for k in keys}
    order = sorted(keys, key=lambda k: ready[k])
    last_use: dict[int, int] = defaultdict(int)
    for k in order:
        for f in (k, *stereo[k], *anchor_nb[k]):
            last_use[f] = max(last_use[f], ready[k])

    cap, scale, size = _open(vid)
    mono_scale = INPUT_SIZE[0] / RGB_WIDTH
    to_mono = INPUT_SIZE[0] / WORK_WIDTH
    buf: dict[int, tuple] = {}
    pts, nrm, idx, maps, fused_frac = [], [], [], {}, []
    fits: list[tuple] = []
    pending: dict[int, tuple] = {}
    world: dict[int, np.ndarray] = {}
    dense: dict[int, np.ndarray] = {}
    segments: dict[int, list] = {}
    labels: dict[int, np.ndarray] = {}
    plab: list[np.ndarray] = []
    stats = {"time_offset_s": round(delay, 4), "time_offset_curve": curve, "keyframes": len(order),
             "keyframes_fused": 0, "pairs_tried": 0, "pairs_aligned": 0, "depth_source": "mono" if mono else "stereo"}
    q = 0
    for i in range(len(p.t)):
        if i in last_use:
            ok, frame = cap.read()
            if not ok:
                break
            img = _gray(frame, size)
            small = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_AREA) if i in mono_keys else None
            buf[i] = (img, features(img), small)
        elif not cap.grab():
            break
        while q < len(order) and ready[order[q]] <= i:
            k = order[q]
            q += 1
            seg = detect_segments(buf[k][0])
            segments[k] = [seg, np.full(len(seg), np.nan), np.ones(len(seg), bool)]
            if segmenter is not None and buf[k][2] is not None:
                labels[k] = segmenter.predict(buf[k][2], up=p.R[k].T @ np.array([0.0, 1.0, 0.0]))
                segments[k][2] = structural(seg, labels[k], WORK_WIDTH)
            maps_k = []
            for j in stereo[k]:
                stats["pairs_tried"] += 1
                m = pair_points(buf[k][0], buf[j][0], p.K(k, scale), p.R[k], p.C[k],
                                p.K(j, scale), p.R[j], p.C[j], buf[k][1], buf[j][1])
                if m is not None:
                    stats["pairs_aligned"] += 1
                    maps_k.append(m)
                    if len(maps_k) == 2:
                        break
            fused = fuse_pair_maps(*maps_k[0], *maps_k[1]) if len(maps_k) == 2 else None
            if fused is not None:
                stats["keyframes_fused"] += 1
                fused_frac.append(float(fused[1].mean()))

            if mono is None:
                if fused is None:
                    continue
                P, ok_px = fused
                segments[k][1] = _depth_at(P[..., 2], ok_px, seg, 1.0)
                pk, nk = organised_points(P, ok_px, p.R[k], p.C[k], pixel_stride, Z_RANGE,
                                          step=NORMAL_STEP, P_normals=smoothed(P, ok_px))
            elif k not in mono_keys:
                continue
            else:
                uv, z = [np.empty((0, 2))], [np.empty(0)]
                if fused is not None:
                    a_uv, a_z = _anchors_from_map(*fused)
                    uv.append(a_uv)
                    z.append(a_z)
                for j in anchor_nb[k]:
                    t_uv, t_z = triangulate_anchors(p, k, j, buf[k][1], buf[j][1], scale)
                    uv.append(t_uv)
                    z.append(t_z)
                uv, z = np.concatenate(uv), np.concatenate(z)
                if len(z):
                    Kw = p.K(k, scale)
                    rays = np.c_[(uv - Kw[:2, 2]) / Kw[0, 0], np.ones(len(uv))]
                    world[k] = (rays * z[:, None]) @ p.R[k].T + p.C[k]
                uv = (uv + 0.5) * to_mono - 0.5
                d = mono.predict(buf[k][2])
                res = _mono_points(d, uv, z, p, k, mono_scale, labels=labels.get(k))
                if res is None:
                    half = cv2.resize(d, (d.shape[1] // 2, d.shape[0] // 2), interpolation=cv2.INTER_AREA)
                    pending[k] = (half.astype(np.float16), (uv + 0.5) / 2 - 0.5, z)
                    continue
                pk, nk, P, ok_px, fit, lk = res
                fits.append((len(z), *fit))
                dense[k] = pk[::DENSE_ANCHOR_SUBSAMPLE]
                segments[k][1] = _depth_at(P[..., 2], ok_px, seg, to_mono)
                if lk is not None:
                    plab.append(lk)
            if keep_maps is not None and k in keep_maps:
                maps[k] = (P[..., 2].copy(), ok_px.copy())
            pts.append(pk)
            nrm.append(nk)
            idx.append(np.full(len(pk), k, dtype=np.int32))
        for f in [f for f in buf if last_use[f] <= i]:
            del buf[f]
    cap.release()

    # Second pass: keyframes without enough anchors of their own borrow measured (stereo/triangulated)
    # anchors of other keyframes and the metric points of directly anchored ones (one hop only, so
    # scale errors cannot chain), projected through the ARKit poses.
    n_direct = len(fits)
    for k, (d, uv, z) in pending.items():
        near = [j for j in set(world) | set(dense) if j != k and abs(j - k) <= PROPAGATE_WINDOW]
        a_uv, a_z = _project_anchors(p, k, mono_scale / 2, d.shape,
                                     [world[j] for j in near if j in world] + [dense[j] for j in near if j in dense])
        uv_all, z_all = np.concatenate([uv, a_uv]), np.concatenate([z, a_z])
        res = _mono_points(d.astype(np.float32), uv_all, z_all, p, k, mono_scale / 2, res_div=2,
                           min_inliers=PROPAGATED_MIN_INLIERS, labels=labels.get(k))
        if res is None:
            continue
        pk, nk, P, ok_px, fit, lk = res
        fits.append((len(z_all), *fit))
        if lk is not None:
            plab.append(lk)
        segments[k][1] = _depth_at(P[..., 2], ok_px, segments[k][0], to_mono / 2)
        if keep_maps is not None and k in keep_maps:
            maps[k] = (P[..., 2].copy(), ok_px.copy())
        pts.append(pk)
        nrm.append(nk)
        idx.append(np.full(len(pk), k, dtype=np.int32))

    if not pts:
        raise RuntimeError("No keyframe produced metric depth from the video")
    stats["fused_pixel_fraction"] = round(float(np.mean(fused_frac)), 4) if fused_frac else 0.0
    if mono is not None:
        f = np.array(fits)
        stats.update({"keyframes_mono": len(fits), "keyframes_mono_direct": n_direct,
                      "anchors_median": int(np.median(f[:, 0])),
                      "fit_inlier_frac_median": round(float(np.median(f[:, 3])), 3),
                      "fit_residual_median": round(float(np.median(f[:, 4])), 4)})
    stats["line_segments"] = int(sum(len(s[0]) for s in segments.values()))
    point_labels = np.concatenate(plab) if segmenter is not None and mono is not None else None
    if point_labels is not None:
        stats["line_segments_structural"] = int(sum(s[2].sum() for s in segments.values()))
    return (np.concatenate(pts), np.concatenate(nrm), np.concatenate(idx), p, stats, maps, segments,
            point_labels)


def _depth_at(Z: np.ndarray, ok: np.ndarray, seg: np.ndarray, factor: float) -> np.ndarray:
    """Helper depth at segment midpoints (WORK_WIDTH pixels; `factor` maps them onto Z's grid).
    Median of three points along the segment; nan where the map has no depth."""
    if not len(seg):
        return np.empty(0)
    h, w = Z.shape
    vals = []
    for f in (0.25, 0.5, 0.75):
        pt = seg[:, :2] * (1 - f) + seg[:, 2:] * f
        u = np.clip(np.round((pt[:, 0] + 0.5) * factor - 0.5).astype(int), 0, w - 1)
        v = np.clip(np.round((pt[:, 1] + 0.5) * factor - 0.5).astype(int), 0, h - 1)
        vals.append(np.where(ok[v, u], Z[v, u], np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(vals), axis=0)


def _mono_points(d, uv, z, p: Poses, k: int, kscale: float, res_div: int = 1, min_inliers: float | None = None,
                 labels: np.ndarray | None = None):
    """Fit the model's relative inverse depth to metric anchors and back-project.
    Returns world points, normals, camera point map, validity, the fit and per-point labels (or None);
    None if the fit is rejected."""
    from roomscan.geometry.monodepth import MAX_FIT_RESIDUAL, MIN_INLIER_FRAC, fit_inverse_affine, metric_depth, sample

    inside = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] <= d.shape[1] - 1) & (uv[:, 1] <= d.shape[0] - 1)
    uv, z = uv[inside], z[inside]
    fit = fit_inverse_affine(sample(d, uv), z) if len(z) else None
    if fit is None or fit[2] < (min_inliers or MIN_INLIER_FRAC) or fit[3] > MAX_FIT_RESIDUAL:
        return None
    Z, ok = metric_depth(d, fit[0], fit[1], z)
    K = p.K(k, kscale)
    h, w = Z.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    P = np.stack([(u - K[0, 2]) * Z / K[0, 0], (v - K[1, 2]) * Z / K[1, 1], Z], -1)
    lab = None if labels is None else cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    res = organised_points(P, ok, p.R[k], p.C[k], MONO_PIXEL_STRIDE // res_div, MONO_Z_RANGE,
                           step=max(1, 2 // res_div), extra=lab)
    pk, nk = res[0], res[1]
    return pk, nk, P, ok, fit, (res[2] if lab is not None else None)


def _project_anchors(p: Poses, k: int, kscale: float, shape, clouds: list[np.ndarray], cell: int = 4):
    """Project world points into keyframe k; keep the nearest point per `cell` px (crude occlusion test)."""
    if not clouds:
        return np.empty((0, 2)), np.empty(0)
    X = np.concatenate(clouds)
    x = (X - p.C[k]) @ p.R[k]
    x = x[x[:, 2] > 0.3]
    K = p.K(k, kscale)
    u = x[:, 0] / x[:, 2] * K[0, 0] + K[0, 2]
    v = x[:, 1] / x[:, 2] * K[1, 1] + K[1, 2]
    h, w = shape
    m = (u >= 0) & (v >= 0) & (u <= w - 1) & (v <= h - 1)
    u, v, zc = u[m], v[m], x[m, 2]
    if len(zc) == 0:
        return np.empty((0, 2)), np.empty(0)
    c = (v // cell).astype(np.int64) * (w // cell + 1) + (u // cell).astype(np.int64)
    order = np.lexsort((zc, c))
    first = np.r_[True, c[order][1:] != c[order][:-1]]
    sel = order[first]
    return np.c_[u[sel], v[sel]], zc[sel]
