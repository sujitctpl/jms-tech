"""
Clothing colour analysis for CCTV person crops.

Pipeline (per crop, cached):
  1. Locate the body with a YOLO pose model: the torso is the shoulder->hip quad (shrunk inward so
     arms, collar and background stay out), the legs are thick hip->knee->ankle strokes. Without the
     pose model (missing weights, no person found) a fixed central box is used instead.
  2. White-balance against the bright neutral background (the doorway walls) and express garment
     brightness *relative to that wall* so the same shirt reads the same in the lit doorway and in shadow.
  3. Take the dominant k-means colour cluster in CIE Lab and name it, flag strong two-tone texture as
     "Checked / Patterned", and keep a soft Lab histogram for appearance matching.

Why: the previous HSV heuristics sampled a fixed box that was mostly wall and the person's head on this
camera, so white shirts read Gray/Cream, sage read Cream, and every person's "torso histogram" largely
matched every other person's (the shared wall). Thresholds below were fit on ~90 hand-labelled
snapshots from the Main Gate camera; the feed is heavily desaturated, so chroma thresholds are low.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from app.core.logging import logger

# ---------------------------------------------------------------------------------------------
# Tags (strings are persisted in visitor notes as "Appearance: <tag>")
# ---------------------------------------------------------------------------------------------
TAG_BLACK = "Black / Dark Attire"
TAG_GRAY = "Gray Attire"
TAG_WHITE = "White / Light Shirt"
TAG_CREAM = "Cream / Khaki Shirt"
TAG_BROWN = "Brown / Rust Top"
TAG_RED = "Red / Maroon Top"
TAG_PINK = "Pink / Peach Top"
TAG_YELLOW = "Yellow / Mustard Top"
TAG_GREEN = "Green / Olive Top"
TAG_BLUE = "Blue / Navy Top"
TAG_PURPLE = "Purple / Violet Top"
TAG_PATTERN = "Checked / Patterned Shirt"
TAG_UNKNOWN = "Casual Attire"

# Colour families that one garment can plausibly drift between when the light changes
# (doorway vs. shadow, front vs. back). Anything not listed here is a real contradiction.
_NEIGHBOURS = {
    TAG_BLACK: {TAG_GRAY, TAG_BLUE, TAG_BROWN},
    TAG_GRAY: {TAG_BLACK, TAG_WHITE, TAG_GREEN, TAG_BLUE, TAG_CREAM},
    TAG_WHITE: {TAG_GRAY, TAG_CREAM, TAG_PINK},
    TAG_CREAM: {TAG_WHITE, TAG_YELLOW, TAG_BROWN, TAG_GREEN, TAG_GRAY},
    TAG_BROWN: {TAG_BLACK, TAG_RED, TAG_CREAM, TAG_YELLOW},
    TAG_RED: {TAG_BROWN, TAG_PINK, TAG_PURPLE},
    TAG_PINK: {TAG_RED, TAG_WHITE, TAG_PURPLE},
    TAG_YELLOW: {TAG_CREAM, TAG_BROWN},
    TAG_GREEN: {TAG_GRAY, TAG_CREAM},
    TAG_BLUE: {TAG_BLACK, TAG_GRAY, TAG_PURPLE},
    TAG_PURPLE: {TAG_BLUE, TAG_RED, TAG_PINK},
}

_ACHROMATIC = {TAG_BLACK, TAG_GRAY, TAG_WHITE}

def _pose_model_path() -> str:
    try:
        from app.core.config import settings

        return settings.CLOTHING_POSE_MODEL_PATH
    except Exception:
        return os.path.join("models", "yolo11n-pose.pt")
_KP_L_SHOULDER, _KP_R_SHOULDER, _KP_L_HIP, _KP_R_HIP = 5, 6, 11, 12
_KP_L_KNEE, _KP_R_KNEE, _KP_L_ANKLE, _KP_R_ANKLE = 13, 14, 15, 16
_KP_CONF = 0.40
_MIN_PART_PIXELS = 40

# Soft Lab histogram used for matching: relative lightness x a* x b*.
_HIST_BINS = (8, 6, 6)
_HIST_RANGE = ((0.0, 1.25), (-24.0, 24.0), (-24.0, 24.0))


@dataclass(frozen=True)
class PartColor:
    """Colour summary of one garment region (torso or legs)."""

    tag: str
    rel_l: float  # dominant-cluster lightness relative to the wall (0 = black, ~1 = as bright as wall)
    a: float  # white-balanced CIE a* (green- / red+)
    b: float  # white-balanced CIE b* (blue- / yellow+)
    patterned: bool
    hist: np.ndarray  # normalised soft Lab histogram (L1 = 1)
    pixels: int

    @property
    def chroma(self) -> float:
        return float(np.hypot(self.a, self.b))


@dataclass(frozen=True)
class ClothingDescriptor:
    torso: Optional[PartColor]
    legs: Optional[PartColor]
    source: str  # "pose" | "fallback"
    # Leg colour is trustworthy only when pose saw both knees clearly and both legs agree in lightness.
    # Side-on walkers often get knees placed in the gap between the legs, which samples the floor.
    legs_reliable: bool = False
    # "short" (bare forearms seen), "long" (sleeve fabric on the forearms) or None (not determinable).
    sleeves: Optional[str] = None

    @property
    def tag(self) -> str:
        return self.torso.tag if self.torso is not None else TAG_UNKNOWN


# ---------------------------------------------------------------------------------------------
# Pose model (lazy, process-wide, thread-safe)
# ---------------------------------------------------------------------------------------------
class _PoseLocator:
    def __init__(self) -> None:
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        path = _pose_model_path()
        try:
            from ultralytics import YOLO

            if not os.path.isfile(path):
                # Official Ultralytics release asset (~6 MB); fetched once into models/.
                from ultralytics.utils.downloads import attempt_download_asset

                logger.info("Clothing colour: pose model missing at %s, downloading...", path)
                attempt_download_asset(path)
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
            self._model = YOLO(path)
            logger.info("Clothing colour: pose model loaded from %s", path)
        except Exception as exc:  # degrade to fixed-box geometry, never crash the caller
            self._failed = True
            logger.warning(
                "Clothing colour: pose model unavailable (%s); falling back to a fixed torso box, "
                "clothing tags and exit matching will be less accurate.", exc,
            )
        return self._model

    def keypoints(self, img: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (xy[17,2], conf[17]) of the dominant person in the crop, or None."""
        with self._lock:
            model = self._load()
            if model is None:
                return None
            try:
                res = model.predict(img, imgsz=320, conf=0.25, verbose=False)[0]
            except Exception as exc:
                # Usually GPU contention with the detector / embedders: retry once on CPU.
                logger.warning("Clothing colour: pose inference failed (%s); retrying on CPU.", exc)
                try:
                    res = model.predict(img, imgsz=320, conf=0.25, verbose=False, device="cpu")[0]
                except Exception as exc2:
                    logger.warning("Clothing colour: pose inference failed on CPU too (%s); using fixed box.", exc2)
                    return None
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return None
        xyxy = res.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        # The crop is centred on the tracked person: prefer the biggest, most confident, most central box.
        h, w = img.shape[:2]
        cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        centrality = 1.0 - np.clip(np.abs(cx - w * 0.5) / (w * 0.5), 0.0, 1.0) * 0.5
        k = int(np.argmax(areas * res.boxes.conf.cpu().numpy() * centrality))
        conf = res.keypoints.conf
        if conf is None:
            return None
        return res.keypoints.xy[k].cpu().numpy(), conf[k].cpu().numpy()


_POSE = _PoseLocator()


def _lerp(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    return p + (q - p) * t


_KNEE_RELIABLE_CONF = 0.75


_KP_L_ELBOW, _KP_R_ELBOW, _KP_L_WRIST, _KP_R_WRIST = 7, 8, 9, 10
_ARM_CONF = 0.60


def _part_masks(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], str, list, list]:
    """Torso mask, legs mask (None if legs not visible), geometry source, per-leg masks (only when
    both knees are confidently located) and forearm masks (mid elbow->wrist, outside the torso)."""
    h, w = img.shape[:2]
    torso = np.zeros((h, w), np.uint8)
    legs = np.zeros((h, w), np.uint8)
    per_leg: list = []
    forearms: list = []
    kp = _POSE.keypoints(img)
    if kp is not None:
        xy, cf = kp
        if cf[_KP_L_SHOULDER] > _KP_CONF and cf[_KP_R_SHOULDER] > _KP_CONF:
            s1, s2 = xy[_KP_L_SHOULDER], xy[_KP_R_SHOULDER]
            shoulder_w = float(np.linalg.norm(s1 - s2))
            hips_ok = cf[_KP_L_HIP] > _KP_CONF * 0.9 and cf[_KP_R_HIP] > _KP_CONF * 0.9
            if hips_ok:
                h1, h2 = xy[_KP_L_HIP], xy[_KP_R_HIP]
            else:
                # Hips hidden (counter, sofa, crop edge): extend perpendicular to the shoulder line.
                d = np.array([-(s2 - s1)[1], (s2 - s1)[0]], dtype=np.float32)
                d /= float(np.linalg.norm(d)) + 1e-6
                if d[1] < 0:
                    d = -d
                h1, h2 = s1 + d * shoulder_w * 1.4, s2 + d * shoulder_w * 1.4
            if shoulder_w >= 6:
                top = (_lerp(s1, h1, 0.12), _lerp(s2, h2, 0.12))
                bot = (_lerp(s1, h1, 0.80), _lerp(s2, h2, 0.80))
                quad = np.array(
                    [_lerp(top[0], top[1], 0.18), _lerp(top[1], top[0], 0.18),
                     _lerp(bot[1], bot[0], 0.18), _lerp(bot[0], bot[1], 0.18)],
                    dtype=np.int32,
                )
                cv2.fillPoly(torso, [quad], 1)
            if hips_ok:
                thick = max(3, int(round(shoulder_w * 0.22)))
                both_knees = all(cf[k] >= _KNEE_RELIABLE_CONF for k in (_KP_L_KNEE, _KP_R_KNEE))
                for hip, knee, ankle in ((_KP_L_HIP, _KP_L_KNEE, _KP_L_ANKLE), (_KP_R_HIP, _KP_R_KNEE, _KP_R_ANKLE)):
                    if cf[knee] <= _KP_CONF:
                        continue
                    one = np.zeros((h, w), np.uint8)
                    p0 = _lerp(xy[hip], xy[knee], 0.20)
                    cv2.line(one, tuple(int(v) for v in p0), tuple(int(v) for v in xy[knee]), 1, thick)
                    if cf[ankle] > _KP_CONF:
                        p1 = _lerp(xy[knee], xy[ankle], 0.75)
                        cv2.line(one, tuple(int(v) for v in xy[knee]), tuple(int(v) for v in p1), 1, thick)
                    legs |= one
                    if both_knees:
                        per_leg.append(one.astype(bool) & ~torso.astype(bool))
            arm_thick = max(3, int(round(shoulder_w * 0.12)))
            for elbow, wrist in ((_KP_L_ELBOW, _KP_L_WRIST), (_KP_R_ELBOW, _KP_R_WRIST)):
                if min(cf[elbow], cf[wrist]) < _ARM_CONF:
                    continue
                fm = np.zeros((h, w), np.uint8)
                a0 = _lerp(xy[elbow], xy[wrist], 0.25)
                a1 = _lerp(xy[elbow], xy[wrist], 0.70)  # stop short of the hand
                cv2.line(fm, tuple(int(v) for v in a0), tuple(int(v) for v in a1), 1, arm_thick)
                fm_b = fm.astype(bool) & ~torso.astype(bool)
                if int(fm_b.sum()) >= 15:
                    forearms.append(fm_b)
            if int(torso.sum()) >= _MIN_PART_PIXELS:
                legs_mask = legs.astype(bool) & ~torso.astype(bool)
                return (torso.astype(bool), (legs_mask if int(legs_mask.sum()) >= _MIN_PART_PIXELS else None),
                        "pose", per_leg, forearms)

    # Fallback geometry: central column. A squat crop is a head-and-shoulders bust, chest lower down.
    torso[:] = 0
    is_bust = h < w * 1.2
    if is_bust:
        torso[int(h * 0.45):int(h * 0.85), int(w * 0.30):int(w * 0.70)] = 1
    else:
        torso[int(h * 0.24):int(h * 0.46), int(w * 0.35):int(w * 0.65)] = 1
    legs_mask = None
    if not is_bust and h >= 140:
        legs[:] = 0
        legs[int(h * 0.60):int(h * 0.86), int(w * 0.35):int(w * 0.65)] = 1
        legs_mask = legs.astype(bool)
    return torso.astype(bool), legs_mask, "fallback", [], []


# ---------------------------------------------------------------------------------------------
# Colour naming
# ---------------------------------------------------------------------------------------------
def _name_color(rel_l: float, a: float, b: float) -> str:
    """Map a white-balanced Lab colour (with wall-relative lightness) to a garment tag."""
    chroma = float(np.hypot(a, b))
    hue = float(np.degrees(np.arctan2(b, a)) % 360.0)

    # Strong colour first.
    if chroma >= 12.0:
        if hue >= 335.0 or hue < 40.0:
            return TAG_RED
        if hue < 105.0:
            return TAG_YELLOW
        if hue < 190.0:
            return TAG_GREEN
        if hue < 315.0:  # saturated blue sits near 300 deg in Lab
            return TAG_BLUE
        return TAG_PURPLE
    # Navy: dark garments carry a camera blue cast of b* ~ -4..-5, so require clearly more than that.
    if b <= -6.4 and 220.0 <= hue <= 320.0:
        return TAG_BLUE
    # Muted warm tones.
    if chroma >= 7.0 and (hue >= 320.0 or hue < 20.0) and rel_l >= 0.40:
        return TAG_PINK
    if chroma >= 4.5 and hue < 60.0 and rel_l < 0.50:
        return TAG_BROWN
    if chroma >= 4.5 and 45.0 <= hue < 110.0 and rel_l >= 0.50:
        return TAG_CREAM
    if b >= 2.2 and -1.5 <= a <= 3.0 and b > abs(a) and rel_l >= 0.60:
        return TAG_CREAM  # pale khaki / beige that the camera nearly washes out
    # Sage / olive: faint but consistent green (negative a*).
    if a <= -1.6 and chroma >= 2.5 and 100.0 <= hue < 190.0:
        return TAG_GREEN
    # Achromatic by brightness relative to the wall.
    if rel_l < 0.33:
        return TAG_BLACK
    if rel_l < 0.62:
        return TAG_GRAY
    return TAG_WHITE


def _soft_hist(rel_l: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Clamp into range: histogramdd silently drops out-of-range samples (vivid colours, over-bright pixels).
    eps = 1e-3
    pts = np.stack([
        np.clip(rel_l, _HIST_RANGE[0][0], _HIST_RANGE[0][1] - eps),
        np.clip(a, _HIST_RANGE[1][0], _HIST_RANGE[1][1] - eps),
        np.clip(b, _HIST_RANGE[2][0], _HIST_RANGE[2][1] - eps),
    ], axis=1)
    hist, _ = np.histogramdd(pts, bins=_HIST_BINS, range=_HIST_RANGE)
    hist = hist.astype(np.float32)
    # Spread mass to neighbouring bins so small lighting shifts don't zero the overlap.
    hist = cv2.GaussianBlur(hist.reshape(_HIST_BINS[0], -1), (3, 3), 0.8).reshape(_HIST_BINS)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def _analyse_part(lab: Tuple[np.ndarray, np.ndarray, np.ndarray], mask: np.ndarray, wall_l: float,
                  wb: Tuple[float, float]) -> Optional[PartColor]:
    L, A, B = lab
    n = int(mask.sum())
    if n < _MIN_PART_PIXELS:
        return None
    px_l = L[mask] / wall_l
    px_a = A[mask] - wb[0]
    px_b = B[mask] - wb[1]
    data = np.stack([L[mask], px_a, px_b], axis=1).astype(np.float32)
    k = 3 if n >= 90 else 1
    if k > 1:
        cv2.setRNGSeed(12345)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=k)
        dom = centers[int(np.argmax(counts))]
    else:
        dom = np.median(data, axis=0)
    rel_l = float(dom[0]) / wall_l
    a, b = float(dom[1]), float(dom[2])

    l_std = float(np.std(L[mask]))
    p10, p90 = np.percentile(L[mask], [10, 90])
    patterned = l_std >= 19.0 and (p90 - p10) >= 50.0

    tag = TAG_PATTERN if patterned else _name_color(rel_l, a, b)
    return PartColor(tag=tag, rel_l=rel_l, a=a, b=b, patterned=patterned,
                     hist=_soft_hist(px_l, px_a, px_b), pixels=n)


def _unpad(crop: np.ndarray) -> np.ndarray:
    try:
        from app.vision.body_embedder import BodyEmbedder

        u = BodyEmbedder.unpad_canvas(crop)
        return u if u is not None and u.size > 0 else crop
    except Exception:
        return crop


def _compute(crop: np.ndarray) -> ClothingDescriptor:
    img = _unpad(crop)
    if img.ndim != 3 or img.shape[0] < 30 or img.shape[1] < 10:
        return ClothingDescriptor(None, None, "fallback")
    torso_m, legs_m, source, per_leg, forearms = _part_masks(img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    A = lab[..., 1] - 128.0
    B = lab[..., 2] - 128.0

    body = torso_m | (legs_m if legs_m is not None else False)
    bg = ~body
    # Reference white = the bright end of the background (doorway wall / door); clamp for dark scenes.
    wall_l = float(np.percentile(L[bg], 90)) if int(bg.sum()) > 50 else float(np.percentile(L, 90))
    wall_l = float(np.clip(wall_l, 40.0, 100.0))
    neutral_bg = bg & (L > 55.0) & (np.hypot(A, B) < 15.0)
    if int(neutral_bg.sum()) > 30:
        wb = (float(np.clip(np.median(A[neutral_bg]), -8, 8)), float(np.clip(np.median(B[neutral_bg]), -8, 8)))
    else:
        wb = (0.0, 0.0)

    torso = _analyse_part((L, A, B), torso_m, wall_l, wb)
    legs = _analyse_part((L, A, B), legs_m, wall_l, wb) if legs_m is not None else None
    legs_reliable = False
    if legs is not None and len(per_leg) == 2 and all(int(m.sum()) >= _MIN_PART_PIXELS // 2 for m in per_leg):
        rel = [float(np.median(L[m])) / wall_l for m in per_leg]
        legs_reliable = abs(rel[0] - rel[1]) <= 0.25
    sleeves = _sleeve_length(torso, forearms, A, B, wb)
    return ClothingDescriptor(torso, legs, source, legs_reliable, sleeves)


def _sleeve_length(torso: Optional[PartColor], forearms: list, A: np.ndarray, B: np.ndarray,
                   wb: Tuple[float, float]) -> Optional[str]:
    """Bare skin on this camera is warm (a* >= 5, b* >= 4.5 after white balance) while sleeve fabric
    is neutral or carries the camera's blue cast. Undetermined for warm-coloured tops (skin-like) and
    whenever the forearms disagree (arm in front of the body, hand in the sample, pushed-up sleeves)."""
    if torso is None or not forearms or torso.patterned:
        return None
    if torso.a >= 4.0 and torso.b >= 3.5:
        return None
    skin = fabric = 0
    for m in forearms:
        a = float(np.median(A[m])) - wb[0]
        b = float(np.median(B[m])) - wb[1]
        if a >= 5.0 and b >= 4.5:
            skin += 1
        elif a <= 3.0 or b <= 1.5:
            fabric += 1
    if skin and not fabric:
        return "short"
    if fabric and not skin:
        return "long"
    return None


# Small LRU so the tag, the matcher and the vetoes don't re-run pose on the same crop.
_CACHE: "OrderedDict[bytes, ClothingDescriptor]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_SIZE = 128


def describe(crop: Optional[np.ndarray]) -> ClothingDescriptor:
    """Analyse the garments in a person crop (BGR). Never raises."""
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return ClothingDescriptor(None, None, "fallback")
    key = hashlib.blake2b(np.ascontiguousarray(crop).tobytes(), digest_size=16,
                          person=str(crop.shape).encode()[:16]).digest()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    try:
        desc = _compute(crop)
    except Exception as exc:
        logger.debug("Clothing colour analysis failed: %s", exc)
        desc = ClothingDescriptor(None, None, "fallback")
    with _CACHE_LOCK:
        _CACHE[key] = desc
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return desc


def estimate_tag(crop: Optional[np.ndarray]) -> str:
    return describe(crop).tag


# ---------------------------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------------------------
def hist_similarity(p: Optional[PartColor], q: Optional[PartColor]) -> float:
    """Bhattacharyya coefficient of the two soft Lab histograms (0..1)."""
    if p is None or q is None:
        return 0.0
    return float(np.clip(np.sum(np.sqrt(p.hist * q.hist)), 0.0, 1.0))


def tags_compatible(p: Optional[PartColor], q: Optional[PartColor]) -> bool:
    """True unless the two garments are clearly different colours (lighting drift allowed)."""
    if p is None or q is None:
        return True
    if p.tag == q.tag or TAG_UNKNOWN in (p.tag, q.tag):
        return True
    if p.patterned or q.patterned:
        if p.patterned and q.patterned:
            return True
        # A check/stripe/print mixes light and dark, and is often only on the front (the exit view is
        # the back), so it is compatible with any neutral. Against a real colour, compare its dominant colour.
        plain = q if p.patterned else p
        if plain.tag in _ACHROMATIC:
            return True
        pt = _name_color(p.rel_l, p.a, p.b)
        qt = _name_color(q.rel_l, q.a, q.b)
        return pt == qt or qt in _NEIGHBOURS.get(pt, set()) or pt in _NEIGHBOURS.get(qt, set())
    return q.tag in _NEIGHBOURS.get(p.tag, set()) or p.tag in _NEIGHBOURS.get(q.tag, set())


# Weak-Re-ID band (0.60 <= reid < 0.70): front-in vs side/back-out views of the SAME person land here
# (VIS-2026-1042: 0.642, VIS-2026-1044: 0.650) and so did one wrong match (VIS-2026-1047: 0.685, short vs
# long sleeves). Clothing must then agree strongly and nothing may contradict. The caller additionally
# requires the candidate to clearly out-score every other visitor still in the store.
WEAK_REID_MIN = 0.60
WEAK_REID_TORSO_MIN = 0.80


def weak_reid_support(d1: ClothingDescriptor, d2: ClothingDescriptor) -> Tuple[bool, str]:
    """Whether clothing evidence is strong and contradiction-free enough to back a weak Re-ID score."""
    if d1.torso is None or d2.torso is None:
        return False, "no torso"
    torso_sim = hist_similarity(d1.torso, d2.torso)
    if torso_sim < WEAK_REID_TORSO_MIN:
        return False, f"torso_sim {torso_sim:.2f} < {WEAK_REID_TORSO_MIN}"
    if not tags_compatible(d1.torso, d2.torso):
        return False, f"clothing '{d1.tag}' vs '{d2.tag}'"
    if d1.sleeves and d2.sleeves and d1.sleeves != d2.sleeves:
        return False, f"sleeves {d1.sleeves} vs {d2.sleeves}"
    if d1.legs is not None and d2.legs is not None and d1.legs_reliable and d2.legs_reliable:
        if abs(d1.legs.rel_l - d2.legs.rel_l) >= 0.40:
            return False, f"trousers {d1.legs.rel_l:.2f} vs {d2.legs.rel_l:.2f}"
    return True, f"torso_sim {torso_sim:.2f}, sleeves {d1.sleeves}/{d2.sleeves}"
