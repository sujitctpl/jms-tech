import os
from typing import Optional, Tuple
import cv2
import numpy as np
from app.core.logging import logger

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class BodyEmbedder:
    """
    OSNet (Omni-Scale Network) Person Re-Identification embedding extraction engine
    with Dual-Part (Upper Torso + Lower Legs) 3D Color Histogram Verification.
    """

    def __init__(self, model_path: Optional[str] = None):
        default_ain = os.path.join("models", "osnet_ain_x1_0_msmt17.onnx")
        default_x025 = os.path.join("models", "osnet_x0_25.onnx")
        if model_path:
            self.model_path = model_path
        elif os.path.isfile(default_ain):
            self.model_path = default_ain
        else:
            self.model_path = default_x025
        self.session = None
        self._initialize_session()

    def _initialize_session(self) -> None:
        if ort is None:
            logger.error("onnxruntime is not installed. Body embedding disabled.")
            return

        if not os.path.isfile(self.model_path):
            logger.error("OSNet ReID model file not found at: %s", self.model_path)
            return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            active_prov = self.session.get_providers()
            logger.info("Loaded OSNet Person ReID model from %s [Active: %s]", self.model_path, active_prov)
        except Exception:
            try:
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                logger.info("Loaded OSNet Person ReID model on CPU fallback from %s", self.model_path)
            except Exception as e2:
                logger.error("Failed to initialize OSNet session: %s", e2)
                self.session = None

    @staticmethod
    def unpad_canvas(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Removes dark UI canvas borders / letterboxing (BGR [33, 24, 20] or black)
        so ReID embeddings and color histograms isolate only the actual person image.
        """
        if crop is None or crop.size == 0:
            return crop
        try:
            diff1 = np.abs(crop.astype(np.int16) - np.array([20, 24, 33], dtype=np.int16))
            diff2 = np.abs(crop.astype(np.int16) - np.array([33, 24, 20], dtype=np.int16))
            is_bg = np.all(diff1 < 22, axis=2) | np.all(diff2 < 22, axis=2) | (np.max(crop, axis=2) < 14)
            non_bg_y, non_bg_x = np.where(~is_bg)
            if len(non_bg_y) > 0 and len(non_bg_x) > 0:
                y1, y2 = int(non_bg_y.min()), int(non_bg_y.max())
                x1, x2 = int(non_bg_x.min()), int(non_bg_x.max())
                if (y2 - y1) >= 40 and (x2 - x1) >= 20:
                    return crop[y1:y2 + 1, x1:x2 + 1]
        except Exception:
            pass
        return crop

    def extract_embedding(self, body_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a 512-dim L2-normalized ReID embedding vector from a full-body person image crop.
        Accepts BGR image (OpenCV format).
        """
        body_crop = self.unpad_canvas(body_crop)
        if self.session is None or body_crop is None or body_crop.size == 0:
            return None

        h, w = body_crop.shape[:2]
        if h < 20 or w < 10:
            return None

        try:
            # 1. Convert BGR to RGB
            img_rgb = cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB)
            # 2. Resize to canonical OSNet dimensions (width=128, height=256)
            img_resized = cv2.resize(img_rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
            # 3. Standard ImageNet / ReID normalization
            img_f = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_f - mean) / std
            # 4. Transpose HWC -> CHW and expand to NCHW (1, 3, 256, 128)
            tensor = np.transpose(img_norm, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0)

            # 5. ONNX Inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: tensor})
            embedding = outputs[0][0].astype(np.float32)

            # 6. L2 Normalization
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm

            return embedding
        except Exception as e:
            logger.warning("Error extracting body ReID embedding: %s", e)
            return None

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two L2-normalized 512-dim embeddings.
        """
        if emb1 is None or emb2 is None:
            return 0.0
        v1 = np.asarray(emb1, dtype=np.float32).flatten()
        v2 = np.asarray(emb2, dtype=np.float32).flatten()
        if v1.shape != v2.shape or v1.size == 0:
            return 0.0
        return float(np.dot(v1, v2))

    @staticmethod
    def extract_body_part_hist(
        crop: Optional[np.ndarray],
        y1_pct: float,
        y2_pct: float,
        x1_pct: float = 0.30,
        x2_pct: float = 0.70,
    ) -> Optional[np.ndarray]:
        """
        Extracts a normalized 3D (Hue-Saturation-Value) color histogram of a specific body region
        (torso or legs) to isolate clothing and distinguish both chromatic (colors) and achromatic
        (white, gray, black) garments. Insets 30% from sides to eliminate background floor/walls.
        """
        crop = BodyEmbedder.unpad_canvas(crop)
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 30 or w < 15:
            return None
        try:
            part = crop[int(h * y1_pct):int(h * y2_pct), int(w * x1_pct):int(w * x2_pct)]
            if part.size == 0:
                return None
            hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
            # 8 Hue x 8 Sat x 8 Val = 512 bins (captures full color and luminance distribution)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
            return hist
        except Exception:
            return None

    @staticmethod
    def extract_salient_torso_patch(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Locates the core person torso patch within a full-body or bust CCTV frame.
        Slides across horizontal columns in the torso band to isolate the actual person
        and eliminate background white walls, door reflections, and floor mats.
        Includes border masking and center-prior weighting.
        """
        crop = BodyEmbedder.unpad_canvas(crop)
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 30 or w < 10:
            return None

        ar = float(h) / float(max(1, w))
        if ar < 1.20:
            # Portrait / bust crop: torso is in lower half (35% - 85%)
            band = crop[int(h * 0.35):int(h * 0.85), :]
        else:
            # Full body crop: torso is in upper body band (18% - 50%)
            band = crop[int(h * 0.18):int(h * 0.50), :]

        bh, bw = band.shape[:2]
        if bh < 10 or bw < 10:
            return None

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        # Torso window width (35% bw) centered on person to avoid capturing doorway walls
        win_w = max(20, int(bw * 0.35))
        min_x = int(bw * 0.08)
        max_x = max(min_x + 1, int(bw * 0.92) - win_w)
        valid_mask = (v >= 25)

        col_scores = []
        center_x = bw * 0.50
        for x in range(min_x, max_x + 1, 2):
            w_s = s[:, x:x + win_w]
            w_v = v[:, x:x + win_w]
            w_mask = valid_mask[:, x:x + win_w]
            if np.sum(w_mask) < (win_w * bh * 0.35):
                continue
            mean_s = float(np.mean(w_s[w_mask]))
            mean_v = float(np.mean(w_v[w_mask]))
            # Clothing prominence: color saturation + darkness contrast against white doorway walls
            prominence = mean_s + max(0.0, 140.0 - mean_v) * 0.40
            c_dist = abs((x + win_w * 0.5) - center_x) / (bw * 0.5)
            score = prominence * (1.0 - 0.40 * c_dist)
            col_scores.append((score, x))

        if not col_scores:
            cx = max(0, int(center_x - win_w * 0.5))
            return band[:, cx:cx + win_w]

        best_score, best_x = max(col_scores, key=lambda it: it[0])
        return band[:, best_x:best_x + win_w]

    @staticmethod
    def extract_salient_legs_patch(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Locates the core person legs/pants patch within a full-body CCTV frame.
        Slides across horizontal columns in the lower-body band (54% - 90% height)
        to isolate trousers/jeans/skirts and eliminate background floor tiles and walls.
        """
        crop = BodyEmbedder.unpad_canvas(crop)
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 50 or w < 10:
            return None
        band = crop[int(h * 0.54):int(h * 0.90), :]
        bh, bw = band.shape[:2]
        if bh < 10 or bw < 10:
            return None

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        win_w = max(20, int(bw * 0.35))
        min_x = int(bw * 0.08)
        max_x = max(min_x + 1, int(bw * 0.92) - win_w)
        valid_mask = (v >= 20)

        col_scores = []
        center_x = bw * 0.50
        for x in range(min_x, max_x + 1, 2):
            w_s = s[:, x:x + win_w]
            w_v = v[:, x:x + win_w]
            w_mask = valid_mask[:, x:x + win_w]
            if np.sum(w_mask) < (win_w * bh * 0.35):
                continue
            mean_s = float(np.mean(w_s[w_mask]))
            mean_v = float(np.mean(w_v[w_mask]))
            prominence = mean_s + max(0.0, 130.0 - mean_v) * 0.40
            c_dist = abs((x + win_w * 0.5) - center_x) / (bw * 0.5)
            score = prominence * (1.0 - 0.40 * c_dist)
            col_scores.append((score, x))

        if not col_scores:
            cx = max(0, int(center_x - win_w * 0.5))
            return band[:, cx:cx + win_w]

        best_score, best_x = max(col_scores, key=lambda it: it[0])
        return band[:, best_x:best_x + win_w]

    @staticmethod
    def extract_torso_color_hist(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Extracts 3D HSV histogram of the core torso, eliminating background walls."""
        patch = BodyEmbedder.extract_salient_torso_patch(crop)
        if patch is None or patch.size == 0:
            return None
        try:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mask = None
            is_wall = (hsv[:, :, 1] < 20) & (hsv[:, :, 2] > 140)
            fg = ~is_wall
            if np.sum(fg) >= (patch.shape[0] * patch.shape[1] * 0.25):
                mask = fg.astype(np.uint8) * 255
            hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
            return hist
        except Exception:
            return None

    @staticmethod
    def estimate_clothing_appearance(crop: Optional[np.ndarray]) -> str:
        """
        Upper-garment colour tag (e.g. "White / Light Shirt") for visitor notes and exit matching.
        Pose-located torso, wall-relative lightness and white balance; see app/vision/clothing_color.py.
        """
        from app.vision.clothing_color import estimate_tag

        return estimate_tag(crop)

    @staticmethod
    def compute_color_similarity(hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
        """
        Computes correlation similarity between two clothing color histograms (0.0 to 1.0).
        """
        if hist1 is None or hist2 is None:
            return 0.0
        try:
            h1 = np.asarray(hist1, dtype=np.float32).flatten()
            h2 = np.asarray(hist2, dtype=np.float32).flatten()
            corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            return float(max(0.0, corr))
        except Exception:
            return 0.0

    def verify_appearance(
        self,
        crop1: Optional[np.ndarray],
        crop2: Optional[np.ndarray],
        emb1: Optional[np.ndarray] = None,
        emb2: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, bool]:
        """
        Multi-Factor Appearance Verification combining:
          1. Deep OSNet Person Re-ID embedding (512-dim cosine similarity)
          2. Upper Torso clothing 3D HSV histogram correlation
          3. Lower Legs/Pants clothing 3D HSV histogram correlation
        Returns (reid_sim, clothing_sim, combined_score, is_match).
        """
        crop1 = self.unpad_canvas(crop1)
        crop2 = self.unpad_canvas(crop2)

        if emb1 is None and crop1 is not None:
            emb1 = self.extract_embedding(crop1)
        if emb2 is None and crop2 is not None:
            emb2 = self.extract_embedding(crop2)

        reid_sim = self.compute_similarity(emb1, emb2)

        # Biometric Face Conflict Veto: only if BOTH crops have a detectable, high-quality near-frontal face
        # (eye distance >= 16px). Extreme profile / back-of-head angles produce false low similarities and must not veto appearance!
        if crop1 is not None and crop2 is not None:
            try:
                from app.vision.face_embedder import FaceEmbedder
                fe = getattr(self, "_cached_face_embedder", None)
                if fe is None:
                    fe = FaceEmbedder()
                    self._cached_face_embedder = fe
                f1, l1, r1 = fe.detect_and_crop_face(crop1)
                f2, l2, r2 = fe.detect_and_crop_face(crop2)
                if f1 is not None and f2 is not None and f1.size > 0 and f2.size > 0 and l1 and l2:
                    eye_d1 = abs(l1[1][0] - l1[0][0])
                    eye_d2 = abs(l2[1][0] - l2[0][0])
                    if eye_d1 >= 16 and eye_d2 >= 16 and f1.shape[1] >= 42 and f2.shape[1] >= 42:
                        e1 = fe.extract_embedding(f1, l1, full_image=crop1, raw_landmarks=r1)
                        e2 = fe.extract_embedding(f2, l2, full_image=crop2, raw_landmarks=r2)
                        f_sim = FaceEmbedder.compute_similarity(e1, e2)
                        if f_sim < 0.40:
                            logger.warning("Biometric face conflict in verify_appearance (sim=%.3f < 0.40). Rejecting appearance match.", f_sim)
                            return reid_sim, 0.0, float(reid_sim * 0.15), False
            except Exception:
                pass

        if crop1 is None or crop2 is None:
            # Embedding-only comparison: no clothing evidence, never enough on its own.
            return reid_sim, 0.0, float(reid_sim * 0.40), False

        # Garment evidence from pose-located torso/legs (see app/vision/clothing_color.py).
        # Fixed-fraction boxes on this doorway camera were mostly wall, so every pair looked alike.
        from app.vision import clothing_color as cc

        d1 = cc.describe(crop1)
        d2 = cc.describe(crop2)
        torso_sim = cc.hist_similarity(d1.torso, d2.torso)
        both_have_legs = d1.legs is not None and d2.legs is not None
        legs_sim = cc.hist_similarity(d1.legs, d2.legs) if both_have_legs else 0.0
        # Trousers are dark for most visitors, so legs carry less identity than the torso.
        clothing_sim = float(0.65 * torso_sim + 0.35 * legs_sim) if both_have_legs else float(torso_sim)

        # VETO 1: clearly different upper-garment colours (lighting drift between neighbours is allowed).
        if not cc.tags_compatible(d1.torso, d2.torso) and reid_sim < 0.82:
            logger.info(
                "Appearance veto: incompatible clothing '%s' vs '%s' (reid=%.3f < 0.82, torso_sim=%.3f).",
                d1.tag, d2.tag, reid_sim, torso_sim,
            )
            return reid_sim, min(clothing_sim, 0.20), float(reid_sim * 0.15), False

        # VETO 2: dark trousers vs light trousers/jeans (wall-relative lightness, robust to exposure).
        # Only on reliably located legs (side-on walkers get knees placed between the legs, sampling the
        # floor), and never against strong body + shirt agreement (VIS-2026-1041: reid 0.81, torso ~0.9).
        strong_upper_match = reid_sim >= 0.78 and torso_sim >= 0.85
        if (both_have_legs and d1.legs_reliable and d2.legs_reliable
                and reid_sim < 0.85 and not strong_upper_match):
            l1, l2 = d1.legs.rel_l, d2.legs.rel_l
            if abs(l1 - l2) >= 0.40 and min(l1, l2) <= 0.35:
                logger.info(
                    "Appearance veto: lower-body contradiction (legs rel L %.2f vs %.2f, reid=%.3f < 0.85).",
                    l1, l2, reid_sim,
                )
                return reid_sim, clothing_sim, float(reid_sim * 0.20), False

        # Decision. Validated on Main Gate entry/exit pairs: same-person turnarounds keep reid >= 0.70
        # with torso_sim >= 0.43 even front-vs-back; strangers in the same colour are left to reid.
        # No low-reid "clothing agrees" branch: dark tops + jeans are too common here, and a reid 0.62-0.70
        # branch departed the wrong visitor (VIS-2026-1047 at 12:46, reid 0.685, long vs short sleeves).
        if reid_sim >= 0.70 and torso_sim >= 0.40:
            is_match = True
            combined = float(0.60 * reid_sim + 0.40 * clothing_sim)
        else:
            is_match = False
            # Keep non-matches under the backend's 0.48 exit-eligibility bar so the logged score never
            # reads like a confident match.
            combined = min(0.45, float(0.40 * reid_sim + 0.60 * clothing_sim))

        return reid_sim, clothing_sim, combined, is_match
