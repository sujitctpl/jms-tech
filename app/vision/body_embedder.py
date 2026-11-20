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
        self.model_path = model_path or os.path.join("models", "osnet_x0_25.onnx")
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

    def extract_embedding(self, body_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a 512-dim L2-normalized ReID embedding vector from a full-body person image crop.
        Accepts BGR image (OpenCV format).
        """
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
    def extract_torso_color_hist(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Backwards-compatible helper returning 3D upper-torso histogram."""
        return BodyEmbedder.extract_body_part_hist(crop, 0.22, 0.52)

    @staticmethod
    def estimate_clothing_appearance(crop: Optional[np.ndarray]) -> str:
        """
        Estimates the primary upper-torso clothing appearance (color/attire tag).
        Used for customer session notes and quick reception identification.
        Filters out low-saturation background floors (e.g. wood laminate/tiles)
        to prevent floor color contamination from corrupting clothing classification.
        """
        if crop is None or crop.size == 0:
            return "Casual Attire"
        try:
            h, w = crop.shape[:2]
            if h < 30 or w < 15:
                return "Casual Attire"
            # Central chest/torso patch: tight inset (24% to 52% height, 30% to 70% width)
            # strictly isolates the chest fabric and eliminates door frame / floor bleed
            ty1 = int(h * 0.24)
            ty2 = int(h * 0.52)
            tx1 = int(w * 0.30)
            tx2 = int(w * 0.70)
            torso = crop[ty1:ty2, tx1:tx2]
            if torso.size == 0:
                return "Casual Attire"
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            h_chan = hsv[:, :, 0].flatten()
            s_chan = hsv[:, :, 1].flatten()
            v_chan = hsv[:, :, 2].flatten()

            mean_v = float(np.mean(v_chan))
            mean_s = float(np.mean(s_chan))

            # Saturated chromatic pixels (sufficient saturation S >= 35 and brightness V >= 45)
            chroma_mask = (s_chan >= 35) & (v_chan >= 45)
            chroma_pct = float(np.sum(chroma_mask)) / float(len(h_chan)) if len(h_chan) > 0 else 0.0

            # Chromatic classification: requires >= 25% chromatic pixels AND mean saturation >= 26
            # This cleanly rejects dark/black/gray garments where a small 5-10% wood frame or skin patch
            # might otherwise corrupt the classification
            if chroma_pct >= 0.25 and mean_s >= 26:
                median_h = float(np.median(h_chan[chroma_mask]))
                # Hue ranges in OpenCV HSV: [0, 180]
                if median_h < 12 or median_h >= 165:
                    return "Red / Maroon Top"
                if 12 <= median_h < 22:
                    return "Orange / Rust Top"
                if 22 <= median_h < 38:
                    return "Yellow / Mustard Top"
                if 38 <= median_h < 85:
                    return "Green Top"
                if 85 <= median_h < 135:
                    return "Blue / Navy Top"
                if 135 <= median_h < 165:
                    return "Purple / Violet Top"

            # Achromatic garments (Black, Gray, White, Cream)
            # Under indoor CCTV illumination, black/dark shirts typically exhibit V in 40-118 with low saturation
            if mean_v < 118 and mean_s < 26:
                return "Black / Dark Attire"
            if mean_s < 25 and mean_v >= 120:
                return "White / Light Shirt"
            if mean_s < 26:
                return "Gray Attire"
            if mean_s < 38 and mean_v >= 105:
                return "Cream / Khaki Shirt"

            return "Casual Attire"
        except Exception:
            return "Casual Attire"

    @staticmethod
    def compute_color_similarity(hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
        """
        Computes correlation similarity between two clothing color histograms (0.0 to 1.0).
        """
        if hist1 is None or hist2 is None:
            return 0.0
        try:
            corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
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
        if emb1 is None and crop1 is not None:
            emb1 = self.extract_embedding(crop1)
        if emb2 is None and crop2 is not None:
            emb2 = self.extract_embedding(crop2)

        reid_sim = self.compute_similarity(emb1, emb2)

        torso_sim = 0.0
        legs_sim = 0.0
        if crop1 is not None and crop2 is not None:
            t1 = self.extract_body_part_hist(crop1, 0.18, 0.52)
            t2 = self.extract_body_part_hist(crop2, 0.18, 0.52)
            torso_sim = self.compute_color_similarity(t1, t2)

            l1 = self.extract_body_part_hist(crop1, 0.58, 0.90)
            l2 = self.extract_body_part_hist(crop2, 0.58, 0.90)
            legs_sim = self.compute_color_similarity(l1, l2)

        # Check high-level clothing appearance tags (e.g. Red / Maroon Top vs Blue / Navy Top)
        tag1 = None
        tag2 = None
        if crop1 is not None and crop2 is not None:
            tag1 = BodyEmbedder.estimate_clothing_appearance(crop1)
            tag2 = BodyEmbedder.estimate_clothing_appearance(crop2)

            chromatic_tags = {
                "Red / Maroon Top", "Orange / Rust Top", "Yellow / Mustard Top",
                "Green Top", "Blue / Navy Top", "Purple / Violet Top"
            }
            # 1. Hard conflict: two distinct saturated chromatic colors (e.g. Red vs Green, Red vs Blue, Orange vs Blue)
            if tag1 in chromatic_tags and tag2 in chromatic_tags and tag1 != tag2:
                return reid_sim, 0.0, float(reid_sim * 0.25), False

            # 2. Hard chromatic vs achromatic conflict:
            # Saturated chromatic tops (Red, Orange, Yellow, Green, Purple, Blue)
            # NEVER match achromatic tops (Gray Attire, Black / Dark Attire, White / Light Shirt, Cream / Khaki Shirt).
            achromatic = {"Gray Attire", "Black / Dark Attire", "White / Light Shirt", "Cream / Khaki Shirt"}
            if (tag1 in chromatic_tags and tag2 in achromatic) or \
               (tag2 in chromatic_tags and tag1 in achromatic):
                return reid_sim, 0.0, float(reid_sim * 0.20), False

            # 3. Hard contrast conflict: Dark/Black vs White/Light or Cream
            if ("Black" in (tag1 or "") and "White" in (tag2 or "")) or ("White" in (tag1 or "") and "Black" in (tag2 or "")):
                return reid_sim, 0.0, float(reid_sim * 0.20), False
            if ("Black" in (tag1 or "") and "Cream" in (tag2 or "")) or ("Cream" in (tag1 or "") and "Black" in (tag2 or "")):
                return reid_sim, 0.0, float(reid_sim * 0.25), False
            if ("Black" in (tag1 or "") and tag2 in {"Red / Maroon Top", "Yellow / Mustard Top", "Orange / Rust Top"}):
                return reid_sim, 0.0, float(reid_sim * 0.25), False
            if ("Black" in (tag2 or "") and tag1 in {"Red / Maroon Top", "Yellow / Mustard Top", "Orange / Rust Top"}):
                return reid_sim, 0.0, float(reid_sim * 0.25), False

            # Direct Torso Luminance Verification:
            # Prevents light/white garments from matching dark garments even if tags are neutral
            try:
                h1, w1 = crop1.shape[:2]
                h2, w2 = crop2.shape[:2]
                p1 = crop1[int(h1 * 0.24):int(h1 * 0.52), int(w1 * 0.30):int(w1 * 0.70)]
                p2 = crop2[int(h2 * 0.24):int(h2 * 0.52), int(w2 * 0.30):int(w2 * 0.70)]
                if p1.size > 0 and p2.size > 0:
                    v1_vals = cv2.cvtColor(p1, cv2.COLOR_BGR2HSV)[:, :, 2]
                    v2_vals = cv2.cvtColor(p2, cv2.COLOR_BGR2HSV)[:, :, 2]
                    v1_median = float(np.median(v1_vals))
                    v2_median = float(np.median(v2_vals))
                    if abs(v1_median - v2_median) > 65.0:
                        return reid_sim, 0.0, float(reid_sim * 0.25), False
            except Exception:
                pass

        # Composite clothing similarity: Upper-body (torso/shirt) is the primary garment!
        # If torso color correlation is near zero (< 0.20), the shirts are completely different.
        # Pants/jeans similarity alone CANNOT override an unmatching shirt.
        if torso_sim >= 0.20 and legs_sim > 0.0:
            clothing_sim = float(0.65 * torso_sim + 0.35 * legs_sim)
        elif torso_sim >= 0.20:
            clothing_sim = float(torso_sim)
        else:
            # Torso has low or zero correlation -> reject clothing match
            clothing_sim = 0.0

        # Multi-Factor Decision:
        # In single-camera doorway CCTV, static background (door frame, floor tiles)
        # artificially inflates OSNet embeddings (often 0.70 - 0.85 for any two people).
        # Therefore, OSNet alone CANNOT authorize a match without positive clothing correlation.
        tags_identical = bool(tag1 and tag2 and tag1 == tag2 and tag1 not in ("Casual Attire", "Neutral / Casual Attire"))

        # Calibration for Match Decision:
        # Strict clothing color requirement: under NO circumstances may is_match be True
        # if clothing color similarity is below 0.36 or torso_sim is below 0.22!
        if clothing_sim < 0.36 or torso_sim < 0.22:
            is_match = False
            combined = float(reid_sim * 0.40 + clothing_sim * 0.60)
        elif tags_identical and reid_sim >= 0.70 and clothing_sim >= 0.36:
            # Verified identical clothing category with confirmed color similarity
            is_match = True
            combined = float(0.55 * reid_sim + 0.45 * clothing_sim)
        elif reid_sim >= 0.75 and clothing_sim >= 0.36 and torso_sim >= 0.28:
            # High Re-ID with positive multi-part clothing color confirmation
            is_match = True
            combined = float(0.50 * reid_sim + 0.50 * clothing_sim)
        elif reid_sim >= 0.68 and clothing_sim >= 0.48 and torso_sim >= 0.35:
            # Moderate Re-ID with strong clothing correlation
            is_match = True
            combined = float(0.45 * reid_sim + 0.55 * clothing_sim)
        else:
            is_match = False
            combined = float(reid_sim * 0.40 + clothing_sim * 0.60)

        return reid_sim, clothing_sim, combined, is_match
