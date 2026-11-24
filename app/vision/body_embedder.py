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
        Estimates the primary upper-torso clothing appearance (color/attire tag).
        Used for customer session notes and quick reception identification.
        Filters out low-saturation background floors & walls (e.g. white doors/tiles)
        to prevent background bleed from corrupting clothing classification.
        """
        patch = BodyEmbedder.extract_salient_torso_patch(crop)
        if patch is None or patch.size == 0:
            return "Casual Attire"
        try:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            h_chan = hsv[:, :, 0].flatten()
            s_chan = hsv[:, :, 1].flatten()
            v_chan = hsv[:, :, 2].flatten()

            # Exclude bright white door/wall background if clothing pixels are present
            is_wall = (s_chan < 20) & (v_chan > 140)
            fg = ~is_wall
            if np.sum(fg) >= (len(v_chan) * 0.25):
                eff_v = v_chan[fg]
                eff_s = s_chan[fg]
                mean_v = float(np.mean(eff_v))
                mean_s = float(np.mean(eff_s))
                eff_h = h_chan[fg]
            else:
                mean_v = float(np.mean(v_chan))
                mean_s = float(np.mean(s_chan))
                eff_h = h_chan
                eff_s = s_chan
                eff_v = v_chan

            # Saturated chromatic pixels (sufficient saturation S >= 30 and brightness V >= 35)
            chroma_mask = (eff_s >= 30) & (eff_v >= 35)
            chroma_pct = float(np.sum(chroma_mask)) / float(len(eff_h)) if len(eff_h) > 0 else 0.0

            # Chromatic classification: requires >= 45% chromatic pixels AND mean saturation >= 38
            if chroma_pct >= 0.45 and mean_s >= 38:
                h_chroma = eff_h[chroma_mask]
                counts = {
                    "Red / Maroon Top": np.sum((h_chroma < 14) | (h_chroma >= 162)),
                    "Orange / Rust Top": np.sum((h_chroma >= 14) & (h_chroma < 24)),
                    "Yellow / Mustard Top": np.sum((h_chroma >= 24) & (h_chroma < 38)),
                    "Green Top": np.sum((h_chroma >= 38) & (h_chroma < 85)),
                    "Blue / Navy Top": np.sum((h_chroma >= 85) & (h_chroma < 135)),
                    "Purple / Violet Top": np.sum((h_chroma >= 135) & (h_chroma < 162)),
                }
                best_tag, max_cnt = max(counts.items(), key=lambda x: x[1])
                if max_cnt >= 0.45 * len(h_chroma) and max_cnt >= 0.20 * len(eff_h):
                    return best_tag

            # Achromatic garments (Black, Gray, White, Cream) via Perceptual Vector Quantization
            # Instead of rigid step-function thresholds that break under illumination shifts,
            # we classify achromatic garments into their nearest perceptual centroid in [S, V, Warmth] space.
            # Warmth = mean(R) - mean(B) separates neutral/cool grays from warm creams/beiges.
            b_chan = patch[:, :, 0].flatten()
            r_chan = patch[:, :, 2].flatten()
            mean_r = float(np.mean(r_chan))
            mean_b = float(np.mean(b_chan))
            warmth = mean_r - mean_b

            # Feature vector: [Saturation, Value, Warmth]
            feat = np.array([mean_s, mean_v, warmth], dtype=np.float32)

            # Calibrated anchor centroids: [S, V, Warmth]
            # Black: low V, low S, neutral warmth
            # Gray: mid V, low S, neutral/cool warmth
            # White: high V, low S, neutral warmth
            # Cream/Khaki: mid-high V, moderate S, positive warmth (R > B)
            achromatic_anchors = {
                "Black / Dark Attire": np.array([10.0, 50.0, 0.0], dtype=np.float32),
                "Gray Attire": np.array([12.0, 110.0, 0.0], dtype=np.float32),
                "White / Light Shirt": np.array([8.0, 160.0, 0.0], dtype=np.float32),
                "Cream / Khaki Shirt": np.array([24.0, 120.0, 8.0], dtype=np.float32),
            }

            # Perceptual metric weights: [w_s, w_v, w_warmth]
            weights = np.array([1.5, 0.8, 2.0], dtype=np.float32)

            best_tag = "Casual Attire"
            min_dist = float("inf")
            for tag, anchor in achromatic_anchors.items():
                dist = float(np.sqrt(np.sum((weights * (feat - anchor)) ** 2)))
                if dist < min_dist:
                    min_dist = dist
                    best_tag = tag

            return best_tag

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

        torso_sim = 0.0
        legs_sim = 0.0
        hsv1 = None
        hsv2 = None
        hsv_l1 = None
        hsv_l2 = None
        both_have_legs = False
        legs_contradiction = False

        if crop1 is not None and crop2 is not None:
            # 1. Torso Region Color Extraction (inner 60% of bounding box)
            hist_torso1 = self.extract_body_part_hist(crop1, 0.18, 0.52, 0.20, 0.80)
            hist_torso2 = self.extract_body_part_hist(crop2, 0.18, 0.52, 0.20, 0.80)
            torso_sim = self.compute_color_similarity(hist_torso1, hist_torso2)

            h1, w1 = crop1.shape[:2]
            h2, w2 = crop2.shape[:2]
            p1 = crop1[int(h1 * 0.18):int(h1 * 0.52), int(w1 * 0.20):int(w1 * 0.80)]
            p2 = crop2[int(h2 * 0.18):int(h2 * 0.52), int(w2 * 0.20):int(w2 * 0.80)]
            if p1.size > 0 and p2.size > 0:
                hsv1 = cv2.cvtColor(p1, cv2.COLOR_BGR2HSV)
                hsv2 = cv2.cvtColor(p2, cv2.COLOR_BGR2HSV)

            # 2. Lower-Body (Trousers/Skirt) Region Extraction
            both_have_legs = (h1 >= 140 and h2 >= 140)
            if both_have_legs:
                hist_legs1 = self.extract_body_part_hist(crop1, 0.56, 0.90, 0.20, 0.80)
                hist_legs2 = self.extract_body_part_hist(crop2, 0.56, 0.90, 0.20, 0.80)
                legs_sim = self.compute_color_similarity(hist_legs1, hist_legs2)

                leg1 = crop1[int(h1 * 0.56):int(h1 * 0.90), int(w1 * 0.20):int(w1 * 0.80)]
                leg2 = crop2[int(h2 * 0.56):int(h2 * 0.90), int(w2 * 0.20):int(w2 * 0.80)]
                if leg1.size > 0 and leg2.size > 0:
                    hsv_l1 = cv2.cvtColor(leg1, cv2.COLOR_BGR2HSV)
                    hsv_l2 = cv2.cvtColor(leg2, cv2.COLOR_BGR2HSV)

                    # Direct Lower-Body Color & Luminance Contradiction Check:
                    med_l1 = np.median(hsv_l1, axis=(0, 1))
                    med_l2 = np.median(hsv_l2, axis=(0, 1))
                    v1_leg, v2_leg = float(med_l1[2]), float(med_l2[2])
                    h1_leg, h2_leg = float(med_l1[0]), float(med_l2[0])
                    s1_leg, s2_leg = float(med_l1[1]), float(med_l2[1])
                    v1_leg_p25 = float(np.percentile(hsv_l1[:, :, 2], 25))
                    v2_leg_p25 = float(np.percentile(hsv_l2[:, :, 2], 25))

                    is_dark1 = (v1_leg <= 65.0 and v1_leg_p25 <= 50.0)
                    is_dark2 = (v2_leg <= 65.0 and v2_leg_p25 <= 50.0)
                    is_light1 = (v1_leg >= 110.0 and v1_leg_p25 >= 80.0)
                    is_light2 = (v2_leg >= 110.0 and v2_leg_p25 >= 80.0)
                    is_denim1 = (s1_leg >= 25.0 and 85.0 <= h1_leg <= 135.0 and v1_leg >= 85.0)
                    is_denim2 = (s2_leg >= 25.0 and 85.0 <= h2_leg <= 135.0 and v2_leg >= 85.0)

                    # Contradiction: one has dark trousers while other has light/denim pants
                    if (is_dark1 and (is_light2 or is_denim2)) or (is_dark2 and (is_light1 or is_denim1)):
                        legs_contradiction = True

        # Composite clothing similarity: balanced upper torso and lower legs when full body is present
        if both_have_legs:
            clothing_sim = float(0.50 * torso_sim + 0.50 * legs_sim)
        else:
            # Portrait / bust: evaluate based on torso fabric
            clothing_sim = float(torso_sim)

        # Check high-level clothing appearance tags (e.g. Red / Maroon Top vs Blue / Navy Top)
        tag1 = None
        tag2 = None
        if crop1 is not None and crop2 is not None:
            tag1 = BodyEmbedder.estimate_clothing_appearance(crop1)
            tag2 = BodyEmbedder.estimate_clothing_appearance(crop2)

            compatible_clusters = [
                # Warm / Earth / Neutral cluster: sunlight vs indoor shadows shift these continuously
                {"Yellow / Mustard Top", "Orange / Rust Top", "Cream / Khaki Shirt", "White / Light Shirt"},
                # Dark / Deep cluster: navy garments desaturate to black/dark under low light
                {"Black / Dark Attire", "Blue / Navy Top"},
                # Warm chromatic cluster
                {"Red / Maroon Top", "Orange / Rust Top"},
                # Cool chromatic cluster
                {"Blue / Navy Top", "Green Top", "Purple / Violet Top"},
            ]

            # Check if tags are identical, neutral, have strong direct torso & clothing agreement, or in a shared compatible lighting cluster
            is_neutral = (tag1 in ("Casual Attire", "Neutral / Casual Attire", None) or tag2 in ("Casual Attire", "Neutral / Casual Attire", None))
            tag_compatible = (tag1 == tag2) or is_neutral or (torso_sim >= 0.68 and clothing_sim >= 0.60) or any(tag1 in c and tag2 in c for c in compatible_clusters)

            # Polar opposite veto: Only veto if tags belong to completely incompatible families
            # and torso clothing correlation is low (if torso_sim >= 0.60, the color histograms directly agree)
            if not tag_compatible and torso_sim < 0.60 and reid_sim < 0.82:
                logger.info(
                    "Appearance veto: incompatible clothing tags '%s' vs '%s' (reid=%.3f < 0.82, torso_sim=%.3f).",
                    tag1,
                    tag2,
                    reid_sim,
                    torso_sim,
                )
                return reid_sim, min(clothing_sim, 0.20), float(reid_sim * 0.15), False

            # Direct Torso Luminance Verification:
            # Veto if one garment is deep pitch black (V < 45) and the other is bright white/light (V > 155)
            if hsv1 is not None and hsv2 is not None:
                try:
                    v1_median = float(np.median(hsv1[:, :, 2]))
                    v2_median = float(np.median(hsv2[:, :, 2]))
                    if (v1_median < 45.0 and v2_median > 155.0) or (v2_median < 45.0 and v1_median > 155.0):
                        if reid_sim < 0.85:
                            logger.info("Appearance veto: true black vs white conflict (V1=%.1f vs V2=%.1f).", v1_median, v2_median)
                            return reid_sim, clothing_sim, float(reid_sim * 0.20), False
                except Exception:
                    pass

            # HARD VETO: Torso Chromatic Contradiction
            # Veto if one garment is achromatic / white / gray (S_med <= 16, chroma < 12%)
            # while the other is clearly chromatic / colored (S_med >= 28, chroma >= 35%) and torso_sim < 0.65
            if hsv1 is not None and hsv2 is not None:
                try:
                    s1_med = float(np.median(hsv1[:, :, 1]))
                    s2_med = float(np.median(hsv2[:, :, 1]))
                    chroma1 = np.sum((hsv1[:, :, 1] >= 30) & (hsv1[:, :, 2] >= 35)) / float(hsv1[:, :, 0].size)
                    chroma2 = np.sum((hsv2[:, :, 1] >= 30) & (hsv2[:, :, 2] >= 35)) / float(hsv2[:, :, 0].size)
                    chromatic_conflict = (
                        (s1_med <= 16.0 and chroma1 < 0.12 and s2_med >= 28.0 and chroma2 >= 0.35 and torso_sim < 0.65)
                        or (s2_med <= 16.0 and chroma2 < 0.12 and s1_med >= 28.0 and chroma1 >= 0.35 and torso_sim < 0.65)
                    )
                    if chromatic_conflict and reid_sim < 0.85:
                        logger.info(
                            "Appearance veto: torso chromatic conflict (S1=%.1f, chroma1=%.1f%% vs S2=%.1f, chroma2=%.1f%%, torso_sim=%.3f, reid=%.3f).",
                            s1_med,
                            chroma1 * 100,
                            s2_med,
                            chroma2 * 100,
                            torso_sim,
                            reid_sim,
                        )
                        return reid_sim, min(clothing_sim, 0.20), float(reid_sim * 0.15), False
                except Exception:
                    pass

        # HARD VETO: Inverted Attire Polarity Veto (e.g. Light Top + Dark Pants vs Dark Top + Light Pants)
        if both_have_legs and hsv1 is not None and hsv2 is not None and hsv_l1 is not None and hsv_l2 is not None:
            try:
                v1_torso = float(np.median(hsv1[:, :, 2]))
                v2_torso = float(np.median(hsv2[:, :, 2]))
                v1_leg = float(np.median(hsv_l1[:, :, 2]))
                v2_leg = float(np.median(hsv_l2[:, :, 2]))
                contrast1 = v1_torso - v1_leg
                contrast2 = v2_torso - v2_leg
                inverted_polarity = (
                    (v1_torso >= 105.0 and v1_leg <= 75.0 and contrast1 >= 30.0 and v2_torso <= 80.0 and v2_leg >= 105.0 and contrast2 <= -25.0)
                    or (v2_torso >= 105.0 and v2_leg <= 75.0 and contrast2 >= 30.0 and v1_torso <= 80.0 and v1_leg >= 105.0 and contrast1 <= -25.0)
                )
                if inverted_polarity and reid_sim < 0.88:
                    logger.info(
                        "Appearance veto: inverted attire polarity (contrast1=%.1f, contrast2=%.1f, reid=%.3f).",
                        contrast1,
                        contrast2,
                        reid_sim,
                    )
                    return reid_sim, min(clothing_sim, 0.20), float(reid_sim * 0.15), False
            except Exception:
                pass

        # HARD VETO: Lower-Body Contradiction Veto (e.g. black trousers vs light blue jeans)
        if both_have_legs and legs_contradiction and reid_sim < 0.85:
            logger.info(
                "Appearance veto: lower-body clothing contradiction (legs_sim=%.3f, reid=%.3f < 0.85).",
                legs_sim,
                reid_sim,
            )
            return reid_sim, clothing_sim, float(reid_sim * 0.20), False

        # Multi-Factor Decision:
        # In single-camera doorway CCTV, static background (door frame, floor tiles)
        # artificially inflates OSNet embeddings (often 0.70 - 0.85 for any two people).
        # However, 180-degree turnarounds (front entering vs back exiting) reduce OSNet to 0.58-0.75,
        # and front accessories (lanyards/badges/collars) reduce torso histogram to 0.15-0.30.
        if tag_compatible:
            if both_have_legs:
                # Branch 1: High Re-ID (>= 0.70) with verified compatible clothing tags (e.g. VIS-1051)
                if reid_sim >= 0.70 and clothing_sim >= 0.22:
                    is_match = True
                    combined = float(0.60 * reid_sim + 0.40 * clothing_sim)
                # Branch 2: Standard turnaround Re-ID (>= 0.58) with balanced clothing agreement
                elif reid_sim >= 0.58 and clothing_sim >= 0.25 and (torso_sim >= 0.14 or legs_sim >= 0.25):
                    is_match = True
                    combined = float(0.55 * reid_sim + 0.45 * clothing_sim)
                else:
                    is_match = False
                    combined = float(reid_sim * 0.40 + clothing_sim * 0.60)
            else:
                # Portrait / partial crops
                if reid_sim >= 0.68 and torso_sim >= 0.16:
                    is_match = True
                    combined = float(0.60 * reid_sim + 0.40 * torso_sim)
                elif reid_sim >= 0.58 and torso_sim >= 0.22:
                    is_match = True
                    combined = float(0.50 * reid_sim + 0.50 * torso_sim)
                else:
                    is_match = False
                    combined = float(reid_sim * 0.40 + clothing_sim * 0.60)
        else:
            # Incompatible color clusters (e.g. Gray vs Black): requires extraordinary Re-ID proof (>= 0.82)
            # or high clothing correlation, which safely rejects different people in dark attire.
            if (reid_sim >= 0.82 and clothing_sim >= 0.40) or (reid_sim >= 0.74 and clothing_sim >= 0.60):
                is_match = True
                combined = float(0.50 * reid_sim + 0.50 * clothing_sim)
            else:
                is_match = False
                combined = float(reid_sim * 0.40 + clothing_sim * 0.60)

        return reid_sim, clothing_sim, combined, is_match
