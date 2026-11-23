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
        Locates the core person torso patch within a full-body CCTV frame.
        Slides across horizontal columns in the upper-body band (18% - 50% height)
        to isolate the actual person and eliminate background white walls, door reflections,
        and floor mats. Includes border masking and center-prior weighting.
        """
        crop = BodyEmbedder.unpad_canvas(crop)
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 30 or w < 10:
            return None
        band = crop[int(h * 0.18):int(h * 0.50), :]
        bh, bw = band.shape[:2]
        if bh < 10 or bw < 10:
            return None

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        win_w = max(15, int(bw * 0.35))
        min_x = int(bw * 0.10)
        max_x = max(min_x + 1, int(bw * 0.90) - win_w)
        valid_mask = (v >= 30)

        col_scores = []
        center_x = bw * 0.50
        for x in range(min_x, max_x + 1, 2):
            w_s = s[:, x:x + win_w]
            w_v = v[:, x:x + win_w]
            w_mask = valid_mask[:, x:x + win_w]
            if np.sum(w_mask) < (win_w * bh * 0.4):
                continue
            mean_s = float(np.mean(w_s[w_mask]))
            mean_v = float(np.mean(w_v[w_mask]))
            # Clothing prominence: color saturation + darkness contrast against white doorway walls
            prominence = mean_s + max(0.0, 135.0 - mean_v) * 0.35
            c_dist = abs((x + win_w * 0.5) - center_x) / (bw * 0.5)
            score = prominence * (1.0 - 0.25 * c_dist)
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
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
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

            mean_v = float(np.mean(v_chan))
            mean_s = float(np.mean(s_chan))

            # Saturated chromatic pixels (sufficient saturation S >= 22 and brightness V >= 35)
            chroma_mask = (s_chan >= 22) & (v_chan >= 35)
            chroma_pct = float(np.sum(chroma_mask)) / float(len(h_chan)) if len(h_chan) > 0 else 0.0

            # Chromatic classification: requires >= 20% chromatic pixels AND mean saturation >= 20
            if chroma_pct >= 0.20 and mean_s >= 20:
                h_chroma = h_chan[chroma_mask]
                counts = {
                    "Red / Maroon Top": np.sum((h_chroma < 14) | (h_chroma >= 162)),
                    "Orange / Rust Top": np.sum((h_chroma >= 14) & (h_chroma < 24)),
                    "Yellow / Mustard Top": np.sum((h_chroma >= 24) & (h_chroma < 38)),
                    "Green Top": np.sum((h_chroma >= 38) & (h_chroma < 85)),
                    "Blue / Navy Top": np.sum((h_chroma >= 85) & (h_chroma < 135)),
                    "Purple / Violet Top": np.sum((h_chroma >= 135) & (h_chroma < 162)),
                }
                best_tag, max_cnt = max(counts.items(), key=lambda x: x[1])
                if max_cnt >= 0.35 * len(h_chroma):
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

        # Biometric Face Conflict Veto: if both crops have a detectable face and they do not match,
        # they are proven to be two different human beings. Appearance CANNOT override face mismatch!
        if crop1 is not None and crop2 is not None:
            try:
                from app.vision.face_embedder import FaceEmbedder
                fe = getattr(self, "_cached_face_embedder", None)
                if fe is None:
                    fe = FaceEmbedder()
                    self._cached_face_embedder = fe
                f1, l1, r1 = fe.detect_and_crop_face(crop1)
                f2, l2, r2 = fe.detect_and_crop_face(crop2)
                if f1 is not None and f2 is not None and f1.size > 0 and f2.size > 0:
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
        if crop1 is not None and crop2 is not None:
            # Dynamic Salient Torso Extraction: isolates pure clothing fabric, eliminating background white doors and walls
            p1 = self.extract_salient_torso_patch(crop1)
            p2 = self.extract_salient_torso_patch(crop2)
            if p1 is not None and p2 is not None and p1.size > 0 and p2.size > 0:
                hsv1 = cv2.cvtColor(p1, cv2.COLOR_BGR2HSV)
                hsv2 = cv2.cvtColor(p2, cv2.COLOR_BGR2HSV)
                hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                cv2.normalize(hist1, hist1, alpha=1.0, norm_type=cv2.NORM_L1)
                cv2.normalize(hist2, hist2, alpha=1.0, norm_type=cv2.NORM_L1)
                torso_sim = float(max(0.0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)))

            # Lower body (trousers/skirt): 54% to 88% height, center 35% to 65% width to avoid background floor tiles
            h1, w1 = crop1.shape[:2]
            h2, w2 = crop2.shape[:2]
            l1 = crop1[int(h1 * 0.54):int(h1 * 0.88), int(w1 * 0.35):int(w1 * 0.65)]
            l2 = crop2[int(h2 * 0.54):int(h2 * 0.88), int(w2 * 0.35):int(w2 * 0.65)]
            if l1.size > 0 and l2.size > 0:
                hsv_l1 = cv2.cvtColor(l1, cv2.COLOR_BGR2HSV)
                hsv_l2 = cv2.cvtColor(l2, cv2.COLOR_BGR2HSV)
                hist_l1 = cv2.calcHist([hsv_l1], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                hist_l2 = cv2.calcHist([hsv_l2], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                cv2.normalize(hist_l1, hist_l1, alpha=1.0, norm_type=cv2.NORM_L1)
                cv2.normalize(hist_l2, hist_l2, alpha=1.0, norm_type=cv2.NORM_L1)
                legs_sim = float(max(0.0, cv2.compareHist(hist_l1, hist_l2, cv2.HISTCMP_CORREL)))

        # Composite clothing similarity: torso fabric is primary; legs cannot boost score if torso is uncorrelated (< 0.25)
        if legs_sim > 0.0 and torso_sim >= 0.25:
            clothing_sim = float(0.70 * torso_sim + 0.30 * legs_sim)
        else:
            clothing_sim = float(torso_sim)

        # Check high-level clothing appearance tags (e.g. Red / Maroon Top vs Blue / Navy Top)
        tag1 = None
        tag2 = None
        if crop1 is not None and crop2 is not None:
            tag1 = BodyEmbedder.estimate_clothing_appearance(crop1)
            tag2 = BodyEmbedder.estimate_clothing_appearance(crop2)

            compatible_clusters = [
                # Warm / Earth cluster: sunlight vs indoor shadows shift these continuously
                {"Yellow / Mustard Top", "Orange / Rust Top", "Cream / Khaki Shirt", "White / Light Shirt"},
                # Dark / Deep cluster: navy garments desaturate to black/dark under low light
                {"Black / Dark Attire", "Blue / Navy Top", "Gray Attire"},
                # Warm chromatic cluster
                {"Red / Maroon Top", "Orange / Rust Top"},
                # Cool chromatic cluster
                {"Blue / Navy Top", "Green Top", "Purple / Violet Top"},
            ]

            # Check if tags are identical, neutral, or in a shared compatible lighting cluster
            is_neutral = (tag1 in ("Casual Attire", "Neutral / Casual Attire", None) or tag2 in ("Casual Attire", "Neutral / Casual Attire", None))
            tag_compatible = (tag1 == tag2) or is_neutral or any(tag1 in c and tag2 in c for c in compatible_clusters)

            # Polar opposite veto: Only veto if tags belong to completely incompatible families
            # (e.g. vibrant Red vs bright Green, or pure Yellow vs pitch Black) AND ReID doesn't overwhelmingly prove otherwise
            if not tag_compatible and reid_sim < 0.82:
                logger.info(
                    "Appearance veto: incompatible clothing tags '%s' vs '%s' (reid=%.3f < 0.82).",
                    tag1,
                    tag2,
                    reid_sim,
                )
                return reid_sim, clothing_sim, float(reid_sim * 0.20), False

            # Direct Torso Luminance Verification:
            # Only veto if one garment is deep black (V < 45) and the other is brilliant white (V > 165)
            if hsv1 is not None and hsv2 is not None:
                try:
                    v1_median = float(np.median(hsv1[:, :, 2]))
                    v2_median = float(np.median(hsv2[:, :, 2]))
                    if (v1_median < 45.0 and v2_median > 165.0) or (v2_median < 45.0 and v1_median > 165.0):
                        if reid_sim < 0.85:
                            logger.info("Appearance veto: true black vs white conflict (V1=%.1f vs V2=%.1f).", v1_median, v2_median)
                            return reid_sim, clothing_sim, float(reid_sim * 0.20), False
                except Exception:
                    pass

        # Multi-Factor Decision:
        # In single-camera doorway CCTV, static background (door frame, floor tiles)
        # artificially inflates OSNet embeddings (often 0.70 - 0.85 for any two people).
        # However, 180-degree turnarounds (front entering vs back exiting) reduce OSNet from ~0.80 to 0.58-0.66.
        if tag_compatible:
            # Verified compatible clothing category across 180-deg turnaround or indoor illumination
            if reid_sim >= 0.58 and (clothing_sim >= 0.25 or torso_sim >= 0.20):
                is_match = True
                combined = float(0.55 * reid_sim + 0.45 * max(clothing_sim, torso_sim))
            elif reid_sim >= 0.68 and (clothing_sim >= 0.18 or torso_sim >= 0.15):
                # Strong Re-ID match supported by non-conflicting color correlation
                is_match = True
                combined = float(0.65 * reid_sim + 0.35 * max(0.25, clothing_sim))
            else:
                is_match = False
                combined = float(reid_sim * 0.40 + clothing_sim * 0.60)
        else:
            # Different color clusters: requires high Re-ID proof and positive clothing correlation
            if reid_sim >= 0.78 and clothing_sim >= 0.40:
                is_match = True
                combined = float(0.50 * reid_sim + 0.50 * clothing_sim)
            else:
                is_match = False
                combined = float(reid_sim * 0.40 + clothing_sim * 0.60)

        return reid_sim, clothing_sim, combined, is_match
