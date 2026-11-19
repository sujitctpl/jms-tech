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
        x1_pct: float = 0.20,
        x2_pct: float = 0.80,
    ) -> Optional[np.ndarray]:
        """
        Extracts a normalized 3D (Hue-Saturation-Value) color histogram of a specific body region
        (torso or legs) to isolate clothing and distinguish both chromatic (colors) and achromatic
        (white, gray, black) garments.
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
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist
        except Exception:
            return None

    @staticmethod
    def extract_torso_color_hist(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Backwards-compatible helper returning 3D upper-torso histogram."""
        return BodyEmbedder.extract_body_part_hist(crop, 0.18, 0.52)

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

        # Composite clothing similarity across upper and lower body
        if torso_sim >= 0.50 and legs_sim >= 0.50:
            clothing_sim = float(0.5 * torso_sim + 0.5 * legs_sim)
        else:
            clothing_sim = float(max(torso_sim, legs_sim))

        # Strict Multi-Factor Decision:
        # 1. Hard Re-ID Floor: Must be >= 0.81 (OSNet rejects different individuals sharing same clothes)
        # 2. Clothing Consistency: Must be >= 0.60
        # 3. Overall Combined Score: Must be >= 0.80
        if reid_sim < 0.81:
            combined = float(reid_sim * 0.85)
            is_match = False
        else:
            combined = float(0.6 * reid_sim + 0.4 * clothing_sim)
            is_match = bool(combined >= 0.80 and clothing_sim >= 0.60)

        return reid_sim, clothing_sim, combined, is_match
