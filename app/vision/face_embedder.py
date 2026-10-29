import hashlib
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
from app.core.config import settings
from app.core.logging import logger

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class FaceEmbedder:
    """
    ArcFace / InsightFace face embedding extraction engine and quality assessor.
    Extracts L2-normalized 512-dimensional feature vectors.
    """

    # Canonical ArcFace 112x112 5-point landmark template (left eye, right eye, nose, mouth left, mouth right)
    ARCFACE_REF_PTS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_size: Optional[int] = None,
        min_confidence: Optional[float] = None,
        blur_threshold: Optional[float] = None,
    ):
        self.model_path = model_path or settings.ARCFACE_MODEL_PATH
        self.min_size = min_size if min_size is not None else settings.FACE_MIN_SIZE
        self.min_confidence = (
            min_confidence if min_confidence is not None else settings.FACE_MIN_CONFIDENCE
        )
        self.blur_threshold = (
            blur_threshold if blur_threshold is not None else settings.FACE_BLUR_THRESHOLD
        )

        self.session = None
        self.input_name = None
        self.output_name = None
        self._init_session()

    def _init_session(self) -> None:
        """Initializes ONNX Runtime session if model exists."""
        if ort is None:
            return

        resolved = self.model_path
        if not os.path.exists(resolved):
            # Check models directory
            alt = os.path.join("models", os.path.basename(resolved))
            if os.path.exists(alt):
                resolved = alt

        if os.path.exists(resolved):
            try:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                available = ort.get_available_providers()
                active = [p for p in providers if p in available]
                self.session = ort.InferenceSession(resolved, providers=active)
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                logger.info("ArcFace embedding model loaded successfully from %s", resolved)
            except Exception as e:
                logger.warning("Could not load ArcFace model from %s: %s", resolved, str(e))
                self.session = None

    def is_face_reliable(
        self,
        face_crop: Optional[np.ndarray],
        confidence: float,
        landmarks: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[bool, str]:
        """
        Validates whether a face detection is sufficiently clear and reliable
        for persistent identity recovery.

        Returns (is_reliable, reason).
        """
        if face_crop is None or face_crop.size == 0:
            return False, "empty_image"

        h, w = face_crop.shape[:2]

        # 1. Minimum Resolution Check
        if h < self.min_size or w < self.min_size:
            return False, f"too_small_{w}x{h}"

        # 2. Aspect Ratio Check (Faces are roughly square or slightly taller than wide)
        ar = float(w) / float(h)
        if ar < 0.45 or ar > 1.8:
            return False, f"invalid_aspect_ratio_{ar:.2f}"

        # 3. Detection Confidence Check
        if confidence < self.min_confidence:
            return False, f"low_confidence_{confidence:.2f}"

        # 4. Blur / Sharpness Check using Laplacian Variance
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.blur_threshold:
            return False, f"blurry_laplacian_{laplacian_var:.1f}"

        # 5. Landmark Geometry & Pose Check (if landmarks available)
        if landmarks and len(landmarks) >= 5:
            # Standard 5 landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
            lx, ly = landmarks[0]
            rx, ry = landmarks[1]
            nx, ny = landmarks[2]

            eye_dist = np.hypot(rx - lx, ry - ly)
            if eye_dist < 3.5:
                return False, f"eye_distance_too_small_{eye_dist:.1f}"

            # Profile pose check: nose x must lie reasonably between eyes
            min_x = min(lx, rx) - 8
            max_x = max(lx, rx) + 8
            if not (min_x <= nx <= max_x):
                return False, "extreme_profile_pose"

        return True, "reliable"

    def extract_embedding(
        self,
        face_crop: np.ndarray,
        landmarks: Optional[List[Tuple[int, int]]] = None,
        full_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extracts an L2-normalized 512-dimensional face embedding.
        If full_image and landmarks (in full_image coords) are provided, performs
        lossless affine alignment directly from the full image to avoid crop border artifacts.
        """
        if face_crop is None or face_crop.size == 0:
            raise ValueError("Invalid empty face crop provided for embedding extraction")

        if self.session is not None:
            # 1. Align face with canonical 5-point landmark template if available
            resized = None
            if landmarks is not None and len(landmarks) == 5:
                src_pts = np.array(landmarks, dtype=np.float32)
                M, _ = cv2.estimateAffinePartial2D(src_pts, self.ARCFACE_REF_PTS)
                if M is not None:
                    src_mat = full_image if full_image is not None else face_crop
                    resized = cv2.warpAffine(src_mat, M, (112, 112), borderValue=0.0)

            if resized is None:
                resized = cv2.resize(face_crop, (112, 112))

            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            elif resized.shape[2] == 4:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)

            # ArcFace expects RGB with (x - 127.5) / 128.0
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = ((rgb.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)
            blob = np.expand_dims(normalized, axis=0)

            outs = self.session.run([self.output_name], {self.input_name: blob})
            raw_embedding = outs[0].flatten().astype(np.float32)
        else:
            # Deterministic, repeatable fallback embedding generator based on image spatial features
            # Used in unit testing and environments without pre-downloaded 200MB ArcFace weights
            raw_embedding = self._generate_synthetic_embedding(face_crop)

        # Strictly L2-normalize so cosine similarity equals inner product
        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            normalized_embedding = (raw_embedding / norm).astype(np.float32)
        else:
            normalized_embedding = np.zeros(512, dtype=np.float32)
            normalized_embedding[0] = 1.0

        return normalized_embedding

    def detect_and_crop_face(
        self,
        image: np.ndarray,
        margin_ratio: float = 0.12,
    ) -> Tuple[np.ndarray, Optional[List[Tuple[int, int]]]]:
        """
        Detects the primary face in a full-frame photo / webcam snapshot and crops it
        so ArcFace receives an aligned, tight face crop matching live detection feeds.
        If no face is detected or if image is already a small crop, returns the original image.
        """
        if image is None or image.size == 0:
            return image, None

        h, w = image.shape[:2]
        if h <= 128 and w <= 128:
            return image, None

        # Check for OpenCV YuNet face detection model
        candidate_paths = [
            os.path.join("models", "face_detection_yunet_2023mar.onnx"),
            "face_detection_yunet_2023mar.onnx",
        ]
        yunet_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                yunet_path = p
                break

        if yunet_path and hasattr(cv2, "FaceDetectorYN_create"):
            try:
                scale = 1.0
                max_dim = 640
                det_img = image
                if max(h, w) > max_dim:
                    scale = max_dim / float(max(h, w))
                    new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
                    det_img = cv2.resize(image, (new_w, new_h))
                else:
                    new_w, new_h = w, h

                detector = cv2.FaceDetectorYN_create(yunet_path, "", (new_w, new_h), score_threshold=0.30)
                _, raw_faces = detector.detect(det_img)

                if raw_faces is not None and len(raw_faces) > 0:
                    # Select face with greatest area
                    best_face = None
                    best_area = -1.0
                    for f in raw_faces:
                        area = float(f[2]) * float(f[3])
                        if area > best_area:
                            best_area = area
                            best_face = f

                    if best_face is not None:
                        fx = float(best_face[0]) / scale
                        fy = float(best_face[1]) / scale
                        fw = float(best_face[2]) / scale
                        fh = float(best_face[3]) / scale

                        mx = fw * margin_ratio
                        my = fh * margin_ratio
                        x1 = max(0, int(round(fx - mx)))
                        y1 = max(0, int(round(fy - my)))
                        x2 = min(w, int(round(fx + fw + mx)))
                        y2 = min(h, int(round(fy + fh + my)))

                        if x2 > x1 + 10 and y2 > y1 + 10:
                            cropped = image[y1:y2, x1:x2]
                            landmarks = []
                            for idx in range(4, 14, 2):
                                lx = int(round(float(best_face[idx]) / scale))
                                ly = int(round(float(best_face[idx + 1]) / scale))
                                landmarks.append((lx - x1, ly - y1))
                            logger.info("Successfully detected and cropped face (%dx%d) for embedding enrollment", x2 - x1, y2 - y1)
                            return cropped, landmarks
            except Exception as e:
                logger.warning("Face cropping detection encountered warning: %s", str(e))

        return image, None

    def extract_face_embedding_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Detects, crops, and extracts an L2-normalized 512-dim ArcFace embedding from a photo.
        Ensures consistency between portrait enrollment and edge detection feeds.
        """
        cropped_face, landmarks = self.detect_and_crop_face(image)
        return self.extract_embedding(cropped_face, landmarks)

    @staticmethod
    def _generate_synthetic_embedding(face_crop: np.ndarray) -> np.ndarray:
        """
        Generates a 512-dimensional deterministic vector from face visual features
        for robust testing and fallback operation.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        resized = cv2.resize(gray, (32, 32)).astype(np.float32)

        # Compute DCT-like frequency projections + mean block statistics
        block_means = cv2.resize(resized, (16, 16)).flatten()  # 256 features
        grad_x = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(grad_x, grad_y)
        grad_stats = cv2.resize(grad_mag, (16, 16)).flatten()  # 256 features

        vec = np.concatenate([block_means, grad_stats]).astype(np.float32)
        # Add deterministic pseudo-random projection to decorrelate
        rng = np.random.RandomState(42)
        proj = rng.standard_normal((512, 512)).astype(np.float32)
        vec = np.dot(proj, vec)
        return vec

    @staticmethod
    def compute_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
        """
        Computes cosine similarity between two L2-normalized embeddings.
        Returns value in range [-1.0, 1.0].
        """
        if e1 is None or e2 is None:
            return 0.0
        return float(np.dot(e1.flatten(), e2.flatten()))

