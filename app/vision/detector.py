import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
import cv2
import numpy as np

# Ensure CUDA & cuDNN DLLs from PyTorch are discoverable by ONNX Runtime on Windows
if os.name == "nt":
    try:
        import torch
        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(_torch_lib):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(_torch_lib)
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.core.config import settings
from app.core.logging import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from ultralytics import RTDETR
except ImportError:
    RTDETR = None

try:
    import rfdetr
    from rfdetr import (
        RFDETRNano,
        RFDETRSmall,
        RFDETRMedium,
        RFDETRLarge,
        RFDETRBase,
    )
except ImportError:
    rfdetr = None
    RFDETRNano = None
    RFDETRSmall = None
    RFDETRMedium = None
    RFDETRLarge = None
    RFDETRBase = None


@dataclass
class Detection:
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int


class BaseDetector(ABC):
    """
    Abstract Base Class for all object / person detectors in the v2 vision pipeline.
    """

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        person_class_id: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.DETECTION_CONFIDENCE_THRESHOLD
        )
        self.person_class_id = (
            person_class_id
            if person_class_id is not None
            else settings.PERSON_CLASS_ID
        )
        self.device = device or settings.DEVICE
        self._model = None

    @abstractmethod
    def _initialize_model(self) -> None:
        """Loads and initializes model on the target device."""
        pass

    @abstractmethod
    def detect(self, frame: Optional[np.ndarray]) -> List[Detection]:
        """
        Runs object detection on a given BGR video frame.
        Returns a list of Detection objects.
        """
        pass

    @property
    def model(self) -> Any:
        if self._model is None:
            self._initialize_model()
        return self._model


class YOLODetector(BaseDetector):
    """
    Person detection engine powered by Ultralytics YOLO (v8, v11, v26, etc.).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        person_class_id: Optional[int] = None,
        device: Optional[str] = None,
    ):
        super().__init__(confidence_threshold, person_class_id, device)
        self.model_path = model_path or settings.YOLO_MODEL_PATH

    def _initialize_model(self) -> None:
        if YOLO is None:
            logger.warning("Ultralytics YOLO is not installed; deferring model loading.")
            return

        try:
            logger.info("Initializing YOLO model '%s' on device '%s'...", self.model_path, self.device)
            self._model = YOLO(self.model_path)
            if self.device != "cpu":
                self._model.to(self.device)
            logger.info("YOLO model initialized successfully.")
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", str(e), exc_info=True)
            self._model = None

    def detect(self, frame: Optional[np.ndarray]) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        if self._model is None:
            self._initialize_model()
            if self._model is None:
                logger.error("YOLO model unavailable for inference.")
                return []

        try:
            imgsz = getattr(settings, "DETECTION_IMGSZ", 960)
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                classes=[self.person_class_id],
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )

            detections: List[Detection] = []
            if not results:
                return detections

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    if cls_id != self.person_class_id or conf < self.confidence_threshold:
                        continue

                    coords = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            box=(coords[0], coords[1], coords[2], coords[3]),
                            confidence=conf,
                            class_id=cls_id,
                        )
                    )

            return detections
        except Exception as e:
            logger.error("Error during YOLO person detection inference: %s", str(e), exc_info=True)
            return []


class RFDETRDetector(BaseDetector):
    """
    Person detection engine powered by Roboflow RF-DETR (Real-time Detection Transformer).
    Based on DINOv2 vision transformer backbone with state-of-the-art accuracy/speed trade-offs.
    Supports variants: nano, small, medium, large, base, or custom checkpoint weights.
    """

    VARIANT_MAP: Dict[str, Any] = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
        "base": RFDETRBase,
    }

    def __init__(
        self,
        variant: Optional[str] = None,
        weights_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        person_class_id: Optional[int] = None,
        device: Optional[str] = None,
    ):
        # Note: Pretrained Roboflow RF-DETR follows standard 1-indexed COCO categories (1 = person, whereas YOLO is 0).
        effective_person_class_id = (
            person_class_id
            if person_class_id is not None
            else getattr(settings, "PERSON_CLASS_ID", 1)
        )
        super().__init__(confidence_threshold, effective_person_class_id, device)
        self.variant = (
            variant
            or getattr(settings, "RF_DETR_VARIANT", "medium")
            or "medium"
        ).lower()
        self.weights_path = (
            weights_path
            or getattr(settings, "RF_DETR_WEIGHTS_PATH", None)
        )

    def _initialize_model(self) -> None:
        if rfdetr is None:
            logger.warning(
                "Package 'rfdetr' is not installed. Run `pip install rfdetr supervision` to enable RF-DETR."
            )
            return

        try:
            logger.info(
                "Initializing RF-DETR variant '%s' (weights='%s') on device '%s'...",
                self.variant,
                self.weights_path,
                self.device,
            )

            variant_cls = self.VARIANT_MAP.get(self.variant, RFDETRMedium)

            # Check if weights_path is a valid RF-DETR checkpoint (ignore default YOLO paths like yolo26s.pt)
            is_valid_rf_weights = (
                self.weights_path
                and os.path.exists(self.weights_path)
                and (
                    "rf-detr" in self.weights_path.lower()
                    or "rfdetr" in self.weights_path.lower()
                    or self.weights_path.endswith(".pth")
                )
                and "yolo" not in os.path.basename(self.weights_path).lower()
            )

            if is_valid_rf_weights:
                # Load custom checkpoint weights
                if hasattr(variant_cls, "from_checkpoint"):
                    self._model = variant_cls.from_checkpoint(path=self.weights_path, trust_checkpoint=True)
                elif hasattr(rfdetr, "from_checkpoint"):
                    self._model = rfdetr.from_checkpoint(path=self.weights_path, trust_checkpoint=True)
                else:
                    try:
                        self._model = variant_cls(pretrain_weights=self.weights_path, device=self.device, trust_checkpoint=True)
                    except Exception:
                        self._model = variant_cls(pretrain_weights=self.weights_path, trust_checkpoint=True)
            else:
                # Load standard pre-trained COCO model
                try:
                    self._model = variant_cls(device=self.device, trust_checkpoint=True)
                except Exception:
                    try:
                        self._model = variant_cls(trust_checkpoint=True)
                    except Exception:
                        self._model = variant_cls()

            logger.info("RF-DETR (%s) initialized successfully.", self.variant)
        except Exception as e:
            logger.error("Failed to load RF-DETR model: %s", str(e), exc_info=True)
            self._model = None

    def detect(self, frame: Optional[np.ndarray]) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        if self._model is None:
            self._initialize_model()
            if self._model is None:
                logger.error("RF-DETR model unavailable for inference.")
                return []

        try:
            # RF-DETR predict accepts numpy array and returns supervision.Detections
            res = self._model.predict(frame, threshold=self.confidence_threshold)

            detections: List[Detection] = []
            if res is None:
                return detections

            # Unpack results if returned inside a single-item list
            if isinstance(res, list):
                res = res[0] if len(res) > 0 else None
                if res is None:
                    return detections

            boxes = getattr(res, "xyxy", None)
            confidences = getattr(res, "confidence", None)
            class_ids = getattr(res, "class_id", None)

            if boxes is None or len(boxes) == 0:
                return detections

            for i in range(len(boxes)):
                cls_id = int(class_ids[i]) if class_ids is not None else 0
                conf = float(confidences[i]) if confidences is not None else 1.0

                # Roboflow RF-DETR uses standard COCO category 1 for person (YOLO uses 0).
                # Support both default (1) and custom/fine-tuned (0) class IDs.
                is_person = (cls_id == self.person_class_id) or (self.person_class_id == 1 and cls_id in (0, 1))
                if not is_person or conf < self.confidence_threshold:
                    continue

                b = boxes[i]
                detections.append(
                    Detection(
                        box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        confidence=conf,
                        class_id=cls_id,
                    )
                )

            return detections
        except Exception as e:
            logger.error("Error during RF-DETR person detection inference: %s", str(e), exc_info=True)
            return []


class RTDETRDetector(BaseDetector):
    """
    Person detection engine powered by Ultralytics RT-DETR (.pt checkpoints e.g. rtdetr-l.pt).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        person_class_id: Optional[int] = None,
        device: Optional[str] = None,
    ):
        super().__init__(confidence_threshold, person_class_id, device)
        self.model_path = model_path or "rtdetr-l.pt"

    def _initialize_model(self) -> None:
        if RTDETR is None:
            logger.warning("Ultralytics RTDETR is not available; deferring model loading.")
            return

        try:
            logger.info("Initializing RT-DETR model '%s' on device '%s'...", self.model_path, self.device)
            self._model = RTDETR(self.model_path)
            if self.device != "cpu":
                self._model.to(self.device)
            logger.info("RT-DETR model initialized successfully.")
        except Exception as e:
            logger.error("Failed to load RT-DETR model: %s", str(e), exc_info=True)
            self._model = None

    def detect(self, frame: Optional[np.ndarray]) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        if self._model is None:
            self._initialize_model()
            if self._model is None:
                logger.error("RT-DETR model unavailable for inference.")
                return []

        try:
            imgsz = getattr(settings, "DETECTION_IMGSZ", 960)
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                classes=[self.person_class_id],
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )

            detections: List[Detection] = []
            if not results:
                return detections

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    if cls_id != self.person_class_id or conf < self.confidence_threshold:
                        continue

                    coords = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            box=(coords[0], coords[1], coords[2], coords[3]),
                            confidence=conf,
                            class_id=cls_id,
                        )
                    )

            return detections
        except Exception as e:
            logger.error("Error during RT-DETR detection inference: %s", str(e), exc_info=True)
            return []


class RTMDetDetector(BaseDetector):
    """
    Person detection engine powered by OpenMMLab RTMDet ONNX runtime.
    Accelerated with CUDAExecutionProvider (GPU) with seamless fallback to CPU.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        person_class_id: Optional[int] = None,
        device: Optional[str] = None,
        imgsz: int = 640,
        iou_thresh: float = 0.45,
    ):
        super().__init__(confidence_threshold, person_class_id, device)
        self.model_path = model_path or "models/rtmdet-m-person.onnx"
        self.imgsz = imgsz
        self.iou_thresh = iou_thresh
        self._session = None
        self._input_name = None
        self._is_person_only = False

    def _initialize_model(self) -> None:
        if ort is None:
            logger.warning("ONNX Runtime is not available; cannot load RTMDet.")
            return

        resolved_path = self.model_path
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.join("models", resolved_path),
                os.path.join("models", os.path.basename(resolved_path)),
                "models/rtmdet-m-person.onnx",
                "models/rtmdet-s-coco.onnx",
            ]
            for c in candidates:
                if os.path.exists(c):
                    resolved_path = c
                    break

        if not os.path.exists(resolved_path):
            logger.error("RTMDet model file not found: %s", self.model_path)
            return

        self.model_path = resolved_path
        self._is_person_only = "person" in os.path.basename(resolved_path).lower()

        providers = ["CPUExecutionProvider"]
        if self.device != "cpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            logger.info("Initializing RTMDet model '%s' with providers %s...", self.model_path, providers)
            self._session = ort.InferenceSession(self.model_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            active_provider = self._session.get_providers()[0]
            logger.info("RTMDet initialized successfully with provider: %s", active_provider)
        except Exception as e:
            logger.error("Failed to initialize RTMDet model: %s", str(e), exc_info=True)
            self._session = None

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize with aspect ratio and normalize with standard RTMDet mean and std.
        """
        orig_h, orig_w = frame.shape[:2]
        ratio = min(self.imgsz / orig_w, self.imgsz / orig_h)
        new_w = max(1, int(orig_w * ratio))
        new_h = max(1, int(orig_h * ratio))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        padded = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.float32)
        padded[:new_h, :new_w] = resized_rgb.astype(np.float32)

        # RTMDet normalization: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        tensor = (padded - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))
        return np.expand_dims(tensor, axis=0).astype(np.float32), ratio

    def detect(self, frame: Optional[np.ndarray]) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        if self._session is None:
            self._initialize_model()
            if self._session is None:
                logger.error("RTMDet session unavailable for inference.")
                return []

        try:
            tensor, ratio = self._preprocess(frame)
            outputs = self._session.run(None, {self._input_name: tensor})

            # outputs[0]: shape (1, num_dets, 5) -> [x1, y1, x2, y2, score]
            # outputs[1]: shape (1, num_dets) -> class_id
            raw_dets = outputs[0][0]
            raw_labels = outputs[1][0]

            boxes = raw_dets[:, :4]
            scores = raw_dets[:, 4]
            labels = raw_labels

            # Filter by score
            score_mask = scores >= self.confidence_threshold
            boxes = boxes[score_mask]
            scores = scores[score_mask]
            labels = labels[score_mask]

            # Filter by class (person only)
            if not self._is_person_only:
                person_mask = (labels == self.person_class_id)
                boxes = boxes[person_mask]
                scores = scores[person_mask]
                labels = labels[person_mask]

            if len(boxes) == 0:
                return []

            # Non-Maximum Suppression (NMS) via OpenCV
            nms_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
            keep_indices = cv2.dnn.NMSBoxes(
                bboxes=nms_boxes,
                scores=[float(s) for s in scores],
                score_threshold=float(self.confidence_threshold),
                nms_threshold=float(self.iou_thresh),
            )

            if len(keep_indices) == 0:
                return []

            indices = keep_indices.flatten() if hasattr(keep_indices, "flatten") else [int(k) for k in keep_indices]

            h, w = frame.shape[:2]
            detections: List[Detection] = []
            for idx in indices:
                x1, y1, x2, y2 = boxes[idx] / ratio
                # Clip to frame boundary
                x1 = max(0.0, min(float(w), float(x1)))
                y1 = max(0.0, min(float(h), float(y1)))
                x2 = max(0.0, min(float(w), float(x2)))
                y2 = max(0.0, min(float(h), float(y2)))

                if (x2 - x1) < 14 or (y2 - y1) < 18:
                    continue

                detections.append(
                    Detection(
                        box=(x1, y1, x2, y2),
                        confidence=float(scores[idx]),
                        class_id=int(labels[idx]) if not self._is_person_only else 0,
                    )
                )

            return detections
        except Exception as e:
            logger.error("Error during RTMDet detection inference: %s", str(e), exc_info=True)
            return []


def create_detector(
    model_type: Optional[str] = None,
    model_path: Optional[str] = None,
    rf_detr_variant: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    person_class_id: Optional[int] = None,
    device: Optional[str] = None,
) -> BaseDetector:
    """
    Factory function to instantiate the appropriate detector backend:
    - 'rtmdet': OpenMMLab RTMDet ONNX (default)
    - 'yolo': Ultralytics YOLO26m / YOLO26s
    - 'rf-detr' / 'rfdetr': Roboflow RF-DETR Transformer
    - 'rtdetr': Ultralytics RT-DETR (.pt)
    """
    resolved_type = (model_type or getattr(settings, "DETECTION_MODEL_TYPE", "rtmdet")).lower()

    # Automatically infer model type from file path if model_type is auto or default
    if resolved_type in ("auto", "default"):
        if model_path:
            lp = model_path.lower()
            if "rtmdet" in lp:
                resolved_type = "rtmdet"
            elif "rf-detr" in lp or "rfdetr" in lp or lp.endswith(".pth"):
                resolved_type = "rf-detr"
            elif "rtdetr" in lp:
                resolved_type = "rtdetr"
            elif "yolo" in lp:
                resolved_type = "yolo"
            else:
                resolved_type = getattr(settings, "DETECTION_MODEL_TYPE", "rtmdet")
        else:
            resolved_type = getattr(settings, "DETECTION_MODEL_TYPE", "rtmdet")

    if resolved_type in ("rf-detr", "rfdetr", "rf_detr"):
        weights = model_path
        if weights and ("yolo" in os.path.basename(weights).lower()):
            weights = None
        return RFDETRDetector(
            variant=rf_detr_variant,
            weights_path=weights,
            confidence_threshold=confidence_threshold,
            person_class_id=person_class_id,
            device=device,
        )
    elif resolved_type in ("rtmdet", "rtm-det", "rtm_det"):
        return RTMDetDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            person_class_id=person_class_id,
            device=device,
        )
    elif resolved_type in ("rtdetr", "rt-detr", "rt_detr"):
        return RTDETRDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            person_class_id=person_class_id,
            device=device,
        )
    else:
        return YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            person_class_id=person_class_id,
            device=device,
        )


class PersonDetector:
    """
    Unified Person Detection Engine facade.
    Maintains 100% backward compatibility with Phase 1 v1 API while enabling
    switching between YOLO, RF-DETR, and RT-DETR in v2.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        person_class_id: Optional[int] = None,
        device: Optional[str] = None,
        model_type: Optional[str] = None,
        rf_detr_variant: Optional[str] = None,
    ):
        self.detector = create_detector(
            model_type=model_type,
            model_path=model_path,
            rf_detr_variant=rf_detr_variant,
            confidence_threshold=confidence_threshold,
            person_class_id=person_class_id,
            device=device,
        )
        self.model_path = model_path or getattr(self.detector, "model_path", None)
        self.confidence_threshold = self.detector.confidence_threshold
        self.person_class_id = self.detector.person_class_id
        self.device = self.detector.device

    @property
    def _model(self):
        return self.detector.model

    @_model.setter
    def _model(self, val):
        self.detector._model = val

    def detect(self, frame: Optional[np.ndarray]) -> List[Detection]:
        return self.detector.detect(frame)
