from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from app.core.config import settings
from app.core.logging import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int


class PersonDetector:
    """
    Person detection engine powered by Ultralytics YOLO.
    Filters exclusively for person class (class_id=0) and confidence threshold.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        person_class_id: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
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

    def _initialize_model(self) -> None:
        if YOLO is None:
            logger.warning("Ultralytics YOLO is not yet imported; deferring model loading.")
            return

        try:
            logger.info("Initializing YOLO model '%s' on device '%s'...", self.model_path, self.device)
            self._model = YOLO(self.model_path)
            # Move to target device if supported
            if self.device != "cpu":
                self._model.to(self.device)
            logger.info("YOLO model initialized successfully.")
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", str(e), exc_info=True)
            self._model = None

    def detect(self, frame: Optional[np.ndarray]) -> List[Detection]:
        """
        Detects persons in a given BGR video frame.
        Returns a list of Detection objects.
        """
        if frame is None or frame.size == 0:
            return []

        if self._model is None:
            self._initialize_model()
            if self._model is None:
                logger.error("Model unavailable for inference.")
                return []

        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                classes=[self.person_class_id],
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

                    # Strictly enforce person class and confidence threshold
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
            logger.error("Error during person detection inference: %s", str(e), exc_info=True)
            return []
