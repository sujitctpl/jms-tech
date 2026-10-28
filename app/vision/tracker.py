from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from app.core.config import settings
from app.core.logging import logger
from app.vision.detector import Detection

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class TrackedPerson:
    tracker_id: str
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Calculates the bottom-center point of the bounding box (feet/ground contact point)."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> Tuple[float, float]:
        """Calculates center point of the bounding box."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PersonTracker:
    """
    Visitor tracking using ByteTrack integrated with Ultralytics YOLO.
    Maintains temporary track IDs across frames.
    Also provides a lightweight tracker fallback for testing tracking continuity.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        person_class_id: Optional[int] = None,
        tracker_type: Optional[str] = None,
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
        self.tracker_type = tracker_type or settings.TRACKER_TYPE
        self.device = device or settings.DEVICE
        self._model = None

        # Fallback tracker state for unit testing with synthetic detections
        self._next_synthetic_id = 1
        self._prev_tracks: Dict[str, Tuple[float, float, float, float]] = {}

    def _get_model(self):
        if self._model is None and YOLO is not None:
            try:
                self._model = YOLO(self.model_path)
                if self.device != "cpu":
                    self._model.to(self.device)
            except Exception as e:
                logger.error("Error loading YOLO model for tracking: %s", str(e))
                self._model = None
        return self._model

    def track_frame(self, frame: Optional[np.ndarray]) -> List[TrackedPerson]:
        """
        Runs YOLO + ByteTrack tracking on a single frame.
        """
        if frame is None or frame.size == 0:
            return []

        model = self._get_model()
        if model is None:
            logger.warning("Tracking model unavailable.")
            return []

        try:
            results = model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_type,
                conf=self.confidence_threshold,
                classes=[self.person_class_id],
                device=self.device,
                verbose=False,
            )

            tracked_persons: List[TrackedPerson] = []
            if not results:
                return tracked_persons

            for r in results:
                boxes = r.boxes
                if boxes is None or boxes.id is None:
                    continue

                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if cls_id != self.person_class_id or conf < self.confidence_threshold:
                        continue

                    # box.id may be None if ByteTrack did not assign an ID
                    if box.id is None or len(box.id) == 0:
                        continue

                    track_id = str(int(box.id[0].item()))
                    coords = box.xyxy[0].tolist()

                    tracked_persons.append(
                        TrackedPerson(
                            tracker_id=track_id,
                            box=(coords[0], coords[1], coords[2], coords[3]),
                            confidence=conf,
                            class_id=cls_id,
                        )
                    )

            return tracked_persons
        except Exception as e:
            logger.error("Error during frame tracking: %s", str(e), exc_info=True)
            return []

    def track_detections_mock(self, detections: List[Detection]) -> List[TrackedPerson]:
        """
        Lightweight IoU-based tracking matcher for unit testing without YOLO model weights.
        """
        matched: List[TrackedPerson] = []
        new_tracks: Dict[str, Tuple[float, float, float, float]] = {}

        for det in detections:
            best_id = None
            best_iou = 0.25  # minimum IoU threshold to match

            for trk_id, prev_box in self._prev_tracks.items():
                iou = self._calculate_iou(det.box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = trk_id

            if best_id is None:
                best_id = str(self._next_synthetic_id)
                self._next_synthetic_id += 1

            new_tracks[best_id] = det.box
            matched.append(
                TrackedPerson(
                    tracker_id=best_id,
                    box=det.box,
                    confidence=det.confidence,
                    class_id=det.class_id,
                )
            )

        self._prev_tracks = new_tracks
        return matched

    @staticmethod
    def _calculate_iou(
        boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]
    ) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denominator = float(boxAArea + boxBArea - interArea)
        if denominator <= 0:
            return 0.0
        return interArea / denominator
