import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.vision.detector import (
    BaseDetector,
    Detection,
    RFDETRDetector,
    YOLODetector,
    create_detector,
)

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
    person_id: Optional[str] = None  # Persistent visitor identity

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Calculates the bottom-center point of the bounding box (ground contact point)."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> Tuple[float, float]:
        """Calculates center point of the bounding box."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PersonTracker:
    """
    Modular Multi-Object Tracking Engine for Showroom Visitor Intelligence.
    Seamlessly supports both:
    1. RF-DETR (Roboflow Detection Transformer) + Kalman/IoU Association
    2. Ultralytics YOLO + ByteTrack / BoT-SORT
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        person_class_id: Optional[int] = None,
        tracker_type: Optional[str] = None,
        device: Optional[str] = None,
        model_type: Optional[str] = None,
        rf_detr_variant: Optional[str] = None,
        detector: Optional[BaseDetector] = None,
    ):
        self.model_type = (
            model_type
            or getattr(settings, "DETECTION_MODEL_TYPE", "rtmdet")
        ).lower()
        self.rf_detr_variant = (
            rf_detr_variant
            or getattr(settings, "RF_DETR_VARIANT", "small")
        )
        if self.model_type in ("rf-detr", "rfdetr", "rf_detr"):
            default_path = getattr(settings, "RF_DETR_WEIGHTS_PATH", None)
        elif self.model_type in ("rtmdet", "rtm-det", "rtm_det"):
            default_path = getattr(settings, "RTMDet_MODEL_PATH", "models/rtmdet-m-person.onnx")
        elif self.model_type in ("rtdetr", "rt-detr", "rt_detr"):
            default_path = "models/rtdetr-l.pt"
        else:
            default_path = getattr(settings, "YOLO_MODEL_PATH", "models/yolo26s.pt")
        self.model_path = model_path or default_path
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(settings, "DETECTION_CONFIDENCE_THRESHOLD", 0.35)
        )
        self.person_class_id = (
            person_class_id
            if person_class_id is not None
            else getattr(settings, "PERSON_CLASS_ID", 0)
        )
        raw_tracker = tracker_type or getattr(settings, "TRACKER_TYPE", "bytetrack.yaml")
        if not raw_tracker.endswith(".yaml"):
            raw_tracker = f"{raw_tracker}.yaml"
        if os.path.exists(raw_tracker):
            self.tracker_type = os.path.abspath(raw_tracker)
        else:
            self.tracker_type = raw_tracker
        self.device = device or getattr(settings, "DEVICE", "cuda:0")

        # Initialize detector backend
        if detector is not None:
            self.detector = detector
        else:
            self.detector = create_detector(
                model_type=self.model_type,
                model_path=self.model_path,
                rf_detr_variant=self.rf_detr_variant,
                confidence_threshold=self.confidence_threshold,
                person_class_id=self.person_class_id,
                device=self.device,
            )

        # Multi-object association state (for RF-DETR and non-YOLO models, and mock tracking)
        self._next_id = 1
        self._next_synthetic_id = 1
        self._active_tracks: Dict[str, Tuple[float, float, float, float]] = {}
        self._disappeared: Dict[str, int] = {}
        self._velocities: Dict[str, Tuple[float, float]] = {}
        self._prev_tracks: Dict[str, Tuple[float, float, float, float]] = {}
        self.max_disappeared = 30
        self.iou_threshold = 0.25

    def _get_yolo_model(self):
        if isinstance(self.detector, YOLODetector):
            return self.detector.model
        if YOLO is not None and self.model_path:
            try:
                m = YOLO(self.model_path)
                if self.device != "cpu":
                    m.to(self.device)
                return m
            except Exception as e:
                logger.error("Error loading YOLO model: %s", str(e))
        return None

    def track_frame(self, frame: Optional[np.ndarray]) -> List[TrackedPerson]:
        """
        Runs detection and tracking on a single frame.
        Routes to native YOLO track if YOLO is active, or uses RF-DETR + unified tracker.
        """
        if frame is None or frame.size == 0:
            return []

        # If using YOLO backend, we can optionally use native YOLO track
        if isinstance(self.detector, YOLODetector):
            yolo_model = self._get_yolo_model()
            if yolo_model is not None:
                try:
                    return self._track_yolo_native(yolo_model, frame)
                except Exception as e:
                    logger.warning("Native YOLO tracking encountered error, falling back to modular tracker: %s", str(e))

        # Modular Tracking Pipeline: Detect with RF-DETR / Detector -> Associate Tracks
        detections = self.detector.detect(frame)
        return self._associate_detections(detections)

    def _track_yolo_native(self, model: Any, frame: np.ndarray) -> List[TrackedPerson]:
        imgsz = getattr(settings, "DETECTION_IMGSZ", 960)
        results = model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.confidence_threshold,
            classes=[self.person_class_id],
            imgsz=imgsz,
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

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id != self.person_class_id or conf < self.confidence_threshold:
                    continue

                if box.id is None or len(box.id) == 0:
                    continue

                track_id = str(int(box.id[0].item()))
                coords = box.xyxy[0].tolist()
                bw = coords[2] - coords[0]
                bh = coords[3] - coords[1]
                # Filter out non-human horizontal shapes
                if bw < 14 or bh < 18 or (bh / max(1.0, bw)) < 0.50:
                    continue

                tracked_persons.append(
                    TrackedPerson(
                        tracker_id=track_id,
                        box=(coords[0], coords[1], coords[2], coords[3]),
                        confidence=conf,
                        class_id=cls_id,
                    )
                )

        return tracked_persons

    def _associate_detections(self, detections: List[Detection]) -> List[TrackedPerson]:
        """
        Associates detections across frames using motion-predicted Hungarian/IoU matching.
        Used for RF-DETR, RT-DETR, and custom detectors.
        """
        # Filter out degenerate/non-human boxes
        valid_dets = [
            d for d in detections
            if (d.box[2] - d.box[0]) >= 14 and (d.box[3] - d.box[1]) >= 18
        ]

        if not self._active_tracks:
            # First frame or all tracks lost: register all detections
            tracked: List[TrackedPerson] = []
            for det in valid_dets:
                tid = str(self._next_id)
                self._next_id += 1
                self._active_tracks[tid] = det.box
                self._disappeared[tid] = 0
                self._velocities[tid] = (0.0, 0.0)
                tracked.append(
                    TrackedPerson(
                        tracker_id=tid,
                        box=det.box,
                        confidence=det.confidence,
                        class_id=det.class_id,
                    )
                )
            return tracked

        track_ids = list(self._active_tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(valid_dets)

        if num_dets == 0:
            # Mark all active tracks as missing for this frame
            for tid in list(self._active_tracks.keys()):
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self.max_disappeared:
                    del self._active_tracks[tid]
                    del self._disappeared[tid]
                    self._velocities.pop(tid, None)
            return []

        # Build Cost Matrix (IoU + Motion Projection)
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            prev_box = self._active_tracks[tid]
            vx, vy = self._velocities.get(tid, (0.0, 0.0))
            pred_box = (
                prev_box[0] + vx,
                prev_box[1] + vy,
                prev_box[2] + vx,
                prev_box[3] + vy,
            )

            for j, det in enumerate(valid_dets):
                iou_prev = self._calculate_iou(prev_box, det.box)
                iou_pred = self._calculate_iou(pred_box, det.box)
                best_iou = max(iou_prev, iou_pred)
                cost_matrix[i, j] = 1.0 - best_iou

        # Greedy / Bipartite matching with IoU threshold
        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()
        matches: List[Tuple[int, int]] = []

        pairs = []
        for r in range(num_tracks):
            for c in range(num_dets):
                if cost_matrix[r, c] <= (1.0 - self.iou_threshold):
                    pairs.append((cost_matrix[r, c], r, c))
        pairs.sort(key=lambda x: x[0])

        for _, r, c in pairs:
            if r not in matched_tracks and c not in matched_dets:
                matched_tracks.add(r)
                matched_dets.add(c)
                matches.append((r, c))

        tracked_persons: List[TrackedPerson] = []

        # Update matched tracks
        for r, c in matches:
            tid = track_ids[r]
            det = valid_dets[c]
            prev_b = self._active_tracks[tid]
            curr_b = det.box

            # Update velocity
            vx = 0.6 * (curr_b[0] - prev_b[0]) + 0.4 * self._velocities.get(tid, (0.0, 0.0))[0]
            vy = 0.6 * (curr_b[1] - prev_b[1]) + 0.4 * self._velocities.get(tid, (0.0, 0.0))[1]
            self._velocities[tid] = (vx, vy)
            self._active_tracks[tid] = curr_b
            self._disappeared[tid] = 0

            tracked_persons.append(
                TrackedPerson(
                    tracker_id=tid,
                    box=curr_b,
                    confidence=det.confidence,
                    class_id=det.class_id,
                )
            )

        # Unmatched existing tracks
        for r in range(num_tracks):
            if r not in matched_tracks:
                tid = track_ids[r]
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self.max_disappeared:
                    del self._active_tracks[tid]
                    del self._disappeared[tid]
                    self._velocities.pop(tid, None)

        # Unmatched detections -> spawn new tracks
        for c in range(num_dets):
            if c not in matched_dets:
                det = valid_dets[c]
                tid = str(self._next_id)
                self._next_id += 1
                self._active_tracks[tid] = det.box
                self._disappeared[tid] = 0
                self._velocities[tid] = (0.0, 0.0)
                tracked_persons.append(
                    TrackedPerson(
                        tracker_id=tid,
                        box=det.box,
                        confidence=det.confidence,
                        class_id=det.class_id,
                    )
                )

        return tracked_persons

    def track_detections_mock(self, detections: List[Detection]) -> List[TrackedPerson]:
        """
        Lightweight IoU-based tracking matcher for unit testing without model weights.
        """
        matched: List[TrackedPerson] = []
        new_tracks: Dict[str, Tuple[float, float, float, float]] = {}

        for det in detections:
            best_id = None
            best_iou = 0.10

            for trk_id, prev_box in self._prev_tracks.items():
                vx, vy = self._velocities.get(trk_id, (0.0, 0.0))
                pred_box = (prev_box[0] + vx, prev_box[1] + vy, prev_box[2] + vx, prev_box[3] + vy)
                iou = max(self._calculate_iou(det.box, prev_box), self._calculate_iou(det.box, pred_box))
                if iou > best_iou:
                    best_iou = iou
                    best_id = trk_id

            if best_id is None:
                best_id = str(self._next_synthetic_id)
                self._next_synthetic_id += 1
                self._velocities[best_id] = (0.0, 0.0)
            else:
                prev_b = self._prev_tracks[best_id]
                self._velocities[best_id] = (det.box[0] - prev_b[0], det.box[1] - prev_b[1])

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
