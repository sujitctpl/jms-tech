"""
Local AI Edge Video Processor & Person Detection Engine
Showroom Intelligence & CCTV Footfall Analytics

Processes video locally using YOLOv11 and sends real-time counting events
to the central FastAPI dashboard without streaming video over the network.
"""

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from app.core.config import settings

# Force TCP transport for RTSP streams globally to prevent packet loss & H264 macroblock decode corruption
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

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

import cv2
import numpy as np

try:
    from ultralytics.trackers.utils.matching import linear_assignment
except Exception:
    linear_assignment = None

# Handle PyInstaller onefile directory resolution
def get_resource_path(relative_path: Optional[str]) -> str:
    """Get absolute path to resource, works for dev, models folder, and PyInstaller."""
    if not relative_path:
        return ""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    target = os.path.join(base_path, relative_path)
    if os.path.exists(target):
        return target
    # Fallback to local cwd
    cwd_target = os.path.join(os.getcwd(), relative_path)
    if os.path.exists(cwd_target):
        return cwd_target
    # Check inside models/ directory
    filename = os.path.basename(relative_path)
    models_target = os.path.join(base_path, "models", filename)
    if os.path.exists(models_target):
        return models_target
    cwd_models = os.path.join(os.getcwd(), "models", filename)
    if os.path.exists(cwd_models):
        return cwd_models
    return relative_path


# ---------------------------------------------------------------------------
# Background HTTP Sync Client (Non-blocking Queue)
# ---------------------------------------------------------------------------
class DashboardSyncClient:
    """Non-blocking background client sending footfall events & telemetry to FastAPI."""

    def __init__(self, base_url: str, camera_id: str):
        self.base_url = base_url.rstrip("/")
        self.camera_id = camera_id
        self.queue: queue.Queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.is_connected = False
        self.last_heartbeat_time = 0.0
        self.total_synced_entries = 0
        self.last_error = ""

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def send_entry_event(self, tracker_id: str, confidence: float = 0.92, section_id: Optional[str] = None, section_name: Optional[str] = None, person_id: Optional[str] = None) -> None:
        """Enqueue an entry event detected locally."""
        payload = {
            "tracker_id": str(tracker_id),
            "confidence": float(confidence),
            "section_id": section_id,
            "section_name": section_name,
            "person_id": str(person_id) if person_id else None,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.queue.put_nowait(("entry", payload))
        except queue.Full:
            pass

    def send_section_entry_event(self, tracker_id: str, section_id: str, section_name: str) -> None:
        """Enqueue a section visit event detected locally."""
        payload = {
            "tracker_id": str(tracker_id),
            "confidence": 0.95,
            "section_id": str(section_id),
            "section_name": str(section_name),
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.queue.put_nowait(("section_entry", payload))
        except queue.Full:
            pass

    def send_employee_punch_event(self, employee_id: str, event_type: str = "CHECK_IN", confidence: float = 0.95) -> None:
        """Enqueue an employee check-in/out punch event detected locally."""
        payload = {
            "employee_id": str(employee_id),
            "camera_id": self.camera_id if self.camera_id and len(self.camera_id) == 36 else None,
            "event_type": event_type,
            "confidence": float(confidence),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.queue.put_nowait(("employee_punch", payload))
        except queue.Full:
            pass

    def send_heartbeat(
        self,
        fps: float,
        frames_processed: int,
        current_persons: int,
        total_entries: int,
        section_counts: Optional[dict] = None,
        section_occupancy: Optional[dict] = None,
        staff_dwell: Optional[dict] = None,
    ) -> None:
        """Enqueue periodic heartbeat telemetry with multi-zone stats."""
        payload = {
            "is_processing": True,
            "fps": round(fps, 1),
            "frames_processed": frames_processed,
            "current_persons": current_persons,
            "total_entries": total_entries,
            "section_counts": section_counts or {},
            "section_occupancy": section_occupancy or {},
            "staff_dwell": staff_dwell or {},
        }
        try:
            self.queue.put_nowait(("heartbeat", payload))
        except queue.Full:
            pass

    def _post(self, endpoint: str, data: dict, timeout: float = 2.5) -> bool:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "Capermint-Local-Edge/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status in (200, 201):
                    self.is_connected = True
                    return True
        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            return False
        return False

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                event_type, payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if event_type in ("entry", "section_entry"):
                endpoint = f"/api/v1/cameras/{self.camera_id}/footfall"
                ok = self._post(endpoint, payload)
                if ok:
                    self.total_synced_entries += 1
            elif event_type == "employee_punch":
                endpoint = "/api/v1/attendance/punch"
                self._post(endpoint, payload)
            elif event_type == "heartbeat":
                endpoint = f"/api/v1/cameras/{self.camera_id}/heartbeat"
                self._post(endpoint, payload)

            self.queue.task_done()

    def stop(self) -> None:
        self.stop_event.set()


# ---------------------------------------------------------------------------
# High-Accuracy Multi-Object Tracking Fallback Engine
# ---------------------------------------------------------------------------
def solve_linear_assignment(
    cost_matrix: np.ndarray, thresh: float = 0.70
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Optimal bipartite matching on cost matrix with distance/gating threshold.
    Returns (matches, unmatched_a, unmatched_b).
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    if linear_assignment is not None:
        try:
            matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh=thresh)
            matches = [(int(r), int(c)) for r, c in matched_indices]
            return matches, [int(i) for i in unmatched_a], [int(i) for i in unmatched_b]
        except Exception:
            pass

    # High-accuracy bipartite matching with gating fallback
    num_rows, num_cols = cost_matrix.shape
    used_rows: Set[int] = set()
    used_cols: Set[int] = set()
    matches: List[Tuple[int, int]] = []

    pairs = []
    for r in range(num_rows):
        for c in range(num_cols):
            val = float(cost_matrix[r, c])
            if val <= thresh:
                pairs.append((val, r, c))
    pairs.sort(key=lambda x: x[0])

    for val, r, c in pairs:
        if r not in used_rows and c not in used_cols:
            used_rows.add(r)
            used_cols.add(c)
            matches.append((r, c))

    unmatched_a = [r for r in range(num_rows) if r not in used_rows]
    unmatched_b = [c for c in range(num_cols) if c not in used_cols]
    return matches, unmatched_a, unmatched_b


class RobustFallbackTracker:
    """
    Advanced multi-object tracking fallback engine.
    Addresses key multi-object tracking failure modes:
    1. Object Velocity & Linear Motion Prediction: Projects tracks along their velocity vector (vx, vy),
       eliminating ID swaps when two persons cross paths.
    2. Visual Appearance Embeddings: Multi-region normalized HSV color histograms provide appearance
       signatures across frames, preventing identity switches when bounding boxes overlap.
    3. Optimal Bipartite Matching: Global cost matrix optimization with spatial distance, IoU overlap,
       and appearance cosine distance gating.
    4. Seamless Primary Tracker Sync: Continuously learns track velocities and appearance
       signatures from ByteTrack / BoT-SORT, ensuring instant, zero-drop recovery if the primary tracker drops.
    """

    def __init__(self, max_disappeared: int = 26, max_distance: float = 120.0, iou_thresh: float = 0.20):
        self.next_id = 10001
        self.objects: Dict[int, Tuple[float, float, float, float]] = {}  # id -> (x1, y1, x2, y2)
        self.disappeared: Dict[int, int] = {}
        self.velocities: Dict[int, Tuple[float, float]] = {}  # id -> (vx, vy)
        self.prev_centroids: Dict[int, Tuple[float, float]] = {}  # id -> (cx, cy)
        self.appearances: Dict[int, np.ndarray] = {}  # id -> 32-dim normalized HSV embedding
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_thresh = iou_thresh

    @staticmethod
    def _compute_iou(
        boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]
    ) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        denom = float(boxAArea + boxBArea - interArea)
        return (interArea / denom) if denom > 0.0 else 0.0

    @staticmethod
    def extract_appearance(
        frame: Optional[np.ndarray], box: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        """Extracts a normalized spatial HSV color appearance signature of the person crop (upper & lower body)."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < 8 or (y2 - y1) < 16:
            return None
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mid_y = max(1, hsv.shape[0] // 2)
            upper = hsv[:mid_y, :]
            lower = hsv[mid_y:, :]
            hist_upper = cv2.calcHist([upper], [0, 1], None, [8, 2], [0, 180, 0, 256]).flatten()
            hist_lower = cv2.calcHist([lower], [0, 1], None, [8, 2], [0, 180, 0, 256]).flatten()
            feat = np.concatenate([hist_upper, hist_lower])
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                return (feat / norm).astype(np.float32)
        except Exception:
            pass
        return None

    def predict_box(self, oid: int) -> Tuple[float, float, float, float]:
        """Projects bounding box position along velocity vector."""
        box = self.objects[oid]
        vx, vy = self.velocities.get(oid, (0.0, 0.0))
        return (box[0] + vx, box[1] + vy, box[2] + vx, box[3] + vy)

    def sync_active_tracks(
        self,
        active_objects: Dict[int, Tuple[float, float, float, float]],
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """
        Continuously synchronizes tracked objects from ByteTrack / BoT-SORT,
        estimating velocities and extracting appearance signatures for zero-lag fallback.
        """
        now_tids = set(active_objects.keys())
        for oid, box in active_objects.items():
            cx = (box[0] + box[2]) * 0.5
            cy = box[3]  # bottom-center ground contact point
            if oid in self.prev_centroids:
                old_cx, old_cy = self.prev_centroids[oid]
                vx = cx - old_cx
                vy = cy - old_cy
                prev_vx, prev_vy = self.velocities.get(oid, (0.0, 0.0))
                self.velocities[oid] = (0.7 * vx + 0.3 * prev_vx, 0.7 * vy + 0.3 * prev_vy)
            else:
                self.velocities[oid] = (0.0, 0.0)

            self.prev_centroids[oid] = (cx, cy)
            self.objects[oid] = box
            self.disappeared[oid] = 0

            if frame is not None:
                emb = self.extract_appearance(frame, box)
                if emb is not None:
                    if oid in self.appearances:
                        self.appearances[oid] = 0.85 * self.appearances[oid] + 0.15 * emb
                    else:
                        self.appearances[oid] = emb

        # Age and prune tracks no longer in active_objects
        for oid in list(self.objects.keys()):
            if oid not in now_tids:
                self.disappeared[oid] = self.disappeared.get(oid, 0) + 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)
                    self.velocities.pop(oid, None)
                    self.prev_centroids.pop(oid, None)
                    self.appearances.pop(oid, None)

        max_id = max(active_objects.keys(), default=0)
        if max_id >= self.next_id:
            self.next_id = max_id + 1

    def coast_tracks(self) -> Dict[int, Tuple[float, float, float, float]]:
        """Coasts active tracks forward along velocity vectors on cadence-skipped frames without aging disappeared counter."""
        for oid in list(self.objects.keys()):
            if self.disappeared.get(oid, 0) == 0:
                self.objects[oid] = self.predict_box(oid)
        return {oid: box for oid, box in self.objects.items() if self.disappeared.get(oid, 0) == 0}

    def update(
        self,
        detections: List[Tuple[float, float, float, float]],
        frame: Optional[np.ndarray] = None,
        only_active: bool = True,
    ) -> Dict[int, Tuple[float, float, float, float]]:
        """
        Updates tracks using object velocity prediction, appearance embeddings,
        and optimal bipartite matching.
        """
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] <= self.max_disappeared and oid in self.objects:
                    # Coast box along velocity trajectory
                    self.objects[oid] = self.predict_box(oid)
                elif self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)
                    self.velocities.pop(oid, None)
                    self.prev_centroids.pop(oid, None)
                    self.appearances.pop(oid, None)
            return {oid: box for oid, box in self.objects.items() if self.disappeared.get(oid, 0) == 0} if only_active else self.objects

        if len(self.objects) == 0:
            for box in detections:
                self.objects[self.next_id] = box
                self.disappeared[self.next_id] = 0
                cx = (box[0] + box[2]) * 0.5
                cy = box[3]
                self.prev_centroids[self.next_id] = (cx, cy)
                self.velocities[self.next_id] = (0.0, 0.0)
                if frame is not None:
                    emb = self.extract_appearance(frame, box)
                    if emb is not None:
                        self.appearances[self.next_id] = emb
                self.next_id += 1
            return self.objects

        # 1. Predict track positions along velocity vectors
        object_ids = list(self.objects.keys())
        pred_boxes = [self.predict_box(oid) for oid in object_ids]
        pred_centroids = [(((b[0] + b[2]) / 2.0), b[3]) for b in pred_boxes]

        det_centroids = [(((b[0] + b[2]) / 2.0), b[3]) for b in detections]
        det_appearances = [self.extract_appearance(frame, b) if frame is not None else None for b in detections]

        # 2. Build cost matrix combining predicted spatial distance, IoU, and appearance cosine distance
        num_tracks = len(object_ids)
        num_dets = len(detections)
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, oid in enumerate(object_ids):
            pred_b = pred_boxes[i]
            pcx, pcy = pred_centroids[i]
            track_app = self.appearances.get(oid)

            for j, det_b in enumerate(detections):
                dcx, dcy = det_centroids[j]
                spatial_dist = math.hypot(pcx - dcx, pcy - dcy)
                norm_dist = min(1.0, spatial_dist / self.max_distance)
                iou = self._compute_iou(pred_b, det_b)
                iou_cost = 1.0 - iou

                det_app = det_appearances[j]
                if track_app is not None and det_app is not None:
                    sim = max(0.0, float(np.dot(track_app, det_app)))
                    app_cost = 1.0 - sim
                    cost = 0.40 * norm_dist + 0.35 * iou_cost + 0.25 * app_cost
                else:
                    cost = 0.55 * norm_dist + 0.45 * iou_cost

                # Hard gating: reject if both distance exceeds max and IoU is negligible
                if spatial_dist > self.max_distance and iou < 0.10:
                    cost = 1.0

                cost_matrix[i, j] = cost

        # 3. Solve optimal bipartite matching
        matches, unmatched_tracks, unmatched_dets = solve_linear_assignment(cost_matrix, thresh=0.68)

        # 4. Update matched tracks
        for row, col in matches:
            oid = object_ids[row]
            det_box = detections[col]
            cx = (det_box[0] + det_box[2]) * 0.5
            cy = det_box[3]
            old_cx, old_cy = self.prev_centroids.get(oid, (cx, cy))
            vx = cx - old_cx
            vy = cy - old_cy
            prev_vx, prev_vy = self.velocities.get(oid, (0.0, 0.0))
            self.velocities[oid] = (0.7 * vx + 0.3 * prev_vx, 0.7 * vy + 0.3 * prev_vy)
            self.prev_centroids[oid] = (cx, cy)
            self.objects[oid] = det_box
            self.disappeared[oid] = 0

            det_app = det_appearances[col]
            if det_app is not None:
                if oid in self.appearances:
                    self.appearances[oid] = 0.85 * self.appearances[oid] + 0.15 * det_app
                else:
                    self.appearances[oid] = det_app

        # 5. Handle unmatched tracks
        for row in unmatched_tracks:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] <= self.max_disappeared:
                self.objects[oid] = self.predict_box(oid)
            else:
                self.objects.pop(oid, None)
                self.disappeared.pop(oid, None)
                self.velocities.pop(oid, None)
                self.prev_centroids.pop(oid, None)
                self.appearances.pop(oid, None)

        # 6. Handle unmatched detections (spawn new track)
        for col in unmatched_dets:
            det_box = detections[col]
            new_id = self.next_id
            self.next_id += 1
            self.objects[new_id] = det_box
            self.disappeared[new_id] = 0
            cx = (det_box[0] + det_box[2]) * 0.5
            cy = det_box[3]
            self.prev_centroids[new_id] = (cx, cy)
            self.velocities[new_id] = (0.0, 0.0)
            det_app = det_appearances[col]
            if det_app is not None:
                self.appearances[new_id] = det_app

        return {oid: box for oid, box in self.objects.items() if self.disappeared.get(oid, 0) == 0} if only_active else self.objects


# Backward compatibility alias
LocalTracker = RobustFallbackTracker


# ---------------------------------------------------------------------------
# Anti-Jitter EMA Track Smoother & Face Hysteresis Filter
# ---------------------------------------------------------------------------
class TrackSmoother:
    """
    Temporal filter & hysteresis manager for tracked persons and facial features.
    1. Anti-jitter Exponential Moving Average (EMA) with velocity deadband filter.
       Freezes bounding boxes when centroid movement < deadband (default 3.5px),
       eliminating sub-pixel micro-jitter on seated employees or stationary individuals.
    2. Face State Hysteresis & Track Memory Buffer:
       Retains neural face detections across momentary drops (up to N frames)
       to eliminate rapid color-flashing between Magenta 'Face' and Lavender 'Head (Away)'.
    3. Facial Landmark EMA smoothing (stabilizes the 5 landmark points).
    4. Head Fallback Box smoothing.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        deadband_px: float = 6.0,
        face_persistence: int = 8,
        body_persistence: int = 26,
        max_visual_coast: int = 20,
        enabled: bool = True,
    ):
        self.alpha = float(alpha)
        self.deadband_px = float(deadband_px)
        self.face_persistence = int(face_persistence)
        self.body_persistence = int(body_persistence)
        self.max_visual_coast = int(max_visual_coast)
        self.enabled = enabled

        # Person body boxes: obj_id -> (x1, y1, x2, y2)
        self.body_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        # Person previous centroids: obj_id -> (cx, cy)
        self.prev_centroids: Dict[int, Tuple[float, float]] = {}

        # Face state: obj_id -> dict
        self.face_tracks: Dict[int, dict] = {}

        # Head fallback boxes: obj_id -> (fx1, fy1, fx2, fy2)
        self.head_boxes: Dict[int, Tuple[float, float, float, float]] = {}

        # Track disappeared counter for GC
        self.disappeared: Dict[int, int] = {}
        # Track velocity vector: obj_id -> (vx, vy)
        self.velocities: Dict[int, Tuple[float, float]] = {}
        # Persistent ID alias mapping to maintain consistent IDs across frames
        self.id_aliases: Dict[int, int] = {}

    @staticmethod
    def _compute_iou(
        boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]
    ) -> float:
        """Calculates Intersection-over-Union (IoU) between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        denom = float(boxAArea + boxBArea - interArea)
        return (interArea / denom) if denom > 0.0 else 0.0

    def recover_missing_tracks(
        self,
        unmatched_boxes: List[Tuple[float, float, float, float]],
        max_dist: float = 110.0,
        min_iou: float = 0.20,
    ) -> Dict[int, Tuple[float, float, float, float]]:
        """
        Attempts to associate detections without IDs to recently missing tracks
        in the smoother persistence buffer, preventing duplicate tracks and identity switches.
        """
        recovered: Dict[int, Tuple[float, float, float, float]] = {}
        if not self.enabled or not unmatched_boxes:
            return recovered

        missing_ids = [
            tid
            for tid, count in self.disappeared.items()
            if 0 < count <= self.body_persistence and tid in self.body_boxes
        ]
        if not missing_ids:
            return recovered

        used_missing: Set[int] = set()
        for box in unmatched_boxes:
            ncx = (box[0] + box[2]) * 0.5
            ncy = (box[1] + box[3]) * 0.5
            best_mid = None
            best_score = float("inf")

            for mid in missing_ids:
                if mid in used_missing:
                    continue
                pbox = self.body_boxes[mid]
                count = self.disappeared.get(mid, 1)
                vx, vy = self.velocities.get(mid, (0.0, 0.0))
                mcx, mcy = self.prev_centroids.get(mid, ((pbox[0] + pbox[2]) * 0.5, (pbox[1] + pbox[3]) * 0.5))
                pred_mcx = mcx + vx * count
                pred_mcy = mcy + vy * count
                pred_pbox = (pbox[0] + vx * count, pbox[1] + vy * count, pbox[2] + vx * count, pbox[3] + vy * count)

                iou = self._compute_iou(box, pred_pbox)
                dist = math.hypot(ncx - pred_mcx, ncy - pred_mcy)

                if iou >= min_iou or dist < max_dist:
                    score = dist - (iou * 100.0)
                    if score < best_score:
                        best_score = score
                        best_mid = mid

            if best_mid is not None:
                recovered[best_mid] = box
                used_missing.add(best_mid)
                self.disappeared[best_mid] = 0

        return recovered

    def smooth_body_boxes(
        self,
        tracked_objects: Dict[int, Tuple[float, float, float, float]],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[int, Tuple[float, float, float, float]]:
        """Applies EMA smoothing, ultra-low-latency persistence coasting across drops, and velocity deadband to person body boxes."""
        if not self.enabled:
            return tracked_objects

        fw, fh = frame_shape if frame_shape is not None else (None, None)

        def _is_near_edge(box: Tuple[float, float, float, float], margin: float = 25.0) -> bool:
            if fw is None or fh is None:
                return False
            return (
                box[0] <= margin
                or box[1] <= margin
                or box[2] >= (fw - margin)
                or box[3] >= (fh - margin)
            )

        # 0. Apply persistent ID aliases to maintain consistent IDs across frames
        if self.id_aliases:
            remapped_tracked = {}
            for tid, box in tracked_objects.items():
                target_id = self.id_aliases.get(tid, tid)
                remapped_tracked[target_id] = box
            tracked_objects = remapped_tracked

        # 1. Re-associate new ByteTrack IDs to recently missing/active tracks not in incoming tracked_objects
        missing_ids = [
            tid
            for tid in list(self.body_boxes.keys())
            if tid not in tracked_objects and self.disappeared.get(tid, 0) <= self.body_persistence
        ]
        remap: Dict[int, int] = {}
        if missing_ids:
            for new_id, box in tracked_objects.items():
                if new_id not in self.body_boxes and new_id not in self.id_aliases:
                    ncx = (box[0] + box[2]) * 0.5
                    ncy = (box[1] + box[3]) * 0.5
                    best_match = None
                    best_dist = 110.0  # max re-association pixel distance
                    for mid in missing_ids:
                        mcx, mcy = self.prev_centroids.get(mid, (0.0, 0.0))
                        d = math.hypot(ncx - mcx, ncy - mcy)
                        if d < best_dist:
                            best_dist = d
                            best_match = mid
                    if best_match is not None:
                        remap[new_id] = best_match
                        self.id_aliases[new_id] = best_match
                        missing_ids.remove(best_match)

        if remap:
            remapped_tracked = {}
            for tid, box in tracked_objects.items():
                target_id = remap.get(tid, tid)
                remapped_tracked[target_id] = box
            tracked_objects = remapped_tracked

        active_ids = set(tracked_objects.keys())

        # 2. Update disappearance counters & purge tracks exceeding cutoff or exiting image edge
        for old_id in list(self.body_boxes.keys()):
            if old_id not in active_ids:
                self.disappeared[old_id] = self.disappeared.get(old_id, 0) + 1
                old_box = self.body_boxes[old_id]
                # If person was near the frame boundary when detection dropped, they exited the camera!
                # Purge immediately so the green box disappears with ZERO delay.
                if _is_near_edge(old_box) or self.disappeared[old_id] > self.body_persistence:
                    self.body_boxes.pop(old_id, None)
                    self.prev_centroids.pop(old_id, None)
                    self.face_tracks.pop(old_id, None)
                    self.head_boxes.pop(old_id, None)
                    self.disappeared.pop(old_id, None)
                    self.velocities.pop(old_id, None)
                    self.id_aliases = {k: v for k, v in self.id_aliases.items() if v != old_id and k != old_id}
            else:
                self.disappeared[old_id] = 0

        smoothed_dict: Dict[int, Tuple[float, float, float, float]] = {}

        # 3. Smooth active detections with EMA and velocity deadband
        for obj_id, raw_box in tracked_objects.items():
            rx1, ry1, rx2, ry2 = raw_box
            rcx = (rx1 + rx2) * 0.5
            rcy = (ry1 + ry2) * 0.5

            if obj_id not in self.body_boxes:
                self.body_boxes[obj_id] = raw_box
                self.prev_centroids[obj_id] = (rcx, rcy)
                self.velocities[obj_id] = (0.0, 0.0)
                smoothed_dict[obj_id] = raw_box
                continue

            px1, py1, px2, py2 = self.body_boxes[obj_id]
            pcx, pcy = self.prev_centroids[obj_id]

            vx = rcx - pcx
            vy = rcy - pcy
            prev_vx, prev_vy = self.velocities.get(obj_id, (0.0, 0.0))
            self.velocities[obj_id] = (0.7 * vx + 0.3 * prev_vx, 0.7 * vy + 0.3 * prev_vy)

            dist = math.hypot(rcx - pcx, rcy - pcy)
            dw = abs((rx2 - rx1) - (px2 - px1))
            dh = abs((ry2 - ry1) - (py2 - py1))

            # Stationary velocity deadband: if motion is negligible, freeze the box completely
            if dist < self.deadband_px and dw < 4.0 and dh < 4.0:
                smoothed_dict[obj_id] = self.body_boxes[obj_id]
                continue

            # Exponential Moving Average for fluid, lag-free motion
            sx1 = self.alpha * rx1 + (1.0 - self.alpha) * px1
            sy1 = self.alpha * ry1 + (1.0 - self.alpha) * py1
            sx2 = self.alpha * rx2 + (1.0 - self.alpha) * px2
            sy2 = self.alpha * ry2 + (1.0 - self.alpha) * py2

            smoothed_box = (sx1, sy1, sx2, sy2)
            self.body_boxes[obj_id] = smoothed_box
            self.prev_centroids[obj_id] = ((sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5)
            smoothed_dict[obj_id] = smoothed_box

        # 4. Ultra-responsive Persistence Coasting: Retain momentarily missed detections within a tight visual window (max 1-2 frames / ~60-90ms)
        # to smooth out single-frame detector blips, but NEVER display ghost boxes for 2-3 seconds when someone leaves!
        for old_id, miss_count in list(self.disappeared.items()):
            if 0 < miss_count <= self.max_visual_coast and old_id in self.body_boxes and old_id not in smoothed_dict:
                old_box = self.body_boxes[old_id]
                if not _is_near_edge(old_box):
                    vx, vy = self.velocities.get(old_id, (0.0, 0.0))
                    coasted_box = (old_box[0] + vx, old_box[1] + vy, old_box[2] + vx, old_box[3] + vy)
                    # Never coast a missing track if an active box already covers the same physical space
                    is_covered = any(
                        self._compute_iou(coasted_box, abox) > 0.30 for abox in smoothed_dict.values()
                    )
                    if not is_covered:
                        smoothed_dict[old_id] = coasted_box

        # 5. Overlap De-duplication (Track NMS): eliminate duplicate / ghost bounding boxes on the same individual
        deduped_dict: Dict[int, Tuple[float, float, float, float]] = {}
        sorted_candidates = sorted(
            smoothed_dict.items(),
            key=lambda item: (self.disappeared.get(item[0], 0), item[0]),
        )
        for tid, box in sorted_candidates:
            has_duplicate = False
            for kept_id, kept_box in deduped_dict.items():
                if self._compute_iou(box, kept_box) > 0.38:
                    has_duplicate = True
                    break
            if not has_duplicate:
                deduped_dict[tid] = box

        return deduped_dict

    def update_face(
        self,
        obj_id: int,
        matched_face: Optional[dict],
        person_box: Tuple[float, float, float, float],
        face_scale: float,
        frame_w: int,
        frame_h: int,
    ) -> Optional[dict]:
        """
        Updates face state for obj_id with EMA coordinate smoothing, landmark smoothing,
        and temporal persistence buffer to eliminate flicker.
        """
        if not self.enabled:
            if matched_face is not None:
                mfx, mfy, mfw, mfh = matched_face["box"]
                pad_w = int(mfw * 0.05 * face_scale)
                pad_h = int(mfh * 0.05 * face_scale)
                bx1 = max(0, int(mfx - pad_w))
                by1 = max(0, int(mfy - pad_h))
                bx2 = min(frame_w, int(mfx + mfw + pad_w))
                by2 = min(frame_h, int(mfy + mfh + pad_h))
                return {
                    "box": (bx1, by1, bx2, by2),
                    "conf": matched_face["conf"],
                    "landmarks": matched_face.get("landmarks", []),
                    "is_buffered": False,
                }
            return None

        px1, py1, px2, py2 = person_box
        curr_body_cx = (px1 + px2) * 0.5
        curr_body_cy = (py1 + py2) * 0.5

        if matched_face is not None:
            # Fresh detection from SCRFD or YuNet
            mfx, mfy, mfw, mfh = matched_face["box"]
            pad_w = int(mfw * 0.05 * face_scale)
            pad_h = int(mfh * 0.05 * face_scale)
            raw_bx1 = max(0.0, float(mfx - pad_w))
            raw_by1 = max(0.0, float(mfy - pad_h))
            raw_bx2 = min(float(frame_w), float(mfx + mfw + pad_w))
            raw_by2 = min(float(frame_h), float(mfy + mfh + pad_h))
            raw_conf = float(matched_face["conf"])
            raw_landmarks = matched_face.get("landmarks", [])

            prev_state = self.face_tracks.get(obj_id)
            if prev_state is not None and prev_state.get("active", False):
                # Apply smoothing to face box coordinates
                pbx1, pby1, pbx2, pby2 = prev_state["box"]
                pfcx = (pbx1 + pbx2) * 0.5
                pfcy = (pby1 + pby2) * 0.5
                nfcx = (raw_bx1 + raw_bx2) * 0.5
                nfcy = (raw_by1 + raw_by2) * 0.5
                fdist = math.hypot(nfcx - pfcx, nfcy - pfcy)

                if fdist < 2.5:
                    # Deadband: hold face box coordinates steady
                    sbx1, sby1, sbx2, sby2 = pbx1, pby1, pbx2, pby2
                else:
                    sbx1 = self.alpha * raw_bx1 + (1.0 - self.alpha) * pbx1
                    sby1 = self.alpha * raw_by1 + (1.0 - self.alpha) * pby1
                    sbx2 = self.alpha * raw_bx2 + (1.0 - self.alpha) * pbx2
                    sby2 = self.alpha * raw_by2 + (1.0 - self.alpha) * pby2
                    sbx2 = max(sbx1 + 10.0, sbx2)
                    sby2 = max(sby1 + 10.0, sby2)

                # Smooth facial landmarks
                prev_lms = prev_state.get("landmarks", [])
                if len(prev_lms) == len(raw_landmarks) and len(raw_landmarks) > 0:
                    smoothed_lms = []
                    for (rx, ry), (px, py) in zip(raw_landmarks, prev_lms):
                        slx = int(round(self.alpha * rx + (1.0 - self.alpha) * px))
                        sly = int(round(self.alpha * ry + (1.0 - self.alpha) * py))
                        smoothed_lms.append((slx, sly))
                else:
                    smoothed_lms = raw_landmarks

                s_conf = 0.75 * raw_conf + 0.10 * prev_state.get("conf", raw_conf)
                hit_count = prev_state.get("hit_count", 0) + 1
            else:
                sbx1, sby1, sbx2, sby2 = raw_bx1, raw_by1, raw_bx2, raw_by2
                smoothed_lms = raw_landmarks
                s_conf = raw_conf
                hit_count = 1

            face_res = {
                "box": (int(round(sbx1)), int(round(sby1)), int(round(sbx2)), int(round(sby2))),
                "conf": s_conf,
                "landmarks": smoothed_lms,
                "is_buffered": False,
                "active": True,
                "miss_count": 0,
                "hit_count": hit_count,
                "last_body_cx": curr_body_cx,
                "last_body_cy": curr_body_cy,
            }
            self.face_tracks[obj_id] = face_res
            return face_res

        # If no face was detected in this frame by SCRFD
        prev_state = self.face_tracks.get(obj_id)
        if prev_state is not None and prev_state.get("active", False):
            miss_count = prev_state.get("miss_count", 0) + 1
            prev_state["miss_count"] = miss_count

            if miss_count <= self.face_persistence:
                # Retain face via track memory buffer!
                # Translate with person's body movement if body shifted
                last_bcx = prev_state.get("last_body_cx", curr_body_cx)
                last_bcy = prev_state.get("last_body_cy", curr_body_cy)
                dx = curr_body_cx - last_bcx
                dy = curr_body_cy - last_bcy

                pbx1, pby1, pbx2, pby2 = prev_state["box"]
                nbx1 = int(round(max(0, min(frame_w - 1, pbx1 + dx))))
                nby1 = int(round(max(0, min(frame_h - 1, pby1 + dy))))
                nbx2 = int(round(max(0, min(frame_w, pbx2 + dx))))
                nby2 = int(round(max(0, min(frame_h, pby2 + dy))))

                # Shift landmarks along with body movement
                shifted_lms = [(int(lx + dx), int(ly + dy)) for (lx, ly) in prev_state.get("landmarks", [])]

                prev_state["box"] = (nbx1, nby1, nbx2, nby2)
                prev_state["landmarks"] = shifted_lms
                prev_state["last_body_cx"] = curr_body_cx
                prev_state["last_body_cy"] = curr_body_cy
                prev_state["is_buffered"] = True
                return prev_state
            else:
                # Persistent window expired, deactivate face so head fallback takes over cleanly
                prev_state["active"] = False

        return None

    def smooth_head_box(
        self,
        obj_id: int,
        raw_head_box: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        """Smooths anatomical head fallback box to eliminate jitter when looking away."""
        if not self.enabled:
            return raw_head_box

        rx1, ry1, rx2, ry2 = raw_head_box
        if obj_id not in self.head_boxes:
            self.head_boxes[obj_id] = (float(rx1), float(ry1), float(rx2), float(ry2))
            return raw_head_box

        px1, py1, px2, py2 = self.head_boxes[obj_id]
        hcx = (rx1 + rx2) * 0.5
        hcy = (ry1 + ry2) * 0.5
        phcx = (px1 + px2) * 0.5
        phcy = (py1 + py2) * 0.5

        if math.hypot(hcx - phcx, hcy - phcy) < 2.5:
            return (int(round(px1)), int(round(py1)), int(round(px2)), int(round(py2)))

        sx1 = self.alpha * rx1 + (1.0 - self.alpha) * px1
        sy1 = self.alpha * ry1 + (1.0 - self.alpha) * py1
        sx2 = self.alpha * rx2 + (1.0 - self.alpha) * px2
        sy2 = self.alpha * ry2 + (1.0 - self.alpha) * py2

        self.head_boxes[obj_id] = (sx1, sy1, sx2, sy2)
        return (int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2)))


# ---------------------------------------------------------------------------
# Employee Department Dwell & Presence Tracker
# ---------------------------------------------------------------------------
class EmployeeDepartmentTracker:
    """
    Real-Time Employee Department & Section Dwell Time Tracker.
    - Accurately tracks when recognized showroom staff enter a department, how much continuous
      time they spend inside, and when they exit.
    - Emits high-visibility, formatted console logs:
      * Entry:
        [--> EMPLOYEE ENTERED DEPARTMENT] Staff {name} ({emp_id}) entered '{dept_name}' at {time_str}
      * Active Dwell Updates (default every 10s):
        [⏱️ EMPLOYEE IN DEPARTMENT] Staff {name} ({emp_id}) in '{dept_name}' | Time spent: {dwell_str} ({seconds}s) | Total today: {cumul_str}
      * Exit:
        [<-- EMPLOYEE LEFT DEPARTMENT] Staff {name} ({emp_id}) left '{dept_name}' | Total time spent: {dwell_str} ({seconds}s) | Total today: {cumul_str} (Visits: {visits})
    - Keyed by persistent employee_id, ensuring zero timer resets or losses on ByteTrack ID flips.
    - Seamlessly transfers pre-recognition candidate track dwell time once biometric verification succeeds.
    - Debounces exits by 2.0s to prevent false exits due to momentary edge crossings or occlusions.
    - Formats digital timers (MM:SS) for bounding box badges & HUD telemetry.
    """

    def __init__(self, log_interval: float = 10.0, exit_debounce: float = 2.0):
        self.log_interval = float(log_interval)
        self.exit_debounce = float(exit_debounce)

        # Active employee department sessions: emp_id -> dict
        self.active_sessions: Dict[str, dict] = {}

        # Pending exit timestamps: emp_id -> timestamp when first not seen in dept
        self.pending_exits: Dict[str, float] = {}

        # Cumulative time spent per employee per department: emp_id -> { dept_id: total_seconds }
        self.cumulative_dwell: Dict[str, Dict[str, float]] = {}

        # Visit counts per employee per department: emp_id -> { dept_id: count }
        self.visit_counts: Dict[str, Dict[str, int]] = {}

        # Unconfirmed track history before biometric confirmation:
        # track_id (str) -> { "dept_id": str, "dept_name": str, "entry_time": float, "last_seen_time": float }
        self.unconfirmed_tracks: Dict[str, dict] = {}

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Formats duration into human-readable representation: e.g. '45s', '2m 15s', '1h 05m 20s'."""
        secs = int(max(0, round(seconds)))
        if secs < 60:
            return f"{secs}s"
        mins = secs // 60
        rem_secs = secs % 60
        if mins < 60:
            return f"{mins}m {rem_secs:02d}s"
        hrs = mins // 60
        rem_mins = mins % 60
        return f"{hrs}h {rem_mins:02d}m {rem_secs:02d}s"

    @staticmethod
    def format_digital(seconds: float) -> str:
        """Formats duration into digital clock format: MM:SS or HH:MM:SS."""
        secs = int(max(0, round(seconds)))
        mins = secs // 60
        rem_secs = secs % 60
        if mins < 60:
            return f"{mins:02d}:{rem_secs:02d}"
        hrs = mins // 60
        rem_mins = mins % 60
        return f"{hrs:02d}:{rem_mins:02d}:{rem_secs:02d}"

    def update_employee_presence(
        self,
        emp_id: str,
        emp_name: str,
        current_dept: Optional[Tuple[str, str]],
        now_ts: float,
        track_id: Optional[str] = None,
    ) -> None:
        """
        Updates employee presence in a department.
        current_dept: (dept_id, dept_name) or None if outside all departments.
        """
        emp_id = str(emp_id)
        if not emp_name:
            emp_name = emp_id

        if current_dept is not None:
            dept_id, dept_name = current_dept
            # Employee confirmed inside department: cancel any pending exit timer
            self.pending_exits.pop(emp_id, None)

            if emp_id in self.active_sessions:
                session = self.active_sessions[emp_id]
                if session["dept_id"] == dept_id:
                    # Ongoing dwell in same department
                    session["last_seen_time"] = now_ts
                    session["emp_name"] = emp_name
                    if track_id:
                        session["track_id"] = str(track_id)

                    time_spent = max(0.0, now_ts - session["entry_time"])
                    if now_ts - session["last_log_time"] >= self.log_interval:
                        session["last_log_time"] = now_ts
                        dur_str = self.format_duration(time_spent)
                        cumul_so_far = self.cumulative_dwell.get(emp_id, {}).get(dept_id, 0.0) + time_spent
                        cumul_str = self.format_duration(cumul_so_far)
                        print(
                            f"[⏱️ EMPLOYEE IN DEPARTMENT] Staff {emp_name} ({emp_id}) in '{dept_name}' | "
                            f"Time spent: {dur_str} ({time_spent:.1f}s) | Total today: {cumul_str}"
                        )
                else:
                    # Transitioned directly into a different department
                    old_dept_id = session["dept_id"]
                    old_dept_name = session["dept_name"]
                    spent = session["last_seen_time"] - session["entry_time"]
                    self._finalize_exit(emp_id, session["emp_name"], old_dept_id, old_dept_name, spent)

                    # Start new department session
                    self._start_session(emp_id, emp_name, dept_id, dept_name, now_ts, track_id)
            else:
                # New entry into department
                entry_time = now_ts
                if track_id and str(track_id) in self.unconfirmed_tracks:
                    cand = self.unconfirmed_tracks.pop(str(track_id))
                    if cand.get("dept_id") == dept_id:
                        entry_time = cand.get("entry_time", now_ts)

                self._start_session(emp_id, emp_name, dept_id, dept_name, entry_time, track_id, current_ts=now_ts)

        else:
            # Employee is not inside any department zone in this frame
            if emp_id in self.active_sessions:
                if emp_id not in self.pending_exits:
                    self.pending_exits[emp_id] = now_ts
                else:
                    outside_dur = now_ts - self.pending_exits[emp_id]
                    if outside_dur >= self.exit_debounce:
                        session = self.active_sessions.pop(emp_id)
                        self.pending_exits.pop(emp_id, None)
                        spent = session["last_seen_time"] - session["entry_time"]
                        self._finalize_exit(emp_id, session["emp_name"], session["dept_id"], session["dept_name"], spent)

    def _start_session(
        self,
        emp_id: str,
        emp_name: str,
        dept_id: str,
        dept_name: str,
        entry_time: float,
        track_id: Optional[str] = None,
        current_ts: Optional[float] = None,
    ) -> None:
        now_ts = current_ts or entry_time
        self.active_sessions[emp_id] = {
            "emp_id": emp_id,
            "emp_name": emp_name,
            "dept_id": dept_id,
            "dept_name": dept_name,
            "entry_time": entry_time,
            "last_seen_time": now_ts,
            "last_log_time": now_ts,
            "track_id": str(track_id) if track_id else None,
        }
        time_str = datetime.now().strftime("%H:%M:%S")
        tid_tag = f" [Track #{track_id}]" if track_id else ""
        dwell_prior = now_ts - entry_time
        if dwell_prior > 1.5:
            dwell_str = self.format_duration(dwell_prior)
            print(
                f"[--> EMPLOYEE DETECTED IN DEPARTMENT] Staff {emp_name} ({emp_id}){tid_tag} "
                f"detected inside '{dept_name}' at {time_str} (Dwell so far: {dwell_str})"
            )
        else:
            print(
                f"[--> EMPLOYEE ENTERED DEPARTMENT] Staff {emp_name} ({emp_id}){tid_tag} "
                f"entered '{dept_name}' at {time_str}"
            )

    def _finalize_exit(self, emp_id: str, emp_name: str, dept_id: str, dept_name: str, time_spent: float) -> None:
        if time_spent < 1.0:
            return
        emp_cumul = self.cumulative_dwell.setdefault(emp_id, {})
        emp_cumul[dept_id] = emp_cumul.get(dept_id, 0.0) + time_spent

        emp_visits = self.visit_counts.setdefault(emp_id, {})
        emp_visits[dept_id] = emp_visits.get(dept_id, 0) + 1

        dur_str = self.format_duration(time_spent)
        tot_cumul_str = self.format_duration(emp_cumul[dept_id])
        num_visits = emp_visits[dept_id]
        print(
            f"[<-- EMPLOYEE LEFT DEPARTMENT] Staff {emp_name} ({emp_id}) left '{dept_name}' | "
            f"Total time spent: {dur_str} ({time_spent:.1f}s) | "
            f"Total today: {tot_cumul_str} (Visits: {num_visits})"
        )

    def update_unconfirmed_track(self, track_id: str, dept: Optional[Tuple[str, str]], now_ts: float) -> None:
        """Tracks department presence for candidate tracks before facial recognition confirms identity."""
        tid = str(track_id)
        if dept is not None:
            dept_id, dept_name = dept
            if tid in self.unconfirmed_tracks and self.unconfirmed_tracks[tid]["dept_id"] == dept_id:
                self.unconfirmed_tracks[tid]["last_seen_time"] = now_ts
            else:
                self.unconfirmed_tracks[tid] = {
                    "dept_id": dept_id,
                    "dept_name": dept_name,
                    "entry_time": now_ts,
                    "last_seen_time": now_ts,
                }
        else:
            if tid in self.unconfirmed_tracks:
                if now_ts - self.unconfirmed_tracks[tid]["last_seen_time"] >= self.exit_debounce:
                    self.unconfirmed_tracks.pop(tid, None)

    def promote_track_to_employee(self, track_id: str, emp_id: str, emp_name: str, now_ts: float) -> None:
        """Promotes a track to an identified employee, transferring active department dwell if applicable."""
        tid = str(track_id)
        emp_id = str(emp_id)
        if tid in self.unconfirmed_tracks:
            cand = self.unconfirmed_tracks.pop(tid)
            if emp_id not in self.active_sessions:
                self._start_session(
                    emp_id=emp_id,
                    emp_name=emp_name,
                    dept_id=cand["dept_id"],
                    dept_name=cand["dept_name"],
                    entry_time=cand["entry_time"],
                    track_id=tid,
                    current_ts=now_ts,
                )

    def tick_inactive_employees(self, visible_staff_ids: Set[str], now_ts: float) -> None:
        """Checks for employees in active sessions who were not seen in this frame."""
        for emp_id in list(self.active_sessions.keys()):
            if emp_id not in visible_staff_ids:
                if emp_id not in self.pending_exits:
                    self.pending_exits[emp_id] = now_ts
                else:
                    if (now_ts - self.pending_exits[emp_id]) >= self.exit_debounce:
                        session = self.active_sessions.pop(emp_id)
                        self.pending_exits.pop(emp_id, None)
                        spent = session["last_seen_time"] - session["entry_time"]
                        self._finalize_exit(emp_id, session["emp_name"], session["dept_id"], session["dept_name"], spent)

    def on_employee_lost(self, emp_id: str, now_ts: float) -> None:
        """Immediate cleanup when an employee track expires or is explicitly lost."""
        emp_id = str(emp_id)
        self.pending_exits.pop(emp_id, None)
        if emp_id in self.active_sessions:
            session = self.active_sessions.pop(emp_id)
            spent = session["last_seen_time"] - session["entry_time"]
            self._finalize_exit(emp_id, session["emp_name"], session["dept_id"], session["dept_name"], spent)

    def get_active_dwell_info(self, emp_id: str, now_ts: float) -> Optional[Tuple[str, str, str]]:
        """Returns (dept_name, digital_str, duration_str) or None."""
        session = self.active_sessions.get(str(emp_id))
        if not session:
            return None
        time_spent = max(0.0, now_ts - session["entry_time"])
        return session["dept_name"], self.format_digital(time_spent), self.format_duration(time_spent)

    def get_staff_in_dept(self, dept_id: str, now_ts: float) -> List[Tuple[str, str]]:
        """Returns list of (emp_name, digital_timer_str) for staff currently active in dept_id."""
        res = []
        for emp_id, s in self.active_sessions.items():
            if s["dept_id"] == dept_id:
                dur = self.format_digital(max(0.0, now_ts - s["entry_time"]))
                res.append((s["emp_name"], dur))
        return res

    def get_all_active_summary(self, now_ts: float) -> str:
        """Returns a concise summary string of all staff currently inside departments for HUD ribbon."""
        items = []
        for emp_id, s in self.active_sessions.items():
            dur = self.format_digital(max(0.0, now_ts - s["entry_time"]))
            short_dept = s["dept_name"].split()[0]
            short_name = s["emp_name"].split()[0]
            items.append(f"{short_name} ({short_dept} {dur})")
        return "  |  ".join(items)

    def get_all_active_sessions_payload(self, now_ts: float) -> Dict[str, dict]:
        """Returns JSON-serializable dictionary of active and cumulative employee dwell for heartbeat telemetry."""
        payload = {}
        all_emp_ids = set(self.active_sessions.keys()).union(self.cumulative_dwell.keys())
        for emp_id in all_emp_ids:
            is_active = emp_id in self.active_sessions
            session = self.active_sessions.get(emp_id)
            spent = max(0.0, now_ts - session["entry_time"]) if session else 0.0

            if session:
                dept_id = session["dept_id"]
                dept_name = session["dept_name"]
                emp_name = session["emp_name"]
            else:
                emp_depts = self.cumulative_dwell.get(emp_id, {})
                dept_id = max(emp_depts.keys(), key=lambda d: emp_depts[d]) if emp_depts else "ai_department"
                dept_name = dept_id.replace("_", " ").title()
                emp_name = emp_id

            cumul_total = self.cumulative_dwell.get(emp_id, {}).get(dept_id, 0.0) + spent

            payload[emp_id] = {
                "name": emp_name,
                "department_id": dept_id,
                "department_name": dept_name,
                "is_active": is_active,
                "dwell_seconds": int(round(spent)),
                "dwell_formatted": self.format_digital(spent),
                "total_seconds": int(round(cumul_total)),
                "total_formatted": self.format_duration(cumul_total),
                "visits": self.visit_counts.get(emp_id, {}).get(dept_id, 1 if is_active else 0),
            }
        return payload

    def finalize_all(self, now_ts: float) -> None:
        """Finalizes all active sessions on shutdown and prints a comprehensive summary table."""
        for emp_id in list(self.active_sessions.keys()):
            session = self.active_sessions.pop(emp_id)
            spent = session["last_seen_time"] - session["entry_time"]
            self._finalize_exit(emp_id, session["emp_name"], session["dept_id"], session["dept_name"], spent)
        self.pending_exits.clear()

        # Print summary if any dwell was recorded
        if self.cumulative_dwell:
            print("\n" + "=" * 65)
            print("  SHOWROOM EMPLOYEE DEPARTMENT DWELL TIME SUMMARY")
            print("=" * 65)
            for emp_id, depts in self.cumulative_dwell.items():
                print(f"[*] Employee: {emp_id}")
                for d_id, total_secs in depts.items():
                    v_count = self.visit_counts.get(emp_id, {}).get(d_id, 0)
                    print(f"    - {d_id}: {self.format_duration(total_secs)} ({total_secs:.1f}s) across {v_count} visit(s)")
            print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Interactive Multi-Zone & Dynamic User-Drawn Entry Line Manager
# ---------------------------------------------------------------------------
class ZoneManager:
    """
    Optimized Dynamic User-Drawn Entry Line and Department Manager.
    - True Vector Normal Directional Crossing: calculates 2D cross product against line normal vector,
      supporting angled, horizontal, vertical, and diagonal entrances accurately.
    - Flip Direction [D]: Instantly reverses entry/exit orientation with live perpendicular arrow.
    - Multi-Department Coverage: Sales, Service, and Accounts with anti-jitter dwell-time verification.
    - Zero-Lag In-Place ROI Blending: eliminates full-frame memory clones for high FPS.
    - Interactive Clickable Toolbar & Keyboard Shortcuts (1: Line, 2: Sales, 3: Service, 4: Accounts, D: Flip, R: Clear).
    - Persistent Identity Linking: prevents multi-counting visitors across ByteTrack ID switches.
    """

    def __init__(
        self,
        config_path: str = "camera_config.json",
        camera_channel: str = "201",
        entry_line_cfg: Optional[dict] = None,
        sections_cfg: Optional[list] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        line_y_ratio: float = 0.35,
        direction: str = "forward",
        api_url: str = "http://127.0.0.1:8000",
    ):
        self.config_path = config_path
        self.camera_channel = str(camera_channel)
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.api_url = str(api_url).rstrip("/")

        # Load entry line only if explicitly saved by user previously
        self.entry_line = None
        if entry_line_cfg and isinstance(entry_line_cfg, dict) and "p1" in entry_line_cfg and "p2" in entry_line_cfg and entry_line_cfg.get("enabled", True):
            p1 = entry_line_cfg["p1"]
            p2 = entry_line_cfg["p2"]
            if len(p1) == 2 and len(p2) == 2:
                self.entry_line = {
                    "enabled": True,
                    "p1": [int(p1[0]), int(p1[1])],
                    "p2": [int(p2[0]), int(p2[1])],
                    "direction": entry_line_cfg.get("direction", direction),
                }

        # Load sections only if explicitly saved by user previously
        self.sections: List[dict] = []
        if sections_cfg and isinstance(sections_cfg, list):
            for s in sections_cfg:
                if isinstance(s, dict):
                    self.sections.append({
                        "id": str(s.get("id", "section")),
                        "name": str(s.get("name", "Section")),
                        "color": tuple(s.get("color", [0, 215, 255])),
                        "box": [int(s["box"][0]), int(s["box"][1]), int(s["box"][2]), int(s["box"][3])] if ("box" in s and len(s["box"]) == 4) else [0, 0, 0, 0],
                    })

        # Multi-Zone Counters
        self.attendance_total = 0
        self.section_totals: Dict[str, int] = {s["id"]: 0 for s in self.sections}
        self.live_occupancy: Dict[str, int] = {s["id"]: 0 for s in self.sections}

        # Track history & persistent identity linking
        self.counted_attendance_ids: Set[int] = set()
        self.counted_person_ids: Set[str] = set()
        self.track_identities: Dict[int, str] = {}
        self.person_visited_sections: Dict[int, Set[str]] = {}
        self.counted_person_sections: Dict[str, Set[str]] = {}
        self.person_section_dwell: Dict[Tuple[int, str], int] = {}
        self.track_positions: Dict[int, Tuple[float, float]] = {}  # obj_id -> prev feet (x, y)
        self.section_occupants: Dict[str, list] = {s["id"]: [] for s in self.sections}

        # Visual Flash Timers
        self.line_flash_timer = 0
        self.section_flash_timers: Dict[str, int] = {s["id"]: 0 for s in self.sections}

        # Interactive Drawing State
        self.active_tool = "entry_line"
        self.is_mouse_down = False
        self.mouse_start = (0, 0)
        self.mouse_current = (0, 0)
        self.status_msg = "CLICK & DRAG TO DRAW ENTRY LINE OR SELECT A DEPARTMENT TOOL ABOVE"
        self.status_timer = 250
        self.dept_tracker: Optional[Any] = None

        # Dynamic Toolbar button layout (computes dynamic department buttons)
        self.btn_sections: List[Tuple[int, int, int, int, dict]] = []
        resolved_cfg = get_resource_path(self.config_path)
        self._last_config_mtime = os.path.getmtime(resolved_cfg) if os.path.exists(resolved_cfg) else 0.0
        self._compute_toolbar_layout()

    def _compute_toolbar_layout(self):
        """Dynamically computes top toolbar button boundaries for entry line and all configured departments."""
        self.btn_line = (10, 8, 135, 46)
        curr_x = 143
        self.btn_sections = []
        for i, s in enumerate(self.sections):
            short_name = s["name"].split()[0][:8]
            btn_w = max(80, min(120, len(short_name) * 9 + 44))
            x1 = curr_x
            x2 = curr_x + btn_w
            self.btn_sections.append((x1, 8, x2, 46, s))
            curr_x = x2 + 8

        self.btn_flip = (curr_x, 8, curr_x + 115, 46)
        curr_x += 123
        self.btn_reset = (curr_x, 8, curr_x + 95, 46)
        curr_x += 103
        self.btn_clear = (curr_x, 8, curr_x + 85, 46)

    def find_section_for_box(self, box: Tuple[float, float, float, float]) -> Optional[Tuple[str, str]]:
        """
        Returns (section_id, section_name) if the bounding box torso center or body volume is genuinely inside a department.
        Uses torso center and overlap ratio to avoid false triggers from background passersby.
        """
        x1, y1, x2, y2 = box
        feet_x = (x1 + x2) * 0.5
        feet_y = y2
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        person_area = max(1.0, (x2 - x1) * (y2 - y1))

        best_match = None
        max_overlap = 0.0

        for s in self.sections:
            bx1, by1, bx2, by2 = s["box"]
            if (bx2 - bx1) <= 15 or (by2 - by1) <= 15:
                continue

            # Compute intersection overlap
            inter_w = max(0.0, min(x2, bx2) - max(x1, bx1))
            inter_h = max(0.0, min(y2, by2) - max(y1, by1))
            overlap_ratio = (inter_w * inter_h) / person_area

            # Torso center is inside department box
            center_inside = (bx1 <= cx <= bx2) and (by1 <= cy <= by2)
            # Feet inside department box and person's center is not far above the box in the background
            feet_inside = (bx1 <= feet_x <= bx2) and (by1 <= feet_y <= by2) and (cy >= by1 - 15)

            if center_inside or overlap_ratio >= 0.25 or feet_inside:
                score = overlap_ratio if overlap_ratio > 0 else (1.0 if center_inside else 0.5)
                if score > max_overlap:
                    max_overlap = score
                    best_match = (s["id"], s["name"])

        return best_match

    def find_section_for_point(self, pt: Tuple[float, float]) -> Optional[Tuple[str, str]]:
        """
        Returns (section_id, section_name) if the point (x, y) is inside a valid drawn department.
        """
        px, py = pt
        for s in self.sections:
            bx1, by1, bx2, by2 = s["box"]
            if (bx2 - bx1) <= 15 or (by2 - by1) <= 15:
                continue
            if bx1 <= px <= bx2 and by1 <= py <= by2:
                return s["id"], s["name"]
        return None

    def check_and_reload_config(self) -> bool:
        """Dynamically reloads departments and entry line if camera_config.json was modified."""
        try:
            resolved = get_resource_path(self.config_path)
            if not os.path.exists(resolved):
                return False
            mtime = os.path.getmtime(resolved)
            if mtime <= self._last_config_mtime:
                return False
            self._last_config_mtime = mtime

            with open(resolved, "r", encoding="utf-8") as f:
                data = json.load(f)
            cams = data.get("cameras", {})
            cam_data = cams.get(self.camera_channel, {})
            sections_cfg = cam_data.get("sections")
            if sections_cfg is not None and isinstance(sections_cfg, list):
                # Preserve drawn boxes in memory if config box is empty [0, 0, 0, 0]
                mem_boxes = {s["id"]: s["box"] for s in self.sections if s.get("box") and s["box"] != [0, 0, 0, 0]}
                new_sections = []
                for s in sections_cfg:
                    if isinstance(s, dict):
                        sid = str(s.get("id", "section"))
                        s_box = [int(s["box"][0]), int(s["box"][1]), int(s["box"][2]), int(s["box"][3])] if ("box" in s and len(s["box"]) == 4) else [0, 0, 0, 0]
                        if s_box == [0, 0, 0, 0] and sid in mem_boxes:
                            s_box = mem_boxes[sid]
                        new_sections.append({
                            "id": sid,
                            "name": str(s.get("name", "Section")),
                            "color": tuple(s.get("color", [0, 215, 255])),
                            "box": s_box,
                        })
                old_ids = [s["id"] for s in self.sections]
                new_ids = [s["id"] for s in new_sections]
                if old_ids != new_ids or len(self.sections) != len(new_sections):
                    self.sections = new_sections
                    for s in self.sections:
                        self.section_totals.setdefault(s["id"], 0)
                        self.live_occupancy.setdefault(s["id"], 0)
                        self.section_flash_timers.setdefault(s["id"], 0)
                    self._compute_toolbar_layout()
                    self.status_msg = f"[+] Departments synced with Dashboard ({len(self.sections)} active)"
                    self.status_timer = 140
                    print(f"[*] Dynamically reloaded {len(self.sections)} departments from {self.config_path}")
                    return True
        except Exception:
            pass
        return False

    def reset_counts(self):
        """Clears and resets Entry count and section visits to 0 without clearing drawn zones."""
        self.attendance_total = 0
        self.counted_attendance_ids.clear()
        self.counted_person_ids.clear()
        self.person_visited_sections.clear()
        self.counted_person_sections.clear()
        self.person_section_dwell.clear()
        for sid in self.section_occupants:
            self.section_occupants[sid].clear()
        for sid in self.section_totals:
            self.section_totals[sid] = 0
        self.status_msg = "[+] Entry count cleared and reset to 0!"
        self.status_timer = 150
        print("[+] Entry count cleared and reset to 0.")

    def set_identity(self, obj_id: int, person_id: Optional[str]) -> None:
        """Links temporary ByteTrack ID to persistent person ID to prevent duplicate entry counts."""
        if person_id:
            pid = str(person_id)
            self.track_identities[obj_id] = pid
            if obj_id in self.counted_attendance_ids:
                self.counted_person_ids.add(pid)
            if pid in self.counted_person_sections:
                self.person_visited_sections.setdefault(obj_id, set()).update(self.counted_person_sections[pid])
            for sid, occupants in self.section_occupants.items():
                for occ in occupants:
                    if occ.get("track_id") == obj_id:
                        occ["pid"] = pid

    def flip_direction(self):
        """Cycles entry line crossing direction between forward, backward, and bidirectional (both)."""
        if not self.entry_line:
            self.status_msg = "[!] No Entry Line drawn yet. Draw one first!"
            self.status_timer = 120
            return
        curr_dir = self.entry_line.get("direction", "forward")
        if curr_dir == "forward":
            new_dir = "backward"
            dir_label = "BACKWARD (➔ Inward)"
        elif curr_dir == "backward":
            new_dir = "both"
            dir_label = "BOTH WAYS (◄➔ Bidirectional)"
        else:
            new_dir = "forward"
            dir_label = "FORWARD (➔ Inward)"
        self.entry_line["direction"] = new_dir
        self.status_msg = f"[+] Entry Direction switched to: {dir_label}"
        self.status_timer = 160
        print(f"[+] Entry Line direction switched to: {new_dir} ({dir_label})")
        self.save_to_config()

    @staticmethod
    def _draw_entry_arrow(frame: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], direction: str, is_flashing: bool, count: int = 0):
        """Draws high-visibility, user-friendly directional arrows and badge along the entry line."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length <= 0:
            return

        # Direction vector u pointing in the exact entry crossing direction:
        # Cross product cp(p1, p2, p) < 0 side is reached when moving along (dy/length, -dx/length).
        # Forward check: cp_prev >= 0 and cp_curr < 0 -> moves towards cp < 0.
        # Backward check: cp_prev <= 0 and cp_curr > 0 -> moves towards cp > 0.
        ux = dy / length
        uy = -dx / length

        if direction == "backward":
            ux, uy = -ux, -uy
        elif direction == "top_to_bottom":
            if uy < 0:
                ux, uy = -ux, -uy
        elif direction == "bottom_to_top":
            if uy > 0:
                ux, uy = -ux, -uy

        arrow_color = (0, 255, 128) if is_flashing else (0, 215, 255)
        bg_outline = (15, 23, 42)

        # 1. Chevrons / arrows along the line (28%, 50%, and 72% of segment length)
        fractions = [0.28, 0.50, 0.72] if length > 140 else [0.50]
        arrow_len = 34 if length > 140 else 24

        for frac in fractions:
            mx = int(p1[0] + dx * frac)
            my = int(p1[1] + dy * frac)

            if direction == "both":
                # Double-headed arrow (bidirectional)
                p_fwd = (int(mx + ux * arrow_len), int(my + uy * arrow_len))
                p_bwd = (int(mx - ux * arrow_len), int(my - uy * arrow_len))
                cv2.arrowedLine(frame, (mx, my), p_fwd, bg_outline, 5, tipLength=0.36)
                cv2.arrowedLine(frame, (mx, my), p_bwd, bg_outline, 5, tipLength=0.36)
                cv2.arrowedLine(frame, (mx, my), p_fwd, arrow_color, 2, tipLength=0.36)
                cv2.arrowedLine(frame, (mx, my), p_bwd, arrow_color, 2, tipLength=0.36)
            else:
                start_pt = (int(mx - ux * 4), int(my - uy * 4))
                end_pt = (int(mx + ux * arrow_len), int(my + uy * arrow_len))
                # High-contrast outline for any floor color (carpet, marble, tiles)
                cv2.arrowedLine(frame, start_pt, end_pt, bg_outline, 5, tipLength=0.38)
                cv2.arrowedLine(frame, start_pt, end_pt, arrow_color, 2, tipLength=0.38)

        # 2. Prominent Direction Badge at Center Arrow Tip
        center_mx = int((p1[0] + p2[0]) / 2)
        center_my = int((p1[1] + p2[1]) / 2)

        if direction == "both":
            badge_text = "◄ BIDIRECTIONAL ENTRY ►"
            bx = max(10, min(frame.shape[1] - 220, center_mx - 90))
            by = max(60, min(frame.shape[0] - 30, center_my - 18))
        else:
            dir_tag = "ENTRY DIRECTION" if not is_flashing else "ENTRY DETECTED!"
            badge_text = f"➔ {dir_tag}"
            bx = max(10, min(frame.shape[1] - 180, int(center_mx + ux * (arrow_len + 16)) - 50))
            by = max(60, min(frame.shape[0] - 30, int(center_my + uy * (arrow_len + 16)) - 10))

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(frame, (bx - 6, by - 4), (bx + tw + 6, by + th + 6), bg_outline, -1)
        cv2.rectangle(frame, (bx - 6, by - 4), (bx + tw + 6, by + th + 6), arrow_color, 1)
        cv2.putText(frame, badge_text, (bx, by + th), cv2.FONT_HERSHEY_SIMPLEX, 0.44, arrow_color, 1, cv2.LINE_AA)

    @staticmethod
    def _cross_product(A, B, C):
        """2D Cross product of vector AB and AC."""
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def _segments_intersect(A, B, C, D):
        cp1 = ZoneManager._cross_product(A, B, C)
        cp2 = ZoneManager._cross_product(A, B, D)
        cp3 = ZoneManager._cross_product(C, D, A)
        cp4 = ZoneManager._cross_product(C, D, B)
        return (cp1 * cp2 <= 0) and (cp3 * cp4 <= 0)

    def process(
        self, tracked_objects: Dict[int, Tuple[float, float, float, float]]
    ) -> Tuple[List[int], List[Tuple[int, str, str]]]:
        new_attendance_crossings: List[int] = []
        new_section_entries: List[Tuple[int, str, str]] = []

        # Reset live occupancy in each frame
        for s in self.sections:
            self.live_occupancy[s["id"]] = 0

        has_entry_line = (
            self.entry_line is not None
            and self.entry_line.get("enabled", True)
            and "p1" in self.entry_line
            and "p2" in self.entry_line
        )

        p1 = tuple(self.entry_line["p1"]) if has_entry_line else None
        p2 = tuple(self.entry_line["p2"]) if has_entry_line else None
        direction = self.entry_line.get("direction", "forward") if has_entry_line else "forward"

        for obj_id, box in tracked_objects.items():
            feet_x = (box[0] + box[2]) * 0.5
            feet_y = box[3]
            curr_pos = (feet_x, feet_y)

            # 1. Check Entry Line Crossing
            pid = self.track_identities.get(obj_id)
            is_already_counted = (obj_id in self.counted_attendance_ids) or (pid and pid in self.counted_person_ids)
            if has_entry_line and not is_already_counted:
                prev_pos = self.track_positions.get(obj_id)
                if prev_pos is not None:
                    # Sensitive displacement filter of 1.5px to reliably capture slow/smooth crossings
                    disp = math.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                    if disp >= 1.5 and self._segments_intersect(p1, p2, prev_pos, curr_pos):
                        cp_prev = self._cross_product(p1, p2, prev_pos)
                        cp_curr = self._cross_product(p1, p2, curr_pos)

                        crossed = False
                        if direction == "forward":
                            crossed = (cp_prev >= 0 and cp_curr < 0) or (cp_prev > 0 and cp_curr <= 0)
                        elif direction == "backward":
                            crossed = (cp_prev <= 0 and cp_curr > 0) or (cp_prev < 0 and cp_curr >= 0)
                        elif direction == "top_to_bottom":
                            crossed = (curr_pos[1] >= prev_pos[1]) or (cp_prev >= 0 and cp_curr < 0)
                        elif direction == "bottom_to_top":
                            crossed = (curr_pos[1] <= prev_pos[1]) or (cp_prev <= 0 and cp_curr > 0)
                        else:  # "both" (bidirectional entry)
                            crossed = True

                        if crossed:
                            self.counted_attendance_ids.add(obj_id)
                            if pid:
                                self.counted_person_ids.add(pid)
                            self.attendance_total += 1
                            self.line_flash_timer = 25
                            new_attendance_crossings.append(obj_id)
            elif has_entry_line and (obj_id in self.counted_attendance_ids):
                # Reverse crossing: person walked back out, re-arm for next entry
                prev_pos = self.track_positions.get(obj_id)
                if prev_pos is not None and self._segments_intersect(p1, p2, prev_pos, curr_pos):
                    cp_prev = self._cross_product(p1, p2, prev_pos)
                    cp_curr = self._cross_product(p1, p2, curr_pos)
                    rev_crossed = False
                    if direction == "forward":
                        rev_crossed = (cp_prev <= 0 and cp_curr > 0)
                    elif direction == "backward":
                        rev_crossed = (cp_prev >= 0 and cp_curr < 0)
                    elif direction == "top_to_bottom":
                        rev_crossed = (curr_pos[1] < prev_pos[1])
                    elif direction == "bottom_to_top":
                        rev_crossed = (curr_pos[1] > prev_pos[1])
                    elif direction == "both":
                        rev_crossed = True
                    if rev_crossed:
                        self.counted_attendance_ids.discard(obj_id)
                        if pid and pid in self.counted_person_ids:
                            self.counted_person_ids.discard(pid)

            self.track_positions[obj_id] = curr_pos

            # 2. Check Sections (AI Dept, BDE, Marketing, etc.) with Torso-Anchored Dwell Verification
            if self.sections:
                visited = self.person_visited_sections.setdefault(obj_id, set())
                matched_sec = self.find_section_for_box(box)
                for s in self.sections:
                    sid = s["id"]
                    dwell_key = (obj_id, sid)
                    is_in_this_sec = (matched_sec is not None and matched_sec[0] == sid)

                    if is_in_this_sec:
                        self.live_occupancy[sid] = self.live_occupancy.get(sid, 0) + 1
                        # Track consecutive frames inside department zone
                        self.person_section_dwell[dwell_key] = self.person_section_dwell.get(dwell_key, 0) + 1
                        # Require at least 3 consecutive frames (~0.12s) to register genuine entry
                        if self.person_section_dwell[dwell_key] >= 3:
                            if sid not in visited:
                                visited.add(sid)
                                if pid:
                                    self.counted_person_sections.setdefault(pid, set()).add(sid)
                                self.section_totals[sid] = self.section_totals.get(sid, 0) + 1
                                self.section_flash_timers[sid] = 25
                                new_section_entries.append((obj_id, sid, s["name"]))
                    else:
                        if dwell_key in self.person_section_dwell:
                            self.person_section_dwell[dwell_key] = 0
                        # Re-arm visit if person has been outside this section for >= 15 frames (~0.6s)
                        if sid in visited:
                            outside_key = (obj_id, f"{sid}_outside")
                            outside_cnt = self.person_section_dwell.get(outside_key, 0) + 1
                            self.person_section_dwell[outside_key] = outside_cnt
                            if outside_cnt >= 15:
                                visited.discard(sid)
                                self.person_section_dwell.pop(outside_key, None)

        # Decrement flash timers
        if self.line_flash_timer > 0:
            self.line_flash_timer -= 1
        for sid in list(self.section_flash_timers.keys()):
            if self.section_flash_timers[sid] > 0:
                self.section_flash_timers[sid] -= 1
        if self.status_timer > 0:
            self.status_timer -= 1

        return new_attendance_crossings, new_section_entries

    def handle_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check Toolbar Button Clicks (Top Bar)
            if 8 <= y <= 46:
                if self.btn_line[0] <= x <= self.btn_line[2]:
                    self.active_tool = "entry_line"
                    self.status_msg = "Selected: [1. DRAW ENTRY LINE] - Click & Drag across entrance"
                    self.status_timer = 150
                    print("[*] Tool selected: [Entry Line]")
                    return
                for i, (bx1, by1, bx2, by2, sec) in enumerate(self.btn_sections):
                    if bx1 <= x <= bx2:
                        self.active_tool = sec["id"]
                        self.status_msg = f"Selected: [{i+2}. DRAW {sec['name'].upper()}] - Click & Drag box over area"
                        self.status_timer = 150
                        print(f"[*] Tool selected: [{sec['name']}]")
                        return
                if self.btn_flip[0] <= x <= self.btn_flip[2]:
                    self.flip_direction()
                    return
                elif self.btn_reset[0] <= x <= self.btn_reset[2]:
                    self.reset_counts()
                    return
                elif self.btn_clear[0] <= x <= self.btn_clear[2]:
                    self.clear_all()
                    return

            # Clicked on video area: start drawing
            self.is_mouse_down = True
            self.mouse_start = (x, y)
            self.mouse_current = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.is_mouse_down:
            self.mouse_current = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.is_mouse_down:
            self.is_mouse_down = False
            self.mouse_current = (x, y)
            sx, sy = self.mouse_start
            ex, ey = x, y

            if self.active_tool == "entry_line":
                if math.hypot(ex - sx, ey - sy) > 20:
                    curr_dir = self.entry_line.get("direction", "forward") if self.entry_line else "forward"
                    self.entry_line = {
                        "enabled": True,
                        "p1": [int(sx), int(sy)],
                        "p2": [int(ex), int(ey)],
                        "direction": curr_dir,
                    }
                    self.status_msg = f"[+] Entry Line saved: ({sx},{sy}) -> ({ex},{ey})!"
                    self.status_timer = 150
                    print(f"[+] Entry Line drawn and saved: [{sx}, {sy}] -> [{ex}, {ey}]")
                    self.save_to_config()
            else:
                x1, x2 = min(sx, ex), max(sx, ex)
                y1, y2 = min(sy, ey), max(sy, ey)
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    sec_id = self.active_tool
                    found_sec = next((s for s in self.sections if s["id"] == sec_id), None)
                    sec_name = found_sec["name"] if found_sec else sec_id.replace("_", " ").title()
                    color = found_sec["color"] if found_sec else (0, 215, 255)

                    if found_sec:
                        found_sec["box"] = [int(x1), int(y1), int(x2), int(y2)]
                    else:
                        self.sections.append({
                            "id": sec_id,
                            "name": sec_name,
                            "color": color,
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                        })
                        self._compute_toolbar_layout()

                    if sec_id not in self.section_totals:
                        self.section_totals[sec_id] = 0
                    if sec_id not in self.live_occupancy:
                        self.live_occupancy[sec_id] = 0

                    self.status_msg = f"[+] {sec_name} box saved: [{x1},{y1},{x2},{y2}]!"
                    self.status_timer = 150
                    print(f"[+] {sec_name} drawn and saved: [{x1}, {y1}, {x2}, {y2}]")
                    self.save_to_config()
                    self._sync_box_to_backend(sec_id, [int(x1), int(y1), int(x2), int(y2)])

    def clear_all(self):
        self.entry_line = None
        for s in self.sections:
            s["box"] = [0, 0, 0, 0]
        self.attendance_total = 0
        self.counted_attendance_ids.clear()
        self.counted_person_ids.clear()
        self.person_visited_sections.clear()
        self.counted_person_sections.clear()
        self.person_section_dwell.clear()
        for sid in self.section_totals:
            self.section_totals[sid] = 0
        self.live_occupancy.clear()
        self.status_msg = "[*] All drawn lines & section boxes cleared! Draw new ones anytime."
        self.status_timer = 150
        print("[*] Cleared drawn entry lines and department boxes.")
        self.save_to_config()

    def _sync_box_to_backend(self, sec_id: str, box: list):
        """Sends updated department bounding box to backend API so Dashboard stays in sync."""
        def _bg_patch():
            try:
                import urllib.request
                req_data = json.dumps({"box": box}).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.api_url}/api/v1/departments/{sec_id}",
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                    method="PATCH",
                )
                urllib.request.urlopen(req, timeout=1.5)
            except Exception:
                pass
        import threading
        threading.Thread(target=_bg_patch, daemon=True).start()

    def save_to_config(self) -> bool:
        try:
            resolved = get_resource_path(self.config_path)
            data = {}
            if os.path.exists(resolved):
                with open(resolved, "r", encoding="utf-8") as f:
                    data = json.load(f)
            cams = data.setdefault("cameras", {})
            cam_data = cams.setdefault(self.camera_channel, {})
            cam_data["entry_line"] = self.entry_line

            # Safely merge: keep all existing configured departments, update boxes for self.sections
            existing_sections = cam_data.get("sections", [])
            merged = []
            seen_ids = set()
            for s in self.sections:
                merged.append({
                    "id": s["id"],
                    "name": s["name"],
                    "color": list(s["color"]),
                    "box": list(s["box"]),
                })
                seen_ids.add(s["id"])

            if isinstance(existing_sections, list):
                for es in existing_sections:
                    if isinstance(es, dict) and es.get("id") not in seen_ids:
                        merged.append(es)
                        seen_ids.add(es.get("id"))

            cam_data["sections"] = merged
            self.sections = merged
            self._compute_toolbar_layout()

            with open(resolved, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            self._last_config_mtime = os.path.getmtime(resolved)
            return True
        except Exception as e:
            print(f"[!] Warning: Failed to save config: {e}")
            return False

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # 1. Zero-Lag In-Place ROI Blending for User-Drawn Departments
        if self.sections:
            for s in self.sections:
                x1, y1, x2, y2 = s["box"]
                if (x2 - x1) <= 15 or (y2 - y1) <= 15:
                    continue  # Undrawn zone, skip rendering on floor
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                if x2_c > x1_c and y2_c > y1_c:
                    roi = frame[y1_c:y2_c, x1_c:x2_c]
                    color_block = np.full_like(roi, s["color"], dtype=np.uint8)
                    cv2.addWeighted(color_block, 0.16, roi, 0.84, 0, roi)

            for s in self.sections:
                x1, y1, x2, y2 = s["box"]
                if (x2 - x1) <= 15 or (y2 - y1) <= 15:
                    continue  # Undrawn zone, badge not displayed until drawn
                color = s["color"]
                is_flashing = self.section_flash_timers.get(s["id"], 0) > 0
                border_color = (0, 255, 128) if is_flashing else color
                thickness = 3 if is_flashing else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness)

                # Department Badge with live occupancy & total count
                occ = self.live_occupancy.get(s["id"], 0)
                tot = self.section_totals.get(s["id"], 0)
                badge_text = f"{s['name'].upper()} | Active: {occ} | Visits: {tot}"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.rectangle(frame, (x1, y1), (x1 + tw + 16, y1 + 24), (15, 23, 42), -1)
                cv2.rectangle(frame, (x1, y1), (x1 + tw + 16, y1 + 24), border_color, 1)
                cv2.putText(frame, badge_text, (x1 + 8, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Render User-Drawn Entry Line with Prominent High-Contrast Directional Indicator
        if self.entry_line and self.entry_line.get("enabled", True):
            p1 = tuple(self.entry_line["p1"])
            p2 = tuple(self.entry_line["p2"])
            direction = self.entry_line.get("direction", "forward")
            is_flashing = self.line_flash_timer > 0
            line_color = (0, 255, 128) if is_flashing else (0, 215, 255)
            line_thickness = 4 if is_flashing else 2

            # Background drop-shadow line for contrast on any floor
            cv2.line(frame, p1, p2, (15, 23, 42), line_thickness + 2)
            cv2.line(frame, p1, p2, line_color, line_thickness)

            # Draw True Perpendicular Direction Arrow & High-Visibility Badge
            ZoneManager._draw_entry_arrow(frame, p1, p2, direction, is_flashing, self.attendance_total)

            dir_display = "BOTH" if direction == "both" else direction.upper()
            line_label = f"*** ENTRY: {self.attendance_total} (+1) ***" if is_flashing else f"ENTRY LINE [{dir_display}] | Entry: {self.attendance_total}  (Press [D] to flip)"
            (lw, lh), _ = cv2.getTextSize(line_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            lx = max(10, min(p1[0], p2[0]) + 10)
            ly = max(75, min(p1[1], p2[1]) - 10)
            cv2.rectangle(frame, (lx - 6, ly - lh - 4), (lx + lw + 6, ly + 6), (15, 23, 42), -1)
            cv2.rectangle(frame, (lx - 6, ly - lh - 4), (lx + lw + 6, ly + 6), line_color, 1)
            cv2.putText(frame, line_label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.48, line_color, 1, cv2.LINE_AA)

        # 3. Live Rubber-Band Feedback While Mouse Dragging
        if self.is_mouse_down:
            sx, sy = self.mouse_start
            cx, cy = self.mouse_current
            if self.active_tool == "entry_line":
                cv2.line(frame, (sx, sy), (cx, cy), (0, 255, 255), 2)
                cv2.circle(frame, (sx, sy), 5, (0, 255, 0), -1)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                dist = int(math.hypot(cx - sx, cy - sy))
                curr_dir = self.entry_line.get("direction", "forward") if self.entry_line else "forward"
                if dist > 20:
                    ZoneManager._draw_entry_arrow(frame, (sx, sy), (cx, cy), curr_dir, False)
                cv2.putText(frame, f"Release to set Entry Line ({dist}px) - Press [D] to flip direction", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                x1, x2 = min(sx, cx), max(sx, cx)
                y1, y2 = min(sy, cy), max(sy, cy)
                w_box = x2 - x1
                h_box = y2 - y1
                found_sec = next((s for s in self.sections if s["id"] == self.active_tool), None)
                sec_name = found_sec["name"] if found_sec else self.active_tool.replace("_", " ").title()
                color = found_sec["color"] if found_sec else (0, 215, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Release to set {sec_name} ({w_box}x{h_box}px)", (x1 + 8, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # 4. Zero-Lag Fast ROI Header Toolbar
        tb_roi = frame[0:52, 0:w]
        tb_bg = np.full_like(tb_roi, (12, 18, 30), dtype=np.uint8)
        cv2.addWeighted(tb_bg, 0.88, tb_roi, 0.12, 0, tb_roi)

        # Button 1: Draw Entry Line
        b1_active = self.active_tool == "entry_line"
        b1_col = (0, 255, 255) if b1_active else (70, 85, 105)
        cv2.rectangle(frame, (self.btn_line[0], self.btn_line[1]), (self.btn_line[2], self.btn_line[3]), (25, 35, 55), -1)
        cv2.rectangle(frame, (self.btn_line[0], self.btn_line[1]), (self.btn_line[2], self.btn_line[3]), b1_col, 2 if b1_active else 1)
        b1_txt = "-> [1] Line" if b1_active else "[1] Line"
        cv2.putText(frame, b1_txt, (self.btn_line[0] + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255) if b1_active else (180, 195, 215), 1, cv2.LINE_AA)

        # Dynamic Department Buttons (from Database & camera_config.json)
        for i, (bx1, by1, bx2, by2, sec) in enumerate(self.btn_sections):
            b_active = self.active_tool == sec["id"]
            sec_col = sec["color"] if b_active else (70, 85, 105)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (25, 35, 55), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), sec_col, 2 if b_active else 1)
            short_name = sec["name"].split()[0][:8]
            b_txt = f"-> [{i+2}] {short_name}" if b_active else f"[{i+2}] {short_name}"
            cv2.putText(frame, b_txt, (bx1 + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sec["color"] if b_active else (180, 195, 215), 1, cv2.LINE_AA)

        # Button 5: Flip Entry Line Direction
        curr_dir = self.entry_line.get("direction", "forward") if self.entry_line else "forward"
        dir_short = "FWD" if curr_dir == "forward" else ("BWD" if curr_dir == "backward" else "BOTH")
        b5_col = (0, 215, 255) if self.entry_line else (70, 85, 105)
        cv2.rectangle(frame, (self.btn_flip[0], self.btn_flip[1]), (self.btn_flip[2], self.btn_flip[3]), (25, 35, 55), -1)
        cv2.rectangle(frame, (self.btn_flip[0], self.btn_flip[1]), (self.btn_flip[2], self.btn_flip[3]), b5_col, 1)
        cv2.putText(frame, f"[D] Dir: {dir_short}", (self.btn_flip[0] + 6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, b5_col, 1, cv2.LINE_AA)

        # Button 6: Reset Entry Count
        cv2.rectangle(frame, (self.btn_reset[0], self.btn_reset[1]), (self.btn_reset[2], self.btn_reset[3]), (25, 35, 55), -1)
        cv2.rectangle(frame, (self.btn_reset[0], self.btn_reset[1]), (self.btn_reset[2], self.btn_reset[3]), (0, 200, 255), 1)
        cv2.putText(frame, "[C] Reset", (self.btn_reset[0] + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

        # Button 7: Clear All
        cv2.rectangle(frame, (self.btn_clear[0], self.btn_clear[1]), (self.btn_clear[2], self.btn_clear[3]), (25, 35, 55), -1)
        cv2.rectangle(frame, (self.btn_clear[0], self.btn_clear[1]), (self.btn_clear[2], self.btn_clear[3]), (70, 85, 105), 1)
        cv2.putText(frame, "[R] Clear", (self.btn_clear[0] + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 160, 185), 1, cv2.LINE_AA)

        # Guidance status message ribbon right below buttons
        if self.status_timer > 0 and self.status_msg:
            (smw, _), _ = cv2.getTextSize(self.status_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (10, 56), (20 + smw, 78), (10, 15, 26), -1)
            cv2.putText(frame, self.status_msg, (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        return frame


# ---------------------------------------------------------------------------
# Asynchronous Threaded RTSP Frame Grabber (Zero-Lag & TCP Transport)
# ---------------------------------------------------------------------------
class ThreadedRTSPReader:
    """
    Background frame grabber with TCP transport.
    Forces TCP transport for RTSP to prevent H264 macroblock decode corruption,
    and continuously drains the buffer to eliminate lag completely.
    """

    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else str(source)
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.width = 1280
        self.height = 720
        self.is_opened = False
        self._connect()

        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        # Force TCP transport for RTSP (eliminates UDP packet loss & H264 decode errors)
        if isinstance(self.source, str) and self.source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened() and isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            self.is_opened = True

    def _grab_loop(self):
        while not self.stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.5)
                continue
            try:
                ret, frame = self.cap.read()
            except Exception:
                break
            if self.stop_event.is_set():
                break
            if not ret or frame is None:
                # Video file loop
                if isinstance(self.source, str) and not self.source.startswith("rtsp://"):
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception:
                        pass
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is not None:
                try:
                    return True, self.latest_frame.copy()
                except Exception:
                    return True, self.latest_frame
            return False, None

    def release(self):
        self.stop_event.set()
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# SCRFD Neural Face Detector (InsightFace - High Accuracy & Small Faces)
# ---------------------------------------------------------------------------
class SCRFDDetector:
    """
    High-accuracy InsightFace SCRFD face detector running natively on NVIDIA GPU.
    Uses PyTorch CUDA on cuda:0 (10ms inference) with ONNX Runtime & OpenCV DNN fallbacks.
    """

    def __init__(self, model_path: str = "models/scrfd_10g_bnkps.onnx", conf_thresh: float = 0.35, nms_thresh: float = 0.40):
        import torch
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_torch = False
        self.use_ort = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Check for Native PyTorch GPU Model (.pt)
        resolved_path = get_resource_path(model_path)
        if resolved_path.endswith(".pt") and os.path.exists(resolved_path) and torch.cuda.is_available():
            try:
                self.torch_model = torch.load(resolved_path, weights_only=False).to(self.device).eval()
                self.use_torch = True
                self.engine_name = f"GPU ({self.device.upper()})"
                print(f"[+] SCRFD Neural Face Detector loaded natively on {self.engine_name}!")
                return
            except Exception as e:
                self.use_torch = False

        # Resolve ONNX file
        resolved_onnx = resolved_path if resolved_path.endswith(".onnx") else resolved_path.replace(".pt", ".onnx")
        if not os.path.exists(resolved_onnx):
            for cand in ["models/scrfd_10g_bnkps.onnx", "models/scrfd_2.5g_bnkps.onnx"]:
                r_cand = get_resource_path(cand)
                if os.path.exists(r_cand):
                    resolved_onnx = r_cand
                    break
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            available = ort.get_available_providers()
            active = [p for p in providers if p in available]
            self.session = ort.InferenceSession(resolved_onnx, providers=active)
            self.input_name = self.session.get_inputs()[0].name
            self.out_names = [o.name for o in self.session.get_outputs()]
            self.use_ort = True
            self.engine_name = "GPU (CUDA)" if "CUDAExecutionProvider" in self.session.get_providers() else "CPU (ORT)"
            print(f"[+] SCRFD 10G loaded via ONNX Runtime on {self.engine_name} ({os.path.basename(resolved_onnx)})!")
            return
        except Exception:
            self.use_ort = False

        # Fallback to OpenCV DNN
        self.net = cv2.dnn.readNet(resolved_onnx)
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.engine_name = "CPU (DNN)"
        print("[+] SCRFD loaded via OpenCV DNN engine.")

    def detect(self, img: np.ndarray, input_size=(640, 640)) -> List[dict]:
        import torch
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        if self.use_torch:
            tensor = torch.from_numpy(det_img).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
            tensor = (tensor - 127.5) / 128.0
            with torch.no_grad():
                net_outs_torch = self.torch_model(tensor)
            net_outs = [o.detach().cpu().numpy() for o in net_outs_torch]
        elif self.use_ort:
            blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)
            net_outs = self.session.run(self.out_names, {self.input_name: blob})
        else:
            blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)
            self.net.setInput(blob)
            net_outs = self.net.forward(self.out_names)

        stride_outs = {}
        for out in net_outs:
            dim = out.shape[1]
            n_rows = out.shape[0]
            stride = None
            for s in self._feat_stride_fpn:
                expected_n = (input_size[1] // s) * (input_size[0] // s) * self._num_anchors
                if n_rows == expected_n:
                    stride = s
                    break
            if stride not in stride_outs:
                stride_outs[stride] = {}
            if dim == 1:
                stride_outs[stride]["score"] = out
            elif dim == 4:
                stride_outs[stride]["bbox"] = out
            elif dim == 10:
                stride_outs[stride]["kps"] = out

        scores_list, bboxes_list, kpss_list = [], [], []
        for stride in self._feat_stride_fpn:
            s_dict = stride_outs.get(stride)
            if not s_dict or "score" not in s_dict or "bbox" not in s_dict:
                continue
            scores = s_dict["score"]
            bboxes = s_dict["bbox"]
            kpss = s_dict.get("kps")

            feat_h = input_size[1] // stride
            feat_w = input_size[0] // stride
            grid_y, grid_x = np.mgrid[:feat_h, :feat_w]
            grid_x = grid_x.reshape(-1) * stride
            grid_y = grid_y.reshape(-1) * stride
            anchor_centers = np.stack([grid_x, grid_y], axis=-1)
            anchor_centers = np.repeat(anchor_centers, self._num_anchors, axis=0)

            pos_inds = np.where(scores >= self.conf_thresh)[0]
            if len(pos_inds) == 0:
                continue

            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            pos_anchors = anchor_centers[pos_inds]

            x1 = (pos_anchors[:, 0] - pos_bboxes[:, 0] * stride) / det_scale
            y1 = (pos_anchors[:, 1] - pos_bboxes[:, 1] * stride) / det_scale
            x2 = (pos_anchors[:, 0] + pos_bboxes[:, 2] * stride) / det_scale
            y2 = (pos_anchors[:, 1] + pos_bboxes[:, 3] * stride) / det_scale
            decoded_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)

            scores_list.append(pos_scores.ravel())
            bboxes_list.append(decoded_boxes)

            if kpss is not None:
                pos_kpss = kpss[pos_inds]
                decoded_kps = np.zeros((len(pos_inds), 5, 2), dtype=np.float32)
                for i in range(5):
                    decoded_kps[:, i, 0] = (pos_anchors[:, 0] + pos_kpss[:, 2 * i] * stride) / det_scale
                    decoded_kps[:, i, 1] = (pos_anchors[:, 1] + pos_kpss[:, 2 * i + 1] * stride) / det_scale
                kpss_list.append(decoded_kps)

        if len(scores_list) == 0:
            return []

        all_scores = np.concatenate(scores_list, axis=0)
        all_bboxes = np.concatenate(bboxes_list, axis=0)
        all_kpss = np.concatenate(kpss_list, axis=0) if kpss_list else None

        indices = cv2.dnn.NMSBoxes(
            bboxes=all_bboxes.tolist(),
            scores=all_scores.tolist(),
            score_threshold=self.conf_thresh,
            nms_threshold=self.nms_thresh,
        )

        results = []
        for idx in indices:
            i = idx if isinstance(idx, (int, np.integer)) else idx[0]
            bx, by, bw, bh = all_bboxes[i]
            conf = float(all_scores[i])
            landmarks = [(int(pt[0]), int(pt[1])) for pt in all_kpss[i]] if all_kpss is not None else []
            results.append({
                "box": (float(bx), float(by), float(bw), float(bh)),
                "conf": conf,
                "landmarks": landmarks,
                "center": (float(bx + bw / 2.0), float(by + bh / 2.0)),
                "matched": False,
            })
        return results


# ---------------------------------------------------------------------------
# YuNet Deep Learning Neural Face Detector (OpenCV Model Zoo)
# ---------------------------------------------------------------------------
def load_yunet_face_detector(model_path: str = "face_detection_yunet_2023mar.onnx", score_threshold: float = 0.35):
    """Loads OpenCV YuNet deep learning face detector (auto-downloads if missing)."""
    resolved = get_resource_path(model_path)
    if not os.path.exists(resolved):
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        print(f"[*] Downloading YuNet neural face model from {url}...")
        try:
            urllib.request.urlretrieve(url, resolved)
            print("[+] YuNet face detector model downloaded successfully!")
        except Exception as e:
            print(f"[!] Warning: Failed to download YuNet face model: {e}")
            return None

    try:
        detector = cv2.FaceDetectorYN_create(
            resolved,
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        return detector
    except Exception as e:
        print(f"[!] Warning: Could not initialize YuNet face detector: {e}")
        return None


def load_face_detector(face_model_path: str, face_confidence: float):
    """Loads SCRFD (InsightFace) or YuNet based on model configuration."""
    model_name = os.path.basename(face_model_path).lower()
    if "scrfd" in model_name:
        try:
            print(f"[*] Loading InsightFace SCRFD Neural Face Detector: {face_model_path}...")
            detector = SCRFDDetector(face_model_path, conf_thresh=face_confidence)
            print("[+] InsightFace SCRFD Face Detector loaded successfully!")
            return ("scrfd", detector)
        except Exception as e:
            print(f"[!] Warning: Could not initialize SCRFD: {e}, falling back to YuNet.")
    print(f"[*] Loading YuNet Neural Face Detector: {face_model_path}...")
    detector = load_yunet_face_detector(face_model_path, score_threshold=face_confidence)
    if detector is not None:
        print("[+] YuNet Neural Face Detector loaded successfully!")
        return ("yunet", detector)
    return (None, None)


def is_valid_facial_geometry(landmarks: List[Tuple[int, int]], min_eye_dist: float = 7.0) -> bool:
    """
    Validates that 5 facial landmarks form a physically valid human face.
    Strictly filters out bags, backpacks, cushions, clothing textures, and reflections.
    Landmarks order: [left_eye, right_eye, nose, left_mouth, right_mouth].
    """
    if not landmarks or len(landmarks) < 5:
        return False
    lx, ly = landmarks[0]
    rx, ry = landmarks[1]
    nx, ny = landmarks[2]
    mx1, my1 = landmarks[3]
    mx2, my2 = landmarks[4]

    # 1. Eye-to-eye distance check
    eye_dist = math.hypot(rx - lx, ry - ly)
    if eye_dist < min_eye_dist:
        return False

    # 2. Eye horizontal alignment (eyes cannot be stacked vertically like bag folds)
    eye_dx = abs(rx - lx)
    eye_dy = abs(ry - ly)
    if eye_dx < eye_dy * 1.15:
        return False

    # 3. Nose position: below eyes, above mouth
    avg_eye_y = (ly + ry) * 0.5
    avg_mouth_y = (my1 + my2) * 0.5
    if ny <= avg_eye_y - 2.0 or ny >= avg_mouth_y + 4.0:
        return False

    # 4. Nose horizontal centering: between left and right eyes (with modest margin)
    min_eye_x = min(lx, rx) - eye_dist * 0.45
    max_eye_x = max(lx, rx) + eye_dist * 0.45
    if not (min_eye_x <= nx <= max_eye_x):
        return False

    # 5. Mouth span below eyes
    if avg_mouth_y <= avg_eye_y + 3.0:
        return False

    return True


# ---------------------------------------------------------------------------
# Main Local Detection Engine
# ---------------------------------------------------------------------------
def run_local_detector(
    source: str,
    camera_id: str,
    api_url: str,
    line_ratio: float,
    direction: str,
    confidence_thresh: float,
    model_path: str,
    headless: bool,
    show_entry_line: bool = True,
    face_scale: float = 1.05,
    face_model_path: str = "models/scrfd_2.5g_bnkps.pt",
    face_confidence: float = 0.30,
    show_head_fallback: bool = False,
    allow_standalone_faces: bool = False,
    imgsz: int = 960,
    box_smoothing: bool = True,
    smoothing_alpha: float = 0.35,
    face_persistence: int = 8,
    body_persistence: int = 26,
    max_visual_coast: int = 20,
    deadband_px: float = 6.0,
    min_feet_y: float = 0.24,
    camera_channel: str = "201",
    similarity_thresh: Optional[float] = None,
    entry_line_cfg: Optional[dict] = None,
    sections_cfg: Optional[list] = None,
    exclusion_zones: Optional[list] = None,
    tracker_name: Optional[str] = None,
    model_type: str = "auto",
    rf_detr_variant: str = "medium",
):
    print("=" * 70)
    print("  CAPERMINT SHOWROOM INTELLIGENCE - LOCAL AI EDGE DETECTOR")
    print("=" * 70)
    print(f"[*] Camera ID      : {camera_id}")
    print(f"[*] Video Source   : {source}")
    print(f"[*] Backend API    : {api_url}")
    print(f"[*] Model Type     : {model_type}")
    model_label = model_path if model_path else f"Pretrained COCO ({rf_detr_variant})"
    print(f"[*] Detection Model: {model_label} (imgsz={imgsz})")
    print(f"[*] Face AI Model  : {face_model_path} (threshold: {face_confidence:.2f})")
    print(f"[*] Confidence     : {confidence_thresh:.2f}")
    from app.core.config import settings
    effective_sim_thresh = similarity_thresh if similarity_thresh is not None else settings.FACE_SIMILARITY_THRESHOLD
    effective_tracker = tracker_name or settings.TRACKER_TYPE
    print(f"[*] Biometric Sim  : {effective_sim_thresh:.2f} (ceiling calibrated)")
    print(f"[*] MOT Tracker    : {effective_tracker} (primary + RobustFallbackTracker)")
    print(f"[*] Face Box Scale : {face_scale:.2f}x (adjust live with +/-)")
    print(f"[*] Anti-Jitter    : {'ON (EMA alpha=' + str(smoothing_alpha) + ', deadband=' + str(deadband_px) + 'px)' if box_smoothing else 'OFF'}")
    print(f"[*] Face Hysteresis: {'ON (' + str(face_persistence) + ' frames buffer)' if box_smoothing else 'OFF'}")
    print(f"[*] Track Persistence: {'ON (buffer=' + str(body_persistence) + ' frames, visual coast=' + str(max_visual_coast) + ' frames / ~0.75s)' if box_smoothing else 'OFF'}")
    print(f"[*] Show Entry Line: {show_entry_line}")
    print(f"[*] Standalone Face: {allow_standalone_faces} (strict screen filtering)")
    print(f"[*] Headless Mode  : {headless}")

    # Detect compute device (NVIDIA CUDA GPU vs CPU)
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"[*] Compute Device : {device.upper()} ({device_name})")

    def is_in_exclusion(bx1: float, by1: float, bx2: float, by2: float) -> bool:
        if not exclusion_zones:
            return False
        bcx = (bx1 + bx2) * 0.5
        bcy = (by1 + by2) * 0.5
        barea = max(1.0, (bx2 - bx1) * (by2 - by1))
        for ez in exclusion_zones:
            ebox = ez if isinstance(ez, (list, tuple)) else ez.get("box", [0, 0, 0, 0])
            ex1, ey1, ex2, ey2 = float(ebox[0]), float(ebox[1]), float(ebox[2]), float(ebox[3])
            if ex1 <= bcx <= ex2 and ey1 <= bcy <= ey2:
                return True
            ix1, iy1 = max(bx1, ex1), max(by1, ey1)
            ix2, iy2 = min(bx2, ex2), min(by2, ey2)
            if ix2 > ix1 and iy2 > iy1:
                iarea = (ix2 - ix1) * (iy2 - iy1)
                if (iarea / barea) > 0.25:
                    return True
        return False

    model_type_str = str(model_type or "rtmdet").lower()
    model_path_str = str(model_path or "").lower()

    if model_type_str in ("rf-detr", "rfdetr") or ("rf-detr" in model_path_str or "rfdetr" in model_path_str):
        resolved_family = "rf-detr"
    elif model_type_str in ("rtmdet", "rtm-det", "rtm_det") or "rtmdet" in model_path_str:
        resolved_family = "rtmdet"
    elif model_type_str in ("rtdetr", "rt-detr", "rt_detr") or "rtdetr" in model_path_str:
        resolved_family = "rtdetr"
    elif model_type_str in ("yolo",) or "yolo" in model_path_str:
        resolved_family = "yolo"
    else:
        resolved_family = "rtmdet"

    custom_detector = None
    model = None

    if resolved_family == "rtmdet":
        resolved_model = get_resource_path(model_path) if model_path else None
        if not resolved_model or not os.path.exists(resolved_model):
            for cand in [
                model_path,
                os.path.join("models", model_path) if model_path else None,
                "models/rtmdet-m-person.onnx",
                "models/rtmdet-s-coco.onnx",
                "rtmdet-m-person.onnx",
            ]:
                if cand and os.path.exists(cand):
                    resolved_model = cand
                    break
            if not resolved_model:
                resolved_model = "models/rtmdet-m-person.onnx"

        print(f"[*] Loading OpenMMLab RTMDet model: {resolved_model} on {device.upper()}...")
        from app.vision.detector import create_detector
        custom_detector = create_detector(
            model_type="rtmdet",
            model_path=resolved_model,
            confidence_threshold=confidence_thresh,
            device=device,
        )
        _ = custom_detector.model
        print(f"[+] RTMDet model loaded successfully on {device_name}!")

    elif resolved_family == "rf-detr":
        custom_weights = get_resource_path(model_path) if (model_path and os.path.exists(model_path) and "yolo" not in os.path.basename(model_path).lower()) else None
        print(f"[*] Loading Roboflow RF-DETR model ({rf_detr_variant}) on {device.upper()}...")
        from app.vision.detector import create_detector
        custom_detector = create_detector(
            model_type="rf-detr",
            model_path=custom_weights,
            rf_detr_variant=rf_detr_variant,
            confidence_threshold=confidence_thresh,
            device=device,
        )
        _ = custom_detector.model
        print(f"[+] RF-DETR model loaded successfully on {device_name}!")

    elif resolved_family == "rtdetr":
        resolved_model = get_resource_path(model_path) if model_path else "models/rtdetr-l.pt"
        if not os.path.exists(resolved_model) and os.path.exists("models/rtdetr-l.pt"):
            resolved_model = "models/rtdetr-l.pt"
        print(f"[*] Loading Ultralytics RT-DETR model: {resolved_model} on {device.upper()}...")
        from app.vision.detector import create_detector
        custom_detector = create_detector(
            model_type="rtdetr",
            model_path=resolved_model,
            confidence_threshold=confidence_thresh,
            device=device,
        )
        _ = custom_detector.model
        print(f"[+] RT-DETR model loaded successfully on {device_name}!")

    else:
        # Resolve YOLO model weights file
        resolved_model = get_resource_path(model_path) if model_path else None
        if not resolved_model or not os.path.exists(resolved_model):
            if model_path and os.path.exists(os.path.join("models", model_path)):
                resolved_model = os.path.join("models", model_path)
            elif os.path.exists("models/yolo26m.pt"):
                resolved_model = "models/yolo26m.pt"
            elif os.path.exists("models/yolo26s.pt"):
                resolved_model = "models/yolo26s.pt"
            elif os.path.exists("models/yolo11s.pt"):
                resolved_model = "models/yolo11s.pt"
            elif os.path.exists("models/yolo11n.pt"):
                resolved_model = "models/yolo11n.pt"
            elif os.path.exists("models/yolo11m.pt"):
                resolved_model = "models/yolo11m.pt"
            else:
                resolved_model = model_path or "models/yolo26m.pt"

        print(f"[*] Loading YOLO model: {resolved_model} on {device.upper()}...")
        from ultralytics import YOLO

        model = YOLO(resolved_model)
        model.to(device)
        print(f"[+] YOLO model loaded successfully on {device_name}!")

    # Load Face Detector (SCRFD or YuNet)
    face_detector_type, face_detector = load_face_detector(face_model_path, face_confidence)
    if face_detector is None:
        print("[!] Running with adaptive anatomical head estimation only.")

    # Open Video Source with Threaded TCP Reader
    print(f"[*] Connecting to video feed with zero-latency TCP stream reader...")
    reader = ThreadedRTSPReader(source)

    # Wait up to 3 seconds for initial frame
    t_wait = time.time()
    while not reader.is_opened and time.time() - t_wait < 3.0:
        time.sleep(0.1)

    if not reader.is_opened:
        print(f"\n[!] Error: Unable to open video source: {source}")
        if str(source).isdigit():
            print("    -> No physical or virtual webcam was detected at index " + str(source) + ".")
            print("    -> To use a CCTV stream, pass your RTSP URL with --source \"rtsp://...\"")
            print("    -> Or pass a video file path with --source \"path/to/video.mp4\"")
        sys.exit(1)

    width = reader.width
    height = reader.height
    line_y = int(height * line_ratio)

    tracker = LocalTracker(max_disappeared=body_persistence, max_distance=120.0)
    smoother = TrackSmoother(
        alpha=smoothing_alpha,
        deadband_px=deadband_px,
        face_persistence=face_persistence,
        body_persistence=body_persistence,
        max_visual_coast=max_visual_coast,
        enabled=box_smoothing,
    )
    zone_manager = ZoneManager(
        config_path="camera_config.json",
        camera_channel=camera_channel,
        entry_line_cfg=entry_line_cfg,
        sections_cfg=sections_cfg,
        frame_width=width,
        frame_height=height,
        line_y_ratio=line_ratio,
        direction=direction,
        api_url=api_url,
    )
    employee_dept_tracker = EmployeeDepartmentTracker(log_interval=10.0, exit_debounce=2.0)
    zone_manager.dept_tracker = employee_dept_tracker
    sync_client = DashboardSyncClient(base_url=api_url, camera_id=camera_id)

    # Face Embedding & Identity Recovery System (ArcFace + pgvector + 14-Day Retention)
    from app.vision.face_embedder import FaceEmbedder
    from app.vision.vector_store import PgVectorStoreManager
    from app.vision.identity_tracker import IdentityRecoveryManager
    from app.vision.employee_recognizer import EmployeeRecognitionManager

    # Open database session if PostgreSQL is running locally
    db_session = None
    try:
        from app.db.session import SessionLocal
        db_session = SessionLocal()
    except Exception as e:
        print(f"[!] Warning: Could not create direct DB session in detector: {e}")

    from app.core.config import settings
    face_embedder = FaceEmbedder(
        min_size=settings.FACE_MIN_SIZE,
        min_confidence=min(face_confidence, settings.FACE_MIN_CONFIDENCE),
        blur_threshold=settings.FACE_BLUR_THRESHOLD,
    )
    sim_thresh = similarity_thresh if similarity_thresh is not None else settings.FACE_SIMILARITY_THRESHOLD
    visitor_vector_store = PgVectorStoreManager(db=db_session, similarity_threshold=sim_thresh)
    employee_vector_store = PgVectorStoreManager(db=db_session, similarity_threshold=sim_thresh)
    identity_manager = IdentityRecoveryManager(
        embedder=face_embedder,
        vector_store=visitor_vector_store,
        similarity_threshold=sim_thresh,
        max_attempts=5,
        cooldown_frames=6,
    )
    employee_recognizer = EmployeeRecognitionManager(
        embedder=face_embedder,
        vector_store=employee_vector_store,
        similarity_threshold=sim_thresh,
        min_confirm_frames=settings.EMPLOYEE_MIN_CONFIRM_FRAMES,
        vote_window=settings.EMPLOYEE_VOTE_WINDOW,
    )

    # Cooldown buffer for employee check-in punches (punch at most once per 5 mins per staff)
    punched_employee_ids: Dict[str, float] = {}
    track_last_recognition_attempt: Dict[str, float] = {}
    announced_staff_tracks: Set[int] = set()

    # Preload registered employees and names into recognizer
    employee_names: Dict[str, str] = {}
    if db_session:
        try:
            from app.repositories.employee_repo import EmployeeRepository
            emp_repo = EmployeeRepository(db_session)
            active_embs = emp_repo.get_all_active_embeddings()
            loaded = employee_recognizer.load_active_employees(active_embs)
            active_emps = emp_repo.list_employees(is_active=True)
            for ae in active_emps:
                employee_names[ae.employee_id] = ae.name
            print(f"[+] Biometric Pipeline: Loaded {loaded} registered employee face embeddings into index.")
        except Exception as e:
            print(f"[!] Warning: Could not load employee embeddings from database: {e}")

    window_name = f"Capermint Edge AI Detector - Cam {camera_channel}"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 576)
        cv2.setMouseCallback(window_name, zone_manager.handle_mouse)

    frame_counter = 0
    fps = 0.0
    fps_timer = time.time()
    last_heartbeat_time = time.time()
    last_employee_reload_time = time.time()
    show_hud = True
    track_start_times: Dict[int, float] = {}
    track_last_seen: Dict[int, float] = {}
    seated_staff_tracks: Dict[str, dict] = {}
    staff_spatial_anchors: Dict[str, dict] = {}
    bytetrack_empty_streak = 0

    # Resolve primary tracker configuration (ByteTrack or BoT-SORT)
    chosen_tracker = tracker_name or settings.TRACKER_TYPE
    if not chosen_tracker.endswith(".yaml"):
        chosen_tracker = f"{chosen_tracker}.yaml"

    tracker_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), chosen_tracker)
    if not os.path.exists(tracker_config):
        tracker_config = os.path.join(os.getcwd(), chosen_tracker)
    if not os.path.exists(tracker_config):
        tracker_config = chosen_tracker

    print("\n[+] Edge Video Processor running. Press 'q' or 'ESC' in the window to quit.\n")

    try:
        while True:
            ret, frame = reader.read_latest()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_counter += 1

            # Check for dynamic dashboard updates to camera_config.json every 90 frames (~3s)
            if frame_counter % 90 == 0:
                zone_manager.check_and_reload_config()

            # Calculate FPS every 15 frames
            if frame_counter % 15 == 0:
                elapsed = time.time() - fps_timer
                if elapsed > 0:
                    fps = 15.0 / elapsed
                fps_timer = time.time()

            h, w = frame.shape[:2]

            # YOLO tracking (class 0 = person) with ByteTrack primary & LocalTracker controlled fallback
            tracked_objects: Dict[int, Tuple[float, float, float, float]] = {}
            detections: List[Tuple[float, float, float, float]] = []
            unmatched_boxes: List[Tuple[float, float, float, float]] = []
            bytetrack_failed = False

            is_cadence_skip = False
            if custom_detector is not None:
                bytetrack_failed = True
                # 2-Frame Detection Cadence: Run RTMDet neural inference on alternate frames
                # or immediately if there are currently no tracked persons.
                run_detector_now = (frame_counter % 2 == 0) or (len(tracker.objects) == 0)
                if run_detector_now:
                    try:
                        res_dets = custom_detector.detect(frame)
                        for d in res_dets:
                            x1, y1, x2, y2 = d.box
                            bw = x2 - x1
                            bh = y2 - y1
                            if bw < 14 or bh < 18 or (bh / max(1.0, bw)) < 0.60 or (y2 / max(1.0, h)) < min_feet_y or is_in_exclusion(x1, y1, x2, y2):
                                continue
                            detections.append((x1, y1, x2, y2))
                            unmatched_boxes.append((x1, y1, x2, y2))
                    except Exception:
                        pass
                else:
                    is_cadence_skip = True
            else:
                try:
                    with torch.inference_mode():
                        results = model.track(
                            source=frame,
                            classes=[0],
                            conf=confidence_thresh,
                            imgsz=imgsz,
                            persist=True,
                            tracker=tracker_config,
                            verbose=False,
                        )
                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            bw = x2 - x1
                            bh = y2 - y1
                            if bw < 14 or bh < 18 or (bh / max(1.0, bw)) < 0.60 or (y2 / max(1.0, h)) < min_feet_y or is_in_exclusion(x1, y1, x2, y2):
                                continue
                            detections.append((x1, y1, x2, y2))
                            if box.id is not None:
                                try:
                                    track_id = int(box.id[0].item())
                                    tracked_objects[track_id] = (x1, y1, x2, y2)
                                except Exception:
                                    unmatched_boxes.append((x1, y1, x2, y2))
                            else:
                                unmatched_boxes.append((x1, y1, x2, y2))
                except Exception:
                    bytetrack_failed = True
                    try:
                        with torch.inference_mode():
                            results = model.predict(
                                source=frame,
                                classes=[0],
                                conf=confidence_thresh,
                                imgsz=imgsz,
                                verbose=False,
                            )
                        if results and len(results) > 0 and results[0].boxes is not None:
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                bw = x2 - x1
                                bh = y2 - y1
                                if bw < 14 or bh < 18 or (bh / max(1.0, bw)) < 0.60 or (y2 / max(1.0, h)) < min_feet_y or is_in_exclusion(x1, y1, x2, y2):
                                    continue
                                detections.append((x1, y1, x2, y2))
                                unmatched_boxes.append((x1, y1, x2, y2))
                    except Exception:
                        pass

            # Controlled fallback & unmatched detection handling
            if bytetrack_failed:
                if is_cadence_skip:
                    # On cadence-skipped frames: project active tracks forward along velocity in 0.02ms!
                    tracked_objects = tracker.coast_tracks()
                elif detections:
                    # Active detection frame: update tracker positions and re-estimate velocity
                    tracked_objects = tracker.update(detections, frame=frame)
                else:
                    tracked_objects = tracker.update([], frame=frame)
            else:
                # Primary tracker (ByteTrack or BoT-SORT) is active
                if tracked_objects:
                    bytetrack_empty_streak = 0
                    # Handle unmatched detections without mixing IDs or creating duplicate identities
                    if unmatched_boxes:
                        # 1. Filter out duplicate / overlapping detections of already tracked persons
                        novel_unmatched = []
                        for ubox in unmatched_boxes:
                            is_dup = any(
                                smoother._compute_iou(ubox, tbox) > 0.45 for tbox in tracked_objects.values()
                            )
                            if not is_dup:
                                novel_unmatched.append(ubox)

                        # 2. Re-associate valid unmatched detections to recently lost tracks in persistence buffer
                        if novel_unmatched:
                            recovered = smoother.recover_missing_tracks(novel_unmatched)
                            tracked_objects.update(recovered)

                            # 3. Associate any remaining novel detections via RobustFallbackTracker
                            still_unmatched = [
                                ubox for ubox in novel_unmatched
                                if not any(smoother._compute_iou(ubox, tb) > 0.40 for tb in tracked_objects.values())
                            ]
                            if still_unmatched:
                                fb = tracker.update(still_unmatched, frame=frame)
                                for fid, fbox in fb.items():
                                    if not any(smoother._compute_iou(fbox, tb) > 0.40 for tb in tracked_objects.values()):
                                        tracked_objects[fid] = fbox

                    # Keep RobustFallbackTracker state synchronized with primary tracks (learning velocities and appearances)
                    tracker.sync_active_tracks(tracked_objects, frame=frame)

                elif detections:
                    # Primary tracker returned no tracks despite detections present in the frame
                    # 1. Recover recently lost tracks from smoother persistence buffer
                    recovered = smoother.recover_missing_tracks(unmatched_boxes)
                    if recovered:
                        tracked_objects.update(recovered)

                    # 2. Immediately track remaining detections via RobustFallbackTracker (no 5-frame silence gap!)
                    remaining_dets = [
                        b for b in detections
                        if not any(smoother._compute_iou(b, tb) > 0.40 for tb in tracked_objects.values())
                    ]
                    if remaining_dets:
                        fb = tracker.update(remaining_dets, frame=frame)
                        for fid, fbox in fb.items():
                            if not any(smoother._compute_iou(fbox, tb) > 0.40 for tb in tracked_objects.values()):
                                tracked_objects[fid] = fbox

                    bytetrack_empty_streak = 0
                    tracker.sync_active_tracks(tracked_objects, frame=frame)
                else:
                    # No detections in this frame
                    bytetrack_empty_streak = 0
                    tracker.update([], frame=frame)

            # Suppress duplicate overlapping person tracks (filters out duplicate boxes on the same seated person)
            if len(tracked_objects) > 1:
                deduped_tracked = {}
                sorted_tracks = sorted(tracked_objects.items(), key=lambda it: (it[1][2]-it[1][0])*(it[1][3]-it[1][1]), reverse=True)
                suppressed_tids = set()
                for i, (tid1, box1) in enumerate(sorted_tracks):
                    if tid1 in suppressed_tids:
                        continue
                    deduped_tracked[tid1] = box1
                    for j in range(i + 1, len(sorted_tracks)):
                        tid2, box2 = sorted_tracks[j]
                        if tid2 in suppressed_tids:
                            continue
                        if smoother._compute_iou(box1, box2) > 0.40:
                            suppressed_tids.add(tid2)
                tracked_objects = deduped_tracked

            # Filter out any tracks violating min_feet_y, abnormal aspect ratio, or located inside exclusion zones (e.g. ceiling lamp, wall decor, edge slivers)
            if tracked_objects:
                tracked_objects = {
                    tid: box for tid, box in tracked_objects.items()
                    if (box[3] / max(1.0, h)) >= min_feet_y
                    and ((box[3] - box[1]) / max(1.0, box[2] - box[0])) >= 0.60
                    and not is_in_exclusion(box[0], box[1], box[2], box[3])
                }

            # Apply Anti-Jitter Smoothing & Deadband to Person Bounding Boxes
            tracked_objects = smoother.smooth_body_boxes(tracked_objects, frame_shape=(w, h))

            # Record track timestamps for on-screen live dwell timer
            now_ts = time.time()
            for obj_id in tracked_objects.keys():
                if obj_id not in track_start_times:
                    track_start_times[obj_id] = now_ts
                track_last_seen[obj_id] = now_ts

            # Expire track timers and release employee locks when tracks disappear (> 25s for biometrics across pauses, > 30s for full cleanup)
            for old_id in list(track_start_times.keys()):
                inactive_sec = now_ts - track_last_seen.get(old_id, 0.0)
                if inactive_sec > 25.0:
                    employee_recognizer.on_track_lost(str(old_id))
                if inactive_sec > 30.0:
                    track_start_times.pop(old_id, None)
                    track_last_seen.pop(old_id, None)
                    identity_manager.on_track_lost(str(old_id))
                    track_last_recognition_attempt.pop(str(old_id), None)
                    employee_dept_tracker.unconfirmed_tracks.pop(str(old_id), None)

            # Evaluate Dynamic Entry Line Crossing & Multi-Zone Section Visits
            new_attendance, new_sections = zone_manager.process(tracked_objects)
            for crossed_id in new_attendance:
                # Strictly suppress visitor entry count if the person is a recognized showroom employee
                if employee_recognizer.is_track_employee(str(crossed_id)):
                    emp_id = employee_recognizer.get_employee_id(str(crossed_id))
                    print(f"[*] Staff member #{crossed_id} (EMP: {emp_id}) crossed entry line - visitor footfall count suppressed.")
                    continue

                pid = identity_manager.get_person_id(crossed_id)
                pid_desc = f" (PID: {pid})" if pid else " (PID: Unresolved)"
                print(f"[--> ENTRY DETECTED] Person #{crossed_id}{pid_desc} crossed Entry Line! Total Entry: {zone_manager.attendance_total}")
                sync_client.send_entry_event(tracker_id=str(crossed_id), confidence=0.94, person_id=pid)

            for (p_id, s_id, s_name) in new_sections:
                sec_tot = zone_manager.section_totals.get(s_id, 0)
                if employee_recognizer.is_track_employee(str(p_id)):
                    emp_id = employee_recognizer.get_employee_id(str(p_id))
                    emp_label = employee_names.get(emp_id, emp_id)
                    print(f"[--> STAFF VISIT] Staff member {emp_label} (#{p_id}) entered {s_name}! Total {s_name} Visits: {sec_tot}")
                else:
                    print(f"[--> SECTION VISIT] Person #{p_id} entered {s_name}! Total {s_name} Visits: {sec_tot}")
                sync_client.send_section_entry_event(tracker_id=str(p_id), section_id=s_id, section_name=s_name)

            # Send Telemetry Heartbeat every 2.5 seconds
            now = time.time()
            if now - last_heartbeat_time > 2.5:
                sync_client.send_heartbeat(
                    fps=fps,
                    frames_processed=frame_counter,
                    current_persons=len(tracked_objects),
                    total_entries=zone_manager.attendance_total,
                    section_counts=zone_manager.section_totals,
                    section_occupancy=zone_manager.live_occupancy,
                    staff_dwell=employee_dept_tracker.get_all_active_sessions_payload(now),
                )
                last_heartbeat_time = now

            # Periodically synchronize employee face embeddings from database (every 15s)
            if db_session and (now - last_employee_reload_time > 15.0):
                last_employee_reload_time = now
                try:
                    from app.repositories.employee_repo import EmployeeRepository
                    emp_repo = EmployeeRepository(db_session)
                    active_embs = emp_repo.get_all_active_embeddings()
                    loaded = employee_recognizer.load_active_employees(active_embs)
                    active_emps = emp_repo.list_employees(is_active=True)
                    employee_names.clear()
                    for ae in active_emps:
                        employee_names[ae.employee_id] = ae.name
                except Exception:
                    pass

            # Run Deep Learning Neural Face Detection (InsightFace SCRFD or OpenCV YuNet)
            # High-Performance Optimization: Execute face inference only when persons are actively tracked
            # AND cadence on every 2nd frame. TrackSmoother seamlessly coasts and translates faces on skipped frames.
            all_detected_faces = []
            should_detect_faces = (
                face_detector is not None
                and (len(tracked_objects) > 0 or allow_standalone_faces)
                and (frame_counter % 3 == 0)
            )
            if should_detect_faces:
                if face_detector_type == "scrfd":
                    det_w = min(w, 640)
                    det_h = min(h, 640)
                    all_detected_faces = face_detector.detect(frame, input_size=(det_w, det_h))
                elif face_detector_type == "yunet":
                    face_detector.setInputSize((w, h))
                    _, raw_faces = face_detector.detect(frame)
                    if raw_faces is not None and len(raw_faces) > 0:
                        for f in raw_faces:
                            fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                            conf = float(f[14])
                            landmarks = [(int(f[i]), int(f[i+1])) for i in range(4, 14, 2)]
                            all_detected_faces.append({
                                "box": (fx, fy, fw, fh),
                                "conf": conf,
                                "landmarks": landmarks,
                                "center": (fx + fw / 2.0, fy + fh / 2.0),
                                "matched": False,
                            })

            # If Headless, skip GUI rendering
            if headless:
                time.sleep(0.001)
                continue

            # ---------------------------------------------------------------
            # Visual GUI Overlay Rendering (OpenCV)
            # ---------------------------------------------------------------
            # 1. Draw Bounding Boxes for each Tracked Person (Body: Emerald Green, Face: Vivid Magenta)
            matched_faces_count = 0
            embeddings_extracted_this_frame = 0 
            visible_staff_this_frame: Set[str] = set()
            for obj_id, (x1, y1, x2, y2) in tracked_objects.items():
                ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                box_w = max(1, ix2 - ix1)
                box_h = max(1, iy2 - iy1)

                # Body box: High-visibility Neon Emerald Green
                body_color = (0, 240, 110)  # BGR Neon Green
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), body_color, 2)

                # Track duration / dwell timer calculation
                dwell_sec = max(0.0, now_ts - track_start_times.get(obj_id, now_ts))
                dwell_tot = int(dwell_sec)
                dwell_h = dwell_tot // 3600
                dwell_m = (dwell_tot % 3600) // 60
                dwell_s = dwell_tot % 60
                timer_str = f"{dwell_h}:{dwell_m:02d}:{dwell_s:02d}" if dwell_h > 0 else f"{dwell_m:02d}:{dwell_s:02d}"

                # 1. Match and process neural face or head fallback FIRST
                matched_face = None
                best_face_dist = 999999.0
                ar = box_h / max(1.0, box_w)
                # Adaptive vertical search window: standing persons (ar >= 2.0) have heads in top 45%;
                # seated/desk-occluded persons (ar < 2.0) have faces spanning up to 75% of the visible torso box.
                max_face_y_ratio = 0.45 if ar >= 2.0 else (0.62 if ar >= 1.4 else 0.75)
                expected_fcy = iy1 + box_h * (0.18 if ar >= 2.0 else 0.35)

                for face in all_detected_faces:
                    if face["matched"]:
                        continue
                    fcx, fcy = face["center"]
                    if (ix1 - 25 <= fcx <= ix2 + 25) and (iy1 - 25 <= fcy <= iy1 + box_h * max_face_y_ratio):
                        dist = math.hypot(fcx - (ix1 + ix2) / 2.0, fcy - expected_fcy)
                        if dist < best_face_dist:
                            best_face_dist = dist
                            matched_face = face

                # Fallback: Targeted upper-body face recovery (disabled by default to prevent hallucinating on back of heads & save 60% GPU latency)
                if False and matched_face is None and face_detector is not None:
                    ub_h = max(24, int(box_h * max_face_y_ratio))
                    uy1 = max(0, iy1 - 8)
                    uy2 = min(h, iy1 + ub_h)
                    ux1 = max(0, ix1 - 12)
                    ux2 = min(w, ix2 + 12)
                    crop_w, crop_h = ux2 - ux1, uy2 - uy1
                    if crop_w > 20 and crop_h > 20:
                        ub_crop = frame[uy1:uy2, ux1:ux2]
                        crop_faces = []
                        if face_detector_type == "scrfd":
                            crop_faces = face_detector.detect(ub_crop, input_size=(320, 320))
                        elif face_detector_type == "yunet":
                            face_detector.setInputSize((crop_w, crop_h))
                            _, cr_raw = face_detector.detect(ub_crop)
                            if cr_raw is not None and len(cr_raw) > 0:
                                for f in cr_raw:
                                    crop_faces.append({
                                        "box": (float(f[0]), float(f[1]), float(f[2]), float(f[3])),
                                        "conf": float(f[14]),
                                        "landmarks": [(int(f[i]), int(f[i+1])) for i in range(4, 14, 2)],
                                    })
                        crop_faces = [cf for cf in crop_faces if cf.get("conf", 0.0) >= 0.55]
                        if crop_faces:
                            best_crop_face = max(crop_faces, key=lambda f: f["conf"])
                            cf_box = best_crop_face["box"]
                            gx = float(cf_box[0] + ux1)
                            gy = float(cf_box[1] + uy1)
                            gw = float(cf_box[2])
                            gh = float(cf_box[3])
                            g_lms = [(int(lx + ux1), int(ly + uy1)) for (lx, ly) in best_crop_face.get("landmarks", [])]
                            matched_face = {
                                "box": (gx, gy, gw, gh),
                                "conf": best_crop_face["conf"],
                                "landmarks": g_lms,
                                "center": (gx + gw / 2.0, gy + gh / 2.0),
                                "matched": True,
                            }

                if matched_face is not None:
                    matched_face["matched"] = True
                    # Attempt Face-Based Recognition (Employees first, then Visitor Recovery)
                    try:
                        m_box = matched_face["box"]
                        cx1 = max(0, int(m_box[0]))
                        cy1 = max(0, int(m_box[1]))
                        cx2 = min(w, int(m_box[0] + m_box[2]))
                        cy2 = min(h, int(m_box[1] + m_box[3]))
                        if cx2 > cx1 + 10 and cy2 > cy1 + 10:
                            face_crop = frame[cy1:cy2, cx1:cx2]
                            raw_lms = matched_face.get("landmarks")
                            crop_lms = None
                            if raw_lms:
                                crop_lms = [(int(lx - cx1), int(ly - cy1)) for (lx, ly) in raw_lms]

                            # 1. Check if recognized as Showroom Employee
                            tid = str(obj_id)
                            is_confirmed = employee_recognizer.is_track_employee(tid)
                            last_attempt = track_last_recognition_attempt.get(tid, 0.0)
                            emp_match = None

                            # Spatial Continuity Check: If track ID changed (e.g. #34 -> #73), check if this person is sitting at a known staff location
                            fcx, fcy = matched_face["center"]
                            if not is_confirmed:
                                active_track_ids_str = {str(k) for k in tracked_objects.keys()}
                                currently_active_staff = {
                                    employee_recognizer.get_employee_id(str(atid))
                                    for atid in tracked_objects.keys()
                                    if str(atid) != tid
                                }
                                for s_eid, anchor in list(staff_spatial_anchors.items()):
                                    if s_eid in currently_active_staff:
                                        continue
                                    last_tid = anchor.get("last_track_id")
                                    if last_tid and last_tid != tid and last_tid in active_track_ids_str:
                                        continue
                                    if (now_ts - anchor["last_seen"]) < 45.0:
                                        if math.hypot(fcx - anchor["center"][0], fcy - anchor["center"][1]) < 60.0:
                                            # Enforce single physical body: clear previous track mappings for s_eid
                                            for otid, oeid in list(employee_recognizer.track_to_employee.items()):
                                                if oeid == s_eid and otid != tid:
                                                    employee_recognizer.track_to_employee.pop(otid, None)
                                            employee_recognizer.track_to_employee[tid] = s_eid
                                            is_confirmed = True
                                            emp_match = (s_eid, 1.0)
                                            anchor["last_seen"] = now_ts
                                            anchor["center"] = (fcx, fcy)
                                            anchor["body_center"] = ((ix1 + ix2) * 0.5, (iy1 + iy2) * 0.5)
                                            anchor["last_track_id"] = tid
                                            break

                            if not is_confirmed and (now_ts - last_attempt >= 0.45) and (embeddings_extracted_this_frame < 1):
                                embeddings_extracted_this_frame += 1
                                track_last_recognition_attempt[tid] = now_ts
                                emp_match = employee_recognizer.evaluate_track_face(
                                    tracker_id=tid,
                                    face_crop=face_crop,
                                    face_conf=float(matched_face["conf"]),
                                    landmarks=crop_lms,
                                    full_image=frame,
                                    raw_landmarks=raw_lms,
                                )

                            if emp_match:
                                emp_id, emp_conf = emp_match
                                # Anchor physical location for spatial continuity across ByteTrack ID flips
                                staff_spatial_anchors[emp_id] = {
                                    "center": (fcx, fcy),
                                    "body_center": ((ix1 + ix2) * 0.5, (iy1 + iy2) * 0.5),
                                    "last_seen": now_ts,
                                    "last_track_id": tid,
                                }
                                # Enforce single physical body: clear previous track mappings for emp_id
                                for otid, oeid in list(employee_recognizer.track_to_employee.items()):
                                    if oeid == emp_id and otid != tid:
                                        employee_recognizer.track_to_employee.pop(otid, None)
                                employee_recognizer.track_to_employee[tid] = emp_id
                                is_confirmed = True

                                if obj_id not in announced_staff_tracks:
                                    announced_staff_tracks.add(obj_id)
                                    emp_label = employee_names.get(emp_id, emp_id)
                                    print(f"[+] Biometrics: Verified Staff {emp_label} ({emp_id}) on Track #{obj_id} (conf: {emp_conf:.2f})")
                                    employee_dept_tracker.promote_track_to_employee(tid, emp_id, emp_label, now_ts)
                                last_punch = punched_employee_ids.get(emp_id, 0.0)
                                if (now_ts - last_punch) > 300.0:
                                    punched_employee_ids[emp_id] = now_ts
                                    sync_client.send_employee_punch_event(emp_id, event_type="CHECK_IN", confidence=emp_conf)
                                    if db_session:
                                        try:
                                            from app.repositories.employee_attendance_repo import EmployeeAttendanceRepository
                                            emp_att_repo = EmployeeAttendanceRepository(db_session)
                                            cam_uuid = uuid.UUID(camera_id) if camera_id and len(camera_id) == 36 else None
                                            emp_att_repo.record_attendance_event(
                                                employee_id=emp_id,
                                                camera_id=cam_uuid,
                                                event_type="CHECK_IN",
                                                confidence=emp_conf,
                                            )
                                        except Exception as db_ex:
                                            pass
                            elif not is_confirmed and not employee_recognizer.is_track_employee(tid):
                                # 2. Process as customer visitor ONLY if definitively NOT a staff member
                                match_res = identity_manager.process_track_face(
                                    tracker_id=str(obj_id),
                                    face_crop=face_crop,
                                    face_conf=float(matched_face["conf"]),
                                    landmarks=crop_lms,
                                    timestamp=now_ts,
                                )
                                if match_res.person_id:
                                    zone_manager.set_identity(obj_id, match_res.person_id)
                    except Exception as e:
                        pass

                # Process face state through Smoother & Hysteresis buffer
                face_info = smoother.update_face(
                    obj_id=obj_id,
                    matched_face=matched_face,
                    person_box=(x1, y1, x2, y2),
                    face_scale=face_scale,
                    frame_w=w,
                    frame_h=h,
                )

                face_tag = ""
                if face_info is not None:
                    matched_faces_count += 1
                    bx1, by1, bx2, by2 = face_info["box"]
                    mconf = face_info["conf"]
                    face_tag = f"Face {int(mconf * 100)}%"

                    # Draw Solid Neural Face Bounding Box & 5 Cyan Landmarks
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 30, 220), 2)
                    for (lx, ly) in face_info.get("landmarks", []):
                        if 0 <= lx < w and 0 <= ly < h:
                            cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)

                elif show_head_fallback:
                    # Adaptive Anatomical Head Estimation when person faces away / occluded
                    ar = box_h / max(1.0, box_w)
                    if ar >= 2.0:
                        face_w_ratio = 0.42 * face_scale
                        face_h_ratio = 0.16 * face_scale
                        top_offset_ratio = 0.015
                    elif ar >= 1.4:
                        face_w_ratio = 0.46 * face_scale
                        face_h_ratio = 0.22 * face_scale
                        top_offset_ratio = 0.02
                    else:
                        face_w_ratio = 0.50 * face_scale
                        face_h_ratio = 0.28 * face_scale
                        top_offset_ratio = 0.02

                    face_w = min(box_w, max(24, int(box_w * face_w_ratio)))
                    face_h = min(box_h, max(26, int(box_h * face_h_ratio)))
                    fx1 = max(0, int(ix1 + (box_w - face_w) / 2))
                    fx2 = min(w, fx1 + face_w)
                    fy1 = max(0, iy1 + max(2, int(box_h * top_offset_ratio)))
                    fy2 = min(iy2, fy1 + face_h)

                    # Smooth anatomical head fallback box (subtle lavender, NO overlapping text badge)
                    fx1, fy1, fx2, fy2 = smoother.smooth_head_box(obj_id, (fx1, fy1, fx2, fy2))
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (190, 100, 190), 1)

                # 2. Render Single Unified Header Badge (Person ID + Persistent ID + Timer + Face Status)
                is_staff = employee_recognizer.is_track_employee(str(obj_id))
                emp_id = employee_recognizer.get_employee_id(str(obj_id)) if is_staff else None
                # Enforce single physical body: ensure this emp_id isn't duplicated on another active track in this frame
                if is_staff and emp_id:
                    for atid in tracked_objects.keys():
                        if atid != obj_id and employee_recognizer.get_employee_id(str(atid)) == emp_id:
                            # If another active track has this employee, demote the non-face track
                            if not face_tag:
                                is_staff = False
                                emp_id = None
                                employee_recognizer.track_to_employee.pop(str(obj_id), None)
                                break

                if not is_staff:
                    # Spatial Continuity Fallback: Check if this body box is seated at an active staff anchor
                    bcx, bcy = (ix1 + ix2) * 0.5, (iy1 + iy2) * 0.5
                    active_track_ids_str = {str(k) for k in tracked_objects.keys()}
                    currently_active_staff = {
                        employee_recognizer.get_employee_id(str(atid))
                        for atid in tracked_objects.keys()
                        if atid != obj_id
                    }
                    for s_eid, anchor in list(staff_spatial_anchors.items()):
                        # 1. Never assign if employee is already actively present on another visible track
                        if s_eid in currently_active_staff:
                            continue
                        # 2. Never steal if the anchor's original track is still present in the camera frame
                        last_tid = anchor.get("last_track_id")
                        if last_tid and last_tid != str(obj_id) and last_tid in active_track_ids_str:
                            continue
                        if (now_ts - anchor["last_seen"]) < 45.0:
                            anchor_pos = anchor.get("body_center", anchor["center"])
                            if math.hypot(bcx - anchor_pos[0], bcy - anchor_pos[1]) < 50.0:
                                # Enforce 1 physical body: clear any other tracks mapped to this employee
                                for otid, oeid in list(employee_recognizer.track_to_employee.items()):
                                    if oeid == s_eid and otid != str(obj_id):
                                        employee_recognizer.track_to_employee.pop(otid, None)
                                is_staff = True
                                emp_id = s_eid
                                employee_recognizer.track_to_employee[str(obj_id)] = s_eid
                                anchor["last_seen"] = now_ts
                                anchor["last_track_id"] = str(obj_id)
                                emp_label = employee_names.get(s_eid, s_eid)
                                employee_dept_tracker.promote_track_to_employee(str(obj_id), s_eid, emp_label, now_ts)
                                break

                if is_staff and emp_id:
                    if emp_id not in employee_names and db_session:
                        try:
                            from app.repositories.employee_repo import EmployeeRepository
                            e_rec = EmployeeRepository(db_session).get_by_employee_id(emp_id)
                            if e_rec and e_rec.name:
                                employee_names[emp_id] = e_rec.name
                        except Exception:
                            pass
                    emp_display = employee_names.get(emp_id, emp_id)
                    visible_staff_this_frame.add(emp_id)

                    # Update employee department dwell and live timer
                    staff_dept = zone_manager.find_section_for_box((x1, y1, x2, y2))
                    employee_dept_tracker.update_employee_presence(emp_id, emp_display, staff_dept, now_ts, track_id=str(obj_id))

                    dwell_info = employee_dept_tracker.get_active_dwell_info(emp_id, now_ts)
                    if dwell_info:
                        d_name, d_dig, _ = dwell_info
                        main_label = f"STAFF [{emp_display}] #{obj_id} | In {d_name}: {d_dig} | {timer_str}"
                    else:
                        main_label = f"STAFF [{emp_display}] #{obj_id} | {timer_str}"
                    body_color = (0, 215, 255)  # Vivid Golden Yellow for verified showroom staff
                    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), body_color, 2)
                else:
                    cand_dept = zone_manager.find_section_for_box((x1, y1, x2, y2))
                    employee_dept_tracker.update_unconfirmed_track(str(obj_id), cand_dept, now_ts)
                    pid_val = identity_manager.get_person_id(obj_id)
                    pid_desc = f" | {pid_val}" if pid_val else " | PID: Unresolved"
                    main_label = f"Person #{obj_id}{pid_desc} | {timer_str}"
                (w_main, _), _ = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)

                w_face = 0
                face_label = ""
                if face_tag:
                    face_label = f"  [{face_tag}]"
                    (w_face, _), _ = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)

                badge_w = w_main + w_face + 10
                badge_h = 20

                # Prevent badge clipping if box is at the very top edge of the frame
                if iy1 >= 22:
                    by_top = iy1 - badge_h
                    by_bot = iy1
                    text_y = iy1 - 5
                else:
                    by_top = iy1
                    by_bot = iy1 + badge_h
                    text_y = iy1 + 15

                # Draw high-contrast sleek dark badge
                badge_border = (255, 30, 220) if face_tag else body_color
                cv2.rectangle(frame, (ix1, by_top), (ix1 + badge_w, by_bot), (15, 23, 42), -1)
                cv2.rectangle(frame, (ix1, by_top), (ix1 + badge_w, by_bot), badge_border, 1)

                # Draw ID and Timer (white)
                cv2.putText(
                    frame,
                    main_label,
                    (ix1 + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Draw Face confidence (electric magenta) if face is detected
                if face_tag:
                    cv2.putText(
                        frame,
                        face_label,
                        (ix1 + 5 + w_main, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (255, 60, 220),
                        1,
                        cv2.LINE_AA,
                    )

                # Feet anchor point (only drawn when entry line is enabled)
                if show_entry_line:
                    cx = int((ix1 + ix2) / 2)
                    cv2.circle(frame, (cx, iy2), 4, (0, 0, 255), -1)

            # ---------------------------------------------------------------
            # Standalone Seated Staff Tracking & Verification
            # Detects and persistently locks onto seated employees behind partitions
            # Strictly filters out bags, backpacks, cushions, clothing textures, and cups
            # ---------------------------------------------------------------
            if allow_standalone_faces:
                # 0. Suppress any detected face that falls within or near ANY active YOLO tracked person
                active_person_boxes = list(tracked_objects.values())
                currently_active_staff = {
                    employee_recognizer.get_employee_id(str(atid))
                    for atid in tracked_objects.keys()
                }

                for face in all_detected_faces:
                    if face.get("matched", False):
                        continue
                    fcx, fcy = face["center"]
                    # If this face is inside or around any active YOLO person box, mark matched to prevent duplicate tracking
                    for pbox in active_person_boxes:
                        px1, py1, px2, py2 = pbox
                        if (px1 - 25 <= fcx <= px2 + 25) and (py1 - 25 <= fcy <= py2 + 25):
                            face["matched"] = True
                            break

                for face in all_detected_faces:
                    if face.get("matched", False):
                        continue

                    sfx, sfy, sfw, sfh = face["box"]
                    lms = face.get("landmarks", [])

                    # 1. Strict facial geometry validation: Rejects non-face objects (bags, cushions, screen reflections)
                    # Standalone faces require stricter eye distance to reject glossy screen glare
                    if not is_valid_facial_geometry(lms, min_eye_dist=12.0):
                        continue

                    # 2. Strict human face dimension & confidence checks (standalone faces must be >= 65% conf to eliminate screen glare)
                    min_standalone_conf = max(0.65, face_confidence)
                    if face["conf"] < min_standalone_conf or sfw < 26 or sfh < 26 or not (0.50 <= sfw / max(1.0, sfh) <= 1.8):
                        continue

                    sbx1 = max(0, int(sfx))
                    sby1 = max(0, int(sfy))
                    sbx2 = min(w, int(sfx + sfw))
                    sby2 = min(h, int(sfy + sfh))
                    if sbx2 <= sbx1 + 10 or sby2 <= sby1 + 10:
                        continue

                    fcx, fcy = (sbx1 + sbx2) * 0.5, (sby1 + sby2) * 0.5

                    # 3. Check if this face matches an existing actively tracked seated employee
                    matched_seated_id = None
                    for eid, s_track in seated_staff_tracks.items():
                        # If this employee is now actively tracked by YOLO person tracker, drop from seated_staff_tracks
                        if eid in currently_active_staff:
                            continue
                        obx1, oby1, obx2, oby2 = s_track["box"]
                        ocx, ocy = (obx1 + obx2) * 0.5, (oby1 + oby2) * 0.5
                        if math.hypot(fcx - ocx, fcy - ocy) < 75.0:
                            matched_seated_id = eid
                            break

                    if matched_seated_id:
                        # Smooth coordinates with EMA and refresh persistence timer
                        old_box = seated_staff_tracks[matched_seated_id]["box"]
                        sm_bx1 = int(0.70 * sbx1 + 0.30 * old_box[0])
                        sm_by1 = int(0.70 * sby1 + 0.30 * old_box[1])
                        sm_bx2 = int(0.70 * sbx2 + 0.30 * old_box[2])
                        sm_by2 = int(0.70 * sby2 + 0.30 * old_box[3])
                        seated_staff_tracks[matched_seated_id]["box"] = (sm_bx1, sm_by1, sm_bx2, sm_by2)
                        seated_staff_tracks[matched_seated_id]["last_seen"] = now_ts
                        seated_staff_tracks[matched_seated_id]["conf"] = face["conf"]
                        if lms:
                            seated_staff_tracks[matched_seated_id]["landmarks"] = lms
                        face["matched"] = True
                        continue

                    # 4. If not yet tracked, run biometric identification against employee database
                    s_crop = frame[sby1:sby2, sbx1:sbx2]
                    s_crop_lms = [(int(lx - sbx1), int(ly - sby1)) for (lx, ly) in lms] if lms else None
                    temp_tid = f"seated_spot_{int(fcx // 40)}_{int(fcy // 40)}"
                    s_last_att = track_last_recognition_attempt.get(temp_tid, 0.0)

                    if now_ts - s_last_att >= 0.40:
                        track_last_recognition_attempt[temp_tid] = now_ts
                        s_match = employee_recognizer.evaluate_track_face(
                            tracker_id=temp_tid,
                            face_crop=s_crop,
                            face_conf=float(face["conf"]),
                            landmarks=s_crop_lms,
                            full_image=frame,
                            raw_landmarks=lms,
                        )

                        if s_match:
                            s_emp_id, s_emp_conf = s_match
                            # NEVER assign if this employee is ALREADY actively tracked by the primary YOLO tracker!
                            if s_emp_id in currently_active_staff:
                                face["matched"] = True
                                continue
                            emp_name = employee_names.get(s_emp_id, s_emp_id)
                            seated_staff_tracks[s_emp_id] = {
                                "box": (sbx1, sby1, sbx2, sby2),
                                "landmarks": lms,
                                "emp_id": s_emp_id,
                                "emp_name": emp_name,
                                "last_seen": now_ts,
                                "conf": face["conf"],
                                "tracker_id": temp_tid,
                            }
                            face["matched"] = True
                            last_punch = punched_employee_ids.get(s_emp_id, 0.0)
                            if (now_ts - last_punch) > 300.0:
                                punched_employee_ids[s_emp_id] = now_ts
                                sync_client.send_employee_punch_event(s_emp_id, event_type="CHECK_IN", confidence=s_emp_conf)
                        elif face["conf"] >= 0.70:
                            # Draw temporary visitor face ONLY if confidence is very high (>= 70%) and geometry is verified
                            cv2.rectangle(frame, (sbx1, sby1), (sbx2, sby2), (255, 30, 220), 2)
                            for (lx, ly) in lms:
                                if 0 <= lx < w and 0 <= ly < h:
                                    cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)
                            slabel = f"Face {int(face['conf'] * 100)}%"
                            (slw, slh), _ = cv2.getTextSize(slabel, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                            cv2.rectangle(frame, (sbx1, max(0, sby1 - 18)), (sbx1 + slw + 8, sby1), (15, 23, 42), -1)
                            cv2.putText(frame, slabel, (sbx1 + 4, max(12, sby1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 30, 220), 1, cv2.LINE_AA)

                # 5. Render & Persist all confirmed Seated Staff across blinks/head-tilts for up to 3.5s
                for eid in list(seated_staff_tracks.keys()):
                    s_track = seated_staff_tracks[eid]
                    time_since_seen = now_ts - s_track["last_seen"]
                    # If this employee is now tracked by the primary YOLO tracker or timed out, cleanly remove
                    if eid in currently_active_staff or time_since_seen > 3.5:
                        employee_recognizer.on_track_lost(s_track.get("tracker_id", f"seated_{eid}"))
                        employee_recognizer.on_track_lost(str(eid))
                        employee_dept_tracker.on_employee_lost(str(eid), now_ts)
                        del seated_staff_tracks[eid]
                        continue

                    bx1, by1, bx2, by2 = s_track["box"]
                    ename = s_track["emp_name"]
                    econf = s_track["conf"]
                    visible_staff_this_frame.add(eid)

                    # Update seated employee department presence & dwell
                    s_dept = zone_manager.find_section_for_box((bx1, by1, bx2, by2))
                    employee_dept_tracker.update_employee_presence(eid, ename, s_dept, now_ts, track_id=s_track.get("tracker_id"))
                    if s_dept:
                        s_sid = s_dept[0]
                        if zone_manager.live_occupancy.get(s_sid, 0) == 0:
                            zone_manager.live_occupancy[s_sid] = 1

                    # Golden Yellow Bounding Box for Staff
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 215, 255), 2)
                    for (lx, ly) in s_track.get("landmarks", []):
                        if 0 <= lx < w and 0 <= ly < h:
                            cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)

                    s_dwell = employee_dept_tracker.get_active_dwell_info(eid, now_ts)
                    if s_dwell:
                        s_dname, s_ddig, _ = s_dwell
                        slabel = f"STAFF [{ename}] | In {s_dname}: {s_ddig} | Face {int(econf * 100)}%"
                    else:
                        slabel = f"STAFF [{ename}] | Face {int(econf * 100)}%"
                    (slw, slh), _ = cv2.getTextSize(slabel, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                    cv2.rectangle(frame, (bx1, max(0, by1 - 18)), (bx1 + slw + 8, by1), (15, 23, 42), -1)
                    cv2.putText(frame, slabel, (bx1 + 4, max(12, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 255), 1, cv2.LINE_AA)

            # Check for inactive staff who left camera view
            employee_dept_tracker.tick_inactive_employees(visible_staff_this_frame, now_ts)

            # 3. Render User-Drawn Sections, Entry Line, and Interactive Clickable Top Toolbar
            frame = zone_manager.render_overlay(frame)

            # 4. Zero-Lag Bottom Status & Telemetry Ribbon
            if show_hud:
                bot_h = 32
                if h > bot_h:
                    bot_roi = frame[h - bot_h:h, 0:w]
                    bot_bg = np.full_like(bot_roi, (10, 15, 26), dtype=np.uint8)
                    cv2.addWeighted(bot_bg, 0.85, bot_roi, 0.15, 0, bot_roi)

                hw_tag = f"[{device.upper()}]"
                sec_badges = "  |  ".join([f"{s['name'].split()[0]}: {zone_manager.section_totals.get(s['id'], 0)}" for s in zone_manager.sections])
                sec_str = f"  |  {sec_badges}" if sec_badges else ""
                sec_shortcuts = "  ".join([f"[{i+2}] {s['name'].split()[0]}" for i, s in enumerate(zone_manager.sections)])
                sec_shortcut_str = f"  {sec_shortcuts} " if sec_shortcuts else ""
                staff_hud = employee_dept_tracker.get_all_active_summary(now_ts)
                staff_hud_str = f"  |  Staff: {staff_hud}" if staff_hud else ""
                stats_text = (
                    f"{hw_tag} FPS: {fps:.1f}  |  "
                    f"Entry: {zone_manager.attendance_total}{sec_str}{staff_hud_str}  |  "
                    f"Keys: [1] Line{sec_shortcut_str} [D] Flip Dir  [C] Reset  [R] Clear"
                )
                cv2.putText(frame, stats_text, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 225, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                break
            elif key == ord('1'):
                zone_manager.active_tool = "entry_line"
                zone_manager.status_msg = "Selected: [1. DRAW ENTRY LINE] - Click & Drag across entrance"
                zone_manager.status_timer = 150
                print("[*] Tool switched to: [Entry Line]")
            elif ord('2') <= key <= ord('9'):
                sec_idx = key - ord('2')
                if sec_idx < len(zone_manager.sections):
                    s = zone_manager.sections[sec_idx]
                    zone_manager.active_tool = s["id"]
                    zone_manager.status_msg = f"Selected: [{sec_idx+2}. DRAW {s['name'].upper()}] - Click & Drag box over area"
                    zone_manager.status_timer = 150
                    print(f"[*] Tool switched to: [{s['name']}]")
            elif key in (ord('d'), ord('D')):
                zone_manager.flip_direction()
            elif key in (ord('c'), ord('C')):
                zone_manager.reset_counts()
            elif key in (ord('r'), ord('R'), ord('x'), ord('X')):
                zone_manager.clear_all()
            elif key in (ord('s'), ord('S')):
                zone_manager.save_to_config()
                zone_manager.status_msg = "[+] Configuration saved to camera_config.json!"
                zone_manager.status_timer = 120
                print("[*] Configuration saved to camera_config.json")
            elif key in (ord('+'), ord('='), ord(']')):  # Increase face box size
                face_scale = round(min(2.5, face_scale + 0.10), 2)
                print(f"[*] Face box scale adjusted: {face_scale:.2f}x")
            elif key in (ord('-'), ord('_'), ord('[')):  # Decrease face box size
                face_scale = round(max(0.6, face_scale - 0.10), 2)
                print(f"[*] Face box scale adjusted: {face_scale:.2f}x")
            elif key in (ord('m'), ord('M')):  # Toggle anti-jitter smoothing
                smoother.enabled = not smoother.enabled
                print(f"[*] Anti-jitter Box Smoothing: {'ON' if smoother.enabled else 'OFF'}")
            elif key in (ord('f'), ord('F')):  # Toggle head fallback
                show_head_fallback = not show_head_fallback
                print(f"[*] Head fallback when face is turned away: {'ON' if show_head_fallback else 'OFF (AI Faces Only)'}")
            elif key in (ord('h'), ord('H')):  # Toggle HUD
                show_hud = not show_hud
                print(f"[*] Virtual entry line & sections: {'ON' if show_entry_line else 'OFF'}")
            elif key in (ord('h'), ord('H')):  # Toggle HUD
                show_hud = not show_hud

    except KeyboardInterrupt:
        print("\n[*] Exiting on user interrupt...")
    finally:
        reader.release()
        if not headless:
            cv2.destroyAllWindows()
        sync_client.stop()
        employee_dept_tracker.finalize_all(time.time())
        print(f"\n[+] Local detection finished. Total Attendance: {zone_manager.attendance_total}, Section visits: {zone_manager.section_totals}")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def load_camera_config() -> dict:
    """Loads camera_config.json (or camera_config.example.json fallback) if available in working directory or bundle."""
    config_path = get_resource_path("camera_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Warning: Failed to parse camera_config.json: {e}")
    # Fallback to example configuration if user config not yet created
    example_path = get_resource_path("camera_config.example.json")
    if os.path.exists(example_path):
        try:
            with open(example_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main():
    config = load_camera_config()
    cameras_map = config.get("cameras", {})
    default_ch = config.get("default_channel", "101")

    parser = argparse.ArgumentParser(
        description="Local AI Video Processor and Person Detection Engine"
    )
    parser.add_argument(
        "channel_args",
        nargs="*",
        default=[],
        help="Showroom camera channel (e.g. 101, 201) and/or optional version keyword ('v2' or 'v1')",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Run in v2 mode using RF-DETR (Roboflow Detection Transformer)",
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="Run in v1 mode using YOLO",
    )
    parser.add_argument(
        "--channel",
        "--cam",
        dest="channel_flag",
        type=str,
        default=None,
        help="Showroom camera channel number (e.g. 101, 201, 301, 401)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: RTSP stream URL, video file path, or webcam index (e.g. '0')",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        help="Camera UUID in backend database",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=config.get("api_url", "http://127.0.0.1:8000"),
        help="FastAPI backend URL for footfall synchronization (default: 'http://127.0.0.1:8000')",
    )
    parser.add_argument(
        "--line-y",
        type=float,
        default=config.get("line_y", 0.55),
        help="Virtual entry line Y position as fraction of frame height (default: 0.55)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["top_to_bottom", "bottom_to_top"],
        default=config.get("direction", "top_to_bottom"),
        help="Entry crossing direction (default: 'top_to_bottom')",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=float(config.get("confidence", 0.31)),
        help="Person detection confidence threshold (default: 0.31)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(config.get("imgsz", 1024)),
        help="Inference input resolution in pixels (default: 1024)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "yolo", "rtmdet", "rf-detr", "rtdetr"],
        default=config.get("model_type", "yolo"),
        help="Detector model architecture: 'yolo' (default), 'rtmdet', 'rf-detr', 'rtdetr', or 'auto'",
    )
    parser.add_argument(
        "--rf-detr-variant",
        type=str,
        choices=["nano", "small", "medium", "base", "large"],
        default=config.get("rf_detr_variant", "small"),
        help="Roboflow RF-DETR model variant (default: 'small')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.get("model", "models/yolo26s.pt"),
        help="Path to detector weights checkpoint (default: 'models/yolo26s.pt')",
    )
    parser.add_argument(
        "--show-line",
        action="store_true",
        default=config.get("show_entry_line", False),
        help="Display and calculate virtual entry line crossings (default: False)",
    )
    parser.add_argument(
        "--face-scale",
        type=float,
        default=float(config.get("face_scale", 1.05)),
        help="Face bounding box scale multiplier (default: 1.05, adjust with +/- in window)",
    )
    parser.add_argument(
        "--face-model",
        type=str,
        default=config.get("face_model", "models/scrfd_2.5g_bnkps.pt"),
        help="Path to SCRFD or YuNet neural face model file (default: 'models/scrfd_2.5g_bnkps.pt')",
    )
    parser.add_argument(
        "--face-conf",
        type=float,
        default=float(config.get("face_confidence", 0.36)),
        help="Neural face detection confidence threshold (default: 0.36)",
    )
    parser.add_argument(
        "--sim-thresh",
        type=float,
        default=float(config.get("face_similarity_threshold", settings.FACE_SIMILARITY_THRESHOLD)),
        help="Biometric face cosine similarity threshold for staff identification (default: 0.32)",
    )
    parser.add_argument(
        "--standalone-faces",
        action="store_true",
        default=config.get("allow_standalone_faces", False),
        help="Enable detection of standalone faces behind cubicle partitions (default: False)",
    )
    parser.add_argument(
        "--head-fallback",
        action="store_true",
        default=bool(config.get("show_head_fallback", False)),
        help="Enable synthetic anatomical head box fallback when face is turned away (default: False)",
    )
    parser.add_argument(
        "--no-head-fallback",
        action="store_true",
        help="Disable anatomical head fallback when face is turned away (shows AI faces only)",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        default=not bool(config.get("box_smoothing", True)),
        help="Disable anti-jitter bounding box smoothing and face hysteresis filter",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=float(config.get("smoothing_alpha", 0.35)),
        help="EMA smoothing alpha factor between 0.1 and 1.0 (default: 0.35)",
    )
    parser.add_argument(
        "--face-persistence",
        type=int,
        default=int(config.get("face_persistence", 8)),
        help="Number of frames to buffer face detections across drops (default: 8)",
    )
    parser.add_argument(
        "--body-persistence",
        type=int,
        default=int(config.get("body_persistence", 26)),
        help="Number of missing frames before an un-detected person track is purged (default: 6)",
    )
    parser.add_argument(
        "--max-visual-coast",
        type=int,
        default=int(config.get("max_visual_coast", 20)),
        help="Max frames to visually display coasted bounding boxes when detection is lost (default: 2)",
    )
    parser.add_argument(
        "--min-feet-y",
        type=float,
        default=float(config.get("min_feet_y", 0.20)),
        help="Minimum normalized ground/feet Y position (filters out hanging decorations/ceiling objects) (default: 0.20)",
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=float(config.get("deadband_px", 6.0)),
        help="Pixel displacement deadband below which stationary boxes freeze (default: 6.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying local OpenCV window (service/background mode)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=config.get("tracker", settings.TRACKER_TYPE),
        help="Multi-object tracker configuration: 'bytetrack', 'botsort', or path to tracker YAML (default: bytetrack.yaml)",
    )

    args = parser.parse_args()

    # Process positional arguments for channel and version keyword (e.g. '201', 'v2')
    raw_pos = [str(a).strip() for a in args.channel_args]
    has_v2_keyword = args.v2 or any(p.lower() == 'v2' for p in raw_pos)
    has_v1_keyword = args.v1 or any(p.lower() == 'v1' for p in raw_pos)

    # Extract channel from positional arguments (ignoring 'v1' / 'v2')
    channel_candidates = [p for p in raw_pos if p.lower() not in ('v1', 'v2')]
    target_channel = args.channel_flag or (channel_candidates[0] if channel_candidates else None)

    # Resolve model type: explicit keyword > CLI flag > settings default
    if has_v2_keyword:
        resolved_model_type = 'rf-detr'
    elif has_v1_keyword:
        resolved_model_type = 'yolo'
    elif args.model_type != 'auto':
        resolved_model_type = args.model_type
    else:
        resolved_model_type = getattr(settings, 'DETECTION_MODEL_TYPE', 'yolo')
    
    # If target channel is provided and matches config
    selected_cam = None
    if target_channel and str(target_channel) in cameras_map:
        selected_cam = cameras_map[str(target_channel)]
        print(f"[*] Selected camera channel: {target_channel} ({selected_cam.get('name', '')})")
    elif not target_channel and not args.source and default_ch in cameras_map:
        target_channel = default_ch
        selected_cam = cameras_map[default_ch]
        ch_name = selected_cam.get('name', default_ch)
        print(f"[*] Using default camera channel: {target_channel} ({ch_name})")

    resolved_source = args.source
    resolved_camera_id = args.camera_id

    entry_line_cfg = None
    sections_cfg = None
    if selected_cam:
        if not resolved_source:
            resolved_source = selected_cam.get("source")
        if not resolved_camera_id:
            resolved_camera_id = selected_cam.get("camera_id")
        entry_line_cfg = selected_cam.get("entry_line")
        sections_cfg = selected_cam.get("sections")
        exclusion_zones_cfg = selected_cam.get("exclusion_zones")
    else:
        exclusion_zones_cfg = config.get("exclusion_zones")

    # Final fallbacks if no channel matched and no args given
    if not resolved_source:
        resolved_source = "0"
    if not resolved_camera_id:
        resolved_camera_id = "88970b31-d885-4fd5-a24a-bdc969c96240"

    resolved_channel = str(target_channel or "201")
    resolved_tracker = selected_cam.get("tracker") if selected_cam else None
    if not resolved_tracker:
        resolved_tracker = args.tracker

    resolved_min_feet_y = float(selected_cam.get("min_feet_y", args.min_feet_y)) if selected_cam and "min_feet_y" in selected_cam else args.min_feet_y
    resolved_conf = float(selected_cam.get("confidence", args.conf)) if selected_cam and "confidence" in selected_cam else args.conf
    resolved_standalone_faces = bool(selected_cam.get("allow_standalone_faces", args.standalone_faces)) if selected_cam and "allow_standalone_faces" in selected_cam else args.standalone_faces

    run_local_detector(
        source=resolved_source,
        camera_id=resolved_camera_id,
        api_url=args.api_url,
        line_ratio=args.line_y,
        direction=args.direction,
        confidence_thresh=resolved_conf,
        model_path=args.model,
        headless=args.headless,
        show_entry_line=args.show_line,
        face_scale=args.face_scale,
        face_model_path=args.face_model,
        face_confidence=args.face_conf,
        show_head_fallback=args.head_fallback and not args.no_head_fallback,
        allow_standalone_faces=resolved_standalone_faces,
        imgsz=args.imgsz,
        box_smoothing=not args.no_smoothing,
        smoothing_alpha=args.smoothing_alpha,
        face_persistence=args.face_persistence,
        body_persistence=args.body_persistence,
        max_visual_coast=args.max_visual_coast,
        deadband_px=args.deadband,
        min_feet_y=resolved_min_feet_y,
        camera_channel=resolved_channel,
        similarity_thresh=args.sim_thresh,
        entry_line_cfg=entry_line_cfg,
        sections_cfg=sections_cfg,
        exclusion_zones=exclusion_zones_cfg,
        tracker_name=resolved_tracker,
        model_type=resolved_model_type,
        rf_detr_variant=args.rf_detr_variant,
    )


if __name__ == "__main__":
    main()

