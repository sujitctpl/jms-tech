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

# Force TCP transport for RTSP streams globally to prevent packet loss & H264 macroblock decode corruption
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np

# Handle PyInstaller onefile directory resolution
def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev, models folder, and PyInstaller."""
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
# Simple Centroid / Bounding Box Tracker
# ---------------------------------------------------------------------------
class LocalTracker:
    """Lightweight Euclidean tracker for continuous person IDs across frames."""

    def __init__(self, max_disappeared: int = 25, max_distance: float = 90.0):
        self.next_id = 10001
        self.objects: Dict[int, Tuple[float, float, float, float]] = {}  # id -> (x1, y1, x2, y2)
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def sync_active_tracks(self, active_objects: Dict[int, Tuple[float, float, float, float]]) -> None:
        """Synchronizes tracked objects from ByteTrack to keep ID continuity if fallback occurs."""
        self.objects = dict(active_objects)
        self.disappeared = {oid: 0 for oid in active_objects.keys()}
        max_id = max(active_objects.keys(), default=0)
        if max_id >= self.next_id:
            self.next_id = max_id + 1

    def update(self, detections: List[Tuple[float, float, float, float]]) -> Dict[int, Tuple[float, float, float, float]]:
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        if len(self.objects) == 0:
            for box in detections:
                self.objects[self.next_id] = box
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return self.objects

        # Match existing objects to new detections based on centroid distance
        object_ids = list(self.objects.keys())
        object_centroids = [
            (((b[0] + b[2]) / 2.0), b[3]) for b in self.objects.values()
        ]  # feet contact point

        det_centroids = [
            (((b[0] + b[2]) / 2.0), b[3]) for b in detections
        ]

        used_rows: Set[int] = set()
        used_cols: Set[int] = set()

        # Compute pairwise distances
        dists = []
        for i, (ox, oy) in enumerate(object_centroids):
            for j, (dx, dy) in enumerate(det_centroids):
                d = math.hypot(ox - dx, oy - dy)
                dists.append((d, i, j))
        dists.sort(key=lambda x: x[0])

        for d, row, col in dists:
            if row in used_rows or col in used_cols:
                continue
            if d > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = detections[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                del self.objects[obj_id]
                del self.disappeared[obj_id]

        # Handle unmatched new detections
        unused_cols = set(range(len(detections))) - used_cols
        for col in unused_cols:
            self.objects[self.next_id] = detections[col]
            self.disappeared[self.next_id] = 0
            self.next_id += 1

        return self.objects


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
        alpha: float = 0.70,
        deadband_px: float = 3.5,
        face_persistence: int = 8,
        body_persistence: int = 40,
        enabled: bool = True,
    ):
        self.alpha = float(alpha)
        self.deadband_px = float(deadband_px)
        self.face_persistence = int(face_persistence)
        self.body_persistence = int(body_persistence)
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
                iou = self._compute_iou(box, pbox)
                mcx, mcy = self.prev_centroids.get(mid, ((pbox[0] + pbox[2]) * 0.5, (pbox[1] + pbox[3]) * 0.5))
                dist = math.hypot(ncx - mcx, ncy - mcy)

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
        self, tracked_objects: Dict[int, Tuple[float, float, float, float]]
    ) -> Dict[int, Tuple[float, float, float, float]]:
        """Applies EMA smoothing, persistence coasting across drops, and velocity deadband to person body boxes."""
        if not self.enabled:
            return tracked_objects

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

        # 2. Update disappearance counters & purge tracks exceeding cutoff
        for old_id in list(self.body_boxes.keys()):
            if old_id not in active_ids:
                self.disappeared[old_id] = self.disappeared.get(old_id, 0) + 1
                if self.disappeared[old_id] > self.body_persistence:
                    # Cutoff exceeded -> clean up so the box disappears
                    self.body_boxes.pop(old_id, None)
                    self.prev_centroids.pop(old_id, None)
                    self.face_tracks.pop(old_id, None)
                    self.head_boxes.pop(old_id, None)
                    self.disappeared.pop(old_id, None)
                    # Clean up aliases pointing to or from this purged old_id
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
                smoothed_dict[obj_id] = raw_box
                continue

            px1, py1, px2, py2 = self.body_boxes[obj_id]
            pcx, pcy = self.prev_centroids[obj_id]

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

        # 4. Persistence Coasting: Retain momentarily missed detections within the cutoff buffer
        for old_id, miss_count in list(self.disappeared.items()):
            if 0 < miss_count <= self.body_persistence and old_id in self.body_boxes and old_id not in smoothed_dict:
                smoothed_dict[old_id] = self.body_boxes[old_id]

        return smoothed_dict

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
# Interactive Multi-Zone & Dynamic User-Drawn Entry Line Manager
# ---------------------------------------------------------------------------
class ZoneManager:
    """
    Real-time Dynamic User-Drawn Entry Line and Section Manager (Sales, Accounts, etc.).
    - Pristine Recording Screen: Starts with NO static preset lines or boxes.
    - Always Interactive: The user draws the entry line and sections directly on the screen.
    - Top Clickable Toolbar:
        [1. Draw Entry Line]  [2. Draw Sales]  [3. Draw Accounts]  [Clear All]
    - Click & Drag on screen draws the line or section box with live visual feedback.
    - Auto-Saves to camera_config.json upon mouse release.
    - Multi-Zone Counting Logic:
        Crossing Entry Line -> Attendance = 1
        Entering Sales Section -> Sales = 1 (and live occupancy updated)
        Entering Accounts Section -> Accounts = 1 (and live occupancy updated)
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
        direction: str = "top_to_bottom",
    ):
        self.config_path = config_path
        self.camera_channel = str(camera_channel)
        self.frame_w = frame_width
        self.frame_h = frame_height

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
                if isinstance(s, dict) and "box" in s and len(s["box"]) == 4:
                    self.sections.append({
                        "id": str(s.get("id", "section")),
                        "name": str(s.get("name", "Section")),
                        "color": tuple(s.get("color", [0, 200, 255])),
                        "box": [int(s["box"][0]), int(s["box"][1]), int(s["box"][2]), int(s["box"][3])],
                    })

        # Multi-Zone Counters
        self.attendance_total = 0
        self.section_totals: Dict[str, int] = {s["id"]: 0 for s in self.sections}
        self.live_occupancy: Dict[str, int] = {s["id"]: 0 for s in self.sections}

        # Track history
        self.counted_attendance_ids: Set[int] = set()
        self.counted_person_ids: Set[str] = set()
        self.track_identities: Dict[int, str] = {}
        self.person_visited_sections: Dict[int, Set[str]] = {}
        self.track_positions: Dict[int, Tuple[float, float]] = {}  # obj_id -> prev feet (x, y)

        # Visual Flash Timers
        self.line_flash_timer = 0
        self.section_flash_timers: Dict[str, int] = {s["id"]: 0 for s in self.sections}

        # Interactive Drawing State (ALWAYS ACTIVE - No toggle needed)
        self.active_tool = "entry_line"  # "entry_line", "sales", "accounts"
        self.is_mouse_down = False
        self.mouse_start = (0, 0)
        self.mouse_current = (0, 0)
        self.status_msg = "CLICK & DRAG TO DRAW ENTRY LINE, OR CLICK BUTTONS ABOVE TO SWITCH TOOL"
        self.status_timer = 250

        # Toolbar button boundaries: (x1, y1, x2, y2)
        self.btn_line = (15, 8, 205, 46)
        self.btn_sales = (215, 8, 395, 46)
        self.btn_accounts = (405, 8, 615, 46)
        self.btn_clear = (625, 8, 745, 46)

    def set_identity(self, obj_id: int, person_id: Optional[str]) -> None:
        """Links temporary ByteTrack ID to persistent person ID to prevent duplicate entry counts."""
        if person_id:
            pid = str(person_id)
            self.track_identities[obj_id] = pid
            if obj_id in self.counted_attendance_ids:
                self.counted_person_ids.add(pid)

    @staticmethod
    def _ccw(A, B, C):
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def _segments_intersect(A, B, C, D):
        return (ZoneManager._ccw(A, B, C) * ZoneManager._ccw(A, B, D) <= 0) and \
               (ZoneManager._ccw(C, D, A) * ZoneManager._ccw(C, D, B) <= 0)

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
        direction = self.entry_line.get("direction", "top_to_bottom") if has_entry_line else "top_to_bottom"

        for obj_id, box in tracked_objects.items():
            feet_x = (box[0] + box[2]) * 0.5
            feet_y = box[3]
            curr_pos = (feet_x, feet_y)

            # 1. Check Attendance Line Crossing (if entry line has been drawn)
            pid = self.track_identities.get(obj_id)
            is_already_counted = (obj_id in self.counted_attendance_ids) or (pid and pid in self.counted_person_ids)
            if has_entry_line and not is_already_counted:
                prev_pos = self.track_positions.get(obj_id)
                if prev_pos is not None:
                    crossed = False
                    if self._segments_intersect(p1, p2, prev_pos, curr_pos):
                        if direction == "top_to_bottom" and curr_pos[1] >= prev_pos[1]:
                            crossed = True
                        elif direction == "bottom_to_top" and curr_pos[1] <= prev_pos[1]:
                            crossed = True
                        elif direction in ("left_to_right", "right_to_left"):
                            crossed = True
                        else:
                            crossed = True

                    if crossed:
                        self.counted_attendance_ids.add(obj_id)
                        if pid:
                            self.counted_person_ids.add(pid)
                        self.attendance_total += 1
                        self.line_flash_timer = 25
                        new_attendance_crossings.append(obj_id)

            self.track_positions[obj_id] = curr_pos

            # 2. Check Sections (Sales, Accounts, etc.)
            if self.sections:
                visited = self.person_visited_sections.setdefault(obj_id, set())
                for s in self.sections:
                    bx1, by1, bx2, by2 = s["box"]
                    if bx1 <= feet_x <= bx2 and by1 <= feet_y <= by2:
                        self.live_occupancy[s["id"]] = self.live_occupancy.get(s["id"], 0) + 1
                        if s["id"] not in visited:
                            visited.add(s["id"])
                            self.section_totals[s["id"]] = self.section_totals.get(s["id"], 0) + 1
                            self.section_flash_timers[s["id"]] = 25
                            new_section_entries.append((obj_id, s["id"], s["name"]))

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
                    self.status_msg = "Selected: [1. DRAW ENTRY LINE] - Click & Drag on video across entrance"
                    self.status_timer = 150
                    print("[*] Tool selected: [Entry Line]")
                    return
                elif self.btn_sales[0] <= x <= self.btn_sales[2]:
                    self.active_tool = "sales"
                    self.status_msg = "Selected: [2. DRAW SALES SECTION] - Click & Drag box over Sales area"
                    self.status_timer = 150
                    print("[*] Tool selected: [Sales Section]")
                    return
                elif self.btn_accounts[0] <= x <= self.btn_accounts[2]:
                    self.active_tool = "accounts"
                    self.status_msg = "Selected: [3. DRAW ACCOUNTS SECTION] - Click & Drag box over Accounts area"
                    self.status_timer = 150
                    print("[*] Tool selected: [Accounts Section]")
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
                    self.entry_line = {
                        "enabled": True,
                        "p1": [int(sx), int(sy)],
                        "p2": [int(ex), int(ey)],
                        "direction": "top_to_bottom",
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
                    sec_name = "Sales Section" if sec_id == "sales" else "Accounts Section"
                    color = (0, 200, 255) if sec_id == "sales" else (255, 140, 0)
                    
                    found = False
                    for s in self.sections:
                        if s["id"] == sec_id:
                            s["box"] = [int(x1), int(y1), int(x2), int(y2)]
                            found = True
                            break
                    if not found:
                        self.sections.append({
                            "id": sec_id,
                            "name": sec_name,
                            "color": color,
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                        })
                    if sec_id not in self.section_totals:
                        self.section_totals[sec_id] = 0
                    if sec_id not in self.live_occupancy:
                        self.live_occupancy[sec_id] = 0

                    self.status_msg = f"[+] {sec_name} box saved: [{x1},{y1},{x2},{y2}]!"
                    self.status_timer = 150
                    print(f"[+] {sec_name} drawn and saved: [{x1}, {y1}, {x2}, {y2}]")
                    self.save_to_config()

    def clear_all(self):
        self.entry_line = None
        self.sections.clear()
        self.attendance_total = 0
        self.counted_attendance_ids.clear()
        self.person_visited_sections.clear()
        self.section_totals.clear()
        self.live_occupancy.clear()
        self.status_msg = "[*] All lines & sections cleared! Draw new ones anytime."
        self.status_timer = 150
        print("[*] Cleared all entry lines and sections.")
        self.save_to_config()

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
            cam_data["sections"] = self.sections
            with open(resolved, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"[!] Warning: Failed to save config: {e}")
            return False

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # 1. Render User-Drawn Section Areas (ONLY if drawn)
        if self.sections:
            overlay = frame.copy()
            for s in self.sections:
                x1, y1, x2, y2 = s["box"]
                color = s["color"]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)

            for s in self.sections:
                x1, y1, x2, y2 = s["box"]
                color = s["color"]
                is_flashing = self.section_flash_timers.get(s["id"], 0) > 0
                border_color = (0, 255, 128) if is_flashing else color
                thickness = 3 if is_flashing else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness)

                # Header badge
                occ = self.live_occupancy.get(s["id"], 0)
                tot = self.section_totals.get(s["id"], 0)
                badge_text = f"{s['name'].upper()} | Active: {occ} | Visits: {tot}"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.rectangle(frame, (x1, y1), (x1 + tw + 16, y1 + 24), (15, 23, 42), -1)
                cv2.rectangle(frame, (x1, y1), (x1 + tw + 16, y1 + 24), border_color, 1)
                cv2.putText(frame, badge_text, (x1 + 8, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Render User-Drawn Entry Line (ONLY if drawn)
        if self.entry_line and self.entry_line.get("enabled", True):
            p1 = tuple(self.entry_line["p1"])
            p2 = tuple(self.entry_line["p2"])
            is_flashing = self.line_flash_timer > 0
            line_color = (0, 255, 128) if is_flashing else (0, 215, 255)
            line_thickness = 4 if is_flashing else 2
            cv2.line(frame, p1, p2, line_color, line_thickness)

            # Direction Arrow
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2
            cv2.arrowedLine(frame, (mx, my - 16), (mx, my + 16), line_color, 2, tipLength=0.35)

            line_label = f"*** ENTRY COUNTED! ***" if is_flashing else f"ENTRY LINE [Attendance: {self.attendance_total}]"
            cv2.putText(frame, line_label, (min(p1[0], p2[0]) + 10, max(75, min(p1[1], p2[1]) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, line_color, 2, cv2.LINE_AA)

        # 3. Live Rubber-Band Feedback While Mouse Dragging
        if self.is_mouse_down:
            sx, sy = self.mouse_start
            cx, cy = self.mouse_current
            if self.active_tool == "entry_line":
                cv2.line(frame, (sx, sy), (cx, cy), (0, 255, 255), 2)
                cv2.circle(frame, (sx, sy), 5, (0, 255, 0), -1)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Release mouse to set Entry Line", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                x1, x2 = min(sx, cx), max(sx, cx)
                y1, y2 = min(sy, cy), max(sy, cy)
                color = (0, 200, 255) if self.active_tool == "sales" else (255, 140, 0)
                sec_name = "Sales Section" if self.active_tool == "sales" else "Accounts Section"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Release mouse to set {sec_name}", (x1 + 8, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # 4. Interactive Clickable Toolbar Buttons (Top Bar)
        tb_bg = frame.copy()
        cv2.rectangle(tb_bg, (0, 0), (w, 54), (10, 15, 26), -1)
        cv2.addWeighted(tb_bg, 0.85, frame, 0.15, 0, frame)

        # Button 1: Draw Entry Line
        b1_active = self.active_tool == "entry_line"
        b1_col = (0, 255, 255) if b1_active else (70, 85, 105)
        cv2.rectangle(frame, (self.btn_line[0], self.btn_line[1]), (self.btn_line[2], self.btn_line[3]), (25, 35, 55), -1)
        cv2.rectangle(frame, (self.btn_line[0], self.btn_line[1]), (self.btn_line[2], self.btn_line[3]), b1_col, 2 if b1_active else 1)
        b1_txt = "-> [1] Entry Line" if b1_active else "[1] Entry Line"
        cv2.putText(frame, b1_txt, (self.btn_line[0] + 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255) if b1_active else (180, 195, 215), 1, cv2.LINE_AA)

        # Button 2: Draw Sales Section
        b2_active = self.active_tool == "sales"
        b2_col = (0, 200, 255) if b2_active else (70, 85, 105)
        cv2.rectangle(frame, (self.btn_sales[0], self.btn_sales[1]), (self.btn_sales[2], self.btn_sales[3]), (25, 35, 55), -1)
        cv2.rectangle(frame, (self.btn_sales[0], self.btn_sales[1]), (self.btn_sales[2], self.btn_sales[3]), b2_col, 2 if b2_active else 1)
        b2_txt = "-> [2] Sales Sec" if b2_active else "[2] Sales Sec"
        cv2.putText(frame, b2_txt, (self.btn_sales[0] + 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255) if b2_active else (180, 195, 215), 1, cv2.LINE_AA)

        # Button 3: Draw Accounts Section
        b3_active = self.active_tool == "accounts"
        b3_col = (255, 140, 0) if b3_active else (70, 85, 105)
        cv2.rectangle(frame, (self.btn_accounts[0], self.btn_accounts[1]), (self.btn_accounts[2], self.btn_accounts[3]), (25, 35, 55), -1)
        cv2.rectangle(frame, (self.btn_accounts[0], self.btn_accounts[1]), (self.btn_accounts[2], self.btn_accounts[3]), b3_col, 2 if b3_active else 1)
        b3_txt = "-> [3] Accounts Sec" if b3_active else "[3] Accounts Sec"
        cv2.putText(frame, b3_txt, (self.btn_accounts[0] + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 140, 0) if b3_active else (180, 195, 215), 1, cv2.LINE_AA)

        # Button 4: Clear All
        cv2.rectangle(frame, (self.btn_clear[0], self.btn_clear[1]), (self.btn_clear[2], self.btn_clear[3]), (25, 35, 55), -1)
        cv2.rectangle(frame, (self.btn_clear[0], self.btn_clear[1]), (self.btn_clear[2], self.btn_clear[3]), (70, 85, 105), 1)
        cv2.putText(frame, "[R] Clear All", (self.btn_clear[0] + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 160, 185), 1, cv2.LINE_AA)

        # Guidance status message ribbon right below buttons
        if self.status_timer > 0 and self.status_msg:
            (smw, _), _ = cv2.getTextSize(self.status_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (15, 58), (25 + smw, 80), (10, 15, 26), -1)
            cv2.putText(frame, self.status_msg, (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

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
                return True, self.latest_frame.copy()
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
    show_head_fallback: bool = True,
    allow_standalone_faces: bool = False,
    imgsz: int = 640,
    box_smoothing: bool = True,
    smoothing_alpha: float = 0.70,
    face_persistence: int = 8,
    body_persistence: int = 40,
    deadband_px: float = 3.5,
    camera_channel: str = "201",
    entry_line_cfg: Optional[dict] = None,
    sections_cfg: Optional[list] = None,
):
    print("=" * 70)
    print("  CAPERMINT SHOWROOM INTELLIGENCE - LOCAL AI EDGE DETECTOR")
    print("=" * 70)
    print(f"[*] Camera ID      : {camera_id}")
    print(f"[*] Video Source   : {source}")
    print(f"[*] Backend API    : {api_url}")
    print(f"[*] Detection Model: {model_path} (imgsz={imgsz})")
    print(f"[*] Face AI Model  : {face_model_path} (threshold: {face_confidence:.2f})")
    print(f"[*] Confidence     : {confidence_thresh:.2f}")
    print(f"[*] Face Box Scale : {face_scale:.2f}x (adjust live with +/-)")
    print(f"[*] Anti-Jitter    : {'ON (EMA alpha=' + str(smoothing_alpha) + ', deadband=' + str(deadband_px) + 'px)' if box_smoothing else 'OFF'}")
    print(f"[*] Face Hysteresis: {'ON (' + str(face_persistence) + ' frames buffer)' if box_smoothing else 'OFF'}")
    print(f"[*] Track Persistence: {'ON (' + str(body_persistence) + ' frames cutoff / ~2.5s)' if box_smoothing else 'OFF'}")
    print(f"[*] Show Entry Line: {show_entry_line}")
    print(f"[*] Standalone Face: {allow_standalone_faces} (strict screen filtering)")
    print(f"[*] Headless Mode  : {headless}")

    # Detect compute device (NVIDIA CUDA GPU vs CPU)
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] Compute Device : {device.upper()} ({device_name})")

    # Resolve model file
    resolved_model = get_resource_path(model_path)
    if not os.path.exists(resolved_model):
        if os.path.exists("models/yolo11m.pt"):
            resolved_model = "models/yolo11m.pt"
        elif os.path.exists("models/yolo11n.pt"):
            resolved_model = "models/yolo11n.pt"
        elif os.path.exists("yolo11m.pt"):
            resolved_model = "yolo11m.pt"
        elif os.path.exists("yolo11n.pt"):
            resolved_model = "yolo11n.pt"
        else:
            resolved_model = model_path

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

    tracker = LocalTracker(max_disappeared=20, max_distance=120.0)
    smoother = TrackSmoother(
        alpha=smoothing_alpha,
        deadband_px=deadband_px,
        face_persistence=face_persistence,
        body_persistence=body_persistence,
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
    )
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
    sim_thresh = settings.FACE_SIMILARITY_THRESHOLD
    vector_store = PgVectorStoreManager(db=db_session, similarity_threshold=sim_thresh)
    identity_manager = IdentityRecoveryManager(
        embedder=face_embedder,
        vector_store=vector_store,
        similarity_threshold=sim_thresh,
        max_attempts=5,
        cooldown_frames=6,
    )
    employee_recognizer = EmployeeRecognitionManager(
        embedder=face_embedder,
        vector_store=vector_store,
        similarity_threshold=sim_thresh,
        min_confirm_frames=3,
        vote_window=5,
    )

    # Cooldown buffer for employee check-in punches (punch at most once per 5 mins per staff)
    punched_employee_ids: Dict[str, float] = {}
    track_last_recognition_attempt: Dict[str, float] = {}

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
    bytetrack_empty_streak = 0

    # Ensure local bytetrack.yaml is resolved directly
    tracker_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack.yaml")
    if not os.path.exists(tracker_config):
        tracker_config = "bytetrack.yaml"

    print("\n[+] Edge Video Processor running. Press 'q' or 'ESC' in the window to quit.\n")

    try:
        while True:
            ret, frame = reader.read_latest()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_counter += 1

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

            try:
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
                # Fallback to model.predict if tracking plugin encounters issue
                try:
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
                            detections.append((x1, y1, x2, y2))
                            unmatched_boxes.append((x1, y1, x2, y2))
                except Exception:
                    pass

            # Controlled fallback & unmatched detection handling
            if bytetrack_failed:
                # Controlled fallback: ByteTrack crashed or failed -> use LocalTracker
                if detections:
                    tracked_objects = tracker.update(detections)
                else:
                    tracked_objects = tracker.update([])
            else:
                # ByteTrack is primary
                if tracked_objects:
                    bytetrack_empty_streak = 0
                    # Handle unmatched detections without mixing LocalTracker IDs or creating duplicate identities
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

                    # Keep LocalTracker state synchronized with ByteTrack tracks (never injecting rogue IDs)
                    tracker.sync_active_tracks(tracked_objects)

                elif detections:
                    # ByteTrack returned no tracks despite detections present in the frame
                    # First check if any unmatched detections recover recently lost ByteTrack tracks
                    recovered = smoother.recover_missing_tracks(unmatched_boxes)
                    if recovered:
                        tracked_objects.update(recovered)
                        bytetrack_empty_streak = 0
                        tracker.sync_active_tracks(tracked_objects)
                    else:
                        bytetrack_empty_streak += 1
                        # Controlled fallback: Only activate LocalTracker after sustained ByteTrack silence (>= 5 frames)
                        if bytetrack_empty_streak >= 5:
                            tracked_objects = tracker.update(detections)
                else:
                    # No detections in this frame
                    bytetrack_empty_streak = 0
                    tracker.update([])

            # Apply Anti-Jitter Smoothing & Deadband to Person Bounding Boxes
            tracked_objects = smoother.smooth_body_boxes(tracked_objects)

            # Record track timestamps for on-screen live dwell timer
            now_ts = time.time()
            for obj_id in tracked_objects.keys():
                if obj_id not in track_start_times:
                    track_start_times[obj_id] = now_ts
                track_last_seen[obj_id] = now_ts

            # Expire track timers inactive for > 15 seconds
            for old_id in list(track_start_times.keys()):
                if now_ts - track_last_seen.get(old_id, 0.0) > 15.0:
                    track_start_times.pop(old_id, None)
                    track_last_seen.pop(old_id, None)
                    identity_manager.on_track_lost(str(old_id))
                    employee_recognizer.on_track_lost(str(old_id))
                    track_last_recognition_attempt.pop(str(old_id), None)

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
                print(f"[--> ATTENDANCE DETECTED] Person #{crossed_id}{pid_desc} crossed Entry Line! Total Attendance: {zone_manager.attendance_total}")
                sync_client.send_entry_event(tracker_id=str(crossed_id), confidence=0.94, person_id=pid)

            for (p_id, s_id, s_name) in new_sections:
                sec_tot = zone_manager.section_totals.get(s_id, 0)
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
                and len(tracked_objects) > 0
                and (frame_counter % 2 == 0)
            )
            if should_detect_faces:
                if face_detector_type == "scrfd":
                    all_detected_faces = face_detector.detect(frame, input_size=(640, 640))
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
                for face in all_detected_faces:
                    if face["matched"]:
                        continue
                    fcx, fcy = face["center"]
                    # Face center must strictly be inside or right next to upper portion of person body (top 38%)
                    if (ix1 - 20 <= fcx <= ix2 + 20) and (iy1 - 20 <= fcy <= iy1 + box_h * 0.38):
                        dist = math.hypot(fcx - (ix1 + ix2) / 2.0, fcy - (iy1 + box_h * 0.15))
                        if dist < best_face_dist:
                            best_face_dist = dist
                            matched_face = face

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

                            # 1. First check if recognized as Showroom Employee (throttles unconfirmed evaluation to max 3 Hz per track)
                            tid = str(obj_id)
                            is_confirmed = employee_recognizer.is_track_employee(tid)
                            last_attempt = track_last_recognition_attempt.get(tid, 0.0)
                            emp_match = None

                            if is_confirmed or (now_ts - last_attempt >= 0.35):
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
                                last_punch = punched_employee_ids.get(emp_id, 0.0)
                                if (now_ts - last_punch) > 300.0:
                                    punched_employee_ids[emp_id] = now_ts
                                    sync_client.send_employee_punch_event(emp_id, event_type="CHECK_IN", confidence=emp_conf)
                                    emp_label = employee_names.get(emp_id, emp_id)
                                    print(f"[+] Biometrics: Verified Staff {emp_label} ({emp_id}) on Track #{obj_id} (conf: {emp_conf:.2f})")
                            else:
                                # 2. Otherwise process as customer visitor
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
                if is_staff:
                    emp_id = employee_recognizer.get_employee_id(str(obj_id))
                    emp_display = employee_names.get(emp_id, emp_id)
                    main_label = f"STAFF [{emp_display}] #{obj_id} | {timer_str}"
                    body_color = (0, 215, 255)  # Vivid Golden Yellow for verified showroom staff
                    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), body_color, 2)
                else:
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

            # Draw standalone faces ONLY if explicitly allowed and strictly verified
            # Filters out false positives on monitors, posters, cups, and wallpapers
            if allow_standalone_faces:
                for face in all_detected_faces:
                    if not face["matched"]:
                        sfx, sfy, sfw, sfh = face["box"]
                        # Strict size, aspect ratio, and confidence check
                        if face["conf"] >= 0.70 and sfw >= 30 and sfh >= 32 and (0.65 <= sfw / max(1.0, sfh) <= 1.4):
                            matched_faces_count += 1
                            sbx1 = max(0, int(sfx))
                            sby1 = max(0, int(sfy))
                            sbx2 = min(w, int(sfx + sfw))
                            sby2 = min(h, int(sfy + sfh))
                            cv2.rectangle(frame, (sbx1, sby1), (sbx2, sby2), (255, 30, 220), 2)
                            for (lx, ly) in face["landmarks"]:
                                if 0 <= lx < w and 0 <= ly < h:
                                    cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)
                            slabel = f"Face {int(face['conf'] * 100)}%"
                            (slw, slh), _ = cv2.getTextSize(slabel, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                            cv2.rectangle(frame, (sbx1, max(0, sby1 - 18)), (sbx1 + slw + 8, sby1), (15, 23, 42), -1)
                            cv2.putText(frame, slabel, (sbx1 + 4, max(12, sby1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 30, 220), 1, cv2.LINE_AA)

            # 3. Render User-Drawn Sections, Entry Line, and Interactive Clickable Top Toolbar
            frame = zone_manager.render_overlay(frame)

            # 4. Bottom Status & Telemetry Ribbon
            if show_hud:
                bot_h = 32
                overlay_bot = frame.copy()
                cv2.rectangle(overlay_bot, (0, h - bot_h), (w, h), (10, 15, 26), -1)
                cv2.addWeighted(overlay_bot, 0.85, frame, 0.15, 0, frame)

                hw_tag = f"[{device.upper()}]"
                sec_badges = "  |  ".join([f"{s['name'].split()[0]}: {zone_manager.section_totals.get(s['id'], 0)}" for s in zone_manager.sections])
                sec_str = f"  |  {sec_badges}" if sec_badges else ""
                stats_text = (
                    f"{hw_tag} FPS: {fps:.1f}  |  "
                    f"Attendance: {zone_manager.attendance_total}{sec_str}  |  "
                    f"Active: {len(tracked_objects)} persons  |  "
                    f"Keys: [1] Line  [2] Sales  [3] Accounts  [R] Clear"
                )
                cv2.putText(frame, stats_text, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 225, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                break
            elif key == ord('1'):
                zone_manager.active_tool = "entry_line"
                zone_manager.status_msg = "Selected: [1. DRAW ENTRY LINE] - Click & Drag on video across entrance"
                zone_manager.status_timer = 150
                print("[*] Tool switched to: [Entry Line]")
            elif key == ord('2'):
                zone_manager.active_tool = "sales"
                zone_manager.status_msg = "Selected: [2. DRAW SALES SECTION] - Click & Drag box over Sales area"
                zone_manager.status_timer = 150
                print("[*] Tool switched to: [Sales Section]")
            elif key == ord('3'):
                zone_manager.active_tool = "accounts"
                zone_manager.status_msg = "Selected: [3. DRAW ACCOUNTS SECTION] - Click & Drag box over Accounts area"
                zone_manager.status_timer = 150
                print("[*] Tool switched to: [Accounts Section]")
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
        "channel",
        nargs="?",
        default=None,
        help="Showroom camera channel number (e.g. 101, 201, 301, 401)",
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
        default=float(config.get("confidence", 0.10)),
        help="YOLO person detection confidence threshold (default: 0.10)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(config.get("imgsz", 640)),
        help="YOLO inference input resolution (default: 640)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.get("model", "models/yolo11m.pt"),
        help="Path to YOLO weights file (default: 'models/yolo11m.pt')",
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
        default=float(config.get("face_confidence", 0.30)),
        help="Neural face detection confidence threshold (default: 0.30)",
    )
    parser.add_argument(
        "--standalone-faces",
        action="store_true",
        default=config.get("allow_standalone_faces", False),
        help="Enable detection of standalone faces behind cubicle partitions (default: False)",
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
        default=float(config.get("smoothing_alpha", 0.70)),
        help="EMA smoothing alpha factor between 0.1 and 1.0 (default: 0.70)",
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
        default=int(config.get("body_persistence", 40)),
        help="Number of missing frames before an un-detected person box disappears (default: 40)",
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=float(config.get("deadband_px", 3.5)),
        help="Pixel displacement deadband below which stationary boxes freeze (default: 3.5)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying local OpenCV window (service/background mode)",
    )

    args = parser.parse_args()

    # Resolve target channel: flag > positional > default
    target_channel = args.channel_flag or args.channel
    
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

    # Final fallbacks if no channel matched and no args given
    if not resolved_source:
        resolved_source = "0"
    if not resolved_camera_id:
        resolved_camera_id = "88970b31-d885-4fd5-a24a-bdc969c96240"

    resolved_channel = str(target_channel or "201")

    run_local_detector(
        source=resolved_source,
        camera_id=resolved_camera_id,
        api_url=args.api_url,
        line_ratio=args.line_y,
        direction=args.direction,
        confidence_thresh=args.conf,
        model_path=args.model,
        headless=args.headless,
        show_entry_line=args.show_line,
        face_scale=args.face_scale,
        face_model_path=args.face_model,
        face_confidence=args.face_conf,
        show_head_fallback=not args.no_head_fallback,
        allow_standalone_faces=args.standalone_faces,
        imgsz=args.imgsz,
        box_smoothing=not args.no_smoothing,
        smoothing_alpha=args.smoothing_alpha,
        face_persistence=args.face_persistence,
        body_persistence=args.body_persistence,
        deadband_px=args.deadband,
        camera_channel=resolved_channel,
        entry_line_cfg=entry_line_cfg,
        sections_cfg=sections_cfg,
    )


if __name__ == "__main__":
    main()

