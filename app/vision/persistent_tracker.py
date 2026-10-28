import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PersistentTrack:
    track_id: int
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    last_seen: float
    first_seen: float = 0.0

    @property
    def face_box(self) -> Tuple[int, int, int, int]:
        """Calculates face/head box in top 22% of bounding box."""
        x1, y1, x2, y2 = map(int, self.box)
        h = y2 - y1
        w = x2 - x1
        face_y2 = y1 + int(h * 0.22)
        face_x1 = x1 + int(w * 0.2)
        face_x2 = x2 - int(w * 0.2)
        return (face_x1, y1, face_x2, face_y2)


class PersistentCameraTracker:
    """
    Maintains detected persons on screen with a configurable Time-To-Live (TTL).
    If a person moves, the bounding box smoothly follows them.
    If the person is briefly occluded or not detected, the box remains visible for ttl_seconds (default 5.0s).
    """

    def __init__(self, ttl_seconds: float = 5.0, iou_thresh: float = 0.15):
        self.ttl_seconds = ttl_seconds
        self.iou_thresh = iou_thresh
        self.tracks: Dict[int, PersistentTrack] = {}
        self._next_id = 1

    def _iou_and_dist(
        self, b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float]:
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        union = area1 + area2 - inter
        iou = inter / union if union > 0 else 0.0

        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
        c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
        dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        diag = math.hypot(b1[2] - b1[0], b1[3] - b1[1])
        return iou, dist, diag

    def update(
        self, detections: List[Tuple[Tuple[float, float, float, float], float]]
    ) -> List[PersistentTrack]:
        """
        Updates persistent tracks with new frame detections.
        detections: list of ((x1, y1, x2, y2), confidence)
        """
        now = time.time()
        matched_ids = set()

        for det_box, conf in detections:
            best_id = None
            best_score = float("inf")

            for tid, trk in self.tracks.items():
                if tid in matched_ids:
                    continue
                iou, dist, diag = self._iou_and_dist(det_box, trk.box)
                # Match if boxes overlap or centers are close relative to bounding box scale
                if iou > self.iou_thresh or dist < (diag * 0.85):
                    # Prefer high IoU and low Euclidean distance
                    score = dist - (iou * 250.0)
                    if score < best_score:
                        best_score = score
                        best_id = tid

            if best_id is not None:
                # Smoothly update position (70% new detection, 30% previous) to eliminate flicker
                old = self.tracks[best_id].box
                self.tracks[best_id].box = (
                    0.7 * det_box[0] + 0.3 * old[0],
                    0.7 * det_box[1] + 0.3 * old[1],
                    0.7 * det_box[2] + 0.3 * old[2],
                    0.7 * det_box[3] + 0.3 * old[3],
                )
                self.tracks[best_id].confidence = conf
                self.tracks[best_id].last_seen = now
                matched_ids.add(best_id)
            else:
                new_id = self._next_id
                self._next_id += 1
                self.tracks[new_id] = PersistentTrack(
                    track_id=new_id,
                    box=det_box,
                    confidence=conf,
                    last_seen=now,
                    first_seen=now,
                )
                matched_ids.add(new_id)

        # Expire tracks that have not been detected for longer than ttl_seconds (5 seconds)
        active_tracks: List[PersistentTrack] = []
        for tid in list(self.tracks.keys()):
            if (now - self.tracks[tid].last_seen) > self.ttl_seconds:
                del self.tracks[tid]
            else:
                active_tracks.append(self.tracks[tid])

        return active_tracks
