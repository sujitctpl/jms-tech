from typing import Any, Dict, List, Optional, Set, Tuple
from app.core.logging import logger
from app.vision.tracker import TrackedPerson


class EntryLineCounter:
    """
    Virtual entry-line crossing detector.
    Tracks person movements across video frames and identifies valid entry crossings
    in the configured direction while strictly preventing duplicate counts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.p1: Optional[Tuple[float, float]] = None
        self.p2: Optional[Tuple[float, float]] = None
        self.direction: str = "top_to_bottom"
        self._parse_config(self.config)

        # Track history: tracker_id -> previous reference point (x, y)
        self.track_positions: Dict[str, Tuple[float, float]] = {}
        # Duplicate prevention set: tracker_ids that have already triggered an entry event
        self.counted_track_ids: Set[str] = set()
        # Duplicate prevention set for persistent person identities across tracking sessions
        self.counted_person_ids: Set[str] = set()

    def update_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._parse_config(config)

    def _parse_config(self, config: Dict[str, Any]) -> None:
        if not config:
            return

        line_coords = config.get("line")
        if line_coords and len(line_coords) == 2:
            self.p1 = (float(line_coords[0][0]), float(line_coords[0][1]))
            self.p2 = (float(line_coords[1][0]), float(line_coords[1][1]))
        elif "pt1" in config and "pt2" in config:
            self.p1 = (float(config["pt1"][0]), float(config["pt1"][1]))
            self.p2 = (float(config["pt2"][0]), float(config["pt2"][1]))

        self.direction = config.get("direction", "top_to_bottom").lower()

    @property
    def is_configured(self) -> bool:
        return self.p1 is not None and self.p2 is not None

    def process_frame(
        self, tracked_persons: List[TrackedPerson]
    ) -> List[TrackedPerson]:
        """
        Evaluates tracked persons against the entry line.
        Returns a list of TrackedPerson instances that just crossed the entry line
        in the configured direction for the FIRST time.
        """
        if not self.is_configured:
            return []

        new_entries: List[TrackedPerson] = []
        current_frame_track_ids: Set[str] = set()

        for person in tracked_persons:
            tid = str(person.tracker_id)
            current_frame_track_ids.add(tid)
            curr_point = person.bottom_center

            # If already counted, never count again (duplicate prevention)
            if tid in self.counted_track_ids:
                self.track_positions[tid] = curr_point
                continue

            # If persistent person identity has already been counted, link and suppress duplicate
            pid = getattr(person, "person_id", None)
            if pid and pid in self.counted_person_ids:
                self.counted_track_ids.add(tid)
                self.track_positions[tid] = curr_point
                continue

            prev_point = self.track_positions.get(tid)
            if prev_point is not None:
                if self._check_crossing(prev_point, curr_point):
                    logger.info(
                        "Valid entry line crossing detected for track ID %s (person: %s) from %s to %s",
                        tid,
                        pid or "unresolved",
                        prev_point,
                        curr_point,
                    )
                    self.counted_track_ids.add(tid)
                    if pid:
                        self.counted_person_ids.add(pid)
                    new_entries.append(person)

            # Update tracked position for next frame
            self.track_positions[tid] = curr_point

        # Clean up stale tracks not seen in recent frames to save memory
        stale_threshold = 300  # if position dictionary grows too large
        if len(self.track_positions) > stale_threshold:
            # retain tracks seen in this frame or recently counted
            for tid in list(self.track_positions.keys()):
                if tid not in current_frame_track_ids and tid not in self.counted_track_ids:
                    del self.track_positions[tid]

        return new_entries

    def _check_crossing(
        self, p_prev: Tuple[float, float], p_curr: Tuple[float, float]
    ) -> bool:
        """
        Tests if trajectory from p_prev to p_curr intersects line segment (p1, p2)
        in the designated direction.
        """
        if self.p1 is None or self.p2 is None:
            return False

        if not self._segments_intersect(self.p1, self.p2, p_prev, p_curr):
            return False

        # Direction verification
        return self._matches_direction(p_prev, p_curr)

    def _matches_direction(
        self, p_prev: Tuple[float, float], p_curr: Tuple[float, float]
    ) -> bool:
        """Validates that movement from p_prev to p_curr matches configured direction."""
        dir_name = self.direction
        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]

        # Geometric cross product with respect to entry line vector (p1 -> p2)
        cp_prev = self._cross_product(self.p1, self.p2, p_prev)
        cp_curr = self._cross_product(self.p1, self.p2, p_curr)

        if dir_name == "left_to_right":
            return dx > 0
        elif dir_name == "right_to_left":
            return dx < 0
        elif dir_name == "top_to_bottom":
            return dy > 0
        elif dir_name == "bottom_to_top":
            return dy < 0
        elif dir_name in ("positive_to_negative", "inward"):
            return cp_prev >= 0 and cp_curr < 0
        elif dir_name in ("negative_to_positive", "outward"):
            return cp_prev <= 0 and cp_curr > 0

        # Default fallback: any crossing of segment
        return True

    @staticmethod
    def _cross_product(
        a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
    ) -> float:
        """Returns 2D cross product of vector AB and AC."""
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @classmethod
    def _segments_intersect(
        cls,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
    ) -> bool:
        """Checks if line segment p1-p2 and p3-p4 intersect."""
        d1 = cls._cross_product(p3, p4, p1)
        d2 = cls._cross_product(p3, p4, p2)
        d3 = cls._cross_product(p1, p2, p3)
        d4 = cls._cross_product(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
            (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
        ):
            return True

        # Collinear checks
        if d1 == 0 and cls._on_segment(p3, p4, p1):
            return True
        if d2 == 0 and cls._on_segment(p3, p4, p2):
            return True
        if d3 == 0 and cls._on_segment(p1, p2, p3):
            return True
        if d4 == 0 and cls._on_segment(p1, p2, p4):
            return True

        return False

    @staticmethod
    def _on_segment(
        pi: Tuple[float, float], pj: Tuple[float, float], pk: Tuple[float, float]
    ) -> bool:
        """Checks if point pk is on line segment pi-pj."""
        return (
            min(pi[0], pj[0]) <= pk[0] <= max(pi[0], pj[0])
            and min(pi[1], pj[1]) <= pk[1] <= max(pi[1], pj[1])
        )

    def reset(self) -> None:
        """Resets tracker state."""
        self.track_positions.clear()
        self.counted_track_ids.clear()
