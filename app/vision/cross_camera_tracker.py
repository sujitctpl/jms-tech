import json
import os
import random
import tempfile
import time
from typing import Dict, Optional, Tuple


class CrossCameraTracker:
    """
    Maintains unified tracking identities and continuous timers across multiple camera channels.
    When a staff member or visitor moves between cameras (e.g. Camera 101 -> Camera 201):
    1. Preserves their initial global track ID (e.g. #7).
    2. Preserves their continuous arrival timer (e.g. 04:24 -> 04:30), preventing stopwatch resets to 00:00.
    Synchronizes across independent Python detector processes using atomic file-backed state.
    """

    def __init__(
        self,
        state_file: str = ".cross_camera_state.json",
        session_timeout: float = 900.0,  # 15 minutes of inactivity concludes visit/movement
        sync_interval: float = 0.5,      # Sync with disk every 500ms for sub-frame latency
    ):
        self.state_file = os.path.abspath(state_file)
        self.session_timeout = float(session_timeout)
        self.sync_interval = float(sync_interval)

        self._memory_state: Dict[str, dict] = {}
        self._last_disk_read: float = 0.0
        self._last_disk_write: float = 0.0
        self._dirty: bool = False

        # Load initial state if file exists
        self._load_from_disk(force=True)

    def _load_from_disk(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_disk_read < self.sync_interval):
            return

        self._last_disk_read = now
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
                if isinstance(disk_data, dict):
                    # Merge disk data into memory state, prioritizing newer timestamps
                    for key, val in disk_data.items():
                        if not isinstance(val, dict):
                            continue
                        mem_val = self._memory_state.get(key)
                        if not mem_val or val.get("last_seen_ts", 0) >= mem_val.get("last_seen_ts", 0):
                            self._memory_state[key] = val
        except Exception:
            # File may be momentarily locked by another camera process; non-critical
            pass

    def _save_to_disk(self, force: bool = False) -> None:
        now = time.time()
        if not self._dirty:
            return
        if not force and (now - self._last_disk_write < self.sync_interval):
            return

        self._last_disk_write = now
        self._dirty = False

        dirname = os.path.dirname(self.state_file) or "."
        os.makedirs(dirname, exist_ok=True)
        tmp_file = os.path.join(
            dirname,
            f".cross_cam_tmp_{os.getpid()}_{random.randint(10000, 99999)}.json"
        )

        try:
            # Purge entries older than 24 hours to keep file tiny (< 10 KB)
            cutoff = now - 86400.0
            clean_state = {
                k: v for k, v in self._memory_state.items()
                if v.get("last_seen_ts", 0) > cutoff
            }
            self._memory_state = clean_state

            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(clean_state, f)

            os.replace(tmp_file, self.state_file)
        except Exception:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    def has_active_visitor(self, person_id: str) -> bool:
        """Checks if person_id is an active recognized visitor who entered via an entrance camera."""
        if not person_id:
            return False
        self._load_from_disk()
        key = f"visitor:{str(person_id).strip()}"
        existing = self._memory_state.get(key)
        now_ts = time.time()
        return bool(existing and (now_ts - existing.get("last_seen_ts", 0) <= self.session_timeout))

    def get_or_register_staff(
        self,
        emp_id: str,
        emp_name: str,
        local_track_id: int,
        local_start_ts: float,
        camera_channel: str,
        now_ts: float,
    ) -> Tuple[int, float]:
        """
        Returns (global_track_id, continuous_first_seen_ts) for a staff member.
        Carries over their original track ID and continuous timer from the first camera.
        """
        self._load_from_disk()

        key = f"staff:{str(emp_id).strip()}"
        existing = self._memory_state.get(key)

        if existing and (now_ts - existing.get("last_seen_ts", 0) <= self.session_timeout):
            # Ongoing active session across cameras: preserve original track ID and start time!
            existing["last_seen_ts"] = now_ts
            existing["last_camera"] = str(camera_channel)
            self._dirty = True
            self._save_to_disk()
            return int(existing.get("global_track_id", local_track_id)), float(existing.get("first_seen_ts", local_start_ts))

        # First observation today or returned after extended absence
        new_entry = {
            "entity_type": "staff",
            "entity_id": str(emp_id),
            "display_name": str(emp_name),
            "global_track_id": int(local_track_id),
            "first_seen_ts": float(local_start_ts),
            "last_seen_ts": float(now_ts),
            "first_camera": str(camera_channel),
            "last_camera": str(camera_channel),
        }
        self._memory_state[key] = new_entry
        self._dirty = True
        self._save_to_disk(force=True)

        return int(local_track_id), float(local_start_ts)

    def get_or_register_visitor(
        self,
        person_id: str,
        local_track_id: int,
        local_start_ts: float,
        camera_channel: str,
        now_ts: float,
    ) -> Tuple[int, float]:
        """
        Returns (global_track_id, continuous_first_seen_ts) for a recognized visitor.
        Carries over their original track ID and continuous visit timer from the entrance camera.
        """
        self._load_from_disk()

        key = f"visitor:{str(person_id).strip()}"
        existing = self._memory_state.get(key)

        if existing and (now_ts - existing.get("last_seen_ts", 0) <= self.session_timeout):
            # Ongoing active visit across cameras
            existing["last_seen_ts"] = now_ts
            existing["last_camera"] = str(camera_channel)
            self._dirty = True
            self._save_to_disk()
            return int(existing.get("global_track_id", local_track_id)), float(existing.get("first_seen_ts", local_start_ts))

        # First observation on entry camera
        new_entry = {
            "entity_type": "visitor",
            "entity_id": str(person_id),
            "global_track_id": int(local_track_id),
            "first_seen_ts": float(local_start_ts),
            "last_seen_ts": float(now_ts),
            "first_camera": str(camera_channel),
            "last_camera": str(camera_channel),
            "waiting_seconds": 0.0,
            "waiting_breach": False,
        }
        self._memory_state[key] = new_entry
        self._dirty = True
        self._save_to_disk(force=True)

        return int(local_track_id), float(local_start_ts)

    def update_visitor_waiting(
        self,
        person_id: str,
        waiting_seconds: float,
        waiting_breach: bool,
        now_ts: float,
    ) -> None:
        """Persists accumulated waiting metrics atomically across cameras."""
        if not person_id:
            return
        self._load_from_disk()
        key = f"visitor:{str(person_id).strip()}"
        existing = self._memory_state.get(key)
        if existing:
            prev_w = float(existing.get("waiting_seconds", 0.0))
            existing["waiting_seconds"] = max(prev_w, float(waiting_seconds))
            existing["waiting_breach"] = bool(existing.get("waiting_breach", False) or waiting_breach)
            existing["last_seen_ts"] = float(now_ts)
            self._dirty = True
            self._save_to_disk()

    def get_visitor_waiting(self, person_id: str) -> Tuple[float, bool]:
        """Retrieves cross-camera continuous waiting metrics for a visitor."""
        if not person_id:
            return (0.0, False)
        self._load_from_disk()
        key = f"visitor:{str(person_id).strip()}"
        existing = self._memory_state.get(key)
        if existing:
            return (float(existing.get("waiting_seconds", 0.0)), bool(existing.get("waiting_breach", False)))
        return (0.0, False)
