import os
import tempfile
import time
import pytest
from app.vision.cross_camera_tracker import CrossCameraTracker


def test_staff_cross_camera_maintains_id_and_timer():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "test_cross_cam.json")
        cam1 = CrossCameraTracker(state_file=state_file, session_timeout=900.0, sync_interval=0.0)
        cam2 = CrossCameraTracker(state_file=state_file, session_timeout=900.0, sync_interval=0.0)

        t0 = time.time() - 264.0  # 4 minutes and 24 seconds ago (04:24)

        # 1. Camera 101 sees Kunal as track #7 at t0
        id_1, start_1 = cam1.get_or_register_staff(
            emp_id="EMP-101",
            emp_name="Kunal",
            local_track_id=7,
            local_start_ts=t0,
            camera_channel="101",
            now_ts=time.time(),
        )
        assert id_1 == 7
        assert abs(start_1 - t0) < 0.01

        # 2. Kunal walks to Camera 201 (assigned local track #46 by ByteTrack)
        t_cam2 = time.time()
        id_2, start_2 = cam2.get_or_register_staff(
            emp_id="EMP-101",
            emp_name="Kunal",
            local_track_id=46,
            local_start_ts=t_cam2,
            camera_channel="201",
            now_ts=t_cam2,
        )

        # Verified: Camera 2 maintains track ID #7 and continuous start time t0!
        assert id_2 == 7  # Maintained from Camera 1!
        assert abs(start_2 - t0) < 0.01  # Timer continues without resetting to 0!


def test_visitor_cross_camera_maintains_id_and_timer():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "test_cross_cam.json")
        cam1 = CrossCameraTracker(state_file=state_file, session_timeout=900.0, sync_interval=0.0)
        cam2 = CrossCameraTracker(state_file=state_file, session_timeout=900.0, sync_interval=0.0)

        t0 = time.time() - 120.0  # 2 minutes ago

        # 1. Camera 101 sees visitor VIS-2026-1041 as track #12
        id_1, start_1 = cam1.get_or_register_visitor(
            person_id="VIS-2026-1041",
            local_track_id=12,
            local_start_ts=t0,
            camera_channel="101",
            now_ts=time.time(),
        )
        assert id_1 == 12
        assert abs(start_1 - t0) < 0.01

        # 2. Visitor walks to Camera 201 (local ByteTrack gives #88)
        t_cam2 = time.time()
        id_2, start_2 = cam2.get_or_register_visitor(
            person_id="VIS-2026-1041",
            local_track_id=88,
            local_start_ts=t_cam2,
            camera_channel="201",
            now_ts=t_cam2,
        )

        # Verified: Camera 2 maintains track ID #12 and continuous start time t0!
        assert id_2 == 12
        assert abs(start_2 - t0) < 0.01
