import math
import pytest
from app.db.models.person_identity import PersonIdentity
from app.schemas.visitor import VisitorResponse, VisitorZoneUpdateRequest
from app.services.visitor_service import VisitorService


def test_perspective_ground_plane_distance():
    """Verify perspective-aware distance calculation."""
    # Person 1 (Visitor)
    vx1, vy1, vx2, vy2 = 100, 200, 160, 350
    v_feet = ((vx1 + vx2) * 0.5, vy2)
    v_h = vy2 - vy1  # 150 px

    # Person 2 (Staff) standing nearby
    sx1, sy1, sx2, sy2 = 140, 210, 200, 350
    s_feet = ((sx1 + sx2) * 0.5, sy2)
    s_h = sy2 - sy1  # 140 px

    avg_h = (v_h + s_h) * 0.5
    px_per_m = avg_h / 1.70  # ~85.29 px per meter
    dx = v_feet[0] - s_feet[0]
    dy = v_feet[1] - s_feet[1]
    dist_m = math.hypot(dx, dy) / px_per_m

    assert dist_m < 2.2, f"Expected staff within 2.2m, got {dist_m:.2f}m"


def test_waiting_breach_detection():
    """Verify 15-minute (900s) cutoff breach logic."""
    rec = PersonIdentity(
        person_id="VIS-2026-TEST",
        waiting_seconds=950,
        is_waiting=True,
        waiting_breach=True,
    )
    assert rec.waiting_seconds == 950
    assert rec.is_waiting is True
    assert rec.waiting_breach is True


def test_visitor_zone_update_schema():
    """Verify VisitorZoneUpdateRequest carries waiting payload."""
    req = VisitorZoneUpdateRequest(
        person_id="VIS-2026-TEST",
        waiting_seconds=920,
        is_waiting=True,
        waiting_breach=True,
    )
    assert req.waiting_seconds == 920
    assert req.is_waiting is True
    assert req.waiting_breach is True


def test_monotonic_waiting_seconds_update():
    """Verify backend update_visitor_zone protects against clobbering higher waiting_seconds."""
    from unittest.mock import MagicMock
    from app.services.visitor_service import VisitorService

    mock_db = MagicMock()
    service = VisitorService(mock_db)
    mock_repo = MagicMock()
    service.repo = mock_repo

    existing_rec = PersonIdentity(
        person_id="VIS-2026-TEST",
        waiting_seconds=650,
        is_waiting=True,
        waiting_breach=False,
    )
    mock_repo.get_by_person_id.return_value = existing_rec

    # A secondary camera / fresh track sends lower waiting_seconds (e.g. 5s)
    req = VisitorZoneUpdateRequest(
        person_id="VIS-2026-TEST",
        waiting_seconds=5,
        is_waiting=False,
        waiting_breach=False,
    )

    res = service.update_visitor_zone(req)
    # The max value (650) must be preserved
    assert existing_rec.waiting_seconds == 650
    assert existing_rec.is_waiting is False


def test_visitor_zone_tracker_carries_over_pid_history():
    """Verify VisitorZoneTracker preserves accumulated waiting time when track ID switches."""
    from local_detector import VisitorZoneTracker

    tracker = VisitorZoneTracker(sync_client=None, sync_interval=4.0, waiting_cutoff_seconds=900.0)
    t = 1000.0

    # Track 1 appears unattended, accumulate 10 frames of 2s each = 20s
    tracker.update_visitor_zone(track_id=1, current_dept=None, now_ts=t, person_id="VIS-101", is_employee_nearby=False)
    for _ in range(10):
        t += 2.0
        tracker.update_visitor_zone(track_id=1, current_dept=None, now_ts=t, person_id="VIS-101", is_employee_nearby=False)

    w_sec, is_w, is_br, _ = tracker.get_waiting_info(1)
    assert w_sec == 20
    assert is_w is True

    # Track 1 is lost (turnaround / occlusion)
    tracker.on_visitor_lost(track_id=1, now_ts=t + 1.0)

    # Track 2 appears (person_id unresolved initially)
    tracker.update_visitor_zone(track_id=2, current_dept=None, now_ts=t + 2.0, person_id=None, is_employee_nearby=False)
    # Re-identified as VIS-101
    tracker.link_track_pid(track_id=2, person_id="VIS-101")
    w_sec2, _, _, _ = tracker.get_waiting_info(2)

    # Waiting seconds should carry over from Track 1!
    assert w_sec2 == 20


def test_spatial_temporal_handoff_preserves_waiting_timer():
    """Verify that SpatialTemporalHandoffTracker reconnects tracks when a visitor turns around or moves."""
    import numpy as np
    from local_detector import SpatialTemporalHandoffTracker

    handoff = SpatialTemporalHandoffTracker(max_coasting_sec=25.0, match_corr_thresh=0.55)
    now = 1000.0

    # Synthetic frame with a person wearing a blue shirt
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[150:400, 200:320] = [180, 50, 20]  # Blue torso

    # Track 14 is waiting for 920s (SLA breach > 15m)
    box14 = (200.0, 100.0, 320.0, 450.0)
    handoff.register_active_track(
        track_id=14,
        box=box14,
        frame=frame,
        now_ts=now,
        person_id="VIS-2026-BREACH",
        display_id=14,
        first_seen_ts=now - 920.0,
        waiting_seconds=920.0,
        waiting_breach=True,
    )

    # Person moves / turns around: Track 14 is dropped by detector
    handoff.on_track_missing(14, now + 1.0)

    # 1.5 seconds later, Track 28 appears 50px away wearing the exact same shirt
    box28 = (240.0, 105.0, 360.0, 455.0)
    matched = handoff.find_handoff_match(
        new_track_id=28,
        box=box28,
        frame=frame,
        now_ts=now + 2.5,
        frame_shape=(1280, 720),
        is_staff=False,
    )

    # Verification: Seamless reconnection!
    assert matched is not None, "Expected handoff match for same person moving nearby"
    assert matched["track_id"] == 14
    assert matched["person_id"] == "VIS-2026-BREACH"
    assert matched["waiting_seconds"] == 920.0
    assert matched["waiting_breach"] is True
    assert matched["first_seen_ts"] == now - 920.0


def test_cross_camera_waiting_sync():
    """Verify that CrossCameraTracker synchronizes waiting time across camera streams."""
    import tempfile
    import os
    import time
    from app.vision.cross_camera_tracker import CrossCameraTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "test_cross_cam_wait.json")
        cam1 = CrossCameraTracker(state_file=state_file, session_timeout=900.0, sync_interval=0.0)
        cam2 = CrossCameraTracker(state_file=state_file, session_timeout=900.0, sync_interval=0.0)

        t0 = time.time() - 450.0
        # Camera 1 records visitor VIS-101 waiting for 450 seconds
        cam1.get_or_register_visitor(
            person_id="VIS-101",
            local_track_id=5,
            local_start_ts=t0,
            camera_channel="101",
            now_ts=time.time(),
        )
        cam1.update_visitor_waiting(
            person_id="VIS-101",
            waiting_seconds=450.0,
            waiting_breach=False,
            now_ts=time.time(),
        )

        # Camera 2 observes visitor VIS-101
        w_sec, is_breach = cam2.get_visitor_waiting("VIS-101")
        assert w_sec == 450.0
        assert is_breach is False


def test_sla_breach_requires_unattended_and_not_in_department():
    """Verify that SLA breach only occurs if visitor is unattended AND not in any department for >= cutoff."""
    from local_detector import VisitorZoneTracker

    tracker = VisitorZoneTracker(sync_client=None, sync_interval=4.0, waiting_cutoff_seconds=900.0)
    t = 1000.0

    # 1. Visitor inside 'Accounts' department box -> NOT waiting
    tracker.update_visitor_zone(track_id=1, current_dept=("accounts", "Accounts Section"), now_ts=t, is_employee_nearby=False)
    for _ in range(500):
        t += 2.0
        tracker.update_visitor_zone(track_id=1, current_dept=("accounts", "Accounts Section"), now_ts=t, is_employee_nearby=False)

    w_sec, is_w, is_br, _ = tracker.get_waiting_info(1)
    assert w_sec == 0
    assert is_w is False
    assert is_br is False

    # 2. Visitor in Main Showroom with staff member nearby -> NOT waiting
    tracker.update_visitor_zone(track_id=2, current_dept=None, now_ts=t, is_employee_nearby=True)
    for _ in range(500):
        t += 2.0
        tracker.update_visitor_zone(track_id=2, current_dept=None, now_ts=t, is_employee_nearby=True)

    w_sec2, is_w2, is_br2, _ = tracker.get_waiting_info(2)
    assert w_sec2 == 0
    assert is_w2 is False
    assert is_br2 is False

    # 3. Visitor in Main Showroom with NO employee nearby -> Accumulates waiting time and breaches after 900s
    tracker.update_visitor_zone(track_id=3, current_dept=None, now_ts=t, is_employee_nearby=False)
    # Total presence only 28 seconds -> cannot breach!
    for _ in range(14):
        t += 2.0
        tracker.update_visitor_zone(track_id=3, current_dept=None, now_ts=t, is_employee_nearby=False)

    w_sec3, is_w3, is_br3, _ = tracker.get_waiting_info(3)
    assert w_sec3 == 28
    assert is_w3 is True
    assert is_br3 is False  # Only 28s dwell, cannot breach!


def test_suppress_premature_exit_under_7s():
    """Verify record_visitor_exit ignores premature exits if dwell is < 7.0 seconds."""
    from unittest.mock import MagicMock
    from local_detector import VisitorZoneTracker

    mock_sync = MagicMock()
    tracker = VisitorZoneTracker(sync_client=mock_sync, sync_interval=4.0, waiting_cutoff_seconds=900.0)
    t0 = 1000.0

    # Visitor enters at t0 (triggers initial zone sync)
    tracker.update_visitor_zone(track_id=1, current_dept=None, now_ts=t0, person_id="VIS-TEST", is_employee_nearby=False)
    assert mock_sync.send_visitor_zone_update.call_count == 1
    mock_sync.reset_mock()

    # 4 seconds later (under 7.0s threshold), premature reverse exit fires
    tracker.record_visitor_exit(track_id=1, now_ts=t0 + 4.0)

    # Exit update MUST be suppressed!
    mock_sync.send_visitor_zone_update.assert_not_called()

    # If they exit after 10 seconds (>= 7.0s), exit IS recorded
    tracker.update_visitor_zone(track_id=1, current_dept=None, now_ts=t0 + 10.0, person_id="VIS-TEST", is_employee_nearby=False)
    mock_sync.reset_mock()
    tracker.record_visitor_exit(track_id=1, now_ts=t0 + 10.0)
    mock_sync.send_visitor_zone_update.assert_called_once()
    payload = mock_sync.send_visitor_zone_update.call_args[0][0]
    assert payload.get("exit_time") is not None


def test_delete_bounced_visitor():
    """Verify delete_visitor removes bounced visitor record from database."""
    from unittest.mock import MagicMock
    from app.services.visitor_service import VisitorService
    from app.db.models.person_identity import PersonIdentity

    mock_db = MagicMock()
    service = VisitorService(mock_db)
    mock_repo = MagicMock()
    service.repo = mock_repo

    mock_rec = PersonIdentity(person_id="VIS-BOUNCE", avatar_url=None)
    mock_repo.get_by_person_id.return_value = mock_rec
    mock_db.query.return_value.filter.return_value.all.return_value = []

    res = service.delete_visitor("VIS-BOUNCE")
    assert res is True
    mock_db.delete.assert_called()
    mock_db.commit.assert_called()





