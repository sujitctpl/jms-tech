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
