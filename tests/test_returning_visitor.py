import base64
from datetime import datetime, timezone, timedelta
import cv2
import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.db.models.person_identity import PersonIdentity
from app.schemas.visitor import VisitorDetectRequest
from app.services.visitor_service import VisitorService, IST


def generate_face_b64(pattern: int = 42) -> str:
    np.random.seed(pattern)
    # Generate 64x64 valid 3-channel face crop
    img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def generate_embedding(seed: int = 42) -> list:
    np.random.seed(seed)
    vec = np.random.randn(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def test_new_visitor_registration(db_session: Session):
    service = VisitorService(db_session)
    emb = generate_embedding(101)

    req = VisitorDetectRequest(
        camera_name="Entrance Camera",
        face_image_base64=generate_face_b64(101),
        face_embedding=emb,
        zone="Entrance",
    )

    resp = service.process_detection(req)
    assert resp.status == "VISITOR_CREATED"
    assert resp.is_new is True
    assert resp.visitor.visitorType == "New"
    assert resp.visitor.exitTime is None


def test_same_day_active_visit_reidentified_not_duplicated(db_session: Session):
    service = VisitorService(db_session)
    emb = generate_embedding(202)

    req1 = VisitorDetectRequest(
        camera_name="Entrance Camera",
        face_image_base64=generate_face_b64(202),
        face_embedding=emb,
        zone="Entrance",
    )
    resp1 = service.process_detection(req1)
    assert resp1.status == "VISITOR_CREATED"
    first_pid = resp1.person_id

    # Second observation 5 minutes later in Sales Lounge
    req2 = VisitorDetectRequest(
        camera_name="Sales Lounge Camera",
        face_image_base64=generate_face_b64(202),
        face_embedding=emb,
        zone="Sales Lounge",
    )
    resp2 = service.process_detection(req2)
    assert resp2.status == "VISITOR_REIDENTIFIED"
    assert resp2.person_id == first_pid
    assert resp2.is_new is False
    assert resp2.visitor.visitorType == "New"


def test_returning_visitor_different_calendar_day(db_session: Session):
    service = VisitorService(db_session)
    emb = generate_embedding(303)

    # 1. First visit was 4 days ago
    past_date = datetime.now(timezone.utc) - timedelta(days=4)
    past_record = PersonIdentity(
        person_id="VIS-2026-PAST01",
        face_embedding=emb,
        is_expired=False,
        created_at=past_date,
        last_seen_at=past_date + timedelta(minutes=25),
        visitor_type="New",
        exit_time=None,  # Exit time was never explicitly written
    )
    db_session.add(past_record)
    db_session.commit()

    # 2. Customer returns TODAY
    req_today = VisitorDetectRequest(
        camera_name="Entrance Camera",
        face_image_base64=generate_face_b64(303),
        face_embedding=emb,
        zone="Entrance",
    )
    resp_today = service.process_detection(req_today)

    assert resp_today.status == "VISITOR_CREATED"
    assert resp_today.is_new is False  # Returning customer
    assert resp_today.visitor.visitorType == "Returning"
    assert resp_today.person_id != "VIS-2026-PAST01"  # Minted fresh visit session for today

    # Verify the past visit's exit_time was closed gracefully
    db_session.refresh(past_record)
    assert past_record.exit_time is not None


def test_returning_visitor_movement_maintains_today_session(db_session: Session):
    service = VisitorService(db_session)
    emb = generate_embedding(404)

    # 1. Past visit 3 days ago
    past_date = datetime.now(timezone.utc) - timedelta(days=3)
    past_record = PersonIdentity(
        person_id="VIS-2026-PAST02",
        face_embedding=emb,
        is_expired=False,
        created_at=past_date,
        last_seen_at=past_date + timedelta(minutes=20),
        visitor_type="New",
        exit_time="04:30 PM",
    )
    db_session.add(past_record)
    db_session.commit()

    # 2. Customer enters today as Returning Visitor
    req_entry = VisitorDetectRequest(
        camera_name="Entrance Camera",
        face_image_base64=generate_face_b64(404),
        face_embedding=emb,
        zone="Entrance",
    )
    resp_entry = service.process_detection(req_entry)
    assert resp_entry.visitor.visitorType == "Returning"
    today_pid = resp_entry.person_id

    # 3. Customer walks into Display Area 10 minutes later
    req_display = VisitorDetectRequest(
        camera_name="Display Camera",
        face_image_base64=generate_face_b64(404),
        face_embedding=emb,
        zone="Display Area",
    )
    resp_display = service.process_detection(req_display)
    assert resp_display.status == "VISITOR_REIDENTIFIED"
    assert resp_display.person_id == today_pid
    assert resp_display.visitor.visitorType == "Returning"


def test_same_day_return_after_exit(db_session: Session):
    service = VisitorService(db_session)
    emb = generate_embedding(505)

    # Morning visit
    req1 = VisitorDetectRequest(
        camera_name="Entrance Camera",
        face_image_base64=generate_face_b64(505),
        face_embedding=emb,
        zone="Entrance",
    )
    resp1 = service.process_detection(req1)
    morning_pid = resp1.person_id

    # Customer exits at 11:30 AM
    morning_rec = service.repo.get_by_person_id(morning_pid)
    morning_rec.exit_time = "11:30 AM"
    db_session.add(morning_rec)
    db_session.commit()

    # Customer returns in the afternoon at 3:00 PM
    req2 = VisitorDetectRequest(
        camera_name="Entrance Camera",
        face_image_base64=generate_face_b64(505),
        face_embedding=emb,
        zone="Entrance",
    )
    resp2 = service.process_detection(req2)
    # Same-day re-entry resumes the SAME record: same ID, a second entry/exit segment.
    assert resp2.status == "VISITOR_REIDENTIFIED"
    assert resp2.is_new is False
    assert resp2.person_id == morning_pid
    assert resp2.visitor.exitTime is None  # inside again
    log = resp2.visitor.visitLog
    assert len(log) == 2
    assert log[0].exitTime is not None
    assert log[1].exitTime is None
    assert "Re-entry #1" in resp2.visitor.notes


def test_same_day_reentry_sums_segments_and_logs_second_exit(db_session: Session):
    from app.schemas.visitor import VisitorZoneUpdateRequest

    service = VisitorService(db_session)
    emb = generate_embedding(606)
    resp1 = service.process_detection(VisitorDetectRequest(
        camera_name="Entrance Camera",
        face_image_base64=generate_face_b64(606),
        face_embedding=emb,
        zone="Entrance",
    ))
    pid = resp1.person_id

    # Rewrite history: first visit 40 min (4:00-4:40 pm IST), departed; last seen 4:40 pm.
    rec = service.repo.get_by_person_id(pid)
    today = datetime.now(IST).replace(hour=16, minute=0, second=0, microsecond=0)
    rec.created_at = today.astimezone(timezone.utc)
    rec.last_seen_at = (today + timedelta(minutes=40)).astimezone(timezone.utc)
    rec.exit_time = "04:40 PM"
    rec.visit_log = None
    db_session.add(rec)
    db_session.commit()

    resp2 = service.process_detection(VisitorDetectRequest(
        camera_name="Entrance Camera",
        face_image_base64=generate_face_b64(606),
        face_embedding=emb,
        zone="Entrance",
    ))
    assert resp2.person_id == pid
    assert len(resp2.visitor.visitLog) == 2
    assert resp2.visitor.visitLog[0].entryTime == "04:00:00 PM"
    assert resp2.visitor.visitLog[0].exitTime == "04:40:00 PM"

    # Second exit is appended to the second segment, and the visitor is departed again.
    service.update_visitor_zone(VisitorZoneUpdateRequest(person_id=pid, exit_time="05:10 PM"))
    rec = service.repo.get_by_person_id(pid)
    assert rec.exit_time == "05:10 PM"
    resp3 = service._to_visitor_response(rec)
    assert len(resp3.visitLog) == 2
    assert resp3.visitLog[1].exitTime is not None
    # Total = first segment (40 min) + a short second one; the idle gap is not counted.
    assert 40 <= resp3.durationMinutes < 50
