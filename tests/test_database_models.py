import uuid
import pytest
from sqlalchemy.exc import IntegrityError
from app.db.models.camera import Camera
from app.db.models.footfall_event import FootfallEvent
from app.db.models.processing_run import ProcessingRun
from app.db.models.visitor_track import VisitorTrack
from app.repositories.footfall_event_repo import FootfallEventRepository


def test_unique_constraint_visitor_track(db_session):
    cam = Camera(name="Test Cam", stream_url="rtsp://localhost/test")
    db_session.add(cam)
    db_session.commit()

    run = ProcessingRun(camera_id=cam.id, status="RUNNING")
    db_session.add(run)
    db_session.commit()

    # Track 1
    t1 = VisitorTrack(camera_id=cam.id, processing_run_id=run.id, tracker_id="T100")
    db_session.add(t1)
    db_session.commit()

    # Attempt duplicate track with same (processing_run_id, tracker_id)
    t2 = VisitorTrack(camera_id=cam.id, processing_run_id=run.id, tracker_id="T100")
    db_session.add(t2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_unique_constraint_footfall_event(db_session):
    cam = Camera(name="Test Cam", stream_url="rtsp://localhost/test")
    db_session.add(cam)
    db_session.commit()

    run = ProcessingRun(camera_id=cam.id, status="RUNNING")
    db_session.add(run)
    db_session.commit()

    track = VisitorTrack(camera_id=cam.id, processing_run_id=run.id, tracker_id="T200")
    db_session.add(track)
    db_session.commit()

    # First entry
    e1 = FootfallEvent(camera_id=cam.id, visitor_track_id=track.id, event_type="entry")
    db_session.add(e1)
    db_session.commit()

    # Duplicate entry for same visitor_track_id must fail database unique constraint
    e2 = FootfallEvent(camera_id=cam.id, visitor_track_id=track.id, event_type="entry")
    db_session.add(e2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_footfall_repo_prevents_duplicate_gracefully(db_session):
    cam = Camera(name="Test Cam", stream_url="rtsp://localhost/test")
    db_session.add(cam)
    db_session.commit()

    run = ProcessingRun(camera_id=cam.id, status="RUNNING")
    db_session.add(run)
    db_session.commit()

    track = VisitorTrack(camera_id=cam.id, processing_run_id=run.id, tracker_id="T300")
    db_session.add(track)
    db_session.commit()

    repo = FootfallEventRepository(db_session)

    # First call records entry
    ev1 = repo.record_entry(camera_id=cam.id, visitor_track_id=track.id)
    assert ev1 is not None

    # Second call returns existing event without raising unhandled exception
    ev2 = repo.record_entry(camera_id=cam.id, visitor_track_id=track.id)
    assert ev2.id == ev1.id

    # Verify count is strictly 1
    total = repo.get_total_entries(camera_id=cam.id)
    assert total == 1
