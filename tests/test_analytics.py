import uuid
from datetime import datetime, timezone, timedelta
from fastapi import status
from app.api.routes import DAILY_FOOTFALL_ANALYTICS, FOOTFALL_ANALYTICS
from app.db.models.camera import Camera
from app.db.models.footfall_event import FootfallEvent
from app.db.models.processing_run import ProcessingRun
from app.db.models.visitor_track import VisitorTrack


def test_footfall_total_and_filtering(client, db_session):
    cam1 = Camera(name="Cam 1", stream_url="rtsp://localhost/1")
    cam2 = Camera(name="Cam 2", stream_url="rtsp://localhost/2")
    db_session.add_all([cam1, cam2])
    db_session.commit()

    run1 = ProcessingRun(camera_id=cam1.id, status="RUNNING")
    run2 = ProcessingRun(camera_id=cam2.id, status="RUNNING")
    db_session.add_all([run1, run2])
    db_session.commit()

    now = datetime.now(timezone.utc)
    t1 = VisitorTrack(camera_id=cam1.id, processing_run_id=run1.id, tracker_id="1")
    t2 = VisitorTrack(camera_id=cam1.id, processing_run_id=run1.id, tracker_id="2")
    t3 = VisitorTrack(camera_id=cam2.id, processing_run_id=run2.id, tracker_id="3")
    db_session.add_all([t1, t2, t3])
    db_session.commit()

    e1 = FootfallEvent(camera_id=cam1.id, visitor_track_id=t1.id, event_type="entry", occurred_at=now)
    e2 = FootfallEvent(camera_id=cam1.id, visitor_track_id=t2.id, event_type="entry", occurred_at=now)
    e3 = FootfallEvent(camera_id=cam2.id, visitor_track_id=t3.id, event_type="entry", occurred_at=now)
    db_session.add_all([e1, e2, e3])
    db_session.commit()

    # Total across all cameras
    resp_all = client.get(FOOTFALL_ANALYTICS)
    assert resp_all.status_code == status.HTTP_200_OK
    assert resp_all.json()["total_entries"] == 3

    # Total filtered by cam1
    resp_cam1 = client.get(f"{FOOTFALL_ANALYTICS}?camera_id={cam1.id}")
    assert resp_cam1.status_code == status.HTTP_200_OK
    assert resp_cam1.json()["total_entries"] == 2
    assert resp_cam1.json()["camera_id"] == str(cam1.id)

    # Total filtered by cam2
    resp_cam2 = client.get(f"{FOOTFALL_ANALYTICS}?camera_id={cam2.id}")
    assert resp_cam2.status_code == status.HTTP_200_OK
    assert resp_cam2.json()["total_entries"] == 1


def test_footfall_date_range_validation(client):
    # start_time > end_time should return 400 Bad Request
    start = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    end = datetime.now(timezone.utc).isoformat()

    resp = client.get(FOOTFALL_ANALYTICS, params={"start_time": start, "end_time": end})
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert "start_time must be less than or equal to end_time" in resp.json()["detail"]

    # Daily date range validation
    resp_daily = client.get(f"{DAILY_FOOTFALL_ANALYTICS}?start_date=2026-10-25&end_date=2026-10-20")
    assert resp_daily.status_code == status.HTTP_400_BAD_REQUEST
    assert "start_date must be less than or equal to end_date" in resp_daily.json()["detail"]


def test_daily_footfall_aggregation(client, db_session):
    cam = Camera(name="Daily Cam", stream_url="rtsp://localhost/daily")
    db_session.add(cam)
    db_session.commit()

    run = ProcessingRun(camera_id=cam.id, status="RUNNING")
    db_session.add(run)
    db_session.commit()

    day1 = datetime(2026, 10, 20, 10, 0, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 10, 21, 14, 0, 0, tzinfo=timezone.utc)

    t1 = VisitorTrack(camera_id=cam.id, processing_run_id=run.id, tracker_id="101")
    t2 = VisitorTrack(camera_id=cam.id, processing_run_id=run.id, tracker_id="102")
    t3 = VisitorTrack(camera_id=cam.id, processing_run_id=run.id, tracker_id="103")
    db_session.add_all([t1, t2, t3])
    db_session.commit()

    e1 = FootfallEvent(camera_id=cam.id, visitor_track_id=t1.id, event_type="entry", occurred_at=day1)
    e2 = FootfallEvent(camera_id=cam.id, visitor_track_id=t2.id, event_type="entry", occurred_at=day1)
    e3 = FootfallEvent(camera_id=cam.id, visitor_track_id=t3.id, event_type="entry", occurred_at=day2)
    db_session.add_all([e1, e2, e3])
    db_session.commit()

    resp = client.get(f"{DAILY_FOOTFALL_ANALYTICS}?camera_id={cam.id}&start_date=2026-10-20&end_date=2026-10-21")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["total_entries"] == 3
    daily_list = data["daily_footfall"]
    assert len(daily_list) == 2
    assert daily_list[0]["date"] == "2026-10-20"
    assert daily_list[0]["entries"] == 2
    assert daily_list[1]["date"] == "2026-10-21"
    assert daily_list[1]["entries"] == 1
