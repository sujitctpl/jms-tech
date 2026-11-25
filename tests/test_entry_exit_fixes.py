"""Regression tests for entry-exit matching, visitor ID minting, and retention fixes."""
from datetime import datetime, timedelta, timezone

import numpy as np

from app.db.models.person_identity import PersonIdentity
from app.repositories.person_identity_repo import PersonIdentityRepository
from app.schemas.visitor import VisitorZoneUpdateRequest, WalkInCreateRequest
from app.services.retention_service import RetentionService
from app.services.visitor_service import VisitorService


def _visitor(db, person_id, notes=None, exit_time=None, created_at=None):
    ts = created_at or datetime.now(timezone.utc)
    rec = PersonIdentity(
        person_id=person_id,
        notes=notes,
        exit_time=exit_time,
        created_at=ts,
        last_seen_at=ts,
    )
    db.add(rec)
    db.commit()
    return rec


def test_exit_tracker_lookup_is_exact_not_substring(db_session):
    """Tracker #4 must not attach its exit to the visitor tagged 'Tracker #42'."""
    _visitor(db_session, "VIS-2026-1042", notes="Walk-in | Tracker #42")
    svc = VisitorService(db_session)

    res = svc.update_visitor_zone(VisitorZoneUpdateRequest(tracker_id="4", exit_time="04:45 PM"))

    assert res is None
    assert svc.repo.get_by_person_id("VIS-2026-1042").exit_time is None


def test_exit_tracker_lookup_matches_exact_tracker(db_session):
    _visitor(db_session, "VIS-2026-1042", notes="Walk-in | Tracker #42")
    svc = VisitorService(db_session)

    res = svc.update_visitor_zone(VisitorZoneUpdateRequest(tracker_id="42", exit_time="04:45 PM"))

    assert res is not None and res.visitorId == "VIS-2026-1042"
    assert res.exitTime is not None


def test_exit_respects_edge_excluded_person(db_session):
    """When the edge ruled a visitor out by face, neither person_id nor tracker lookup may pick it."""
    _visitor(db_session, "VIS-2026-1042", notes="Tracker #42")
    svc = VisitorService(db_session)

    res = svc.update_visitor_zone(
        VisitorZoneUpdateRequest(
            person_id="VIS-2026-1042",
            tracker_id="42",
            exit_time="04:45 PM",
            exclude_person_id="VIS-2026-1042",
        )
    )

    assert res is None
    assert svc.repo.get_by_person_id("VIS-2026-1042").exit_time is None


def test_walk_in_never_reuses_a_live_person_id(db_session):
    """After a delete the row count drops; the next walk-in must not overwrite an existing visitor."""
    for n in (1041, 1042, 1043):
        _visitor(db_session, f"VIS-2026-{n}", notes=f"original {n}")
    svc = VisitorService(db_session)
    assert svc.delete_visitor("VIS-2026-1041")

    res = svc.create_walk_in(WalkInCreateRequest(notes="new walk-in"))

    assert res.visitorId not in ("VIS-2026-1042", "VIS-2026-1043")
    assert svc.repo.get_by_person_id("VIS-2026-1043").notes == "original 1043"


def test_retention_purges_body_only_visitors(db_session):
    """A visitor with only an OSNet body vector (face never captured) must still expire."""
    repo = PersonIdentityRepository(db_session)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    body = np.random.rand(512).astype(np.float32)
    body /= np.linalg.norm(body)
    repo.create_or_update("VIS-BODY-ONLY", None, body, timestamp=old)

    report = RetentionService(db=db_session, retention_days=7).cleanup_expired_embeddings(
        db=db_session, face_crops_dir="nonexistent_dir_for_test"
    )

    rec = repo.get_by_person_id("VIS-BODY-ONLY")
    assert report.expired_count == 1
    assert rec.is_expired is True
    assert rec.body_embedding is None
    assert rec.face_embedding is None
