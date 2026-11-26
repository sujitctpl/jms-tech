"""Exit attribution rules from the 2026-11-26 review (wrong entry-exit matches).

The vision models are stubbed: each visitor gets a one-hot body embedding, and verify_appearance
returns a scripted (reid, clothing, combined, is_match) per visitor, so only the decision logic is tested.
"""
import base64
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from app.db.models.person_identity import PersonIdentity
from app.schemas.visitor import VisitorZoneUpdateRequest
from app.services.visitor_service import VisitorService, latest_exit_match_pct, registration_tracker


def _exit_image_b64() -> str:
    img = np.full((96, 48, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf).decode("ascii")


def _one_hot(i: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[i] = 1.0
    return v


def _visitor(db, idx, person_id, tracker, created_at=None, exit_time=None):
    ts = created_at or (datetime.now(timezone.utc) - timedelta(minutes=10))
    rec = PersonIdentity(
        person_id=person_id,
        notes=f"Appearance: Test | Tracker #{tracker}",
        body_embedding=_one_hot(idx),
        exit_time=exit_time,
        created_at=ts,
        last_seen_at=ts,
    )
    db.add(rec)
    db.commit()
    return rec


@pytest.fixture
def svc_with_scores(db_session, monkeypatch):
    """Returns (service, scores) where scores[idx] = (reid, clothing, combined, is_match)."""
    svc = VisitorService(db_session)
    scores = {}
    monkeypatch.setattr(svc.embedder, "detect_and_crop_face", lambda img: (None, None, None))
    monkeypatch.setattr(svc.body_embedder, "extract_embedding", lambda img: np.ones(512, dtype=np.float32))

    def fake_verify(crop1=None, crop2=None, emb1=None, emb2=None):
        return scores[int(np.argmax(emb2))]

    monkeypatch.setattr(svc.body_embedder, "verify_appearance", fake_verify)
    return svc, scores


def _exit(svc, person_id=None, tracker="999"):
    return svc.update_visitor_zone(
        VisitorZoneUpdateRequest(
            person_id=person_id, tracker_id=tracker, exit_time="04:45 PM", exit_image_base64=_exit_image_b64()
        )
    )


def test_edge_face_hint_without_evidence_falls_to_search(db_session, svc_with_scores):
    """VIS-2026-1089 at 10:44: an edge face-hint ID whose body does not match must not be departed."""
    svc, scores = svc_with_scores
    _visitor(db_session, 1, "VIS-T-A", tracker=10)
    _visitor(db_session, 2, "VIS-T-B", tracker=20)
    scores[1] = (0.469, 0.20, 0.07, False)
    scores[2] = (0.80, 0.85, 0.82, True)

    res = _exit(svc, person_id="VIS-T-A", tracker="7139")

    assert res is not None and res.visitorId == "VIS-T-B"
    assert svc.repo.get_by_person_id("VIS-T-A").exit_time is None


def test_edge_face_hint_without_evidence_and_no_match_leaves_exit_unattached(db_session, svc_with_scores):
    svc, scores = svc_with_scores
    _visitor(db_session, 1, "VIS-T-A", tracker=10)
    scores[1] = (0.41, 0.53, 0.45, False)

    assert _exit(svc, person_id="VIS-T-A", tracker="7391") is None
    assert svc.repo.get_by_person_id("VIS-T-A").exit_time is None


def test_edge_supplied_id_with_body_match_is_attached(db_session, svc_with_scores):
    svc, scores = svc_with_scores
    _visitor(db_session, 1, "VIS-T-A", tracker=10)
    scores[1] = (0.78, 0.80, 0.79, True)

    res = _exit(svc, person_id="VIS-T-A", tracker="555")

    assert res is not None and res.visitorId == "VIS-T-A"
    assert latest_exit_match_pct(svc.repo.get_by_person_id("VIS-T-A").notes) == 79.0


def test_registration_track_exit_stays_trusted(db_session, svc_with_scores):
    """The same continuous track that registered the visitor keeps working on a back-view exit."""
    svc, scores = svc_with_scores
    _visitor(db_session, 1, "VIS-T-A", tracker=10)
    scores[1] = (0.60, 0.50, 0.45, False)  # back view: not is_match, but no hard contradiction

    res = _exit(svc, person_id="VIS-T-A", tracker="10")

    assert res is not None and res.visitorId == "VIS-T-A"


def test_previous_day_visitor_never_receives_exit_by_person_id(db_session, svc_with_scores):
    svc, scores = svc_with_scores
    yesterday = datetime.now(timezone.utc) - timedelta(days=1, hours=2)
    _visitor(db_session, 1, "VIS-T-OLD", tracker=10, created_at=yesterday)
    scores[1] = (0.90, 0.90, 0.90, True)

    assert _exit(svc, person_id="VIS-T-OLD", tracker="10") is None
    assert svc.repo.get_by_person_id("VIS-T-OLD").exit_time != "04:45 PM"


def test_departed_visitor_never_receives_second_exit_by_person_id(db_session, svc_with_scores):
    svc, scores = svc_with_scores
    _visitor(db_session, 1, "VIS-T-A", tracker=10, exit_time="10:16 AM")
    scores[1] = (0.90, 0.90, 0.90, True)

    assert _exit(svc, person_id="VIS-T-A", tracker="10") is None
    assert svc.repo.get_by_person_id("VIS-T-A").exit_time == "10:16 AM"


def test_reid_override_beats_clothing_driven_combined_score(db_session, svc_with_scores):
    """10:56 exit: true visitor reid 0.827 / combined 0.749 vs wrong one reid 0.724 / combined 0.761."""
    svc, scores = svc_with_scores
    _visitor(db_session, 1, "VIS-T-WRONG", tracker=10)
    _visitor(db_session, 2, "VIS-T-TRUE", tracker=20)
    scores[1] = (0.724, 0.816, 0.761, True)
    scores[2] = (0.827, 0.631, 0.749, True)

    res = _exit(svc)

    assert res is not None and res.visitorId == "VIS-T-TRUE"
    assert svc.repo.get_by_person_id("VIS-T-WRONG").exit_time is None


def test_small_reid_lead_does_not_override_combined_score(db_session, svc_with_scores):
    svc, scores = svc_with_scores
    _visitor(db_session, 1, "VIS-T-A", tracker=10)
    _visitor(db_session, 2, "VIS-T-B", tracker=20)
    scores[1] = (0.75, 0.90, 0.81, True)
    scores[2] = (0.78, 0.55, 0.69, True)

    res = _exit(svc)

    assert res is not None and res.visitorId == "VIS-T-A"


def test_close_reid_and_close_score_stays_ambiguous(db_session, svc_with_scores):
    svc, scores = svc_with_scores
    _visitor(db_session, 1, "VIS-T-A", tracker=10)
    _visitor(db_session, 2, "VIS-T-B", tracker=20)
    scores[1] = (0.728, 0.75, 0.737, True)
    scores[2] = (0.708, 0.78, 0.736, True)

    assert _exit(svc) is None


def test_note_helpers():
    notes = "Appearance: X | Tracker #1485 | Tracker #3078 | Exit Match: 75.6% (Re-ID: 70%) | Tracker #7139 | Exit Match: 7.0% (Re-ID: 47%)"
    assert registration_tracker(notes) == "1485"
    assert registration_tracker("Tracker #auto") is None
    assert latest_exit_match_pct(notes) == 7.0
    assert latest_exit_match_pct("no score") is None
