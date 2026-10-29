from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from app.core.config import settings
from app.db.models.camera import Camera
from app.db.models.footfall_event import FootfallEvent
from app.db.models.person_identity import PersonIdentity
from app.db.models.processing_run import ProcessingRun
from app.db.models.visitor_track import VisitorTrack
from app.repositories.footfall_event_repo import FootfallEventRepository
from app.repositories.person_identity_repo import PersonIdentityRepository
from app.repositories.visitor_track_repo import VisitorTrackRepository
from app.schemas.camera import FootfallRecordRequest
from app.services.camera_service import CameraService
from app.services.retention_service import RetentionService
from app.vision.entry_counter import EntryLineCounter
from app.vision.face_embedder import FaceEmbedder
from app.vision.faiss_index import FaissIndexManager
from app.vision.identity_tracker import (
    IdentityMatchResult,
    IdentityMatchStatus,
    IdentityRecoveryManager,
)
from app.vision.tracker import TrackedPerson


# ---------------------------------------------------------------------------
# Test Fixtures & Synthetic Face Generation Helpers
# ---------------------------------------------------------------------------
def create_synthetic_face_crop(width: int = 80, height: int = 80, pattern: int = 1) -> np.ndarray:
    """Generates a non-blank, structured synthetic face image for deterministic testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add gradients and features
    for y in range(height):
        for x in range(width):
            img[y, x, 0] = (x * 3 + pattern * 40) % 256
            img[y, x, 1] = (y * 3 + pattern * 30) % 256
            img[y, x, 2] = ((x + y) * 2 + pattern * 50) % 256
    # Draw simulated facial eye/nose contrast points to pass Laplacian blur test
    cv2_pt1 = (width // 3, height // 3)
    cv2_pt2 = (2 * width // 3, height // 3)
    img[cv2_pt1[1]-5:cv2_pt1[1]+5, cv2_pt1[0]-5:cv2_pt1[0]+5] = 255
    img[cv2_pt2[1]-5:cv2_pt2[1]+5, cv2_pt2[0]-5:cv2_pt2[0]+5] = 255
    return img


# ---------------------------------------------------------------------------
# Phase 1: Face Quality Checks & Embedding Extraction Tests
# ---------------------------------------------------------------------------
def test_face_quality_checks_blur_resolution_aspect_ratio():
    embedder = FaceEmbedder(min_size=36, min_confidence=0.40, blur_threshold=20.0)

    # 1. Reliable face
    good_face = create_synthetic_face_crop(64, 64, pattern=1)
    is_ok, reason = embedder.is_face_reliable(good_face, confidence=0.85)
    assert is_ok is True
    assert reason == "reliable"

    # 2. Too small resolution
    small_face = create_synthetic_face_crop(24, 24, pattern=1)
    is_ok, reason = embedder.is_face_reliable(small_face, confidence=0.85)
    assert is_ok is False
    assert "too_small" in reason

    # 3. Low detector confidence
    is_ok, reason = embedder.is_face_reliable(good_face, confidence=0.25)
    assert is_ok is False
    assert "low_confidence" in reason

    # 4. Extreme aspect ratio (e.g. 100 wide x 50 high, ar = 2.0 > 1.6)
    distorted_face = create_synthetic_face_crop(100, 50, pattern=1)
    is_ok, reason = embedder.is_face_reliable(distorted_face, confidence=0.85)
    assert is_ok is False
    assert "invalid_aspect_ratio" in reason

    # 5. Severely blurred face (flat uniform image has 0 Laplacian variance)
    flat_face = np.full((64, 64, 3), 128, dtype=np.uint8)
    is_ok, reason = embedder.is_face_reliable(flat_face, confidence=0.85)
    assert is_ok is False
    assert "blurry" in reason


def test_face_embedding_normalization_and_similarity():
    embedder = FaceEmbedder()
    face1 = create_synthetic_face_crop(64, 64, pattern=1)
    face2 = create_synthetic_face_crop(64, 64, pattern=2)

    emb1 = embedder.extract_embedding(face1)
    emb2 = embedder.extract_embedding(face2)

    assert emb1.shape == (512,)
    assert emb2.shape == (512,)

    # Must be L2-normalized unit vectors
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    assert np.isclose(norm1, 1.0, atol=1e-4)
    assert np.isclose(norm2, 1.0, atol=1e-4)

    # Self-similarity must be exactly 1.0
    self_sim = embedder.compute_similarity(emb1, emb1)
    assert np.isclose(self_sim, 1.0, atol=1e-4)

    # Distinct faces must have lower similarity
    diff_sim = embedder.compute_similarity(emb1, emb2)
    assert diff_sim < 0.95


# ---------------------------------------------------------------------------
# Phase 1: FAISS Similarity Search & Index Management Tests
# ---------------------------------------------------------------------------
def test_faiss_index_lifecycle_add_search_remove_rebuild():
    index_mgr = FaissIndexManager(similarity_threshold=0.50)

    # Generate 3 distinct normalized embeddings
    rng = np.random.RandomState(100)
    v1 = rng.randn(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = rng.randn(512).astype(np.float32)
    v2 /= np.linalg.norm(v2)

    # Add to index
    index_mgr.add("PID_A", v1)
    index_mgr.add("PID_B", v2)
    assert index_mgr.size() == 2

    # Query with exact match v1
    best_id, score = index_mgr.find_best_match(v1)
    assert best_id == "PID_A"
    assert np.isclose(score, 1.0, atol=1e-4)

    # Query with slight noisy version of v2 (sim ~ 0.98 > threshold 0.50)
    v2_noisy = v2 + 0.05 * rng.randn(512).astype(np.float32)
    v2_noisy /= np.linalg.norm(v2_noisy)
    best_id_noisy, score_noisy = index_mgr.find_best_match(v2_noisy)
    assert best_id_noisy == "PID_B"
    assert score_noisy >= 0.50

    # Test safe deletion
    removed = index_mgr.remove("PID_A")
    assert removed is True
    assert index_mgr.size() == 1

    # Search for PID_A after removal must not find PID_A
    match_after_delete, _ = index_mgr.find_best_match(v1)
    assert match_after_delete != "PID_A"

    # Test batch removal
    index_mgr.remove_ids(["PID_B"])
    assert index_mgr.size() == 0


# ---------------------------------------------------------------------------
# Phase 1: Identity Recovery Across ByteTrack Track Loss
# ---------------------------------------------------------------------------
def test_identity_recovery_when_bytetrack_id_changes():
    """
    Scenario:
    1. Visitor enters under temporary ByteTrack ID '10'.
    2. Face is clear, identity PID_VISITOR_1 is created.
    3. Visitor disappears (tracking lost).
    4. Visitor reappears; ByteTrack assigns a brand new ID '88'.
    5. Identity Recovery recognizes visitor as PID_VISITOR_1.
    """
    embedder = FaceEmbedder(min_size=36, min_confidence=0.40)
    faiss_mgr = FaissIndexManager(similarity_threshold=0.50)
    identity_mgr = IdentityRecoveryManager(
        embedder=embedder,
        faiss_index=faiss_mgr,
        similarity_threshold=0.50,
        max_attempts=5,
        cooldown_frames=0,  # 0 cooldown for fast unit test
    )

    visitor_face = create_synthetic_face_crop(64, 64, pattern=42)

    # Frame 1: ByteTrack track '10' appears (1st observation)
    r1 = identity_mgr.process_track_face("10", visitor_face, face_conf=0.88)
    assert r1.status == IdentityMatchStatus.UNRESOLVED  # Awaiting 2nd confirmation frame

    # Frame 2: ByteTrack track '10' confirmed -> new persistent identity created
    r2 = identity_mgr.process_track_face("10", visitor_face, face_conf=0.88)
    assert r2.status == IdentityMatchStatus.CREATED
    pid_created = r2.person_id
    assert pid_created is not None
    assert pid_created.startswith("PID_")

    # Verify strict separation of IDs:
    assert identity_mgr.get_person_id("10") == pid_created
    assert identity_mgr.get_person_id("88") is None

    # Track '10' is lost
    identity_mgr.on_track_lost("10")

    # Visitor reappears with a NEW ByteTrack ID '88'
    # Same visitor face presented to track '88'
    r3 = identity_mgr.process_track_face("88", visitor_face, face_conf=0.88)
    assert r3.status == IdentityMatchStatus.MATCHED
    assert r3.person_id == pid_created
    assert r3.similarity >= 0.50

    # Both temporary ByteTrack IDs map to the same persistent person ID
    assert identity_mgr.get_person_id("88") == pid_created


def test_unreliable_face_leaves_identity_unresolved():
    embedder = FaceEmbedder(min_size=36, min_confidence=0.40)
    faiss_mgr = FaissIndexManager(similarity_threshold=0.50)
    identity_mgr = IdentityRecoveryManager(
        embedder=embedder,
        faiss_index=faiss_mgr,
        similarity_threshold=0.50,
    )

    # Blurry face
    blurry = np.full((64, 64, 3), 100, dtype=np.uint8)
    res = identity_mgr.process_track_face("99", blurry, face_conf=0.85)

    assert res.status == IdentityMatchStatus.UNRELIABLE_FACE
    assert res.person_id is None
    # Track continues temporary tracking with persistent identity unresolved
    assert identity_mgr.get_person_id("99") is None


def test_duplicate_entry_prevention_across_tracking_sessions():
    """
    Ensures that when an identified visitor has already crossed the entry line,
    reappearing under a new ByteTrack ID does NOT generate a duplicate entry count.
    """
    identity_mgr = IdentityRecoveryManager(similarity_threshold=0.50, cooldown_frames=0)
    face_a = create_synthetic_face_crop(64, 64, pattern=7)

    # Initialize Person A under Track 1
    identity_mgr.process_track_face("1", face_a, face_conf=0.9)
    res = identity_mgr.process_track_face("1", face_a, face_conf=0.9)
    pid_a = res.person_id

    # Track 1 crosses entry line
    should_count_1 = identity_mgr.should_count_entry("1")
    assert should_count_1 is True

    # Immediate second crossing attempt on Track 1 is rejected
    should_count_1_dup = identity_mgr.should_count_entry("1")
    assert should_count_1_dup is False

    # Track 1 lost, visitor reappears under Track 2
    identity_mgr.on_track_lost("1")
    res2 = identity_mgr.process_track_face("2", face_a, face_conf=0.9)
    assert res2.status == IdentityMatchStatus.MATCHED
    assert res2.person_id == pid_a

    # Track 2 (same persistent person) attempts entry crossing -> MUST BE SUPPRESSED
    should_count_2 = identity_mgr.should_count_entry("2")
    assert should_count_2 is False, "Duplicate entry was not suppressed for recovered visitor identity!"


def test_entry_line_counter_persistent_duplicate_prevention():
    cfg = {
        "line": [[100.0, 200.0], [500.0, 200.0]],
        "direction": "top_to_bottom",
    }
    counter = EntryLineCounter(cfg)

    # Frame 1: Person A (tracker_id="t1", person_id="PID_X") moving across line
    p1_f1 = TrackedPerson(tracker_id="t1", box=(150.0, 100.0, 200.0, 190.0), confidence=0.9, class_id=0, person_id="PID_X")
    entries_f1 = counter.process_frame([p1_f1])
    assert len(entries_f1) == 0

    # Frame 2: Person A crosses line
    p1_f2 = TrackedPerson(tracker_id="t1", box=(150.0, 120.0, 200.0, 215.0), confidence=0.9, class_id=0, person_id="PID_X")
    entries_f2 = counter.process_frame([p1_f2])
    assert len(entries_f2) == 1
    assert entries_f2[0].tracker_id == "t1"

    # Frame 3: Tracking lost and reappears with tracker_id="t99", but same person_id="PID_X"
    p2_f1 = TrackedPerson(tracker_id="t99", box=(150.0, 100.0, 200.0, 190.0), confidence=0.9, class_id=0, person_id="PID_X")
    counter.process_frame([p2_f1])
    p2_f2 = TrackedPerson(tracker_id="t99", box=(150.0, 120.0, 200.0, 215.0), confidence=0.9, class_id=0, person_id="PID_X")
    entries_p2 = counter.process_frame([p2_f2])
    # Must NOT count duplicate crossing for PID_X!
    assert len(entries_p2) == 0


# ---------------------------------------------------------------------------
# Phase 2: 14-Day Rolling Biometric Retention Policy Tests
# ---------------------------------------------------------------------------
def test_retention_policy_purges_expired_embeddings_and_preserves_analytics(db_session):
    cam = Camera(name="Retention Test Cam", stream_url="rtsp://localhost/test")
    db_session.add(cam)
    db_session.commit()

    run = ProcessingRun(camera_id=cam.id, status="RUNNING")
    db_session.add(run)
    db_session.commit()

    now = datetime.now(timezone.utc)
    t_20_days_ago = now - timedelta(days=20)
    t_5_days_ago = now - timedelta(days=5)

    faiss_mgr = FaissIndexManager(similarity_threshold=0.50)
    repo = PersonIdentityRepository(db_session)

    # 1. Visitor 1: Seen 20 days ago (EXPIRED)
    emb_1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
    p1 = repo.create_or_update("PID_EXPIRED", emb_1, timestamp=t_20_days_ago)
    faiss_mgr.add("PID_EXPIRED", emb_1)

    # 2. Visitor 2: Seen 5 days ago (ACTIVE / UNEXPIRED)
    emb_2 = np.zeros(512, dtype=np.float32)
    emb_2[0] = 1.0
    p2 = repo.create_or_update("PID_ACTIVE", emb_2, timestamp=t_5_days_ago)
    faiss_mgr.add("PID_ACTIVE", emb_2)

    # 3. Create visitor track & footfall event for the expired visitor
    vt = VisitorTrack(
        camera_id=cam.id,
        processing_run_id=run.id,
        tracker_id="TRK_OLD",
        person_id="PID_EXPIRED",
        person_identity_id=p1.id,
        first_seen_at=t_20_days_ago,
        last_seen_at=t_20_days_ago,
    )
    db_session.add(vt)
    db_session.commit()

    footfall_repo = FootfallEventRepository(db_session)
    event = footfall_repo.record_entry(cam.id, vt.id, occurred_at=t_20_days_ago)
    assert event is not None
    initial_total_footfall = footfall_repo.get_total_entries(camera_id=cam.id)
    assert initial_total_footfall == 1

    # Verify both exist before retention cleanup
    assert faiss_mgr.size() == 2
    assert p1.face_embedding is not None

    # Run 14-day retention cleanup
    retention_svc = RetentionService(db=db_session, faiss_index=faiss_mgr, retention_days=14)
    report = retention_svc.cleanup_expired_embeddings(db=db_session, now=now)

    assert report.success is True
    assert report.expired_count == 1

    # Database verification
    db_session.refresh(p1)
    db_session.refresh(p2)

    # Expired embedding is permanently deleted (None) and marked expired
    assert p1.face_embedding is None
    assert p1.is_expired is True
    assert p1.expired_at is not None

    # Active embedding is strictly preserved
    assert p2.face_embedding is not None
    assert p2.is_expired is False

    # FAISS verification: expired vector removed; active vector remains
    assert faiss_mgr.size() == 1
    match_expired, _ = faiss_mgr.find_best_match(emb_1)
    assert match_expired != "PID_EXPIRED"
    match_active, _ = faiss_mgr.find_best_match(emb_2)
    assert match_active == "PID_ACTIVE"

    # Historical Business Analytics & Footfall Events Verification:
    # Must be 100% PRESERVED!
    total_footfall_after = footfall_repo.get_total_entries(camera_id=cam.id)
    assert total_footfall_after == initial_total_footfall == 1
    track_after = db_session.query(VisitorTrack).filter(VisitorTrack.id == vt.id).first()
    assert track_after is not None
    assert track_after.person_id == "PID_EXPIRED"

    # Idempotency verification: Running cleanup again should purge 0 records
    report_rerun = retention_svc.cleanup_expired_embeddings(db=db_session, now=now)
    assert report_rerun.success is True
    assert report_rerun.expired_count == 0


def test_temporary_track_id_does_not_reset_retention_timer(db_session):
    """
    Requirement 7: Keep temporary tracking IDs from resetting the retention timer;
    only a reliable identity match should update last_seen_at.
    """
    t_10_days_ago = datetime.now(timezone.utc) - timedelta(days=10)
    repo = PersonIdentityRepository(db_session)
    emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
    identity = repo.create_or_update("PID_TIMER_TEST", emb, timestamp=t_10_days_ago)

    initial_last_seen = identity.last_seen_at

    # A temporary ByteTrack detection occurs (no identity match)
    cam_service = CameraService(db_session)
    cam = Camera(name="Cam", stream_url="rtsp://localhost/test")
    db_session.add(cam)
    db_session.commit()

    run = ProcessingRun(camera_id=cam.id, status="RUNNING")
    db_session.add(run)
    db_session.commit()

    # Anonymous temporary track recorded
    req_anon = FootfallRecordRequest(
        tracker_id="TEMP_999",
        confidence=0.92,
        processing_run_id=run.id,
    )
    cam_service.record_footfall(cam.id, req_anon)

    # Verify last_seen_at of persistent identity is UNCHANGED
    db_session.refresh(identity)
    assert identity.last_seen_at == initial_last_seen

    # Now a reliable identity match occurs with person_id="PID_TIMER_TEST"
    req_matched = FootfallRecordRequest(
        tracker_id="TEMP_1000",
        confidence=0.95,
        processing_run_id=run.id,
        person_id="PID_TIMER_TEST",
    )
    cam_service.record_footfall(cam.id, req_matched)

    # Verify last_seen_at IS updated on reliable match
    db_session.refresh(identity)
    assert identity.last_seen_at > initial_last_seen


# ---------------------------------------------------------------------------
# API Endpoints: Retention Status & Manual Cleanup
# ---------------------------------------------------------------------------
def test_retention_api_endpoints(client, db_session):
    # 1. Query status endpoint
    resp_status = client.get("/api/v1/identity/status")
    assert resp_status.status_code == 200
    data = resp_status.json()
    assert data["retention_policy_days"] == 14
    assert "active_face_embeddings" in data
    assert "calibrated_similarity_threshold" in data

    # 2. Trigger cleanup endpoint
    resp_cleanup = client.post("/api/v1/identity/cleanup?retention_days=14")
    assert resp_cleanup.status_code == 200
    cleanup_data = resp_cleanup.json()
    assert cleanup_data["success"] is True
    assert cleanup_data["retention_days"] == 14
    assert "execution_time_seconds" in cleanup_data
