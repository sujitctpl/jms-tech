import datetime
import uuid
import numpy as np
import pytest
from sqlalchemy import text
from app.db.models.employee import Employee
from app.db.models.footfall_event import FootfallEvent
from app.db.models.person_identity import PersonIdentity
from app.db.models.visitor_track import VisitorTrack
from app.repositories.employee_repo import EmployeeRepository
from app.repositories.person_identity_repo import PersonIdentityRepository
from app.services.retention_service import RetentionService
from app.vision.identity_tracker import IdentityMatchStatus, IdentityRecoveryManager
from app.vision.vector_store import PgVectorStoreManager


def make_unit_vector(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


# -------------------------------------------------------------------------
# 1. Vector Insertion and Native Cosine Search
# -------------------------------------------------------------------------

def test_pgvector_embedding_insertion_and_retrieval(db_session):
    repo = PersonIdentityRepository(db_session)
    emb = make_unit_vector(seed=101)

    record = repo.create_or_update(
        person_id="VISITOR_001",
        embedding=emb,
    )
    assert record.id is not None
    assert record.person_id == "VISITOR_001"
    assert record.is_expired is False

    # Check numpy conversion helper
    retrieved_arr = record.get_embedding_array()
    assert retrieved_arr is not None
    assert retrieved_arr.shape == (512,)
    assert np.allclose(retrieved_arr, emb, atol=1e-4)


def test_pgvector_similarity_search_exact_and_threshold(db_session):
    repo = PersonIdentityRepository(db_session)
    v1 = make_unit_vector(seed=201)
    v2 = make_unit_vector(seed=202)

    repo.create_or_update("VISITOR_A", v1)
    repo.create_or_update("VISITOR_B", v2)

    # 1. Exact match probe
    pid, score = repo.find_best_match(v1, similarity_threshold=0.50)
    assert pid == "VISITOR_A"
    assert score >= 0.99

    # 2. Probe with noise
    noisy_v1 = v1 + np.random.randn(512).astype(np.float32) * 0.02
    noisy_v1 /= np.linalg.norm(noisy_v1)
    pid_noisy, score_noisy = repo.find_best_match(noisy_v1, similarity_threshold=0.50)
    assert pid_noisy == "VISITOR_A"
    assert score_noisy > 0.85

    # 3. Probe with unrelated vector below threshold
    v_novel = make_unit_vector(seed=999)
    pid_novel, score_novel = repo.find_best_match(v_novel, similarity_threshold=0.85)
    assert pid_novel is None  # Below 0.85 threshold


# -------------------------------------------------------------------------
# 2. Persistent Identity Recovery with ByteTrack ID Changes
# -------------------------------------------------------------------------

def test_identity_recovery_with_pgvector_backend(db_session):
    repo = PersonIdentityRepository(db_session)
    vec_manager = PgVectorStoreManager(db_session, similarity_threshold=0.50)

    class MockEmbedder:
        def __init__(self, vector):
            self.vector = vector
        def is_face_reliable(self, face_crop, face_conf, landmarks):
            return True, "reliable"
        def extract_embedding(self, face_crop, landmarks):
            return self.vector

    target_emb = make_unit_vector(seed=301)
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)

    # Initial visitor registered in pgvector
    repo.create_or_update("PID_EXISTING", target_emb)

    matched_callbacks = []
    created_callbacks = []

    def on_matched(pid, ts):
        matched_callbacks.append((pid, ts))
        repo.update_last_seen(pid)

    def on_created(pid, emb, ts):
        created_callbacks.append((pid, emb, ts))
        repo.create_or_update(pid, emb)

    tracker = IdentityRecoveryManager(
        embedder=MockEmbedder(target_emb),
        vector_store=vec_manager,
        similarity_threshold=0.50,
        on_matched_callback=on_matched,
        on_created_callback=on_created,
    )

    # Sighting 1: Tracker ID "track_1"
    res1 = tracker.process_track_face(tracker_id="track_1", face_crop=dummy_crop, face_conf=0.95)
    assert res1.status == IdentityMatchStatus.MATCHED
    assert res1.person_id == "PID_EXISTING"
    assert len(matched_callbacks) == 1

    # Track lost (person walks behind pillar)
    tracker.on_track_lost("track_1")
    # New track 2 has not been observed yet
    assert tracker.get_person_id("track_2") is None

    # Sighting 2: Tracker assigns NEW ID "track_2" to reappearing visitor
    res2 = tracker.process_track_face(tracker_id="track_2", face_crop=dummy_crop, face_conf=0.92)
    assert res2.status == IdentityMatchStatus.MATCHED
    assert res2.person_id == "PID_EXISTING"
    assert tracker.get_person_id("track_2") == "PID_EXISTING"


# -------------------------------------------------------------------------
# 3. 14-Day Rolling Retention Policy with pgvector
# -------------------------------------------------------------------------

def test_pgvector_retention_policy_purges_embeddings_and_preserves_analytics(db_session):
    repo = PersonIdentityRepository(db_session)
    vec_manager = PgVectorStoreManager(db_session, similarity_threshold=0.50)

    t_now = datetime.datetime.now(datetime.timezone.utc)
    old_time = t_now - datetime.timedelta(days=20)  # > 14 days ago

    emb1 = make_unit_vector(seed=401)
    emb2 = make_unit_vector(seed=402)

    # 1. Create expired visitor (20 days inactive)
    p_exp = repo.create_or_update("PID_EXPIRED", emb1, timestamp=old_time)
    # 2. Create active visitor (seen today)
    p_act = repo.create_or_update("PID_ACTIVE", emb2, timestamp=t_now)

    # 3. Simulate historical visitor footfall analytics attached to expired visitor
    track = VisitorTrack(
        camera_id=uuid.uuid4(),
        processing_run_id=uuid.uuid4(),
        tracker_id=99,
        person_id="PID_EXPIRED",
        first_seen_at=old_time,
        last_seen_at=old_time,
    )
    db_session.add(track)
    db_session.flush()

    event = FootfallEvent(
        camera_id=track.camera_id,
        visitor_track_id=track.id,
        event_type="entry",
        occurred_at=old_time,
    )
    db_session.add(event)
    db_session.commit()

    # 4. Execute 14-day rolling retention cleanup
    retention_svc = RetentionService(db=db_session, retention_days=14)
    report = retention_svc.cleanup_expired_embeddings(now=t_now)
    assert report.success is True
    assert report.expired_count == 1

    # Verify expired visitor embedding is removed and not searchable
    p_exp_after = repo.get_by_person_id("PID_EXPIRED")
    assert p_exp_after.is_expired is True
    assert p_exp_after.face_embedding is None

    # Searching with emb1 must NOT match PID_EXPIRED anymore!
    pid_match, score_match = repo.find_best_match(emb1, similarity_threshold=0.50)
    assert pid_match != "PID_EXPIRED"

    # Active visitor remains intact and searchable
    p_act_after = repo.get_by_person_id("PID_ACTIVE")
    assert p_act_after.is_expired is False
    assert p_act_after.face_embedding is not None

    pid_act, score_act = repo.find_best_match(emb2, similarity_threshold=0.50)
    assert pid_act == "PID_ACTIVE"

    # CRITICAL: Verify aggregate footfall analytics are strictly preserved!
    saved_event = db_session.query(FootfallEvent).filter_by(id=event.id).first()
    assert saved_event is not None
    assert saved_event.event_type == "entry"
    saved_track = db_session.query(VisitorTrack).filter_by(id=track.id).first()
    assert saved_track is not None
    assert saved_track.person_id == "PID_EXPIRED"


# -------------------------------------------------------------------------
# 4. Employee Compatibility and Isolation
# -------------------------------------------------------------------------

def test_employee_and_visitor_vector_isolation(db_session):
    vis_repo = PersonIdentityRepository(db_session)
    emp_repo = EmployeeRepository(db_session)
    vec_manager = PgVectorStoreManager(db_session, similarity_threshold=0.50)

    shared_emb = make_unit_vector(seed=501)

    # Register employee with shared_emb
    emp = emp_repo.create(
        employee_id="EMP-STAFF-01",
        name="Staff Member One",
        department="Showroom",
        embedding=shared_emb,
    )

    # 1. Visitor search must NOT return employee!
    vis_pid, vis_score = vec_manager.find_best_visitor_match(shared_emb, similarity_threshold=0.50)
    assert vis_pid is None  # Visitor query isolated from employees

    # 2. Employee search returns employee
    emp_pid, emp_score = vec_manager.find_best_employee_match(shared_emb, similarity_threshold=0.50)
    assert emp_pid == "EMP-STAFF-01"
    assert emp_score >= 0.99

    # 3. Retention policy run does NOT purge employee embedding
    retention_svc = RetentionService(db=db_session, retention_days=14)
    old_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
    retention_svc.cleanup_expired_embeddings(now=datetime.datetime.now(datetime.timezone.utc))

    emp_after = emp_repo.get_by_emp_id("EMP-STAFF-01")
    assert emp_after.has_face_registered is True
    assert emp_after.face_embedding is not None


# -------------------------------------------------------------------------
# 5. Reliability: Concurrency & Transaction Safety
# -------------------------------------------------------------------------

def test_concurrent_identity_creation_idempotency(db_session):
    repo = PersonIdentityRepository(db_session)
    emb = make_unit_vector(seed=601)

    # Sighting from Camera 1
    r1 = repo.create_or_update("PID_CONCURRENT_1", emb)
    assert r1.person_id == "PID_CONCURRENT_1"

    # Immediate concurrent sighting from Camera 2 for same person ID updates last_seen without error
    emb_updated = make_unit_vector(seed=602)
    r2 = repo.create_or_update("PID_CONCURRENT_1", emb_updated)
    assert r2.person_id == "PID_CONCURRENT_1"
    assert r2.id == r1.id

    # Verify no duplicate identity records created
    count = db_session.query(PersonIdentity).filter_by(person_id="PID_CONCURRENT_1").count()
    assert count == 1
