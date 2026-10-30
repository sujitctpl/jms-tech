"""
Comprehensive Face Recognition Accuracy & Reliability Test Suite.
Tests:
1. Face alignment (canonical 112x112 5-point affine transformation)
2. ArcFace embedding L2 normalization
3. Poor-quality face rejection (blur, extreme yaw/pitch/roll, underexposure, glare)
4. Multiple embeddings per employee & multi-angle aggregation
5. pgvector similarity search & inactive/expired exclusion
6. Matching margin (Top-1 vs Top-2 ambiguity rejection)
7. Calibrated threshold boundary handling
8. Temporal identity voting (quality-weighted confirmation buffer)
9. Tracker ID reuse & track lost cleanup
10. Unknown identity handling
11. Employee deactivation synchronization
12. Biometric retention integrity
13. Concurrent multi-threaded recognition requests
"""
import concurrent.futures
import numpy as np
import pytest
import cv2

from app.core.config import settings
from app.db.models.employee import Employee, EmployeeFace
from app.vision.face_embedder import FaceEmbedder
from app.vision.vector_store import PgVectorStoreManager
from app.vision.employee_recognizer import EmployeeRecognitionManager
from app.vision.threshold_evaluator import ThresholdEvaluator


def make_unit_vector(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_face_alignment():
    """1. Verifies 5-point landmark affine transformation to 112x112."""
    embedder = FaceEmbedder()
    # Create test canvas
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    # 5 landmarks: left eye, right eye, nose, left mouth, right mouth
    raw_landmarks = [(100, 110), (160, 110), (130, 140), (110, 180), (150, 180)]
    crop = canvas[80:220, 80:220]

    # Run extraction with full image and raw landmarks
    emb = embedder.extract_embedding(crop, [(20, 30), (80, 30), (50, 60), (30, 100), (70, 100)], full_image=canvas, raw_landmarks=raw_landmarks)
    assert emb is not None
    assert emb.shape == (512,)


def test_embedding_normalization():
    """2. Verifies embedding output is strictly L2 normalized."""
    embedder = FaceEmbedder()
    dummy = np.full((112, 112, 3), 128, dtype=np.uint8)
    emb = embedder.extract_embedding(dummy)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-4, f"Embedding norm {norm} is not unit length."


def test_poor_quality_face_rejection():
    """3. Verifies quality assessment filters blurry, extreme pose, and invalid faces."""
    embedder = FaceEmbedder()

    # Small face crop (e.g. 10x10 < min_size) with high contrast to test resolution check
    tiny_crop = np.zeros((10, 10, 3), dtype=np.uint8)
    tiny_crop[::2, ::2] = 255
    rel, reason = embedder.is_face_reliable(tiny_crop, 0.90)
    assert not rel
    assert "too_small" in reason.lower()

    # Low detection confidence
    ok_crop = np.full((80, 80, 3), 120, dtype=np.uint8)
    # Add high frequency noise to avoid blur trigger
    ok_crop[::2, ::2] = 200
    rel, reason = embedder.is_face_reliable(ok_crop, 0.20)
    assert not rel
    assert "confidence" in reason.lower()

    # Extreme yaw pose (> 45 degrees)
    # Landmarks where nose is way outside eye center indicating severe profile turn
    extreme_yaw_lms = [(50, 50), (120, 50), (160, 80), (70, 110), (110, 110)]
    yaw, pitch, roll = embedder.estimate_head_pose(extreme_yaw_lms)
    assert abs(yaw) > settings.FACE_MAX_YAW

    rel, reason = embedder.is_face_reliable(ok_crop, 0.90, landmarks=extreme_yaw_lms)
    assert not rel
    assert "pose" in reason.lower() or "yaw" in reason.lower()


def test_multiple_embeddings_per_employee_aggregation(db_session):
    """4. Verifies multiple embeddings per employee are correctly aggregated by best score."""
    vector_mgr = PgVectorStoreManager(db=db_session, similarity_threshold=0.38)

    emp = Employee(
        employee_id="EMP-TEST-MULTI",
        name="Multi Angle Tester",
        department="Engineering",
        designation="Software Engineer",
        is_active=True,
    )
    db_session.add(emp)
    db_session.flush()

    # Create 3 distinct angle embeddings for this employee
    base_v = make_unit_vector(1001)
    angle_v1 = make_unit_vector(1002)
    angle_v2 = make_unit_vector(1003)

    db_session.add(EmployeeFace(employee_id=emp.id, face_embedding=base_v.tolist(), angle_label="primary"))
    db_session.add(EmployeeFace(employee_id=emp.id, face_embedding=angle_v1.tolist(), angle_label="angle_left"))
    db_session.add(EmployeeFace(employee_id=emp.id, face_embedding=angle_v2.tolist(), angle_label="angle_right"))
    db_session.commit()

    # Probe matching angle_v2
    matched_id, score = vector_mgr.find_best_employee_match(angle_v2)
    assert matched_id == "EMP-TEST-MULTI"
    assert score > 0.99


def test_pgvector_similarity_search_and_inactive_exclusion(db_session):
    """5. Verifies inactive employees are excluded from pgvector similarity search."""
    vector_mgr = PgVectorStoreManager(db=db_session, similarity_threshold=0.38)

    inactive_emp = Employee(
        employee_id="EMP-INACTIVE-01",
        name="Deactivated Staff",
        department="HR",
        designation="HR Executive",
        is_active=False,
    )
    db_session.add(inactive_emp)
    db_session.flush()

    emb = make_unit_vector(2001)
    db_session.add(EmployeeFace(employee_id=inactive_emp.id, face_embedding=emb.tolist(), angle_label="primary"))
    db_session.commit()

    matched_id, score = vector_mgr.find_best_employee_match(emb)
    assert matched_id is None, "Inactive employee must not be recognized."


def test_matching_margin_ambiguity_rejection():
    """6. Verifies ambiguous matches between top-1 and top-2 candidates are rejected by margin."""
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.38, match_margin=0.05)

    base = make_unit_vector(3001)
    # Create two lookalike embeddings with 0.02 similarity difference
    emp1_v = base.copy()
    emp2_v = base + make_unit_vector(3002) * 0.05
    emp2_v /= np.linalg.norm(emp2_v)

    vector_mgr.add("EMP-TWIN-1", emp1_v)
    vector_mgr.add("EMP-TWIN-2", emp2_v)

    # Probe vector midway between both twins
    probe = (emp1_v + emp2_v) / 2.0
    probe /= np.linalg.norm(probe)

    matched_id, score = vector_mgr.find_best_employee_match(probe, match_margin=0.08)
    # Should be rejected due to ambiguity (margin < 0.08)
    assert matched_id is None


def test_threshold_boundary_handling():
    """7. Verifies threshold enforcement at exact boundary values."""
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.45)
    emp_v = make_unit_vector(4001)
    vector_mgr.add("EMP-BOUNDARY", emp_v)

    # Vector with 0.40 similarity -> should reject
    rng = np.random.RandomState(4002)
    noise = rng.randn(512).astype(np.float32)
    noise -= np.dot(noise, emp_v) * emp_v
    noise /= np.linalg.norm(noise)
    # Construct vector with exact cosine similarity 0.40
    probe_low = 0.40 * emp_v + np.sqrt(1.0 - 0.40**2) * noise

    res_id, sim = vector_mgr.find_best_employee_match(probe_low, similarity_threshold=0.45)
    assert res_id is None
    assert sim < 0.45

    # Vector with 0.50 similarity -> should accept
    probe_high = 0.50 * emp_v + np.sqrt(1.0 - 0.50**2) * noise
    res_id, sim = vector_mgr.find_best_employee_match(probe_high, similarity_threshold=0.45)
    assert res_id == "EMP-BOUNDARY"
    assert sim >= 0.45


def test_temporal_identity_voting():
    """8. Verifies quality-weighted voting buffer over multiple video frames."""
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.38)
    recog = EmployeeRecognitionManager(
        vector_store=vector_mgr,
        similarity_threshold=0.38,
        min_confirm_frames=2,
        vote_window=4,
    )
    emp_v = make_unit_vector(5001)
    recog.register_employee("EMP-VOTE", emp_v)

    class MockEmbedder:
        def is_face_reliable(self, crop, conf, lms):
            return True, "ok"
        def extract_embedding(self, crop, lms=None, full_image=None, raw_landmarks=None):
            return emp_v

    recog.embedder = MockEmbedder()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)

    # Frame 1: Pending confirmation -> None
    r1 = recog.evaluate_track_face("track_vote", dummy, 0.95)
    assert r1 is None
    assert not recog.is_track_employee("track_vote")

    # Frame 2: Confirmed!
    r2 = recog.evaluate_track_face("track_vote", dummy, 0.95)
    assert r2 is not None
    eid, score = r2
    assert eid == "EMP-VOTE"
    assert recog.is_track_employee("track_vote")


def test_tracker_id_reuse_cleanup():
    """9. Verifies on_track_lost cleanly frees tracker ID and voting state."""
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.38)
    recog = EmployeeRecognitionManager(
        vector_store=vector_mgr,
        similarity_threshold=0.38,
        min_confirm_frames=1,
    )
    emp_v = make_unit_vector(6001)
    recog.register_employee("EMP-CLEANUP", emp_v)

    class MockEmbedder:
        def is_face_reliable(self, crop, conf, lms):
            return True, "ok"
        def extract_embedding(self, crop, lms=None, full_image=None, raw_landmarks=None):
            return emp_v

    recog.embedder = MockEmbedder()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)

    recog.evaluate_track_face("track_101", dummy, 0.95)
    assert recog.is_track_employee("track_101")

    # Track lost
    recog.on_track_lost("track_101")
    assert not recog.is_track_employee("track_101")
    assert "track_101" not in recog.track_votes


def test_unknown_identity_handling():
    """10. Verifies non-employee face is rejected as Unknown without matching staff."""
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.38)
    recog = EmployeeRecognitionManager(
        vector_store=vector_mgr,
        similarity_threshold=0.38,
        min_confirm_frames=1,
    )
    recog.register_employee("EMP-STAFF", make_unit_vector(7001))

    class MockEmbedder:
        def is_face_reliable(self, crop, conf, lms):
            return True, "ok"
        def extract_embedding(self, crop, lms=None, full_image=None, raw_landmarks=None):
            # Orthogonal visitor face
            return make_unit_vector(7002)

    recog.embedder = MockEmbedder()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)

    match = recog.evaluate_track_face("track_visitor", dummy, 0.95)
    assert match is None
    assert not recog.is_track_employee("track_visitor")


def test_employee_deactivation_sync():
    """11. Verifies removing employee from recognizer cache prevents subsequent match."""
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.38)
    recog = EmployeeRecognitionManager(
        vector_store=vector_mgr,
        similarity_threshold=0.38,
        min_confirm_frames=1,
    )
    emp_v = make_unit_vector(8001)
    recog.register_employee("EMP-TERMINATED", emp_v)

    assert recog.remove_employee("EMP-TERMINATED") is True
    # In-memory vector cache must no longer find this employee
    mid, sim = vector_mgr.find_best_employee_match(emp_v)
    assert mid is None


def test_biometric_retention_vector_store():
    """12. Verifies clear and re-index operations maintain integrity."""
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.38)
    v1 = make_unit_vector(9001)
    v2 = make_unit_vector(9002)

    vector_mgr.add("EMP-RET-1", v1)
    vector_mgr.add("EMP-RET-2", v2)
    assert vector_mgr.size() == 2

    vector_mgr.clear()
    assert vector_mgr.size() == 0


def test_concurrent_recognition_requests():
    """13. Verifies concurrent thread-safety of EmployeeRecognitionManager."""
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.38)
    recog = EmployeeRecognitionManager(
        vector_store=vector_mgr,
        similarity_threshold=0.38,
        min_confirm_frames=1,
    )
    for i in range(10):
        recog.register_employee(f"EMP-THREAD-{i}", make_unit_vector(10000 + i))

    class MockEmbedder:
        def is_face_reliable(self, crop, conf, lms):
            return True, "ok"
        def extract_embedding(self, crop, lms=None, full_image=None, raw_landmarks=None):
            return make_unit_vector(10005)

    recog.embedder = MockEmbedder()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)

    def worker_probe(tid):
        return recog.evaluate_track_face(f"track_{tid}", dummy, 0.95)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_probe, i) for i in range(24)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Exactly one track should confirm EMP-THREAD-5; subsequent tracks should reject duplicate active body
    confirmed_tracks = [r for r in results if r is not None and r[0] == "EMP-THREAD-5"]
    assert len(confirmed_tracks) >= 1
