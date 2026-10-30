import datetime
import uuid
import numpy as np
import pytest
from app.db.models.employee import Employee
from app.db.models.employee_attendance import EmployeeAttendanceLog, EmployeeDailyAttendance
from app.db.models.person_identity import PersonIdentity
from app.repositories.employee_repo import EmployeeRepository
from app.repositories.employee_attendance_repo import EmployeeAttendanceRepository
from app.repositories.person_identity_repo import PersonIdentityRepository
from app.services.retention_service import RetentionService
from app.vision.employee_recognizer import EmployeeRecognitionManager
from app.vision.vector_store import PgVectorStoreManager


def make_random_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# -------------------------------------------------------------------------
# 1. Employee Repository & Management Tests
# -------------------------------------------------------------------------

def test_employee_creation_and_retrieval(db_session):
    repo = EmployeeRepository(db_session)
    emp = repo.create(
        employee_id="EMP-1001",
        name="Alice Sharma",
        department="Sales",
        designation="Sales Executive",
        shift_start_time=datetime.time(9, 30),
        shift_end_time=datetime.time(18, 30),
        late_grace_minutes=15,
        early_departure_grace_minutes=15,
    )
    assert emp.id is not None
    assert emp.employee_id == "EMP-1001"
    assert emp.name == "Alice Sharma"
    assert emp.is_active is True

    fetched = repo.get_by_emp_id("EMP-1001")
    assert fetched is not None
    assert fetched.name == "Alice Sharma"
    assert fetched.shift_start_time == datetime.time(9, 30)


def test_employee_update_and_deactivation(db_session):
    repo = EmployeeRepository(db_session)
    emp = repo.create(
        employee_id="EMP-1002",
        name="Bob Verma",
        department="Inventory",
        designation="Manager",
    )
    updated = repo.update(emp, name="Robert Verma", department="Operations", is_active=False)
    assert updated.name == "Robert Verma"
    assert updated.department == "Operations"
    assert updated.is_active is False

    active_emps = repo.list_employees(is_active=True)
    assert not any(e.employee_id == "EMP-1002" for e in active_emps)


def test_employee_biometric_face_registration(db_session):
    repo = EmployeeRepository(db_session)
    emp = repo.create(
        employee_id="EMP-1003",
        name="Charlie Patel",
        department="Security",
    )
    emb = make_random_embedding(seed=101)
    updated = repo.update_face_embedding(emp, emb)
    assert updated.has_face_registered is True
    assert updated.face_embedding is not None

    stored_emb = updated.get_embedding_array()
    assert stored_emb is not None
    assert np.allclose(stored_emb, emb, atol=1e-5)


# -------------------------------------------------------------------------
# 2. Dedicated Employee Recognition Engine (ArcFace + pgvector)
# -------------------------------------------------------------------------

def test_employee_recognition_manager_matching():
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.55)
    recog_mgr = EmployeeRecognitionManager(vector_store=vector_mgr, similarity_threshold=0.55)

    emp1_emb = make_random_embedding(seed=201)
    emp2_emb = make_random_embedding(seed=202)

    recog_mgr.register_employee("EMP-001", emp1_emb)
    recog_mgr.register_employee("EMP-002", emp2_emb)

    # Mock embedder output
    class MockEmbedder:
        def __init__(self, emb):
            self.emb = emb
        def is_face_reliable(self, face_crop, face_conf, landmarks):
            return True, "reliable"
        def extract_embedding(self, face_crop, landmarks):
            return self.emb

    # Probe with emp1 face
    recog_mgr.embedder = MockEmbedder(emp1_emb)
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    match = recog_mgr.evaluate_track_face(tracker_id="track_10", face_crop=dummy_crop, face_conf=0.95)

    assert match is not None
    emp_id, score = match
    assert emp_id == "EMP-001"
    assert score > 0.95
    assert recog_mgr.is_track_employee("track_10") is True
    assert recog_mgr.get_employee_id("track_10") == "EMP-001"

    # Maintain employee ID across ByteTrack tracking session
    cached_match = recog_mgr.evaluate_track_face(tracker_id="track_10", face_crop=None, face_conf=0.0)
    assert cached_match == ("EMP-001", 1.0)

    # When track disappears and new track ID assigned
    recog_mgr.on_track_lost("track_10")
    assert recog_mgr.is_track_employee("track_10") is False

    # New track 11 for the same employee
    match_new_track = recog_mgr.evaluate_track_face(tracker_id="track_11", face_crop=dummy_crop, face_conf=0.92)
    assert match_new_track is not None
    assert match_new_track[0] == "EMP-001"
    assert recog_mgr.get_employee_id("track_11") == "EMP-001"


def test_employee_recognition_manager_unknown_visitor():
    vector_mgr = PgVectorStoreManager(similarity_threshold=0.60)
    recog_mgr = EmployeeRecognitionManager(vector_store=vector_mgr, similarity_threshold=0.60)

    emp_emb = make_random_embedding(seed=301)
    recog_mgr.register_employee("EMP-ADMIN", emp_emb)

    # Visitor with orthogonal/unrelated embedding
    visitor_emb = make_random_embedding(seed=999)

    class MockEmbedder:
        def is_face_reliable(self, face_crop, face_conf, landmarks):
            return True, "reliable"
        def extract_embedding(self, face_crop, landmarks):
            return visitor_emb

    recog_mgr.embedder = MockEmbedder()
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    match = recog_mgr.evaluate_track_face(tracker_id="track_99", face_crop=dummy_crop, face_conf=0.90)

    # Unknown visitor must NOT be recognized as staff
    assert match is None
    assert recog_mgr.is_track_employee("track_99") is False
    assert recog_mgr.get_employee_id("track_99") is None


# -------------------------------------------------------------------------
# 3. Attendance Rules & Daily Aggregations
# -------------------------------------------------------------------------

def test_attendance_checkin_checkout_hours_and_debounce(db_session):
    emp_repo = EmployeeRepository(db_session)
    att_repo = EmployeeAttendanceRepository(db_session)

    emp = emp_repo.create(
        employee_id="EMP-2001",
        name="Devendra Joshi",
        department="Finance",
        shift_start_time=datetime.time(9, 0),
        shift_end_time=datetime.time(18, 0),
        late_grace_minutes=15,
        early_departure_grace_minutes=15,
    )

    t_base = datetime.datetime(2026, 10, 29, 8, 55, 0, tzinfo=datetime.timezone.utc)

    # 1. First punch (Check-in) on time
    log1, daily1 = att_repo.record_attendance_event(
        employee_id="EMP-2001",
        event_time=t_base,
        event_type="CHECK_IN",
        confidence=0.98,
        debounce_seconds=180,
    )
    assert log1 is not None
    assert daily1 is not None
    assert daily1.status == "PRESENT"
    assert daily1.is_late is False
    assert daily1.late_minutes == 0
    assert daily1.total_work_hours == 0.0

    # 2. Debounced punch within 3 minutes (e.g. at 8:56:30)
    t_dup = t_base + datetime.timedelta(seconds=90)
    log_dup, daily_dup = att_repo.record_attendance_event(
        employee_id="EMP-2001",
        event_time=t_dup,
        debounce_seconds=180,
    )
    assert log_dup is None  # Suppressed duplicate!
    logs_debounced = att_repo.get_employee_logs(emp.id)
    assert len(logs_debounced) == 1

    # 3. Exit punch at 18:05:00 (after 9.1 hours)
    t_exit = datetime.datetime(2026, 10, 29, 18, 5, 0, tzinfo=datetime.timezone.utc)
    log_exit, daily_exit = att_repo.record_attendance_event(
        employee_id="EMP-2001",
        event_time=t_exit,
        event_type="CHECK_OUT",
        debounce_seconds=180,
    )
    assert log_exit is not None
    logs_total = att_repo.get_employee_logs(emp.id)
    assert len(logs_total) == 2
    assert daily_exit.is_early_departure is False
    assert daily_exit.early_departure_minutes == 0
    assert daily_exit.total_work_hours >= 9.15


def test_attendance_late_arrival_and_early_departure(db_session):
    emp_repo = EmployeeRepository(db_session)
    att_repo = EmployeeAttendanceRepository(db_session)

    emp = emp_repo.create(
        employee_id="EMP-2002",
        name="Pooja Mehta",
        department="Marketing",
        shift_start_time=datetime.time(9, 0),
        shift_end_time=datetime.time(18, 0),
        late_grace_minutes=15,
        early_departure_grace_minutes=15,
    )

    # Shift starts at 09:00. Arrival at 09:35 (20 mins beyond 15m grace)
    t_late = datetime.datetime(2026, 10, 29, 9, 35, 0, tzinfo=datetime.timezone.utc)
    log_in, daily = att_repo.record_attendance_event(
        employee_id="EMP-2002",
        event_time=t_late,
        debounce_seconds=60,
    )
    assert daily.is_late is True
    assert daily.late_minutes == 35

    # Exit early at 17:15 (45 mins before 18:00, beyond 15m grace)
    t_early_exit = datetime.datetime(2026, 10, 29, 17, 15, 0, tzinfo=datetime.timezone.utc)
    log_out, daily_updated = att_repo.record_attendance_event(
        employee_id="EMP-2002",
        event_time=t_early_exit,
        debounce_seconds=60,
    )
    assert daily_updated.is_early_departure is True
    assert daily_updated.early_departure_minutes == 45
    assert daily_updated.status == "LATE"


def test_attendance_mark_absences(db_session):
    emp_repo = EmployeeRepository(db_session)
    att_repo = EmployeeAttendanceRepository(db_session)

    # Create 2 active employees and 1 inactive
    e1 = emp_repo.create(employee_id="EMP-ACTIVE-1", name="Emp One")
    e2 = emp_repo.create(employee_id="EMP-ACTIVE-2", name="Emp Two")
    e3 = emp_repo.create(employee_id="EMP-INACTIVE", name="Emp Three", is_active=False)

    target_date = datetime.date(2026, 10, 29)

    # Only e1 checks in
    att_repo.record_attendance_event(
        employee_id=e1.employee_id,
        event_time=datetime.datetime(2026, 10, 29, 9, 0, tzinfo=datetime.timezone.utc),
    )

    # Mark absences for the day
    absent_count = att_repo.mark_absences(target_date)
    assert absent_count == 1  # Only e2 should be marked absent, inactive e3 is ignored

    records = att_repo.get_daily_summary(target_date)
    status_map = {r.employee.employee_id: r.status for r in records}
    assert status_map["EMP-ACTIVE-1"] == "PRESENT"
    assert status_map["EMP-ACTIVE-2"] == "ABSENT"
    assert "EMP-INACTIVE" not in status_map


# -------------------------------------------------------------------------
# 4. Data Retention Isolation: Employee Biometrics are NOT Expired
# -------------------------------------------------------------------------

def test_employee_biometric_data_retention_exemption(db_session):
    emp_repo = EmployeeRepository(db_session)
    ident_repo = PersonIdentityRepository(db_session)
    visitor_vector_store = PgVectorStoreManager(similarity_threshold=0.50)

    # 1. Register employee with face embedding
    emp = emp_repo.create(employee_id="EMP-PERMANENT", name="Staff Member")
    emp_emb = make_random_embedding(seed=777)
    emp_repo.update_face_embedding(emp, emp_emb)

    # 2. Register visitor with face embedding created 20 days ago
    vis_emb = make_random_embedding(seed=888)
    visitor_vector_store.add("VISITOR-EXPIRED", vis_emb)
    old_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=20)
    ident_repo.create_or_update("VISITOR-EXPIRED", vis_emb, old_time)

    # Verify both exist before retention cleanup
    assert ident_repo.get_by_person_id("VISITOR-EXPIRED") is not None
    assert emp_repo.get_by_emp_id("EMP-PERMANENT") is not None
    assert emp.face_embedding is not None

    # 3. Run visitor 14-day rolling retention policy
    retention_svc = RetentionService(
        db_session=db_session,
        vector_store=visitor_vector_store,
        retention_days=14,
    )
    result = retention_svc.cleanup_expired_embeddings()
    assert result.expired_count == 1

    # Visitor embedding must be removed
    assert visitor_vector_store.get_count() == 0
    expired_vis = ident_repo.get_by_person_id("VISITOR-EXPIRED")
    assert expired_vis.face_embedding is None

    # CRITICAL: Employee record and biometric face embedding MUST REMAIN INTACT!
    emp_after = emp_repo.get_by_emp_id("EMP-PERMANENT")
    assert emp_after is not None
    assert emp_after.face_embedding is not None
    assert emp_after.has_face_registered is True
    assert np.allclose(emp_after.get_embedding_array(), emp_emb, atol=1e-5)


# -------------------------------------------------------------------------
# 5. REST APIs for Employee Registration and Attendance
# -------------------------------------------------------------------------

def test_employee_registration_and_listing_api(client):
    # 1. Create employee
    payload = {
        "employee_id": "EMP-REST-01",
        "name": "Sunita Rao",
        "department": "Jewelry Design",
        "designation": "Chief Designer",
        "shift_start_time": "10:00:00",
        "shift_end_time": "19:00:00",
        "late_grace_minutes": 20,
        "early_departure_grace_minutes": 20,
    }
    res = client.post("/api/v1/employees", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["employee_id"] == "EMP-REST-01"
    assert data["name"] == "Sunita Rao"
    assert data["has_face_registered"] is False
    # Biometric data MUST NOT be exposed in response
    assert "face_embedding" not in data

    # 2. Duplicate ID fails
    res_dup = client.post("/api/v1/employees", json=payload)
    assert res_dup.status_code == 409

    # 3. List employees
    res_list = client.get("/api/v1/employees?department=Jewelry Design")
    assert res_list.status_code == 200
    emps = res_list.json()
    assert len(emps) == 1
    assert emps[0]["employee_id"] == "EMP-REST-01"


def test_employee_biometric_enrollment_api(client):
    # Register employee
    client.post("/api/v1/employees", json={
        "employee_id": "EMP-BIO-01",
        "name": "Karan Singhania",
        "department": "Showroom",
    })

    # Enroll face embedding vector
    emb = make_random_embedding(seed=555).tolist()
    res = client.post("/api/v1/employees/EMP-BIO-01/face", json={"embedding": emb})
    assert res.status_code == 200
    data = res.json()
    assert data["has_face_registered"] is True
    assert "face_embedding" not in data


def test_manual_attendance_punch_api(client):
    client.post("/api/v1/employees", json={
        "employee_id": "EMP-PUNCH-01",
        "name": "Rahul Kapoor",
        "department": "Security",
    })

    # Record punch
    punch_payload = {
        "employee_id": "EMP-PUNCH-01",
        "event_type": "CHECK_IN",
        "confidence": 0.96,
    }
    res = client.post("/api/v1/attendance/punch", json=punch_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["employee_id"] == "EMP-PUNCH-01"
    assert data["status"] in ("PRESENT", "LATE")


def test_employee_multi_image_upload_api(client):
    import cv2
    # 1. Register employee
    client.post("/api/v1/employees", json={
        "employee_id": "EMP-MULTI-01",
        "name": "Ananya Sen",
        "department": "Sales",
    })

    # 2. Generate two test portrait image buffers
    img1 = np.full((120, 120, 3), 180, dtype=np.uint8)
    img2 = np.full((120, 120, 3), 210, dtype=np.uint8)
    _, buf1 = cv2.imencode(".jpg", img1)
    _, buf2 = cv2.imencode(".jpg", img2)

    # 3. Upload multiple images with 1st photo as profile picture
    files = [
        ("files", ("avatar_angle1.jpg", buf1.tobytes(), "image/jpeg")),
        ("files", ("avatar_angle2.jpg", buf2.tobytes(), "image/jpeg")),
    ]
    res = client.post("/api/v1/employees/EMP-MULTI-01/face/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["has_face_registered"] is True
    # Verify 1st photo was saved as primary profile picture
    assert data["photo_url"] is not None
    assert "EMP-MULTI-01_profile" in data["photo_url"]


def test_employee_multi_image_base64_enrollment_api(client):
    import base64
    import cv2
    img1 = np.full((100, 100, 3), 150, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
    _, buf1 = cv2.imencode(".jpg", img1)
    _, buf2 = cv2.imencode(".jpg", img2)
    b64_1 = base64.b64encode(buf1.tobytes()).decode("ascii")
    b64_2 = base64.b64encode(buf2.tobytes()).decode("ascii")

    res = client.post("/api/v1/employees", json={
        "employee_id": "EMP-MULTI-B64",
        "name": "Deepak Mehta",
        "department": "Customer Relations",
        "face_images_base64": [b64_1, b64_2],
    })
    assert res.status_code == 201
    data = res.json()
    assert data["has_face_registered"] is True
    assert data["face_photos_count"] >= 1
    assert data["photo_url"] is not None
    assert "EMP-MULTI-B64_profile" in data["photo_url"]


def test_multi_frame_voting_buffer():
    """Point 1: Verifies multi-frame temporal confirmation eliminates single-frame false matches."""
    from app.vision.vector_store import PgVectorStoreManager
    store = PgVectorStoreManager(similarity_threshold=0.55)
    recog_mgr = EmployeeRecognitionManager(
        vector_store=store,
        similarity_threshold=0.55,
        min_confirm_frames=3,
        vote_window=5,
    )

    emp_emb = make_random_embedding(seed=401)
    recog_mgr.register_employee("EMP-VOTE-01", emp_emb)

    class MockEmbedder:
        def is_face_reliable(self, face_crop, face_conf, landmarks):
            return True, "reliable"
        def extract_embedding(self, face_crop, landmarks):
            return emp_emb

    recog_mgr.embedder = MockEmbedder()
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)

    # Frame 1: tentative match -> returns None (pending confirmation)
    f1 = recog_mgr.evaluate_track_face(tracker_id="track_voting", face_crop=dummy_crop, face_conf=0.95)
    assert f1 is None
    assert recog_mgr.is_track_employee("track_voting") is False

    # Frame 2: tentative match -> returns None (still only 2 frames)
    f2 = recog_mgr.evaluate_track_face(tracker_id="track_voting", face_crop=dummy_crop, face_conf=0.95)
    assert f2 is None
    assert recog_mgr.is_track_employee("track_voting") is False

    # Frame 3: 3rd consecutive confirmation -> verified and confirmed!
    f3 = recog_mgr.evaluate_track_face(tracker_id="track_voting", face_crop=dummy_crop, face_conf=0.95)
    assert f3 is not None
    assert f3[0] == "EMP-VOTE-01"
    assert recog_mgr.is_track_employee("track_voting") is True
    assert recog_mgr.get_employee_id("track_voting") == "EMP-VOTE-01"


def test_pgvector_multi_angle_gallery_matching(db_session):
    """Point 2: Verifies pgvector multi-angle gallery matches alternate face angles at maximum score."""
    from app.repositories.employee_repo import EmployeeRepository
    from app.vision.vector_store import PgVectorStoreManager
    repo = EmployeeRepository(db_session)
    store = PgVectorStoreManager(db=db_session, similarity_threshold=0.55)

    front_angle = make_random_embedding(seed=501)
    left_angle = make_random_embedding(seed=502)
    right_angle = make_random_embedding(seed=503)

    emp = repo.create(
        employee_id="EMP-GALLERY-01",
        name="Karan Malhotra",
        department="Sales",
        embedding=front_angle,
        gallery_embeddings=[front_angle, left_angle, right_angle],
    )
    assert emp.faces is not None
    assert len(emp.faces) == 3

    # Probe 1: probe matching left_angle
    best_id, score = store.find_best_employee_match(left_angle, db=db_session)
    assert best_id == "EMP-GALLERY-01"
    assert score >= 0.99

    # Probe 2: probe matching right_angle
    best_id_r, score_r = store.find_best_employee_match(right_angle, db=db_session)
    assert best_id_r == "EMP-GALLERY-01"
    assert score_r >= 0.99


def test_check_in_after_auto_mark_absent(db_session, client):
    """Verifies that an employee pre-marked ABSENT (first_check_in=None) correctly updates to PRESENT on check-in punch."""
    from app.repositories.employee_repo import EmployeeRepository
    from app.repositories.employee_attendance_repo import EmployeeAttendanceRepository
    from datetime import date, datetime, time, timezone

    emp_repo = EmployeeRepository(db_session)
    att_repo = EmployeeAttendanceRepository(db_session)

    emp = emp_repo.create(
        employee_id="EMP-PREABSENT-01",
        name="Aarav Sharma",
        department="Floor",
        shift_start_time=time(9, 0),
        late_grace_minutes=30,
    )

    # 1. Simulate auto-marking absences (e.g. daily report generation before arrival)
    today = date.today()
    absent_count = att_repo.mark_absences(today)
    assert absent_count >= 1

    daily_records = att_repo.get_employee_history(emp.id, start_date=today, end_date=today)
    assert len(daily_records) == 1
    daily_before = daily_records[0]
    assert daily_before.status == "ABSENT"
    assert daily_before.first_check_in is None

    # 2. Employee arrives and is recognized by camera at 09:15 (within grace period)
    punch_time = datetime.combine(today, datetime.strptime("09:15:00", "%H:%M:%S").time()).replace(tzinfo=timezone.utc)
    log, daily_after = att_repo.record_attendance_event(
        employee=emp,
        timestamp=punch_time,
        confidence=0.98,
    )

    # 3. Employee must now be marked PRESENT with valid first_check_in
    assert daily_after.status == "PRESENT"
    assert daily_after.first_check_in is not None
    assert daily_after.is_late is False


def test_add_more_photos_after_registration_api(client):
    import cv2
    # 1. Register employee
    res_create = client.post("/api/v1/employees", json={
        "employee_id": "EMP-ADDPHOTO-01",
        "name": "Pooja Hegde",
        "department": "Sales",
    })
    assert res_create.status_code == 201

    # 2. Upload initial batch of 2 photos
    img1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 120, dtype=np.uint8)
    _, buf1 = cv2.imencode(".jpg", img1)
    _, buf2 = cv2.imencode(".jpg", img2)

    files_initial = [
        ("files", ("photo1.jpg", buf1.tobytes(), "image/jpeg")),
        ("files", ("photo2.jpg", buf2.tobytes(), "image/jpeg")),
    ]
    res_init = client.post("/api/v1/employees/EMP-ADDPHOTO-01/face/upload", files=files_initial)
    assert res_init.status_code == 200
    data_init = res_init.json()
    assert data_init["has_face_registered"] is True
    assert data_init["face_photos_count"] == 2
    initial_avatar = data_init["photo_url"]
    assert initial_avatar is not None

    # 3. Add 2 more photos via POST /api/v1/employees/{employee_id}/photos
    img3 = np.full((100, 100, 3), 150, dtype=np.uint8)
    img4 = np.full((100, 100, 3), 180, dtype=np.uint8)
    _, buf3 = cv2.imencode(".jpg", img3)
    _, buf4 = cv2.imencode(".jpg", img4)

    files_more = [
        ("files", ("photo3.jpg", buf3.tobytes(), "image/jpeg")),
        ("files", ("photo4.jpg", buf4.tobytes(), "image/jpeg")),
    ]
    res_more = client.post("/api/v1/employees/EMP-ADDPHOTO-01/photos", files=files_more)
    assert res_more.status_code == 200
    data_more = res_more.json()
    assert data_more["has_face_registered"] is True
    # Gallery now contains total of 4 photos!
    assert data_more["face_photos_count"] == 4
    # Existing avatar preserved by default
    assert data_more["photo_url"] == initial_avatar

    # 4. Verify employee list returns updated face_photos_count
    res_list = client.get("/api/v1/employees")
    assert res_list.status_code == 200
    matched = [e for e in res_list.json() if e["employee_id"] == "EMP-ADDPHOTO-01"]
    assert len(matched) == 1
    assert matched[0]["face_photos_count"] == 4



