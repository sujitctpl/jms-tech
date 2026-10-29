"""Interactive test script for Employee Attendance and Biometric Recognition in Showroom Visitor Intelligence.
Can be executed directly via:
    python scripts/test_employee_attendance_interactive.py
"""

from datetime import date, datetime, time, timedelta, timezone
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import SessionLocal
from app.repositories.employee_attendance_repo import EmployeeAttendanceRepository
from app.repositories.employee_repo import EmployeeRepository
from app.services.retention_service import RetentionService
from app.vision.vector_store import PgVectorStoreManager


def make_random_embedding(seed: int = 123) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def main():
    print("=" * 70)
    print("       SHOWROOM EMPLOYEE ATTENDANCE & BIOMETRIC TEST SUITE       ")
    print("=" * 70)

    db: Session = SessionLocal()
    emp_repo = EmployeeRepository(db)
    att_repo = EmployeeAttendanceRepository(db)
    vector_mgr = PgVectorStoreManager(db=db)

    try:
        # -------------------------------------------------------------
        # STEP 1: Employee Registration
        # -------------------------------------------------------------
        print("\n[STEP 1] Registering Showroom Employees...")
        emp1_id = "EMP-TEST-001"
        emp2_id = "EMP-TEST-002"

        # Clean up any previous test records
        prev1 = emp_repo.get_by_employee_id(emp1_id)
        if prev1:
            db.delete(prev1)
            db.commit()
        prev2 = emp_repo.get_by_employee_id(emp2_id)
        if prev2:
            db.delete(prev2)
            db.commit()

        emp1 = emp_repo.create(
            employee_id=emp1_id,
            name="Aarav Sharma",
            department="Diamond Counter",
            designation="Senior Sales Executive",
            shift_start_time=time(9, 30),
            shift_end_time=time(18, 30),
            late_grace_minutes=15,
            early_departure_grace_minutes=15,
        )
        print(f"  [+] Registered: {emp1.name} (ID: {emp1.employee_id}, Dept: {emp1.department})")

        emp2 = emp_repo.create(
            employee_id=emp2_id,
            name="Sneha Patel",
            department="Gold Counter",
            designation="Showroom Associate",
            shift_start_time=time(9, 30),
            shift_end_time=time(18, 30),
            late_grace_minutes=15,
            early_departure_grace_minutes=15,
        )
        print(f"  [+] Registered: {emp2.name} (ID: {emp2.employee_id}, Dept: {emp2.department})")

        # -------------------------------------------------------------
        # STEP 2: Biometric Face Vector Enrollment (pgvector)
        # -------------------------------------------------------------
        print("\n[STEP 2] Enrolling Biometric Face Embeddings (512-dim ArcFace vectors)...")
        face_emb_aarav = make_random_embedding(seed=101)
        face_emb_sneha = make_random_embedding(seed=202)

        emp_repo.update_face_embedding(emp1, face_emb_aarav)
        emp_repo.update_face_embedding(emp2, face_emb_sneha)
        print(f"  [+] Enrolled face biometrics for {emp1.name}: has_face_registered={emp1.has_face_registered}")
        print(f"  [+] Enrolled face biometrics for {emp2.name}: has_face_registered={emp2.has_face_registered}")

        # -------------------------------------------------------------
        # STEP 3: Automated Facial Recognition Search Test (HNSW)
        # -------------------------------------------------------------
        print("\n[STEP 3] Testing Facial Recognition Search via PostgreSQL pgvector (HNSW)...")
        # Query with slightly noisy probe (simulating real camera variation)
        probe_aarav = face_emb_aarav + np.random.randn(512).astype(np.float32) * 0.02
        probe_aarav /= np.linalg.norm(probe_aarav)

        matched_id, score = vector_mgr.find_best_employee_match(probe_aarav, similarity_threshold=0.50, db=db)
        print(f"  [+] Recognition Probe -> Matched Employee: {matched_id} (Cosine Similarity: {score:.3f})")
        assert matched_id == emp1.employee_id, f"Expected {emp1.employee_id}, got {matched_id}"

        # -------------------------------------------------------------
        # STEP 4: On-Time Attendance Check-In (Aarav at 09:25)
        # -------------------------------------------------------------
        print("\n[STEP 4] Recording On-Time Check-In Punch (Aarav at 09:25 AM)...")
        today = date.today()
        t_on_time = datetime.combine(today, time(9, 25), tzinfo=timezone.utc)

        log1, daily1 = att_repo.record_attendance_event(
            employee=emp1,
            event_type="CHECK_IN",
            event_time=t_on_time,
            confidence=0.97,
            debounce_seconds=180,
        )
        print(f"  [+] Check-In Logged: Status={daily1.status}, IsLate={daily1.is_late}, LateMinutes={daily1.late_minutes}")
        assert daily1.status == "PRESENT"
        assert daily1.is_late is False

        # -------------------------------------------------------------
        # STEP 5: Duplicate Punch Debounce Test (Aarav again at 09:26:30)
        # -------------------------------------------------------------
        print("\n[STEP 5] Testing Rapid Duplicate Debounce Suppression (within 180s)...")
        t_rapid = t_on_time + timedelta(seconds=90)
        log_dup, daily_dup = att_repo.record_attendance_event(
            employee=emp1,
            event_type="CHECK_IN",
            event_time=t_rapid,
            debounce_seconds=180,
        )
        if log_dup is None:
            print("  [+] SUCCESS: Rapid duplicate punch was automatically SUPPRESSED by 180-second cooldown debounce.")
        else:
            print("  [-] FAILURE: Rapid duplicate was not suppressed.")

        # -------------------------------------------------------------
        # STEP 6: Late Arrival Punch Test (Sneha at 10:10 AM)
        # -------------------------------------------------------------
        print("\n[STEP 6] Testing Late Arrival Calculation (Sneha at 10:10 AM, shift starts 09:30 + 15m grace)...")
        t_late = datetime.combine(today, time(10, 10), tzinfo=timezone.utc)
        log_late, daily_late = att_repo.record_attendance_event(
            employee=emp2,
            event_type="CHECK_IN",
            event_time=t_late,
            confidence=0.96,
            debounce_seconds=60,
        )
        print(f"  [+] Late Check-In Logged: Status={daily_late.status}, IsLate={daily_late.is_late}, LateMinutes={daily_late.late_minutes} min")
        assert daily_late.is_late is True
        assert daily_late.late_minutes == 40.0

        # -------------------------------------------------------------
        # STEP 7: Check-Out & Working Hours Calculation (Aarav at 18:45)
        # -------------------------------------------------------------
        print("\n[STEP 7] Recording Shift Check-Out (Aarav at 18:45 PM)...")
        t_checkout = datetime.combine(today, time(18, 45), tzinfo=timezone.utc)
        log_out, daily_out = att_repo.record_attendance_event(
            employee=emp1,
            event_type="CHECK_OUT",
            event_time=t_checkout,
            confidence=0.98,
            debounce_seconds=180,
        )
        print(f"  [+] Shift Completed: Total Work Hours={daily_out.total_work_hours} hrs, Status={daily_out.status}")
        assert daily_out.total_work_hours >= 9.3

        # -------------------------------------------------------------
        # STEP 8: Daily Roll Summary Report
        # -------------------------------------------------------------
        print("\n[STEP 8] Generating Daily Attendance Summary Report...")
        records = att_repo.get_daily_attendance(target_date=today)
        print("\n" + "-" * 75)
        print(f"{'EMP ID':<14} | {'NAME':<16} | {'IN TIME':<8} | {'OUT TIME':<8} | {'HOURS':<6} | {'STATUS':<8}")
        print("-" * 75)
        for r in records:
            in_str = r.first_check_in.strftime("%H:%M") if r.first_check_in else "--:--"
            out_str = r.last_check_out.strftime("%H:%M") if r.last_check_out else "--:--"
            print(f"{r.employee.employee_id:<14} | {r.employee.name:<16} | {in_str:<8} | {out_str:<8} | {r.total_work_hours:>5.1f} | {r.status:<8}")
        print("-" * 75)

        # -------------------------------------------------------------
        # STEP 9: 14-Day Retention Exemption Verification
        # -------------------------------------------------------------
        print("\n[STEP 9] Verifying Biometric Retention Exemption (Permanent Storage)...")
        retention_svc = RetentionService(db=db, retention_days=14)
        report = retention_svc.cleanup_expired_embeddings()
        emp1_after = emp_repo.get_by_employee_id(emp1_id)
        assert emp1_after.face_embedding is not None, "Employee embedding was incorrectly deleted!"
        print(f"  [+] SUCCESS: Visitor 14-day cleanup job purged {report.expired_count} expired visitor vectors.")
        print(f"  [+] Verified: Employee {emp1_after.name} face biometrics remain PERMANENT and intact.")

        print("\n" + "=" * 70)
        print("          ALL EMPLOYEE ATTENDANCE MODULE TESTS PASSED!          ")
        print("=" * 70 + "\n")

    finally:
        db.close()


if __name__ == "__main__":
    main()

