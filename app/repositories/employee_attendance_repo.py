from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional, Tuple, Union
import uuid
from sqlalchemy import Date, and_, desc, func
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.db.models.employee import Employee
from app.db.models.employee_attendance import EmployeeAttendanceLog, EmployeeDailyAttendance


def ensure_tz_aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class EmployeeAttendanceRepository:
    def __init__(self, db: Session):
        self.db = db

    def record_attendance_event(
        self,
        employee: Optional[Union[Employee, str, uuid.UUID]] = None,
        camera_id: Optional[uuid.UUID] = None,
        event_type: str = "CHECK_IN",
        timestamp: Optional[datetime] = None,
        confidence: float = 0.95,
        debounce_seconds: int = 180,
        employee_id: Optional[str] = None,
        event_time: Optional[datetime] = None,
    ) -> Tuple[Optional[EmployeeAttendanceLog], EmployeeDailyAttendance]:
        """
        Records an employee detection event from a CCTV camera, updates daily attendance,
        and calculates working hours, late arrivals, and early departures.
        Suppresses duplicate rapid logs within debounce_seconds.
        """
        now = event_time or timestamp or datetime.now(timezone.utc)
        now = ensure_tz_aware(now)
        today = now.date()

        # Resolve employee instance if employee_id string or UUID provided
        emp_obj = None
        target = employee if employee is not None else employee_id
        if isinstance(target, Employee):
            emp_obj = target
        elif isinstance(target, uuid.UUID):
            emp_obj = self.db.query(Employee).filter(Employee.id == target).first()
        elif isinstance(target, str):
            try:
                val = uuid.UUID(target)
                emp_obj = self.db.query(Employee).filter(Employee.id == val).first()
            except (ValueError, AttributeError):
                pass
            if not emp_obj:
                emp_obj = self.db.query(Employee).filter(Employee.employee_id == str(target)).first()

        if not emp_obj:
            raise ValueError(f"Could not resolve employee from identifier: {target}")

        # 1. Debounce check: prevent duplicate logs within debounce_seconds
        recent_logs = (
            self.db.query(EmployeeAttendanceLog)
            .filter(EmployeeAttendanceLog.employee_id == emp_obj.id)
            .order_by(desc(EmployeeAttendanceLog.timestamp))
            .limit(10)
            .all()
        )
        last_log = None
        for cand in recent_logs:
            cand_ts = ensure_tz_aware(cand.timestamp)
            if cand_ts and cand_ts >= now - timedelta(seconds=debounce_seconds):
                last_log = cand
                break

        log = None
        if not last_log:
            log = EmployeeAttendanceLog(
                employee_id=emp_obj.id,
                camera_id=camera_id,
                event_type=event_type,
                confidence=float(confidence),
                timestamp=now,
            )
            self.db.add(log)

        # 2. Daily Attendance Roll aggregation
        daily = (
            self.db.query(EmployeeDailyAttendance)
            .filter(
                EmployeeDailyAttendance.employee_id == emp_obj.id,
                EmployeeDailyAttendance.date == today,
            )
            .first()
        )

        shift_start = emp_obj.shift_start_time or time(9, 30)
        shift_end = emp_obj.shift_end_time or time(18, 30)
        late_grace = emp_obj.late_grace_minutes if emp_obj.late_grace_minutes is not None else 15
        early_grace = (
            emp_obj.early_departure_grace_minutes
            if emp_obj.early_departure_grace_minutes is not None
            else 15
        )

        shift_start_dt = datetime.combine(today, shift_start, tzinfo=now.tzinfo)
        shift_end_dt = datetime.combine(today, shift_end, tzinfo=now.tzinfo)
        late_cutoff = shift_start_dt + timedelta(minutes=late_grace)
        early_cutoff = shift_end_dt - timedelta(minutes=early_grace)

        if not daily or daily.first_check_in is None:
            # First sighting of the day -> Check-In
            is_late = now > late_cutoff
            late_mins = max(0.0, (now - shift_start_dt).total_seconds() / 60.0) if is_late else 0.0
            status = "LATE" if is_late else "PRESENT"

            if not daily:
                daily = EmployeeDailyAttendance(
                    employee_id=emp_obj.id,
                    date=today,
                    first_check_in=now,
                    last_check_out=now,
                    total_work_hours=0.0,
                    is_late=is_late,
                    late_minutes=round(late_mins, 1),
                    is_early_departure=False,
                    early_departure_minutes=0.0,
                    status=status,
                )
            else:
                daily.first_check_in = now
                daily.last_check_out = now
                daily.is_late = is_late
                daily.late_minutes = round(late_mins, 1)
                daily.is_early_departure = False
                daily.early_departure_minutes = 0.0
                daily.status = status

            self.db.add(daily)
            logger.info(
                "Recorded first check-in for employee %s (%s) at %s [Status: %s]",
                emp_obj.employee_id,
                emp_obj.name,
                now.isoformat(),
                status,
            )
        else:
            # Subsequent sighting -> update last_check_out and calculate work duration
            first_in = ensure_tz_aware(daily.first_check_in)
            if first_in and now >= first_in:
                daily.last_check_out = now
                work_seconds = (now - first_in).total_seconds()
                daily.total_work_hours = round(max(0.0, work_seconds / 3600.0), 2)

                # Check Early Departure
                if now < early_cutoff:
                    daily.is_early_departure = True
                    daily.early_departure_minutes = round(
                        max(0.0, (shift_end_dt - now).total_seconds() / 60.0), 1
                    )
                else:
                    daily.is_early_departure = False
                    daily.early_departure_minutes = 0.0

                # Evaluate overall status
                if daily.total_work_hours >= 7.0:
                    daily.status = "LATE" if daily.is_late else "PRESENT"
                elif daily.total_work_hours >= 4.0:
                    daily.status = "HALF_DAY"
                else:
                    daily.status = "LATE" if daily.is_late else "PRESENT"

            self.db.add(daily)

        self.db.commit()
        if log:
            self.db.refresh(log)
        self.db.refresh(daily)
        return log, daily

    def get_daily_attendance(
        self,
        target_date: date,
        department: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[EmployeeDailyAttendance]:
        query = (
            self.db.query(EmployeeDailyAttendance)
            .join(Employee, EmployeeDailyAttendance.employee_id == Employee.id)
            .filter(EmployeeDailyAttendance.date == target_date)
        )
        if department:
            query = query.filter(Employee.department == department)
        if status:
            query = query.filter(EmployeeDailyAttendance.status == status)

        return query.order_by(Employee.employee_id.asc()).all()

    def get_daily_summary(
        self,
        target_date: date,
        department: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[EmployeeDailyAttendance]:
        return self.get_daily_attendance(target_date, department=department, status=status)

    def get_employee_history(
        self,
        employee_uuid: uuid.UUID,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[EmployeeDailyAttendance]:
        query = self.db.query(EmployeeDailyAttendance).filter(
            EmployeeDailyAttendance.employee_id == employee_uuid
        )
        if start_date:
            query = query.filter(EmployeeDailyAttendance.date >= start_date)
        if end_date:
            query = query.filter(EmployeeDailyAttendance.date <= end_date)

        return query.order_by(EmployeeDailyAttendance.date.desc()).all()

    def get_employee_logs(
        self,
        employee_uuid: uuid.UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 200,
    ) -> List[EmployeeAttendanceLog]:
        query = self.db.query(EmployeeAttendanceLog).filter(
            EmployeeAttendanceLog.employee_id == employee_uuid
        )
        if start_time:
            query = query.filter(EmployeeAttendanceLog.timestamp >= start_time)
        if end_time:
            query = query.filter(EmployeeAttendanceLog.timestamp <= end_time)

        return query.order_by(EmployeeAttendanceLog.timestamp.desc()).limit(limit).all()

    def mark_absences(self, target_date: date) -> int:
        """
        Creates ABSENT daily attendance records for all active employees
        who have not clocked in on the target_date. Idempotent.
        """
        active_employees = (
            self.db.query(Employee)
            .filter(Employee.is_active.is_(True))
            .all()
        )

        recorded_emp_ids = {
            r.employee_id
            for r in self.db.query(EmployeeDailyAttendance.employee_id)
            .filter(EmployeeDailyAttendance.date == target_date)
            .all()
        }

        absent_count = 0
        for emp in active_employees:
            if emp.id not in recorded_emp_ids:
                absent_record = EmployeeDailyAttendance(
                    employee_id=emp.id,
                    date=target_date,
                    first_check_in=None,
                    last_check_out=None,
                    total_work_hours=0.0,
                    is_late=False,
                    late_minutes=0.0,
                    is_early_departure=False,
                    early_departure_minutes=0.0,
                    status="ABSENT",
                )
                self.db.add(absent_record)
                absent_count += 1

        if absent_count > 0:
            self.db.commit()
            logger.info("Marked %d employees ABSENT for date %s", absent_count, target_date.isoformat())
        return absent_count
