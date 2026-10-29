import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Optional
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid
from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.camera import Camera
    from app.db.models.employee import Employee


class EmployeeAttendanceLog(Base):
    """
    Raw check-in, check-out, and camera presence events detected by CCTV cameras.
    """

    __tablename__ = "employee_attendance_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    employee_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # CHECK_IN, CHECK_OUT, or PRESENCE
    event_type: Mapped[str] = mapped_column(String(20), nullable=False, default="CHECK_IN")
    confidence: Mapped[float] = mapped_column(Float, default=0.95, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    employee: Mapped["Employee"] = relationship("Employee", back_populates="attendance_logs")
    camera: Mapped[Optional["Camera"]] = relationship("Camera")


class EmployeeDailyAttendance(Base):
    """
    Daily aggregated attendance roll for an employee:
    First check-in, last check-out, total work hours, late arrival, and early departure flags.
    """

    __tablename__ = "employee_daily_attendance"
    __table_args__ = (
        UniqueConstraint("employee_id", "date", name="uq_employee_daily_attendance"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    employee_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    first_check_in: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_check_out: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    total_work_hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    is_late: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    late_minutes: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    is_early_departure: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    early_departure_minutes: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # PRESENT, LATE, HALF_DAY, ABSENT
    status: Mapped[str] = mapped_column(String(20), default="PRESENT", nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    employee: Mapped["Employee"] = relationship("Employee", back_populates="daily_attendance")

