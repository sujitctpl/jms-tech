import uuid
from datetime import datetime, time
from typing import TYPE_CHECKING, Any, List, Optional
import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Time, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid
from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.camera import Camera
    from app.db.models.employee_attendance import EmployeeAttendanceLog, EmployeeDailyAttendance


class Employee(Base):
    """
    Employee record with organizational details and biometric face embedding.
    Employee records and face embeddings are permanently retained (exempt from visitor retention).
    """

    __tablename__ = "employees"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    employee_id: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    department: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    designation: Mapped[str] = mapped_column(String(100), nullable=False)
    
    assigned_camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    assigned_location: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)

    # Shift configuration (e.g. 09:30:00 to 18:30:00)
    shift_start_time: Mapped[Optional[time]] = mapped_column(Time, default=time(9, 30))
    shift_end_time: Mapped[Optional[time]] = mapped_column(Time, default=time(18, 30))
    late_grace_minutes: Mapped[int] = mapped_column(Integer, default=15)
    early_departure_grace_minutes: Mapped[int] = mapped_column(Integer, default=15)

    # 512-dim ArcFace embedding stored as pgvector Vector(512) (never exposed in API responses)
    face_embedding: Mapped[Optional[Any]] = mapped_column(Vector(512), nullable=True)

    # Multi-angle photo gallery embeddings stored as JSON list of 512-d normalized vectors
    face_gallery: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    # 1st photo stored as primary profile picture URL / file path
    photo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    camera: Mapped[Optional["Camera"]] = relationship("Camera")
    faces: Mapped[List["EmployeeFace"]] = relationship(
        "EmployeeFace", back_populates="employee", cascade="all, delete-orphan"
    )
    attendance_logs: Mapped[List["EmployeeAttendanceLog"]] = relationship(
        "EmployeeAttendanceLog", back_populates="employee", cascade="all, delete-orphan"
    )
    daily_attendance: Mapped[List["EmployeeDailyAttendance"]] = relationship(
        "EmployeeDailyAttendance", back_populates="employee", cascade="all, delete-orphan"
    )

    @property
    def has_face_registered(self) -> bool:
        if self.face_embedding is None:
            return False
        if isinstance(self.face_embedding, (list, tuple, np.ndarray)):
            return len(self.face_embedding) > 0
        if hasattr(self.face_embedding, "to_numpy"):
            return len(self.face_embedding.to_numpy()) > 0
        return bool(self.face_embedding)

    def get_embedding_array(self) -> Optional[np.ndarray]:
        if self.face_embedding is None:
            return None
        if isinstance(self.face_embedding, np.ndarray):
            return self.face_embedding.astype(np.float32)
        if hasattr(self.face_embedding, "to_numpy"):
            return self.face_embedding.to_numpy().astype(np.float32)
        return np.array(self.face_embedding, dtype=np.float32)

    def get_gallery_embeddings(self) -> List[np.ndarray]:
        """
        Returns all normalized embedding vectors across enrolled gallery photos.
        Combines employee_faces table records, legacy face_gallery JSON, and primary face_embedding.
        """
        gallery: List[np.ndarray] = []
        # 1. From related employee_faces records
        if hasattr(self, "faces") and self.faces:
            for f in self.faces:
                arr = f.get_embedding_array()
                if arr is not None and len(arr) == 512:
                    norm = np.linalg.norm(arr)
                    gallery.append((arr / norm) if norm > 0 else arr)

        # 2. Fallback to JSON face_gallery column
        elif self.face_gallery and isinstance(self.face_gallery, list):
            for item in self.face_gallery:
                arr = np.array(item, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    gallery.append(arr / norm)
                elif len(arr) == 512:
                    gallery.append(arr)

        # 3. Fallback to primary face_embedding
        if not gallery:
            primary = self.get_embedding_array()
            if primary is not None:
                norm = np.linalg.norm(primary)
                gallery.append((primary / norm) if norm > 0 else primary)
        return gallery


class EmployeeFace(Base):
    """
    Stores individual multi-angle face embeddings for an employee using native PostgreSQL pgvector.
    Enables sub-millisecond multi-angle face search (Front, Left, Right, CCTV angle)
    backed by native pgvector HNSW indexing.
    """
    __tablename__ = "employee_faces"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    employee_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    angle_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default="general")
    face_embedding: Mapped[Any] = mapped_column(Vector(512), nullable=False)
    photo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship back to parent Employee
    employee: Mapped["Employee"] = relationship("Employee", back_populates="faces")

    def get_embedding_array(self) -> np.ndarray:
        if isinstance(self.face_embedding, np.ndarray):
            return self.face_embedding.astype(np.float32)
        if hasattr(self.face_embedding, "to_numpy"):
            return self.face_embedding.to_numpy().astype(np.float32)
        return np.array(self.face_embedding, dtype=np.float32)
