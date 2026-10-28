import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid
from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.camera import Camera
    from app.db.models.visitor_track import VisitorTrack
    from app.db.models.detection_event import DetectionEvent


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    camera_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(20), default="RUNNING", nullable=False, index=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    frames_processed: Mapped[int] = mapped_column(
        BigInteger, default=0, nullable=False
    )

    # Relationships
    camera: Mapped["Camera"] = relationship(
        "Camera", back_populates="processing_runs"
    )
    visitor_tracks: Mapped[List["VisitorTrack"]] = relationship(
        "VisitorTrack", back_populates="processing_run", cascade="all, delete-orphan"
    )
    detection_events: Mapped[List["DetectionEvent"]] = relationship(
        "DetectionEvent", back_populates="processing_run", cascade="all, delete-orphan"
    )
