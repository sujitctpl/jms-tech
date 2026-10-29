import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid
from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.camera import Camera
    from app.db.models.processing_run import ProcessingRun
    from app.db.models.detection_event import DetectionEvent
    from app.db.models.footfall_event import FootfallEvent
    from app.db.models.person_identity import PersonIdentity


class VisitorTrack(Base):
    __tablename__ = "visitor_tracks"
    __table_args__ = (
        UniqueConstraint(
            "processing_run_id", "tracker_id", name="uq_processing_run_tracker"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    camera_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    processing_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("processing_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tracker_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20), default="ACTIVE", nullable=False, index=True
    )
    person_identity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("person_identities.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    person_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    camera: Mapped["Camera"] = relationship(
        "Camera", back_populates="visitor_tracks"
    )
    processing_run: Mapped["ProcessingRun"] = relationship(
        "ProcessingRun", back_populates="visitor_tracks"
    )
    detection_events: Mapped[List["DetectionEvent"]] = relationship(
        "DetectionEvent", back_populates="visitor_track", cascade="all, delete-orphan"
    )
    footfall_event: Mapped[Optional["FootfallEvent"]] = relationship(
        "FootfallEvent", back_populates="visitor_track", uselist=False
    )
    person_identity: Mapped[Optional["PersonIdentity"]] = relationship(
        "PersonIdentity", back_populates="visitor_tracks"
    )
