import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, Uuid
from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.processing_run import ProcessingRun
    from app.db.models.visitor_track import VisitorTrack
    from app.db.models.detection_event import DetectionEvent
    from app.db.models.footfall_event import FootfallEvent


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    stream_url: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    entry_line_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON().with_variant(postgresql.JSONB, "postgresql"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    processing_runs: Mapped[List["ProcessingRun"]] = relationship(
        "ProcessingRun", back_populates="camera", cascade="all, delete-orphan"
    )
    visitor_tracks: Mapped[List["VisitorTrack"]] = relationship(
        "VisitorTrack", back_populates="camera", cascade="all, delete-orphan"
    )
    detection_events: Mapped[List["DetectionEvent"]] = relationship(
        "DetectionEvent", back_populates="camera", cascade="all, delete-orphan"
    )
    footfall_events: Mapped[List["FootfallEvent"]] = relationship(
        "FootfallEvent", back_populates="camera", cascade="all, delete-orphan"
    )
