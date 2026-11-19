import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional
import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid
from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.visitor_track import VisitorTrack


class PersonIdentity(Base):
    """
    Persistent identity record representing a recognized visitor across multiple visits/sessions.
    Stores 512-dimensional face embeddings subject to a 14-day rolling retention policy.
    Uses PostgreSQL pgvector with HNSW indexing for sub-millisecond similarity search.
    """

    __tablename__ = "person_identities"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    person_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    # 512-dimensional ArcFace vector; nulled upon 14-day expiration
    face_embedding: Mapped[Optional[Any]] = mapped_column(Vector(512), nullable=True)
    # 512-dimensional OSNet full-body ReID vector
    body_embedding: Mapped[Optional[Any]] = mapped_column(Vector(512), nullable=True)

    is_expired: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    expired_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Visitor profile and journey metadata
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    exit_avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    visit_purpose: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    assigned_salesperson: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    visitor_type: Mapped[str] = mapped_column(String(20), default="New", nullable=False)
    tagging_status: Mapped[str] = mapped_column(String(20), default="Untagged", nullable=False)
    gender_estimated: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    age_group_estimated: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    zone_entered: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    camera_source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    exit_time: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    zone_dwell: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    waiting_seconds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_waiting: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    waiting_breach: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    visitor_tracks: Mapped[List["VisitorTrack"]] = relationship(
        "VisitorTrack", back_populates="person_identity"
    )

    def get_embedding_array(self) -> Optional[np.ndarray]:
        """Returns normalized numpy float32 array or None if expired/unset."""
        if self.face_embedding is None:
            return None
        if isinstance(self.face_embedding, np.ndarray):
            return self.face_embedding.astype(np.float32)
        if hasattr(self.face_embedding, "to_numpy"):
            return self.face_embedding.to_numpy().astype(np.float32)
        return np.array(self.face_embedding, dtype=np.float32)

    get_embedding_vector = get_embedding_array

    def get_body_embedding_array(self) -> Optional[np.ndarray]:
        """Returns normalized numpy float32 array or None if unset."""
        if self.body_embedding is None:
            return None
        if isinstance(self.body_embedding, np.ndarray):
            return self.body_embedding.astype(np.float32)
        if hasattr(self.body_embedding, "to_numpy"):
            return self.body_embedding.to_numpy().astype(np.float32)
        return np.array(self.body_embedding, dtype=np.float32)

    get_body_embedding_vector = get_body_embedding_array

