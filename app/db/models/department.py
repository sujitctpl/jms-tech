import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON, Uuid
from app.db.base import Base


class Department(Base):
    """
    Configurable detection department / zone in the showroom.
    Can be assigned to specific cameras or global across all showroom cameras.
    """

    __tablename__ = "departments"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)  # slug e.g. 'sales', 'service', 'vip_lounge'
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    color: Mapped[str] = mapped_column(String(20), default="#00D7FF", nullable=False)
    camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=True,
    )
    box: Mapped[Optional[List[int]]] = mapped_column(JSON, nullable=True)  # [x1, y1, x2, y2]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

