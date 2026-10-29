from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.api.routes import RETENTION_CLEANUP, RETENTION_STATUS
from app.core.config import settings
from app.db.models.person_identity import PersonIdentity
from app.db.session import get_db
from app.services.retention_service import RetentionService

router = APIRouter(prefix="/api/v1/identity", tags=["Identity & Retention"])


class RetentionCleanupResponse(BaseModel):
    success: bool
    expired_count: int
    retention_days: int
    cutoff_timestamp: str
    execution_time_seconds: float
    message: str


class RetentionStatusResponse(BaseModel):
    retention_policy_days: int
    active_face_embeddings: int
    expired_identities: int
    calibrated_similarity_threshold: float
    auto_cleanup_enabled: bool


@router.post(
    "/cleanup",
    response_model=RetentionCleanupResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger 14-day face embedding retention cleanup",
)
def run_retention_cleanup(
    retention_days: Optional[int] = Query(None, ge=1, le=365, description="Retention window in days (default: 14)"),
    db: Session = Depends(get_db),
) -> Any:
    """
    Executes biometric retention cleanup:
    Purges face embeddings and associated vector representations for visitors
    not reliably recognized within 14 days (or custom window).
    Preserves all aggregate footfall, visitor counts, and analytics.
    """
    service = RetentionService(db=db)
    report = service.cleanup_expired_embeddings(db=db, retention_days=retention_days)
    return RetentionCleanupResponse(
        success=report.success,
        expired_count=report.expired_count,
        retention_days=report.retention_days,
        cutoff_timestamp=report.cutoff_timestamp.isoformat(),
        execution_time_seconds=report.execution_time_seconds,
        message=report.message,
    )


@router.get(
    "/status",
    response_model=RetentionStatusResponse,
    summary="Get biometric identity retention status",
)
def get_retention_status(
    db: Session = Depends(get_db),
) -> Any:
    """Returns biometric retention metrics and current index statistics."""
    active_count = (
        db.query(PersonIdentity)
        .filter(PersonIdentity.face_embedding.isnot(None), PersonIdentity.is_expired.is_(False))
        .count()
    )
    expired_count = (
        db.query(PersonIdentity)
        .filter(PersonIdentity.is_expired.is_(True))
        .count()
    )

    return RetentionStatusResponse(
        retention_policy_days=settings.FACE_RETENTION_DAYS,
        active_face_embeddings=active_count,
        expired_identities=expired_count,
        calibrated_similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD,
        auto_cleanup_enabled=settings.RETENTION_AUTO_CLEANUP_ENABLED,
    )

