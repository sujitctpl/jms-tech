import uuid
from datetime import date, datetime
from typing import Any, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.api.routes import DAILY_FOOTFALL_ANALYTICS, FOOTFALL_ANALYTICS
from app.db.session import get_db
from app.schemas.analytics import DailyFootfallResponse, FootfallTotalResponse
from app.services.analytics_service import AnalyticsService

router = APIRouter(tags=["Analytics"])


@router.get(
    FOOTFALL_ANALYTICS,
    response_model=FootfallTotalResponse,
    summary="Retrieve total entry counts for a selected period",
)
def get_total_footfall(
    camera_id: Optional[uuid.UUID] = Query(None, description="Optional camera ID filter"),
    start_time: Optional[datetime] = Query(None, description="Start timestamp (ISO 8601)"),
    end_time: Optional[datetime] = Query(None, description="End timestamp (ISO 8601)"),
    db: Session = Depends(get_db),
) -> Any:
    """Retrieve total entry count for a selected time period with optional camera filtering."""
    service = AnalyticsService(db)
    return service.get_total_footfall(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
    )


@router.get(
    DAILY_FOOTFALL_ANALYTICS,
    response_model=DailyFootfallResponse,
    summary="Retrieve daily entry counts",
)
def get_daily_footfall(
    camera_id: Optional[uuid.UUID] = Query(None, description="Optional camera ID filter"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
) -> Any:
    """Retrieve daily entry counts grouped by date for a selected date range with optional camera filtering."""
    service = AnalyticsService(db)
    return service.get_daily_footfall(
        camera_id=camera_id,
        start_date=start_date,
        end_date=end_date,
    )
