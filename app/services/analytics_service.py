import uuid
from datetime import date, datetime
from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.repositories.footfall_event_repo import FootfallEventRepository
from app.schemas.analytics import (
    DailyFootfallItem,
    DailyFootfallResponse,
    FootfallTotalResponse,
)


class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self.footfall_repo = FootfallEventRepository(db)

    def get_total_footfall(
        self,
        camera_id: Optional[uuid.UUID] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> FootfallTotalResponse:
        if start_time and end_time and start_time > end_time:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_time must be less than or equal to end_time",
            )

        total = self.footfall_repo.get_total_entries(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
        )

        from app.services.camera_service import _edge_telemetry
        sec_counts: dict = {}
        for cam_telemetry in _edge_telemetry.values():
            for sec_name, count in cam_telemetry.get("section_counts", {}).items():
                sec_counts[sec_name] = sec_counts.get(sec_name, 0) + count

        return FootfallTotalResponse(
            total_entries=total,
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            section_counts=sec_counts,
        )

    def get_daily_footfall(
        self,
        camera_id: Optional[uuid.UUID] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> DailyFootfallResponse:
        if start_date and end_date and start_date > end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_date must be less than or equal to end_date",
            )

        daily_rows = self.footfall_repo.get_daily_entries(
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date,
        )

        items = [
            DailyFootfallItem(date=row_date, entries=count)
            for row_date, count in daily_rows
        ]
        total = sum(item.entries for item in items)

        return DailyFootfallResponse(
            daily_footfall=items,
            total_entries=total,
            camera_id=camera_id,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
        )
