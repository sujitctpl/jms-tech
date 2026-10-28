import uuid
from datetime import date, datetime, timezone
from typing import List, Optional, Tuple
from sqlalchemy import Date, cast, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.db.models.footfall_event import FootfallEvent


class FootfallEventRepository:
    def __init__(self, db: Session):
        self.db = db

    def record_entry(
        self,
        camera_id: uuid.UUID,
        visitor_track_id: uuid.UUID,
        occurred_at: Optional[datetime] = None,
    ) -> Optional[FootfallEvent]:
        """
        Records an entry event with strict duplicate prevention.
        Returns the created FootfallEvent, or None if already recorded for this track.
        """
        ts = occurred_at or datetime.now(timezone.utc)
        # Check application level first
        existing = (
            self.db.query(FootfallEvent)
            .filter(FootfallEvent.visitor_track_id == visitor_track_id)
            .first()
        )
        if existing:
            return existing

        event = FootfallEvent(
            camera_id=camera_id,
            visitor_track_id=visitor_track_id,
            event_type="entry",
            occurred_at=ts,
            created_at=datetime.now(timezone.utc),
        )
        try:
            self.db.add(event)
            self.db.commit()
            self.db.refresh(event)
            return event
        except IntegrityError:
            self.db.rollback()
            logger.warning(
                "Duplicate entry event rejected by database constraint for visitor_track_id: %s",
                visitor_track_id,
            )
            return (
                self.db.query(FootfallEvent)
                .filter(FootfallEvent.visitor_track_id == visitor_track_id)
                .first()
            )

    def get_total_entries(
        self,
        camera_id: Optional[uuid.UUID] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        query = self.db.query(func.count(FootfallEvent.id)).filter(
            FootfallEvent.event_type == "entry"
        )
        if camera_id:
            query = query.filter(FootfallEvent.camera_id == camera_id)
        if start_time:
            query = query.filter(FootfallEvent.occurred_at >= start_time)
        if end_time:
            query = query.filter(FootfallEvent.occurred_at <= end_time)
        return query.scalar() or 0

    def get_daily_entries(
        self,
        camera_id: Optional[uuid.UUID] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Tuple[str, int]]:
        from datetime import time as dt_time

        date_col = func.date(FootfallEvent.occurred_at)
        query = (
            self.db.query(date_col.label("day"), func.count(FootfallEvent.id).label("count"))
            .filter(FootfallEvent.event_type == "entry")
        )
        if camera_id:
            query = query.filter(FootfallEvent.camera_id == camera_id)
        if start_date:
            start_dt = datetime.combine(start_date, dt_time.min, tzinfo=timezone.utc)
            query = query.filter(FootfallEvent.occurred_at >= start_dt)
        if end_date:
            end_dt = datetime.combine(end_date, dt_time.max, tzinfo=timezone.utc)
            query = query.filter(FootfallEvent.occurred_at <= end_dt)

        results = query.group_by(date_col).order_by(date_col.asc()).all()
        return [(str(row.day), int(row.count)) for row in results]
