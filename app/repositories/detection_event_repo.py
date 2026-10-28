import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from app.db.models.detection_event import DetectionEvent


class DetectionEventRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        camera_id: uuid.UUID,
        processing_run_id: uuid.UUID,
        visitor_track_id: uuid.UUID,
        frame_number: int,
        confidence: float,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        detected_at: Optional[datetime] = None,
    ) -> DetectionEvent:
        ts = detected_at or datetime.now(timezone.utc)
        event = DetectionEvent(
            camera_id=camera_id,
            processing_run_id=processing_run_id,
            visitor_track_id=visitor_track_id,
            detected_at=ts,
            frame_number=frame_number,
            confidence=confidence,
            bbox_x1=bbox_x1,
            bbox_y1=bbox_y1,
            bbox_x2=bbox_x2,
            bbox_y2=bbox_y2,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event
