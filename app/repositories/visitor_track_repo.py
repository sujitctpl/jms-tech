import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from app.db.models.visitor_track import VisitorTrack


class VisitorTrackRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_run_and_tracker_id(
        self, processing_run_id: uuid.UUID, tracker_id: str
    ) -> Optional[VisitorTrack]:
        return (
            self.db.query(VisitorTrack)
            .filter(
                VisitorTrack.processing_run_id == processing_run_id,
                VisitorTrack.tracker_id == str(tracker_id),
            )
            .first()
        )

    def get_or_create(
        self,
        camera_id: uuid.UUID,
        processing_run_id: uuid.UUID,
        tracker_id: str,
        timestamp: Optional[datetime] = None,
        person_id: Optional[str] = None,
        person_identity_id: Optional[uuid.UUID] = None,
    ) -> VisitorTrack:
        ts = timestamp or datetime.now(timezone.utc)
        track = self.get_by_run_and_tracker_id(processing_run_id, str(tracker_id))
        if track:
            track.last_seen_at = ts
            if person_id and not track.person_id:
                track.person_id = str(person_id)
            if person_identity_id and not track.person_identity_id:
                track.person_identity_id = person_identity_id
            self.db.add(track)
            self.db.commit()
            self.db.refresh(track)
            return track

        track = VisitorTrack(
            camera_id=camera_id,
            processing_run_id=processing_run_id,
            tracker_id=str(tracker_id),
            first_seen_at=ts,
            last_seen_at=ts,
            status="ACTIVE",
            person_id=str(person_id) if person_id else None,
            person_identity_id=person_identity_id,
        )
        self.db.add(track)
        self.db.commit()
        self.db.refresh(track)
        return track

    def link_person_identity(
        self,
        track_id: uuid.UUID,
        person_id: str,
        person_identity_id: Optional[uuid.UUID] = None,
    ) -> Optional[VisitorTrack]:
        track = self.db.query(VisitorTrack).filter(VisitorTrack.id == track_id).first()
        if track:
            track.person_id = str(person_id)
            if person_identity_id:
                track.person_identity_id = person_identity_id
            self.db.add(track)
            self.db.commit()
            self.db.refresh(track)
        return track

    def mark_lost_or_completed(self, track_id: uuid.UUID, status: str = "COMPLETED") -> None:
        track = self.db.query(VisitorTrack).filter(VisitorTrack.id == track_id).first()
        if track:
            track.status = status
            self.db.add(track)
            self.db.commit()
