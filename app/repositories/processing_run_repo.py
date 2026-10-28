import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from app.db.models.processing_run import ProcessingRun


class ProcessingRunRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, camera_id: uuid.UUID) -> ProcessingRun:
        run = ProcessingRun(
            camera_id=camera_id,
            status="RUNNING",
            frames_processed=0,
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def get_by_id(self, run_id: uuid.UUID) -> Optional[ProcessingRun]:
        return self.db.query(ProcessingRun).filter(ProcessingRun.id == run_id).first()

    def get_active_run(self, camera_id: uuid.UUID) -> Optional[ProcessingRun]:
        return (
            self.db.query(ProcessingRun)
            .filter(
                ProcessingRun.camera_id == camera_id,
                ProcessingRun.status == "RUNNING",
            )
            .order_by(ProcessingRun.started_at.desc())
            .first()
        )

    def finish_run(
        self,
        run_id: uuid.UUID,
        status: str = "STOPPED",
        frames_processed: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Optional[ProcessingRun]:
        run = self.get_by_id(run_id)
        if not run:
            return None

        run.status = status
        run.ended_at = datetime.now(timezone.utc)
        if frames_processed is not None:
            run.frames_processed = frames_processed
        if error_message is not None:
            run.error_message = error_message

        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def update_frame_count(self, run_id: uuid.UUID, count: int) -> None:
        run = self.get_by_id(run_id)
        if run:
            run.frames_processed = count
            self.db.add(run)
            self.db.commit()
