import time
import uuid
from typing import Any, Dict, List
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.db.models.camera import Camera
from app.repositories.camera_repo import CameraRepository
from app.repositories.footfall_event_repo import FootfallEventRepository
from app.repositories.processing_run_repo import ProcessingRunRepository
from app.repositories.visitor_track_repo import VisitorTrackRepository
from app.schemas.camera import (
    CameraCreate,
    CameraHeartbeatRequest,
    CameraStatusResponse,
    CameraUpdate,
    EntryLineConfig,
    FootfallRecordRequest,
    FootfallRecordResponse,
)
from app.workers.video_worker import worker_manager

_edge_telemetry: Dict[uuid.UUID, Dict[str, Any]] = {}


class CameraService:
    def __init__(self, db: Session):
        self.db = db
        self.camera_repo = CameraRepository(db)

    def register_camera(self, camera_in: CameraCreate) -> Camera:
        logger.info("Registering camera: %s", camera_in.name)
        return self.camera_repo.create(camera_in)

    def list_cameras(self) -> List[Camera]:
        return self.camera_repo.list_all()

    def get_camera(self, camera_id: uuid.UUID) -> Camera:
        camera = self.camera_repo.get_by_id(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera with ID {camera_id} not found",
            )
        return camera

    def update_camera(self, camera_id: uuid.UUID, camera_in: CameraUpdate) -> Camera:
        camera = self.get_camera(camera_id)
        logger.info("Updating camera: %s", camera_id)
        return self.camera_repo.update(camera, camera_in)

    def validate_camera_config_for_processing(self, camera: Camera) -> None:
        """
        Validates camera configuration before starting video processing.
        Ensures camera is active, has a stream URL, and has a valid entry line configuration.
        """
        if not camera.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot start processing on an inactive camera. Please activate camera first.",
            )

        if not camera.stream_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Camera has no valid stream URL configured.",
            )

        if not camera.entry_line_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Camera entry line configuration is required before starting processing.",
            )

        # Validate structure via Pydantic model
        try:
            EntryLineConfig(**camera.entry_line_config)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid entry line configuration: {str(e)}",
            )

    def start_camera(self, camera_id: uuid.UUID) -> Dict[str, Any]:
        camera = self.get_camera(camera_id)
        self.validate_camera_config_for_processing(camera)

        started = worker_manager.start_worker(camera)
        if not started:
            return {
                "message": "Video processing is already active for this camera.",
                "camera_id": camera_id,
                "is_processing": True,
            }

        return {
            "message": "Video processing started successfully.",
            "camera_id": camera_id,
            "is_processing": True,
        }

    def stop_camera(self, camera_id: uuid.UUID) -> Dict[str, Any]:
        camera = self.get_camera(camera_id)
        stopped = worker_manager.stop_worker(camera.id)
        return {
            "message": "Video processing stopped." if stopped else "Video processing was not running.",
            "camera_id": camera_id,
            "is_processing": False,
        }

    def get_camera_status(self, camera_id: uuid.UUID) -> CameraStatusResponse:
        camera = self.get_camera(camera_id)
        worker_info = worker_manager.get_status(camera_id)

        edge = _edge_telemetry.get(camera.id)
        is_edge_active = bool(edge and (time.time() - edge.get("last_seen", 0) < 20.0))

        is_processing = worker_info["is_processing"] or (is_edge_active and edge.get("is_processing", False))
        frames_processed = edge.get("frames_processed", worker_info["frames_processed"]) if is_edge_active else worker_info["frames_processed"]
        fps = edge.get("fps") if is_edge_active else None
        current_persons = edge.get("current_persons", 0) if is_edge_active else 0
        total_entries = edge.get("total_entries", 0) if is_edge_active else 0
        section_counts = edge.get("section_counts", {}) if is_edge_active else {}
        section_occupancy = edge.get("section_occupancy", {}) if is_edge_active else {}

        return CameraStatusResponse(
            camera_id=camera.id,
            name=camera.name,
            is_active=camera.is_active,
            is_processing=is_processing,
            active_run_id=worker_info["active_run_id"],
            frames_processed=frames_processed,
            last_error=worker_info["last_error"],
            fps=fps,
            current_persons=current_persons,
            total_entries=total_entries,
            section_counts=section_counts,
            section_occupancy=section_occupancy,
        )

    def record_footfall(
        self, camera_id: uuid.UUID, footfall_in: FootfallRecordRequest
    ) -> FootfallRecordResponse:
        """
        Ingests a footfall entry detected locally by an edge processor / local executable.
        """
        camera = self.get_camera(camera_id)

        run_repo = ProcessingRunRepository(self.db)
        track_repo = VisitorTrackRepository(self.db)
        footfall_repo = FootfallEventRepository(self.db)

        run_id = footfall_in.processing_run_id
        if not run_id:
            active_run = run_repo.get_active_run(camera.id)
            if not active_run:
                active_run = run_repo.create(camera.id)
            run_id = active_run.id

        vt = track_repo.get_or_create(
            camera_id=camera.id,
            processing_run_id=run_id,
            tracker_id=footfall_in.tracker_id,
            timestamp=footfall_in.occurred_at,
        )

        event = footfall_repo.record_entry(
            camera_id=camera.id,
            visitor_track_id=vt.id,
            occurred_at=footfall_in.occurred_at,
        )

        total_count = footfall_repo.get_total_entries(camera_id=camera.id)
        logger.info(
            "Edge Detection Recorded: camera=%s, track=%s, total_count=%d",
            camera.name,
            footfall_in.tracker_id,
            total_count,
        )

        return FootfallRecordResponse(
            success=True,
            camera_id=camera.id,
            visitor_track_id=vt.id,
            footfall_event_id=event.id if event else vt.id,
            total_camera_footfall=total_count,
            message="Entry footfall recorded successfully.",
        )

    def record_heartbeat(
        self, camera_id: uuid.UUID, heartbeat: CameraHeartbeatRequest
    ) -> Dict[str, Any]:
        """
        Records a telemetry heartbeat from a local edge processor / local executable.
        """
        camera = self.get_camera(camera_id)
        _edge_telemetry[camera.id] = {
            "is_processing": heartbeat.is_processing,
            "fps": heartbeat.fps,
            "frames_processed": heartbeat.frames_processed,
            "current_persons": heartbeat.current_persons,
            "total_entries": heartbeat.total_entries,
            "section_counts": heartbeat.section_counts,
            "section_occupancy": heartbeat.section_occupancy,
            "last_seen": time.time(),
        }
        return {"status": "ok", "camera_id": camera.id}

