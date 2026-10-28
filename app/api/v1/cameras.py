import time
import uuid
from typing import Any, Dict, Generator, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.api.routes import (
    CAMERA_FRAME,
    CAMERA_HEARTBEAT,
    CAMERA_STATUS,
    CAMERA_STREAM,
    GET_CAMERA,
    LIST_CAMERAS,
    RECORD_CAMERA_FOOTFALL,
    REGISTER_CAMERAS,
    START_CAMERA,
    STOP_CAMERA,
    UPDATE_CAMERA,
)
from app.db.session import get_db
from app.schemas.camera import (
    CameraCreate,
    CameraHeartbeatRequest,
    CameraResponse,
    CameraStatusResponse,
    CameraUpdate,
    FootfallRecordRequest,
    FootfallRecordResponse,
)
from app.services.camera_service import CameraService
from app.workers.video_worker import worker_manager

router = APIRouter(tags=["Cameras"])


@router.post(
    REGISTER_CAMERAS,
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a CCTV camera",
)
def register_camera(
    camera_in: CameraCreate,
    db: Session = Depends(get_db),
) -> Any:
    """Register a CCTV camera with name, stream URL, location, and entry-line configuration."""
    service = CameraService(db)
    return service.register_camera(camera_in)


@router.get(
    LIST_CAMERAS,
    response_model=List[CameraResponse],
    summary="List configured cameras",
)
def list_cameras(
    db: Session = Depends(get_db),
) -> Any:
    """List configured cameras."""
    service = CameraService(db)
    return service.list_cameras()


@router.get(
    GET_CAMERA,
    response_model=CameraResponse,
    summary="Retrieve camera details",
)
def get_camera(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> Any:
    """Retrieve details for a specific camera."""
    service = CameraService(db)
    return service.get_camera(camera_id)


@router.patch(
    UPDATE_CAMERA,
    response_model=CameraResponse,
    summary="Update camera configuration",
)
def update_camera(
    camera_id: uuid.UUID,
    camera_in: CameraUpdate,
    db: Session = Depends(get_db),
) -> Any:
    """Update camera configuration or perform soft deactivation."""
    service = CameraService(db)
    return service.update_camera(camera_id, camera_in)


@router.post(
    START_CAMERA,
    summary="Start video processing",
)
def start_camera(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Start continuous video processing and person tracking for a camera."""
    service = CameraService(db)
    return service.start_camera(camera_id)


@router.post(
    STOP_CAMERA,
    summary="Stop video processing",
)
def stop_camera(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Stop continuous video processing cleanly for a camera."""
    service = CameraService(db)
    return service.stop_camera(camera_id)


@router.get(
    CAMERA_STATUS,
    response_model=CameraStatusResponse,
    summary="Retrieve camera processing status",
)
def get_camera_status(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> Any:
    """Retrieve current processing status, active run ID, and processed frames count."""
    service = CameraService(db)
    return service.get_camera_status(camera_id)


@router.post(
    RECORD_CAMERA_FOOTFALL,
    response_model=FootfallRecordResponse,
    summary="Record edge footfall entry event",
)
def record_camera_footfall(
    camera_id: uuid.UUID,
    footfall_in: FootfallRecordRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Record an entry event detected locally by the edge detection model."""
    service = CameraService(db)
    return service.record_footfall(camera_id, footfall_in)


@router.post(
    CAMERA_HEARTBEAT,
    summary="Record edge telemetry heartbeat",
)
def record_camera_heartbeat(
    camera_id: uuid.UUID,
    heartbeat_in: CameraHeartbeatRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Record heartbeat telemetry (FPS, frame count, detected persons) from local edge processor."""
    service = CameraService(db)
    return service.record_heartbeat(camera_id, heartbeat_in)


@router.get(
    CAMERA_FRAME,
    summary="Processed frame snapshot endpoint (Disabled for browser)",
)
def get_camera_frame(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> Any:
    """Live frame preview in the browser is disabled. Detections run locally on the edge processor."""
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Browser live frame preview is disabled. Processing and detection run locally on the edge device.",
    )


@router.get(
    CAMERA_STREAM,
    summary="Live MJPEG video stream (Disabled for browser)",
)
def stream_camera(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> Any:
    """Live preview stream in the browser is disabled. Processing and detection run locally on the edge device."""
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Browser live video preview is disabled. Processing and detection run locally on the edge device.",
    )


