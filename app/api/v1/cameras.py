import copy
import json
import os
import time
import uuid
from typing import Any, Dict, Generator, List, Optional
import cv2
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.api.routes import (
    CAMERA_CONFIG,
    CAMERA_FRAME,
    CAMERA_HEARTBEAT,
    CAMERA_STATUS,
    CAMERA_STREAM,
    DELETE_CAMERA_CHANNEL,
    GET_CAMERA,
    LIST_CAMERAS,
    RECORD_CAMERA_FOOTFALL,
    REGISTER_CAMERAS,
    START_CAMERA,
    STOP_CAMERA,
    TEST_CAMERA_STREAM,
    UPDATE_CAMERA,
)
from app.core.logging import logger, sanitize_url
from app.db.models.camera import Camera
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
    CAMERA_CONFIG,
    summary="Retrieve complete camera and edge detection configuration",
)
def get_camera_config(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Retrieves full dynamic configuration from camera_config.json with DB synchronization."""
    config_path = "camera_config.json"
    data: Dict[str, Any] = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to parse camera_config.json: %s", str(e))

    if not data and os.path.exists("camera_config.example.json"):
        try:
            with open("camera_config.example.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    # Ensure robust defaults
    data.setdefault("default_channel", "101")
    data.setdefault("model_type", "yolo")
    data.setdefault("confidence", 0.31)
    data.setdefault("imgsz", 960)
    data.setdefault("face_confidence", 0.36)
    data.setdefault("face_similarity_threshold", 0.65)
    data.setdefault("face_scale", 1.05)
    data.setdefault("box_smoothing", True)
    data.setdefault("smoothing_alpha", 0.35)
    data.setdefault("show_entry_line", True)
    data.setdefault("allow_standalone_faces", False)
    data.setdefault("show_head_fallback", False)
    data.setdefault("deadband_px", 6.0)
    data.setdefault("body_persistence", 26)
    cams = data.setdefault("cameras", {})

    # Only seed initial cameras from database if configuration has NO cameras configured
    if not cams:
        try:
            db_cameras = db.query(Camera).filter(Camera.is_active.is_(True)).all()
            for idx, db_cam in enumerate(db_cameras):
                ch_key = f"{101 + idx * 100}"
                cams[ch_key] = {
                    "name": db_cam.name,
                    "camera_id": str(db_cam.id),
                    "source": db_cam.stream_url,
                    "confidence": data.get("confidence", 0.31),
                    "min_feet_y": 0.08,
                    "allow_standalone_faces": False,
                    "is_active": True,
                    "entry_line": {
                        "enabled": True,
                        "p1": [200, 500],
                        "p2": [1000, 500],
                        "direction": "top_to_bottom",
                    } if db_cam.entry_line_config else None,
                }
        except Exception as e:
            logger.debug("Database camera initial seed skipped: %s", str(e))

    return data


@router.post(
    CAMERA_CONFIG,
    summary="Save dynamic camera and AI settings to camera_config.json and sync with DB",
)
def save_camera_config(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Updates camera_config.json dynamically and synchronizes with PostgreSQL database."""
    config_path = "camera_config.json"
    existing_data: Dict[str, Any] = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            pass

    # Update global config keys
    for key, val in payload.items():
        if key != "cameras":
            existing_data[key] = val

    # Set cameras dictionary directly so deleted cameras are not retained
    if "cameras" in payload and isinstance(payload["cameras"], dict):
        new_cams = payload["cameras"]
        old_cams = existing_data.get("cameras", {})
        archived_profiles = existing_data.get("archived_camera_profiles", {})
        
        # Hardcoded backend calibration profiles for known cameras (e.g. 101 ceiling lamp filter & exclusion zones)
        CHANNEL_CALIBRATIONS = {
            "101": {
                "min_feet_y": 0.32,
                "exclusion_zones": [
                    {"name": "ceiling_lamp", "box": [40, 40, 200, 220]},
                    {"name": "wall_decor", "box": [50, 430, 120, 520]},
                    {"name": "bottom_floor_edge", "box": [550, 620, 800, 720]},
                ],
            }
        }

        # Preserve calibrated edge parameters like exclusion_zones, sections, and min_feet_y in backend
        for ch, cam_dict in new_cams.items():
            if isinstance(cam_dict, dict):
                source_template = old_cams.get(ch) or archived_profiles.get(ch) or CHANNEL_CALIBRATIONS.get(ch, {})
                if isinstance(source_template, dict):
                    if ("exclusion_zones" not in cam_dict or not cam_dict.get("exclusion_zones")) and "exclusion_zones" in source_template:
                        cam_dict["exclusion_zones"] = source_template["exclusion_zones"]
                    if ("sections" not in cam_dict or not cam_dict["sections"]) and "sections" in source_template:
                        cam_dict["sections"] = source_template["sections"]
                    if ("min_feet_y" not in cam_dict or cam_dict["min_feet_y"] is None) and "min_feet_y" in source_template:
                        cam_dict["min_feet_y"] = source_template["min_feet_y"]
                    if not cam_dict.get("camera_id") and "camera_id" in source_template:
                        cam_dict["camera_id"] = source_template["camera_id"]
                
                # Channel-specific hardcoded fallbacks
                if ch in CHANNEL_CALIBRATIONS:
                    calib = CHANNEL_CALIBRATIONS[ch]
                    if "min_feet_y" not in cam_dict or cam_dict["min_feet_y"] == 0.08:
                        cam_dict["min_feet_y"] = calib["min_feet_y"]
                    if "exclusion_zones" not in cam_dict or not cam_dict.get("exclusion_zones"):
                        cam_dict["exclusion_zones"] = calib["exclusion_zones"]

        existing_data["cameras"] = new_cams

        # Sync cameras to PostgreSQL database
        current_cam_ids = set()
        for ch, cam_dict in new_cams.items():
            if isinstance(cam_dict, dict):
                cam_id_str = cam_dict.get("camera_id")
                cam_name = cam_dict.get("name", f"Camera {ch}")
                cam_source = cam_dict.get("source", "")
                cam_loc = cam_dict.get("location") or f"Channel {ch}"
                entry_line_cfg = cam_dict.get("entry_line")

                entry_line_db = None
                if entry_line_cfg and isinstance(entry_line_cfg, dict) and "p1" in entry_line_cfg and "p2" in entry_line_cfg:
                    entry_line_db = {
                        "line": [entry_line_cfg["p1"], entry_line_cfg["p2"]],
                        "direction": entry_line_cfg.get("direction", "top_to_bottom"),
                    }

                try:
                    if cam_id_str:
                        current_cam_ids.add(str(cam_id_str))
                        try:
                            uid = uuid.UUID(str(cam_id_str))
                            db_cam = db.query(Camera).filter(Camera.id == uid).first()
                            if db_cam:
                                db_cam.name = cam_name
                                if cam_source:
                                    db_cam.stream_url = cam_source
                                db_cam.location = cam_loc
                                db_cam.is_active = True
                                if entry_line_db:
                                    db_cam.entry_line_config = entry_line_db
                                db.commit()
                        except ValueError:
                            pass
                except Exception as e:
                    logger.warning("Failed to sync camera channel %s to DB: %s", ch, str(e))

        # Deactivate any active DB cameras that were removed from the camera config
        try:
            active_db_cams = db.query(Camera).filter(Camera.is_active.is_(True)).all()
            for active_cam in active_db_cams:
                if str(active_cam.id) not in current_cam_ids:
                    # Also check if any channel references it
                    is_referenced = any(
                        isinstance(c, dict) and str(c.get("camera_id")) == str(active_cam.id)
                        for c in new_cams.values()
                    )
                    if not is_referenced:
                        active_cam.is_active = False
            db.commit()
        except Exception as e:
            logger.warning("Failed to deactivate removed DB cameras: %s", str(e))

    # Atomically write to camera_config.json
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)
        logger.info("Successfully saved dynamic settings to camera_config.json")
    except Exception as e:
        logger.error("Failed to write to camera_config.json: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to write config: {str(e)}")

    return {
        "success": True,
        "message": "Camera configuration saved and dynamically hot-reloaded for Edge Detector.",
        "config": existing_data,
    }


@router.post(
    TEST_CAMERA_STREAM,
    summary="Test connectivity and retrieve frame resolution/FPS for an RTSP or video stream",
)
def test_camera_stream(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Tests opening a stream URL with OpenCV to verify resolution, FPS, and reachable status."""
    stream_url = str(payload.get("stream_url", "")).strip()
    if not stream_url:
        raise HTTPException(status_code=400, detail="stream_url is required")

    source: Any = int(stream_url) if stream_url.isdigit() else stream_url
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {
                "online": False,
                "error": "Failed to connect to stream. Please verify IP address, port (554), and credentials.",
            }
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return {
                "online": False,
                "error": "Connection opened, but no video frame could be read. The stream may be buffering or offline.",
            }
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return {
            "online": True,
            "resolution": [w, h],
            "fps": round(float(fps), 1),
            "message": f"Connected successfully! Stream resolution: {w}x{h} @ {round(fps, 1)} FPS",
        }
    except Exception as e:
        return {
            "online": False,
            "error": f"Error testing stream: {str(e)}",
        }


@router.delete(
    DELETE_CAMERA_CHANNEL,
    summary="Delete a camera channel from camera_config.json and deactivate in DB",
)
def delete_camera_channel(
    channel_or_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Removes a camera channel from camera_config.json and deactivates it in PostgreSQL."""
    config_path = "camera_config.json"
    deleted = False
    target_cam_id = None

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cams = data.get("cameras", {})
            if channel_or_id in cams:
                if isinstance(cams[channel_or_id], dict):
                    target_cam_id = cams[channel_or_id].get("camera_id")
                    # Archive calibrated profile (exclusion zones, floor filter, entry line) so it can be restored
                    archive = data.setdefault("archived_camera_profiles", {})
                    archive[channel_or_id] = copy.deepcopy(cams[channel_or_id])
                del cams[channel_or_id]
                deleted = True
            else:
                target_key = None
                for k, v in cams.items():
                    if isinstance(v, dict) and str(v.get("camera_id")) == channel_or_id:
                        target_key = k
                        target_cam_id = channel_or_id
                        archive = data.setdefault("archived_camera_profiles", {})
                        archive[k] = copy.deepcopy(v)
                        break
                if target_key:
                    del cams[target_key]
                    deleted = True
            if deleted:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to delete camera from config: %s", str(e))

    # Deactivate in PostgreSQL
    lookup_id = target_cam_id or channel_or_id
    try:
        uid = uuid.UUID(str(lookup_id))
        db_cam = db.query(Camera).filter(Camera.id == uid).first()
        if db_cam:
            db_cam.is_active = False
            db.commit()
            deleted = True
    except Exception:
        pass

    # Also search by name if channel like "101"
    try:
        db_cams = db.query(Camera).filter(Camera.name.ilike(f"%{channel_or_id}%")).all()
        for dc in db_cams:
            dc.is_active = False
        if db_cams:
            db.commit()
            deleted = True
    except Exception:
        pass

    return {
        "success": deleted,
        "message": "Camera channel deleted successfully" if deleted else "Camera not found",
    }


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


