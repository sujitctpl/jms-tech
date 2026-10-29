import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.core.logging import sanitize_url

DirectionLiteral = Literal[
    "left_to_right",
    "right_to_left",
    "top_to_bottom",
    "bottom_to_top",
    "positive_to_negative",
    "negative_to_positive",
    "inward",
    "outward",
]


class EntryLineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line: List[List[float]] = Field(
        ...,
        description="Two 2D points defining the entry line: [[x1, y1], [x2, y2]]",
        min_length=2,
        max_length=2,
    )
    direction: DirectionLiteral = Field(
        ...,
        description="Designated entry crossing direction",
    )

    @field_validator("line")
    @classmethod
    def validate_line_points(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 2:
            raise ValueError("Entry line must contain exactly two points: [[x1, y1], [x2, y2]]")
        for pt in v:
            if len(pt) != 2:
                raise ValueError("Each point in entry line must be a 2D coordinate [x, y]")
            if pt[0] < 0 or pt[1] < 0:
                raise ValueError("Coordinates must be non-negative numbers")
        if v[0] == v[1]:
            raise ValueError("The two points defining the line cannot be identical")
        return v


class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Camera display name")
    stream_url: str = Field(..., min_length=1, description="CCTV RTSP or video stream URL")
    location: Optional[str] = Field(None, max_length=150, description="Showroom location/zone")
    entry_line_config: Optional[EntryLineConfig] = Field(
        None, description="Virtual entry-line configuration"
    )


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    stream_url: Optional[str] = Field(None, min_length=1)
    location: Optional[str] = Field(None, max_length=150)
    is_active: Optional[bool] = Field(None)
    entry_line_config: Optional[EntryLineConfig] = Field(None)


class CameraResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    stream_url: str
    location: Optional[str] = None
    is_active: bool
    entry_line_config: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    @field_validator("stream_url", mode="after")
    @classmethod
    def mask_credentials(cls, v: str) -> str:
        # Keep stream credentials out of API responses
        return sanitize_url(v)


class CameraStatusResponse(BaseModel):
    camera_id: uuid.UUID
    name: str
    is_active: bool
    is_processing: bool
    active_run_id: Optional[uuid.UUID] = None
    frames_processed: int = 0
    last_error: Optional[str] = None
    fps: Optional[float] = None
    current_persons: int = 0
    total_entries: int = 0
    section_counts: Dict[str, int] = Field(default_factory=dict)
    section_occupancy: Dict[str, int] = Field(default_factory=dict)


class FootfallRecordRequest(BaseModel):
    tracker_id: str = Field(..., description="Unique track ID detected by local model")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Detection confidence score")
    occurred_at: Optional[datetime] = Field(None, description="Timestamp of the crossing event")
    processing_run_id: Optional[uuid.UUID] = Field(None, description="Optional active run ID")
    section_id: Optional[str] = Field(None, description="Section ID if entering a section")
    section_name: Optional[str] = Field(None, description="Section Name (e.g. Sales, Accounts)")
    person_id: Optional[str] = Field(None, description="Persistent person ID if identity recovered")


class FootfallRecordResponse(BaseModel):
    success: bool
    camera_id: uuid.UUID
    visitor_track_id: uuid.UUID
    footfall_event_id: uuid.UUID
    total_camera_footfall: int
    message: str


class CameraHeartbeatRequest(BaseModel):
    is_processing: bool = True
    fps: Optional[float] = None
    frames_processed: int = 0
    current_persons: int = 0
    total_entries: int = 0
    section_counts: Dict[str, int] = Field(default_factory=dict)
    section_occupancy: Dict[str, int] = Field(default_factory=dict)

