from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VisitorResponse(BaseModel):
    id: str
    visitorId: str
    avatarUrl: Optional[str] = None
    exitAvatarUrl: Optional[str] = None
    firstSeenTime: str
    lastSeenTime: Optional[str] = None
    exitTime: Optional[str] = None
    visitorType: str = "New"
    visitPurpose: Optional[str] = None
    assignedSalesperson: Optional[str] = None
    durationMinutes: int = 1
    notes: str = ""
    taggingStatus: str = "Untagged"
    date: str
    genderEstimated: Optional[str] = "Unspecified"
    ageGroupEstimated: Optional[str] = "26-35"
    zoneEntered: Optional[str] = "Main Showroom"
    confidenceScore: float = 0.95
    exitMatchConfidence: Optional[float] = None
    cameraSource: Optional[str] = "Entrance Camera"
    zoneDwell: Optional[Dict[str, str]] = None
    waitingSeconds: int = 0
    isWaiting: bool = False
    waitingBreach: bool = False
    waitingFormatted: Optional[str] = None


class VisitorListResponse(BaseModel):
    visitors: List[VisitorResponse]
    total: int
    tagged_count: int
    untagged_count: int


class VisitorTagRequest(BaseModel):
    visit_purpose: str = Field(..., description="E.g. Test Ride, Vehicle Enquiry, Booking, Service, etc.")
    assigned_salesperson: Optional[str] = Field(None, description="Assigned staff name")
    notes: Optional[str] = Field(None, description="Notes on the visit")


class VisitorDetectRequest(BaseModel):
    face_image_base64: Optional[str] = Field(None, description="Base64 encoded JPEG crop of the face")
    face_embedding: Optional[List[float]] = Field(None, description="512-dim ArcFace embedding vector")
    body_embedding: Optional[List[float]] = Field(None, description="512-dim OSNet body ReID embedding vector")
    person_id: Optional[str] = Field(None, description="Persistent visitor ID if already resolved")
    camera_id: Optional[str] = None
    camera_name: Optional[str] = "Main Entrance"
    tracker_id: Optional[str] = None
    confidence: Optional[float] = 0.95
    zone: Optional[str] = "Main Showroom"
    appearance: Optional[str] = Field(None, description="Estimated clothing appearance e.g. Blue Top")
    zone_dwell: Optional[Dict[str, str]] = Field(None, description="Dwell times per detection zone")
    exit_time: Optional[str] = Field(None, description="Departure time formatted string")
    is_entrance: Optional[bool] = Field(None, description="Whether this camera is a store entrance camera")
    waiting_seconds: Optional[int] = Field(None, description="Accumulated unattended waiting time in seconds")
    is_waiting: Optional[bool] = Field(None, description="Whether visitor is currently waiting unattended")
    waiting_breach: Optional[bool] = Field(None, description="Whether waiting time exceeded cutoff (15 mins)")


class VisitorZoneUpdateRequest(BaseModel):
    person_id: Optional[str] = None
    tracker_id: Optional[str] = None
    current_zone: Optional[str] = None
    zone_dwell: Optional[Dict[str, str]] = None
    exit_time: Optional[str] = None
    notes_append: Optional[str] = None
    waiting_seconds: Optional[int] = None
    is_waiting: Optional[bool] = None
    waiting_breach: Optional[bool] = None
    face_image_base64: Optional[str] = Field(None, description="Optional Base64 encoded snapshot captured on SLA breach or zone update")
    exit_image_base64: Optional[str] = Field(None, description="Optional Base64 encoded snapshot captured when visitor departs through entrance")
    body_embedding: Optional[List[float]] = Field(None, description="Optional 512-dim OSNet body ReID embedding vector")
    exclude_person_id: Optional[str] = Field(None, description="Visitor the edge already ruled out for this exit (face contradiction); never attach the exit to it")



class VisitorDetectResponse(BaseModel):
    status: str = Field(..., description="VISITOR_CREATED | VISITOR_REIDENTIFIED | EMPLOYEE_DETECTED")
    is_employee: bool = False
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None
    person_id: Optional[str] = None
    is_new: bool = False
    visitor: Optional[VisitorResponse] = None
    message: str


class WalkInCreateRequest(BaseModel):
    visit_purpose: Optional[str] = None
    assigned_salesperson: Optional[str] = None
    notes: Optional[str] = None
    zone_entered: Optional[str] = "Main Showroom"
    visitor_type: Optional[str] = "New"

