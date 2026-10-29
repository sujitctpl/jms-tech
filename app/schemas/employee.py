from datetime import date, datetime, time
from typing import List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field


class EmployeeBase(BaseModel):
    employee_id: str = Field(..., min_length=1, max_length=50, description="Unique Employee ID (e.g. EMP001)")
    name: str = Field(..., min_length=1, max_length=100, description="Full Name")
    department: str = Field(default="General", min_length=1, max_length=100, description="Department (e.g. Sales, Accounts)")
    designation: str = Field(default="Staff", min_length=1, max_length=100, description="Designation (e.g. Sales Executive)")
    assigned_camera_id: Optional[uuid.UUID] = Field(None, description="Assigned CCTV Camera UUID")
    assigned_location: Optional[str] = Field(None, max_length=150, description="Assigned Location / Showroom Counter")
    shift_start_time: Optional[time] = Field(default=time(9, 30), description="Shift Start Time (default 09:30)")
    shift_end_time: Optional[time] = Field(default=time(18, 30), description="Shift End Time (default 18:30)")
    late_grace_minutes: int = Field(default=15, ge=0, le=120, description="Late arrival grace period in minutes")
    early_departure_grace_minutes: int = Field(default=15, ge=0, le=120, description="Early departure grace in minutes")
    photo_url: Optional[str] = Field(None, max_length=500, description="Profile picture URL or path (1st uploaded photo)")


class FaceEnrollmentRequest(BaseModel):
    face_image_base64: Optional[str] = Field(None, description="Base64 encoded face portrait image")
    face_images_base64: Optional[List[str]] = Field(None, description="Optional list of Base64 photos (1st used as profile picture)")
    embedding_vector: Optional[List[float]] = Field(None, description="512-dim ArcFace embedding vector")
    embedding: Optional[List[float]] = Field(None, description="Alias for embedding_vector")


class EmployeeCreate(EmployeeBase):
    face_embedding: Optional[List[float]] = Field(None, description="Optional 512-dim ArcFace embedding vector")
    face_image_base64: Optional[str] = Field(None, description="Optional Base64 encoded portrait image for embedding extraction")
    face_images_base64: Optional[List[str]] = Field(None, description="Optional list of Base64 encoded images (1st kept as profile picture)")


class EmployeeUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    department: Optional[str] = Field(None, min_length=1, max_length=100)
    designation: Optional[str] = Field(None, min_length=1, max_length=100)
    assigned_camera_id: Optional[uuid.UUID] = None
    assigned_location: Optional[str] = Field(None, max_length=150)
    shift_start_time: Optional[time] = None
    shift_end_time: Optional[time] = None
    late_grace_minutes: Optional[int] = Field(None, ge=0, le=120)
    early_departure_grace_minutes: Optional[int] = Field(None, ge=0, le=120)
    photo_url: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None


class EmployeeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    employee_id: str
    name: str
    department: str
    designation: str
    assigned_camera_id: Optional[uuid.UUID] = None
    assigned_location: Optional[str] = None
    shift_start_time: Optional[time] = None
    shift_end_time: Optional[time] = None
    late_grace_minutes: int = 15
    early_departure_grace_minutes: int = 15
    photo_url: Optional[str] = None
    is_active: bool = True
    has_face_registered: bool = False
    created_at: datetime
    updated_at: datetime


class AttendancePunchRequest(BaseModel):
    employee_id: str = Field(..., description="Registered Employee ID")
    camera_id: Optional[uuid.UUID] = Field(None, description="CCTV Camera ID that captured the punch")
    event_type: str = Field(default="CHECK_IN", description="CHECK_IN, CHECK_OUT, or PRESENCE")
    timestamp: Optional[datetime] = Field(None, description="Event timestamp")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Recognition confidence")


class AttendancePunchResponse(BaseModel):
    success: bool
    employee_id: str
    name: str
    event_type: str
    recorded_at: datetime
    status: str
    first_check_in: Optional[datetime] = None
    last_check_out: Optional[datetime] = None
    total_work_hours: float = 0.0
    is_late: bool = False
    message: str


class DailyAttendanceItem(BaseModel):
    employee_id: str
    name: str
    department: str
    designation: str
    date: date
    first_check_in: Optional[datetime] = None
    last_check_out: Optional[datetime] = None
    total_work_hours: float = 0.0
    is_late: bool = False
    late_minutes: float = 0.0
    is_early_departure: bool = False
    early_departure_minutes: float = 0.0
    status: str


class DailyAttendanceReportResponse(BaseModel):
    date: date
    total_employees: int
    present_count: int
    late_count: int
    absent_count: int
    half_day_count: int
    records: List[DailyAttendanceItem]
