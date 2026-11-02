import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DepartmentCreate(BaseModel):
    id: Optional[str] = Field(None, max_length=50, description="Department slug (e.g. sales, vip_lounge)")
    name: str = Field(..., min_length=1, max_length=100, description="Department display name")
    color: str = Field(default="#00D7FF", max_length=20, description="Hex color (e.g. #00D7FF)")
    camera_id: Optional[uuid.UUID] = Field(None, description="Optional target camera UUID")
    box: Optional[List[int]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    is_active: bool = Field(default=True)


class DepartmentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    color: Optional[str] = Field(None, max_length=20)
    camera_id: Optional[uuid.UUID] = None
    box: Optional[List[int]] = None
    is_active: Optional[bool] = None


class DepartmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    color: str
    camera_id: Optional[uuid.UUID] = None
    box: Optional[List[int]] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

