import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class FootfallTotalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total_entries: int
    camera_id: Optional[uuid.UUID] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    section_counts: Optional[dict] = None


class DailyFootfallItem(BaseModel):
    date: str
    entries: int


class DailyFootfallResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    daily_footfall: List[DailyFootfallItem]
    total_entries: int
    camera_id: Optional[uuid.UUID] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
