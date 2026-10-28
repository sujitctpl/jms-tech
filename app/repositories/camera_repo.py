import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from app.db.models.camera import Camera
from app.schemas.camera import CameraCreate, CameraUpdate


class CameraRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, camera_id: uuid.UUID) -> Optional[Camera]:
        return self.db.query(Camera).filter(Camera.id == camera_id).first()

    def list_all(self, active_only: bool = False) -> List[Camera]:
        query = self.db.query(Camera)
        if active_only:
            query = query.filter(Camera.is_active.is_(True))
        return query.order_by(Camera.created_at.desc()).all()

    def create(self, camera_in: CameraCreate) -> Camera:
        entry_line_dict = (
            camera_in.entry_line_config.model_dump()
            if camera_in.entry_line_config
            else None
        )
        camera = Camera(
            name=camera_in.name,
            stream_url=camera_in.stream_url,
            location=camera_in.location,
            entry_line_config=entry_line_dict,
        )
        self.db.add(camera)
        self.db.commit()
        self.db.refresh(camera)
        return camera

    def update(self, camera: Camera, camera_update: CameraUpdate) -> Camera:
        update_data = camera_update.model_dump(exclude_unset=True)
        if "entry_line_config" in update_data and update_data["entry_line_config"] is not None:
            update_data["entry_line_config"] = update_data["entry_line_config"].copy()

        for field, value in update_data.items():
            setattr(camera, field, value)

        self.db.add(camera)
        self.db.commit()
        self.db.refresh(camera)
        return camera
