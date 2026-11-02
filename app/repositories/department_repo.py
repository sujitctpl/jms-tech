import json
import os
import re
import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.db.models.department import Department
from app.schemas.department import DepartmentCreate, DepartmentUpdate


def hex_to_bgr(hex_color: str) -> List[int]:
    """Convert hex #RRGGBB to OpenCV [B, G, R]."""
    if not hex_color:
        return [0, 215, 255]
    h = hex_color.lstrip("#")
    if len(h) == 6:
        try:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return [b, g, r]
        except ValueError:
            pass
    return [0, 215, 255]


def bgr_to_hex(bgr: List[int]) -> str:
    """Convert OpenCV [B, G, R] to hex #RRGGBB."""
    if isinstance(bgr, (list, tuple)) and len(bgr) >= 3:
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        return f"#{r:02x}{g:02x}{b:02x}".upper()
    return "#00D7FF"


def slugify(text: str) -> str:
    """Generate a clean slug identifier from name."""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:50] or "dept"


class DepartmentRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_departments(self, camera_id: Optional[uuid.UUID] = None, only_active: bool = False) -> List[Department]:
        query = self.db.query(Department)
        if only_active:
            query = query.filter(Department.is_active.is_(True))
        if camera_id is not None:
            query = query.filter((Department.camera_id == camera_id) | (Department.camera_id.is_(None)))
        return query.order_by(Department.created_at.asc()).all()

    def get_department(self, dept_id: str) -> Optional[Department]:
        return self.db.query(Department).filter(Department.id == dept_id).first()

    def create_department(self, data: DepartmentCreate) -> Department:
        dept_id = data.id.strip() if data.id else slugify(data.name)
        # Check if ID already exists
        existing = self.get_department(dept_id)
        if existing:
            dept_id = f"{dept_id}_{int(uuid.uuid4().hex[:4], 16) % 1000}"

        dept = Department(
            id=dept_id,
            name=data.name.strip(),
            color=data.color.strip() if data.color else "#00D7FF",
            camera_id=data.camera_id,
            box=data.box,
            is_active=data.is_active,
        )
        self.db.add(dept)
        self.db.commit()
        self.db.refresh(dept)
        self.sync_to_camera_config()
        return dept

    def update_department(self, dept_id: str, data: DepartmentUpdate) -> Optional[Department]:
        dept = self.get_department(dept_id)
        if not dept:
            return None

        update_dict = data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(dept, key, value)

        self.db.commit()
        self.db.refresh(dept)
        self.sync_to_camera_config()
        return dept

    def delete_department(self, dept_id: str) -> bool:
        dept = self.get_department(dept_id)
        if not dept:
            return False
        self.db.delete(dept)
        self.db.commit()
        self.sync_to_camera_config()
        return True

    def sync_to_camera_config(self, config_path: str = "camera_config.json") -> bool:
        """Synchronizes database departments to camera_config.json sections, preserving existing drawn coordinates."""
        try:
            if not os.path.exists(config_path):
                return False

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            active_depts = self.list_departments(only_active=True)

            cams = config_data.setdefault("cameras", {})
            for cam_key, cam_data in cams.items():
                if isinstance(cam_data, dict):
                    cam_id = cam_data.get("camera_id")
                    existing_sections = cam_data.get("sections", [])
                    # Map existing section id -> box to preserve user-drawn boxes
                    existing_boxes = {}
                    if isinstance(existing_sections, list):
                        for es in existing_sections:
                            if isinstance(es, dict) and "id" in es and "box" in es:
                                if es["box"] and es["box"] != [0, 0, 0, 0]:
                                    existing_boxes[es["id"]] = es["box"]

                    cam_sections = []
                    for d in active_depts:
                        if d.camera_id is None or (cam_id and str(d.camera_id) == str(cam_id)):
                            # Use DB box, or fallback to existing drawn box in camera_config, or [0, 0, 0, 0]
                            box_val = d.box
                            if (not box_val or box_val == [0, 0, 0, 0]) and d.id in existing_boxes:
                                box_val = existing_boxes[d.id]
                                # Save preserved box back to DB as well
                                d.box = box_val

                            cam_sections.append({
                                "id": d.id,
                                "name": d.name,
                                "color": hex_to_bgr(d.color),
                                "box": box_val or [0, 0, 0, 0],
                            })
                    cam_data["sections"] = cam_sections

            # Commit any box updates preserved back to DB
            self.db.commit()

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)

            logger.info("Synchronized %d departments to %s", len(active_depts), config_path)
            return True
        except Exception as e:
            logger.warning("Failed to sync departments to camera_config.json: %s", str(e))
            return False

    def seed_defaults_if_empty(self) -> None:
        """Seeds default showroom departments if table is empty."""
        count = self.db.query(Department).count()
        if count == 0:
            defaults = [
                DepartmentCreate(id="sales", name="Sales Section", color="#00D7FF", box=[0, 242, 423, 618]),
                DepartmentCreate(id="service", name="Service Section", color="#32E178", box=[430, 242, 850, 618]),
                DepartmentCreate(id="accounts", name="Accounts Section", color="#FF8C00", box=[860, 242, 1280, 618]),
            ]
            for d in defaults:
                dept = Department(
                    id=d.id,
                    name=d.name,
                    color=d.color,
                    box=d.box,
                    is_active=True,
                )
                self.db.add(dept)
            self.db.commit()
            self.sync_to_camera_config()
            logger.info("Seeded 3 default departments: Sales, Service, Accounts")
