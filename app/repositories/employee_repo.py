from datetime import datetime, timezone
from typing import List, Optional, Tuple
import uuid
import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.db.models.employee import Employee, EmployeeFace
from app.vision.vector_store import PgVectorStoreManager


class EmployeeRepository:
    def __init__(self, db: Session):
        self.db = db
        self.vector_manager = PgVectorStoreManager(db)

    def get_by_id(self, employee_uuid: uuid.UUID) -> Optional[Employee]:
        return self.db.query(Employee).filter(Employee.id == employee_uuid).first()

    def get_by_employee_id(self, employee_id: str) -> Optional[Employee]:
        return self.db.query(Employee).filter(Employee.employee_id == str(employee_id)).first()

    def get_by_emp_id(self, employee_id: str) -> Optional[Employee]:
        return self.get_by_employee_id(employee_id)

    def find_best_match(
        self,
        embedding: np.ndarray,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """Queries pgvector with HNSW index for the closest active employee face."""
        return self.vector_manager.find_best_employee_match(
            embedding=embedding,
            similarity_threshold=similarity_threshold,
            db=self.db,
        )

    def create(
        self,
        employee_id: str,
        name: str,
        department: str = "General",
        designation: str = "Staff",
        assigned_camera_id: Optional[uuid.UUID] = None,
        assigned_location: Optional[str] = None,
        shift_start_time=None,
        shift_end_time=None,
        late_grace_minutes: int = 15,
        early_departure_grace_minutes: int = 15,
        embedding: Optional[np.ndarray] = None,
        gallery_embeddings: Optional[List[np.ndarray]] = None,
        photo_url: Optional[str] = None,
        is_active: bool = True,
    ) -> Employee:
        vec_data = embedding.astype(np.float32).tolist() if embedding is not None else None
        gallery_data = (
            [g.astype(np.float32).tolist() for g in gallery_embeddings]
            if gallery_embeddings
            else None
        )
        emp = Employee(
            employee_id=str(employee_id),
            name=name,
            department=department,
            designation=designation,
            assigned_camera_id=assigned_camera_id,
            assigned_location=assigned_location,
            shift_start_time=shift_start_time,
            shift_end_time=shift_end_time,
            late_grace_minutes=late_grace_minutes,
            early_departure_grace_minutes=early_departure_grace_minutes,
            face_embedding=vec_data,
            face_gallery=gallery_data,
            photo_url=photo_url,
            is_active=is_active,
        )
        self.db.add(emp)
        self.db.commit()
        self.db.refresh(emp)

        # Sync to pgvector employee_faces table for multi-angle search
        if embedding is not None or gallery_embeddings:
            angles = gallery_embeddings if gallery_embeddings else ([embedding] if embedding is not None else [])
            for idx, ang_emb in enumerate(angles):
                label = "front" if idx == 0 else f"angle_{idx}"
                ef = EmployeeFace(
                    employee_id=emp.id,
                    angle_label=label,
                    face_embedding=ang_emb.astype(np.float32).tolist(),
                    photo_url=photo_url if idx == 0 else None,
                )
                self.db.add(ef)
            self.db.commit()
            self.db.refresh(emp)

        logger.info("Registered new employee %s (%s, %s)", emp.employee_id, emp.name, emp.department)
        return emp

    def list_employees(
        self,
        department: Optional[str] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Employee]:
        query = self.db.query(Employee)
        if department:
            query = query.filter(Employee.department == department)
        if is_active is not None:
            query = query.filter(Employee.is_active.is_(is_active))
        return query.order_by(Employee.employee_id.asc()).offset(skip).limit(limit).all()

    def update(self, employee: Employee, **kwargs) -> Employee:
        for k, v in kwargs.items():
            if hasattr(employee, k) and v is not None:
                setattr(employee, k, v)
        self.db.add(employee)
        self.db.commit()
        self.db.refresh(employee)
        return employee

    def update_face_embedding(
        self,
        employee: Employee,
        embedding: np.ndarray,
        gallery_embeddings: Optional[List[np.ndarray]] = None,
        photo_url: Optional[str] = None,
    ) -> Employee:
        employee.face_embedding = embedding.astype(np.float32).tolist()
        if gallery_embeddings:
            employee.face_gallery = [g.astype(np.float32).tolist() for g in gallery_embeddings]
        if photo_url is not None:
            employee.photo_url = photo_url
        self.db.add(employee)

        # Sync to pgvector employee_faces table for multi-angle HNSW search
        self.db.query(EmployeeFace).filter(EmployeeFace.employee_id == employee.id).delete()
        angles = gallery_embeddings if gallery_embeddings else [embedding]
        for idx, ang_emb in enumerate(angles):
            label = "front" if idx == 0 else f"angle_{idx}"
            ef = EmployeeFace(
                employee_id=employee.id,
                angle_label=label,
                face_embedding=ang_emb.astype(np.float32).tolist(),
                photo_url=photo_url if idx == 0 else None,
            )
            self.db.add(ef)

        self.db.commit()
        self.db.refresh(employee)
        logger.info(
            "Updated biometric face embedding (gallery size: %d) for employee %s",
            len(gallery_embeddings) if gallery_embeddings else 1,
            employee.employee_id,
        )
        return employee

    def set_active_status(self, employee: Employee, is_active: bool) -> Employee:
        employee.is_active = is_active
        self.db.add(employee)
        self.db.commit()
        self.db.refresh(employee)
        logger.info("Set employee %s active status: %s", employee.employee_id, is_active)
        return employee

    def get_all_active_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """
        Retrieves all active registered employees and their face embeddings,
        including all multi-angle gallery photos.
        """
        records = (
            self.db.query(Employee)
            .filter(
                Employee.is_active.is_(True),
                (Employee.face_embedding.isnot(None) | Employee.face_gallery.isnot(None)),
            )
            .all()
        )
        results = []
        for r in records:
            gallery = r.get_gallery_embeddings()
            for g_emb in gallery:
                results.append((r.employee_id, g_emb))
        return results
