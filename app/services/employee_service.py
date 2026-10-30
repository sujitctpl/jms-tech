import base64
from datetime import date, datetime, timezone
import os
from typing import Any, List, Optional, Union
import uuid
import cv2
import numpy as np
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.db.models.employee import Employee
from app.repositories.employee_attendance_repo import EmployeeAttendanceRepository
from app.repositories.employee_repo import EmployeeRepository
from app.schemas.employee import (
    AttendancePunchRequest,
    AttendancePunchResponse,
    DailyAttendanceItem,
    DailyAttendanceReportResponse,
    EmployeeCreate,
    EmployeeResponse,
    EmployeeUpdate,
)
from app.vision.employee_recognizer import EmployeeRecognitionManager
from app.vision.face_embedder import FaceEmbedder

# Global singleton employee recognition manager
global_employee_recognizer = EmployeeRecognitionManager()


class EmployeeService:
    def __init__(self, db: Session, recognizer: Optional[EmployeeRecognitionManager] = None):
        self.db = db
        self.repo = EmployeeRepository(db)
        self.attendance_repo = EmployeeAttendanceRepository(db)
        self.recognizer = recognizer or global_employee_recognizer
        self.embedder = FaceEmbedder()

    def register_employee(self, data: EmployeeCreate) -> EmployeeResponse:
        existing = self.repo.get_by_employee_id(data.employee_id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Employee with ID '{data.employee_id}' is already registered.",
            )

        embedding = None
        gallery_embeddings = None
        photo_url = data.photo_url

        # Check for multiple base64 images or single base64 image
        images_b64 = data.face_images_base64 or ([data.face_image_base64] if data.face_image_base64 else [])
        if images_b64:
            # 1. Keep the 1st photo as their profile picture
            first_b64 = images_b64[0]
            if "," in first_b64:
                first_b64 = first_b64.split(",", 1)[1]
            first_b64 = first_b64.strip()
            missing_padding = len(first_b64) % 4
            if missing_padding:
                first_b64 += "=" * (4 - missing_padding)
            try:
                first_bytes = base64.b64decode(first_b64)
                photo_url = self._save_profile_picture(data.employee_id, first_bytes)
            except Exception as e:
                logger.warning("Could not save initial profile picture for %s: %s", data.employee_id, str(e))

            # 2. Extract embeddings from all provided images for multi-angle gallery
            embeddings = []
            for b64_item in images_b64:
                try:
                    emb = self._extract_embedding_from_base64(b64_item)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning("Failed extracting embedding from image sample: %s", str(e))
            if embeddings:
                embedding = embeddings[0]
                gallery_embeddings = embeddings

        elif data.face_embedding is not None and len(data.face_embedding) == 512:
            arr = np.array(data.face_embedding, dtype=np.float32)
            norm = np.linalg.norm(arr)
            embedding = (arr / norm) if norm > 0 else arr

        emp = self.repo.create(
            employee_id=data.employee_id,
            name=data.name,
            department=data.department,
            designation=data.designation,
            assigned_camera_id=data.assigned_camera_id,
            assigned_location=data.assigned_location,
            shift_start_time=data.shift_start_time,
            shift_end_time=data.shift_end_time,
            late_grace_minutes=data.late_grace_minutes,
            early_departure_grace_minutes=data.early_departure_grace_minutes,
            embedding=embedding,
            gallery_embeddings=gallery_embeddings,
            photo_url=photo_url,
        )

        if embedding is not None:
            self.recognizer.register_employee(
                emp.employee_id,
                embedding=embedding,
                gallery_embeddings=gallery_embeddings,
            )

        return self._to_response(emp)

    def list_employees(
        self,
        department: Optional[str] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[EmployeeResponse]:
        employees = self.repo.list_employees(
            department=department,
            is_active=is_active,
            skip=skip,
            limit=limit,
        )
        return [self._to_response(e) for e in employees]

    def _resolve_employee(self, emp_identifier: Any) -> Employee:
        emp = None
        if isinstance(emp_identifier, uuid.UUID):
            emp = self.repo.get_by_id(emp_identifier)
        else:
            try:
                val = uuid.UUID(str(emp_identifier))
                emp = self.repo.get_by_id(val)
            except (ValueError, AttributeError):
                pass
            if not emp:
                emp = self.repo.get_by_employee_id(str(emp_identifier))
        if not emp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Employee '{emp_identifier}' not found.",
            )
        return emp

    def get_employee(self, employee_id: Any) -> EmployeeResponse:
        emp = self._resolve_employee(employee_id)
        return self._to_response(emp)

    def update_employee(self, employee_id: Any, data: EmployeeUpdate) -> EmployeeResponse:
        emp = self._resolve_employee(employee_id)

        updated = self.repo.update(emp, **data.model_dump(exclude_unset=True))
        if data.is_active is False:
            self.recognizer.remove_employee(emp.employee_id)
        elif data.is_active is True and emp.face_embedding is not None:
            arr = emp.get_embedding_array()
            gallery = emp.get_gallery_embeddings()
            if arr is not None:
                self.recognizer.register_employee(emp.employee_id, embedding=arr, gallery_embeddings=gallery)

        return self._to_response(updated)

    def enroll_face(
        self,
        employee_id: Any,
        face_image_base64: Optional[str] = None,
        face_images_base64: Optional[List[str]] = None,
        embedding_vector: Optional[List[float]] = None,
        append: bool = False,
    ) -> EmployeeResponse:
        emp = self._resolve_employee(employee_id)
        photo_url = None

        images_b64 = face_images_base64 or ([face_image_base64] if face_image_base64 else [])
        if images_b64:
            # 1. Keep 1st photo as profile picture if not appending or no photo set yet
            first_b64 = images_b64[0]
            if "," in first_b64:
                first_b64 = first_b64.split(",", 1)[1]
            try:
                first_bytes = base64.b64decode(first_b64)
                if not append or not emp.photo_url:
                    photo_url = self._save_profile_picture(emp.employee_id, first_bytes)
                else:
                    ts = int(datetime.utcnow().timestamp())
                    self._save_profile_picture(emp.employee_id, first_bytes, suffix=f"gallery_{ts}")
            except Exception as e:
                logger.warning("Failed to save 1st photo as profile picture for %s: %s", emp.employee_id, str(e))

            # 2. Extract embeddings from all images
            embeddings = []
            for b64_item in images_b64:
                try:
                    emb = self._extract_embedding_from_base64(b64_item)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning("Failed extracting embedding from photo: %s", str(e))

            if not embeddings:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Face embedding generation failed: No valid face detected in provided images.",
                )
            embedding = embeddings[0]
            gallery_embeddings = embeddings

        elif embedding_vector is not None and len(embedding_vector) == 512:
            arr = np.array(embedding_vector, dtype=np.float32)
            norm = np.linalg.norm(arr)
            embedding = (arr / norm) if norm > 0 else arr
            gallery_embeddings = [embedding]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either face images (Base64) or a 512-dimensional embedding_vector must be provided.",
            )

        if append:
            updated = self.repo.add_face_embeddings(
                emp,
                new_embeddings=gallery_embeddings,
                new_photo_url=photo_url,
                set_as_primary_profile=False,
            )
            self.recognizer.register_employee(
                updated.employee_id,
                embedding=updated.get_embedding_array(),
                gallery_embeddings=updated.get_gallery_embeddings(),
            )
            return self._to_response(updated)

        updated = self.repo.update_face_embedding(
            emp,
            embedding,
            gallery_embeddings=gallery_embeddings,
            photo_url=photo_url,
        )
        self.recognizer.register_employee(
            updated.employee_id,
            embedding=embedding,
            gallery_embeddings=gallery_embeddings,
        )
        return self._to_response(updated)

    def enroll_faces_from_bytes(
        self,
        employee_id: Any,
        images_bytes: List[bytes],
        append: bool = False,
        set_as_primary: bool = False,
    ) -> EmployeeResponse:
        emp = self._resolve_employee(employee_id)
        if not images_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one image must be provided.",
            )

        # 1. Profile picture handling
        photo_url = None
        if not append or set_as_primary or not emp.photo_url:
            first_photo_bytes = images_bytes[0]
            photo_url = self._save_profile_picture(emp.employee_id, first_photo_bytes)
        else:
            ts = int(datetime.utcnow().timestamp())
            self._save_profile_picture(emp.employee_id, images_bytes[0], suffix=f"gallery_{ts}")

        # 2. Extract embeddings from all uploaded images with quality & duplicate checks
        embeddings: List[np.ndarray] = []
        errors: List[str] = []
        existing_gallery = emp.get_gallery_embeddings() if append else []

        for idx, img_b in enumerate(images_bytes):
            arr = np.frombuffer(img_b, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"Photo #{idx + 1}: Could not decode image.")
                continue

            # Quality validation with graceful fallback to face detection
            is_valid, q_reason, cropped, crop_lms, full_lms = self.embedder.validate_enrollment_quality(img)
            if not is_valid:
                logger.warning("Photo #%d for employee %s failed strict quality check: %s. Attempting fallback face extraction.", idx + 1, emp.employee_id, q_reason)
                cropped, crop_lms, full_lms = self.embedder.detect_and_crop_face(img)
                if cropped is None or cropped.size == 0:
                    errors.append(f"Photo #{idx + 1}: Rejected - {q_reason}")
                    continue

            try:
                emb = self.embedder.extract_embedding(cropped, crop_lms, full_image=img, raw_landmarks=full_lms)

                # Duplicate enrollment check against existing gallery and current batch
                is_duplicate = False
                for ex_vec in existing_gallery + embeddings:
                    if float(np.dot(emb, ex_vec)) > 0.998:
                        is_duplicate = True
                        break

                if is_duplicate:
                    logger.info("Photo #%d for employee %s skipped as duplicate angle (sim > 0.998)", idx + 1, emp.employee_id)
                    continue

                embeddings.append(emb)
            except Exception as e:
                errors.append(f"Photo #{idx + 1}: {e}")

        if not embeddings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Face embedding enrollment failed: {'; '.join(errors) if errors else 'No new valid face angles found.'}",
            )

        if append:
            updated = self.repo.add_face_embeddings(
                emp,
                new_embeddings=embeddings,
                new_photo_url=photo_url,
                set_as_primary_profile=set_as_primary,
            )
            self.recognizer.register_employee(
                updated.employee_id,
                embedding=updated.get_embedding_array(),
                gallery_embeddings=updated.get_gallery_embeddings(),
            )
            return self._to_response(updated)

        primary_embedding = embeddings[0]
        gallery_embeddings = embeddings

        updated = self.repo.update_face_embedding(
            emp,
            primary_embedding,
            gallery_embeddings=gallery_embeddings,
            photo_url=photo_url,
        )
        self.recognizer.register_employee(
            updated.employee_id,
            embedding=primary_embedding,
            gallery_embeddings=gallery_embeddings,
        )
        return self._to_response(updated)

    def add_photos_from_bytes(
        self,
        employee_id: Any,
        images_bytes: List[bytes],
        set_as_primary: bool = False,
    ) -> EmployeeResponse:
        return self.enroll_faces_from_bytes(
            employee_id=employee_id,
            images_bytes=images_bytes,
            append=True,
            set_as_primary=set_as_primary,
        )

    def enroll_face_from_bytes(self, employee_id: Any, image_bytes: bytes) -> EmployeeResponse:
        return self.enroll_faces_from_bytes(employee_id=employee_id, images_bytes=[image_bytes])

    def record_punch(self, punch: AttendancePunchRequest) -> AttendancePunchResponse:
        emp = self.repo.get_by_employee_id(punch.employee_id)
        if not emp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Employee '{punch.employee_id}' not found.",
            )
        if not emp.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Employee '{punch.employee_id}' is deactivated.",
            )

        ts = punch.timestamp or datetime.now(timezone.utc)
        log, daily = self.attendance_repo.record_attendance_event(
            employee=emp,
            camera_id=punch.camera_id,
            event_type=punch.event_type,
            timestamp=ts,
            confidence=punch.confidence,
        )

        msg = "Attendance punch logged successfully." if log else "Duplicate punch suppressed within cooldown window."
        return AttendancePunchResponse(
            success=True,
            employee_id=emp.employee_id,
            name=emp.name,
            event_type=punch.event_type,
            recorded_at=ts,
            status=daily.status,
            first_check_in=daily.first_check_in,
            last_check_out=daily.last_check_out,
            total_work_hours=daily.total_work_hours,
            is_late=daily.is_late,
            message=msg,
        )

    def get_daily_report(
        self,
        report_date: date,
        department: Optional[str] = None,
        auto_mark_absent: bool = True,
    ) -> DailyAttendanceReportResponse:
        if auto_mark_absent:
            self.attendance_repo.mark_absences(report_date)

        rows = self.attendance_repo.get_daily_attendance(
            target_date=report_date,
            department=department,
        )

        items: List[DailyAttendanceItem] = []
        present_cnt = 0
        late_cnt = 0
        absent_cnt = 0
        half_day_cnt = 0

        for r in rows:
            emp = r.employee
            items.append(
                DailyAttendanceItem(
                    employee_id=emp.employee_id,
                    name=emp.name,
                    department=emp.department,
                    designation=emp.designation,
                    date=r.date,
                    first_check_in=r.first_check_in,
                    last_check_out=r.last_check_out,
                    total_work_hours=r.total_work_hours,
                    is_late=r.is_late,
                    late_minutes=r.late_minutes,
                    is_early_departure=r.is_early_departure,
                    early_departure_minutes=r.early_departure_minutes,
                    status=r.status,
                )
            )
            if r.status == "PRESENT":
                present_cnt += 1
            elif r.status == "LATE":
                late_cnt += 1
                present_cnt += 1
            elif r.status == "ABSENT":
                absent_cnt += 1
            elif r.status == "HALF_DAY":
                half_day_cnt += 1
                present_cnt += 1

        return DailyAttendanceReportResponse(
            date=report_date,
            total_employees=len(items),
            present_count=present_cnt,
            late_count=late_cnt,
            absent_count=absent_cnt,
            half_day_count=half_day_cnt,
            records=items,
        )

    def _save_profile_picture(self, employee_id: str, image_bytes: bytes, suffix: Optional[str] = None) -> str:
        upload_dir = os.path.join("static", "uploads", "employees")
        os.makedirs(upload_dir, exist_ok=True)
        ext = ".jpg"
        if image_bytes.startswith(b"\x89PNG"):
            ext = ".png"
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
            ext = ".webp"

        clean_id = str(employee_id).replace("/", "_").replace("\\", "_")
        filename = f"{clean_id}_{suffix}{ext}" if suffix else f"{clean_id}_profile{ext}"
        filepath = os.path.join(upload_dir, filename)
        try:
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            return f"/static/uploads/employees/{filename}"
        except Exception as e:
            logger.warning("Could not save profile picture file to disk: %s", str(e))
            b64 = base64.b64encode(image_bytes).decode("ascii")
            mime = "image/png" if ext == ".png" else "image/jpeg"
            return f"data:{mime};base64,{b64}"

    @staticmethod
    def _combine_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
        if not embeddings:
            raise ValueError("No face embeddings provided to combine.")
        if len(embeddings) == 1:
            return embeddings[0]

        mean_vec = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            return (mean_vec / norm).astype(np.float32)
        return mean_vec.astype(np.float32)

    def _extract_embedding_from_base64(self, b64_str: str) -> np.ndarray:
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            b64_str = b64_str.strip()
            missing_padding = len(b64_str) % 4
            if missing_padding:
                b64_str += "=" * (4 - missing_padding)
            raw_bytes = base64.b64decode(b64_str)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from base64 data.")
            return self.embedder.extract_face_embedding_from_image(img)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process face image: {e}",
            )

    @staticmethod
    def _to_response(emp: Employee) -> EmployeeResponse:
        gallery = emp.get_gallery_embeddings()
        photos_count = len(gallery) if gallery else (1 if emp.face_embedding is not None else 0)
        return EmployeeResponse(
            id=emp.id,
            employee_id=emp.employee_id,
            name=emp.name,
            department=emp.department,
            designation=emp.designation,
            assigned_camera_id=emp.assigned_camera_id,
            assigned_location=emp.assigned_location,
            shift_start_time=emp.shift_start_time,
            shift_end_time=emp.shift_end_time,
            late_grace_minutes=emp.late_grace_minutes,
            early_departure_grace_minutes=emp.early_departure_grace_minutes,
            photo_url=emp.photo_url,
            is_active=emp.is_active,
            has_face_registered=bool(emp.face_embedding is not None),
            face_photos_count=photos_count,
            created_at=emp.created_at,
            updated_at=emp.updated_at,
        )
