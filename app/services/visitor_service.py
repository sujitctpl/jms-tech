import base64
from datetime import datetime, timezone, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple
import uuid
import cv2
import numpy as np
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.db.models.person_identity import PersonIdentity
from app.repositories.employee_attendance_repo import EmployeeAttendanceRepository
from app.repositories.employee_repo import EmployeeRepository
from app.repositories.person_identity_repo import PersonIdentityRepository
from app.schemas.visitor import (
    VisitorDetectRequest,
    VisitorDetectResponse,
    VisitorListResponse,
    VisitorResponse,
    VisitorTagRequest,
    VisitorZoneUpdateRequest,
    WalkInCreateRequest,
)
from app.services.employee_service import global_employee_recognizer
from app.vision.face_embedder import FaceEmbedder

# Indian Standard Time (UTC+05:30)
IST = timezone(timedelta(hours=5, minutes=30))


def to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert any naive or aware datetime to Indian Standard Time (IST)."""
    if not dt:
        return None
    if dt.tzinfo is None:
        # DB naive timestamps represent UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


class VisitorService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = PersonIdentityRepository(db)
        self.emp_repo = EmployeeRepository(db)
        self.attendance_repo = EmployeeAttendanceRepository(db)
        self.embedder = FaceEmbedder()

    def _to_visitor_response(self, record: PersonIdentity) -> VisitorResponse:
        # Convert created_at and last_seen_at to Indian Standard Time (IST) 12-hour AM/PM format
        created_ist = to_ist(record.created_at)
        last_seen_ist = to_ist(record.last_seen_at)
        now_ist = datetime.now(IST)

        first_seen_str = (
            created_ist.strftime("%I:%M %p")
            if created_ist
            else now_ist.strftime("%I:%M %p")
        )
        last_seen_str = (
            last_seen_ist.strftime("%I:%M %p")
            if last_seen_ist
            else None
        )

        date_str = (
            created_ist.strftime("%Y-%m-%d")
            if created_ist
            else now_ist.strftime("%Y-%m-%d")
        )

        # Resolve departure / exit time
        exit_time_val = getattr(record, "exit_time", None)

        # Safety: If exit_time was prematurely set to arrival time and person was seen recently,
        # they are genuinely still browsing inside the showroom!
        if exit_time_val and last_seen_ist:
            is_active_recently = (now_ist - last_seen_ist).total_seconds() < 900
            if exit_time_val == first_seen_str and is_active_recently:
                exit_time_val = None

        if not exit_time_val and created_ist:
            # If the visit is from a previous calendar date (in IST), departure was when last seen
            if created_ist.date() < now_ist.date():
                exit_time_val = last_seen_str or first_seen_str
            elif last_seen_ist:
                # If visitor hasn't been seen in > 15 minutes today, mark departed
                if (now_ist - last_seen_ist).total_seconds() > 900:
                    exit_time_val = last_seen_str

        # Calculate duration in minutes (till exit if departed, or till now if active inside)
        duration = 1
        if created_ist:
            if exit_time_val and last_seen_ist:
                end_time = last_seen_ist
            else:
                end_time = now_ist
            delta_sec = max(0.0, (end_time - created_ist).total_seconds())
            duration = max(1, int(round(delta_sec / 60.0)))

        # Parse and resolve zone dwell breakdown
        zone_dwell_dict: Dict[str, str] = {}
        if getattr(record, "zone_dwell", None):
            try:
                if isinstance(record.zone_dwell, dict):
                    zone_dwell_dict = record.zone_dwell
                elif isinstance(record.zone_dwell, str):
                    zone_dwell_dict = json.loads(record.zone_dwell)
            except Exception:
                pass
        if not zone_dwell_dict:
            primary_z = record.zone_entered or "Main Showroom"
            zone_dwell_dict = {primary_z: f"{duration} mins"}

        return VisitorResponse(
            id=str(record.id),
            visitorId=record.person_id,
            avatarUrl=record.avatar_url,
            firstSeenTime=first_seen_str,
            lastSeenTime=last_seen_str,
            exitTime=exit_time_val,
            visitorType=record.visitor_type or "New",
            visitPurpose=record.visit_purpose,
            assignedSalesperson=record.assigned_salesperson,
            durationMinutes=duration,
            notes=record.notes or "",
            taggingStatus=record.tagging_status or ("Tagged" if record.visit_purpose else "Untagged"),
            date=date_str,
            genderEstimated=record.gender_estimated or "Unspecified",
            ageGroupEstimated=record.age_group_estimated or "26-35",
            zoneEntered=record.zone_entered or "Main Showroom",
            confidenceScore=0.96,
            cameraSource=record.camera_source or "Entrance Camera",
            zoneDwell=zone_dwell_dict,
        )

    def list_visitors(
        self,
        tagging_status: Optional[str] = None,
        visit_purpose: Optional[str] = None,
    ) -> VisitorListResponse:
        records = self.repo.list_visitors(
            tagging_status=tagging_status,
            visit_purpose=visit_purpose,
        )
        all_records = self.repo.list_visitors(tagging_status=None, visit_purpose=None)

        tagged = sum(1 for r in all_records if r.tagging_status == "Tagged" or r.visit_purpose)
        untagged = sum(1 for r in all_records if (r.tagging_status == "Untagged" or not r.tagging_status) and not r.visit_purpose)

        items = [self._to_visitor_response(r) for r in records]
        return VisitorListResponse(
            visitors=items,
            total=len(all_records),
            tagged_count=tagged,
            untagged_count=untagged,
        )

    def tag_visitor(self, person_id: str, payload: VisitorTagRequest) -> VisitorResponse:
        record = self.repo.update_visitor_tag(
            person_id=person_id,
            visit_purpose=payload.visit_purpose,
            assigned_salesperson=payload.assigned_salesperson,
            notes=payload.notes,
        )
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Visitor with ID '{person_id}' not found.",
            )
        return self._to_visitor_response(record)

    def update_visitor_zone(self, payload: VisitorZoneUpdateRequest) -> Optional[VisitorResponse]:
        """Updates real-time zone dwell durations, active zone, and departure time for a visitor."""
        record = None
        if payload.person_id:
            record = self.repo.get_by_person_id(payload.person_id)
        if not record and payload.tracker_id:
            record = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.notes.like(f"%Tracker #{payload.tracker_id}%"),
                    PersonIdentity.is_expired.is_(False),
                )
                .order_by(PersonIdentity.created_at.desc())
                .first()
            )
        if not record:
            return None

        now_utc = datetime.now(timezone.utc)
        record.last_seen_at = now_utc
        if payload.current_zone:
            record.zone_entered = payload.current_zone
        if payload.exit_time:
            record.exit_time = payload.exit_time
        if payload.zone_dwell:
            record.zone_dwell = json.dumps(payload.zone_dwell)

        if payload.notes_append:
            if not record.notes:
                record.notes = payload.notes_append
            elif payload.notes_append not in record.notes:
                record.notes = f"{record.notes} | {payload.notes_append}"

        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return self._to_visitor_response(record)

    def clear_all_visitors(self) -> dict:
        """Deletes all visitor records and cleans up stored visitor snapshots on disk."""
        count = self.db.query(PersonIdentity).delete()
        self.db.commit()
        vis_dir = os.path.join("static", "uploads", "visitors")
        deleted_files = 0
        if os.path.exists(vis_dir):
            for f in os.listdir(vis_dir):
                fp = os.path.join(vis_dir, f)
                if os.path.isfile(fp):
                    try:
                        os.remove(fp)
                        deleted_files += 1
                    except Exception:
                        pass
        return {
            "status": "SUCCESS",
            "deleted_count": count,
            "deleted_files": deleted_files,
            "message": "All visitor records and face snapshots cleared successfully.",
        }

    def create_walk_in(self, payload: WalkInCreateRequest) -> VisitorResponse:
        total_count = self.db.query(PersonIdentity).count()
        new_pid = f"VIS-2026-{1040 + total_count + 1}"

        record = self.repo.create_or_update(
            person_id=new_pid,
            embedding=None,
            avatar_url=None,
            camera_source="Reception Manual Walk-In",
            zone_entered=payload.zone_entered or "Main Showroom",
            notes=payload.notes or "Manual walk-in registered at reception",
            visit_purpose=payload.visit_purpose,
            assigned_salesperson=payload.assigned_salesperson,
        )
        return self._to_visitor_response(record)

    def process_detection(self, req: VisitorDetectRequest) -> VisitorDetectResponse:
        """
        Processes real-time face detection from a camera:
        1. Decodes face image crop & extracts 512-d ArcFace embedding.
        2. Checks against Employee Recognition Vector Store:
           - If staff: Suppresses visitor count, updates attendance, does NOT add to Visit Purpose.
        3. Checks against Visitor Identity Vector Store (pgvector / HNSW):
           - If matched (similarity >= 0.60): Re-identifies visitor across camera views (SAME ID retained).
           - If new: Generates unique visitor ID, saves face photo snapshot, creates 'Untagged' visitor.
        """
        face_img = None
        face_crop_bytes = None

        if req.face_image_base64:
            try:
                b64_str = req.face_image_base64
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                b64_str = b64_str.strip()
                missing_padding = len(b64_str) % 4
                if missing_padding:
                    b64_str += "=" * (4 - missing_padding)
                face_crop_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(face_crop_bytes, np.uint8)
                face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.warning("Failed to decode base64 face crop: %s", str(e))

        # Calibrated check: Accept faces with resolution >= 24x24 so typical CCTV distances are captured
        if face_img is not None:
            fh, fw = face_img.shape[:2]
            if fh < 24 or fw < 24:
                logger.info("Face rejected: resolution %dx%d is below minimum 24x24 required for clear identification.", fw, fh)
                return VisitorDetectResponse(
                    status="LOW_QUALITY_FACE",
                    is_employee=False,
                    is_new=False,
                    message=f"Face crop too small ({fw}x{fh}px). Waiting for closer frame.",
                )

        now_utc = datetime.now(timezone.utc)

        # ---------------------------------------------------------------------
        # Pre-check: If edge detector already identified an existing person_id or tracker
        # ---------------------------------------------------------------------
        existing_rec = None
        if req.person_id:
            existing_rec = self.repo.get_by_person_id(req.person_id)
        if not existing_rec and req.tracker_id:
            # Check if this active track was already logged in the last 60 seconds
            existing_rec = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.notes.like(f"%Tracker #{req.tracker_id}%"),
                    PersonIdentity.is_expired.is_(False),
                )
                .order_by(PersonIdentity.created_at.desc())
                .first()
            )

        if existing_rec:
            existing_rec.last_seen_at = now_utc
            if req.camera_name:
                existing_rec.camera_source = req.camera_name
            if req.zone:
                existing_rec.zone_entered = req.zone
            if getattr(req, "zone_dwell", None):
                existing_rec.zone_dwell = json.dumps(req.zone_dwell)
            if getattr(req, "exit_time", None):
                existing_rec.exit_time = req.exit_time
            if face_img is not None and face_img.size > 0:
                try:
                    os.makedirs("static/uploads/visitors", exist_ok=True)
                    avatar_img = cv2.resize(face_img, (320, 320), interpolation=cv2.INTER_LANCZOS4)
                    filename = f"{existing_rec.person_id}.jpg"
                    filepath = os.path.join("static", "uploads", "visitors", filename)
                    cv2.imwrite(filepath, avatar_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    existing_rec.avatar_url = f"/static/uploads/visitors/{filename}"
                except Exception as e:
                    logger.error("Failed to update visitor face snapshot: %s", str(e))
            if getattr(req, "appearance", None):
                if not existing_rec.notes or "AI Face Detection" in existing_rec.notes:
                    existing_rec.notes = f"Appearance: {req.appearance} | Tracker #{req.tracker_id or 'auto'}"
            self.db.add(existing_rec)
            self.db.commit()
            self.db.refresh(existing_rec)
            return VisitorDetectResponse(
                status="VISITOR_REIDENTIFIED",
                is_employee=False,
                person_id=existing_rec.person_id,
                is_new=False,
                visitor=self._to_visitor_response(existing_rec),
                message=f"Visitor {existing_rec.person_id} updated with enhanced photo.",
            )

        # Resolve 512-d ArcFace embedding
        embedding = None
        if req.face_embedding and len(req.face_embedding) == 512:
            embedding = np.array(req.face_embedding, dtype=np.float32)
        elif face_img is not None and face_img.size > 0:
            try:
                embedding = self.embedder.extract_embedding(face_img)
            except Exception as e:
                logger.warning("Failed to extract embedding from face crop: %s", str(e))

        # ---------------------------------------------------------------------
        # 1. Check against Showroom Employee Roster
        # ---------------------------------------------------------------------
        if embedding is not None:
            active_employees = self.emp_repo.get_all_active_embeddings()
            best_emp_id = None
            best_emp_sim = 0.0

            for emp_id, emp_emb in active_employees:
                sim = FaceEmbedder.compute_similarity(embedding, emp_emb)
                if sim > best_emp_sim:
                    best_emp_sim = sim
                    best_emp_id = emp_id

            threshold = float(getattr(settings, "FACE_SIMILARITY_THRESHOLD", 0.62))
            if best_emp_id and best_emp_sim >= threshold:
                emp_obj = self.emp_repo.get_by_employee_id(best_emp_id)
                emp_name = emp_obj.name if emp_obj else best_emp_id
                logger.info(
                    "Employee recognized: %s (%s) with similarity %.3f. Suppressing from Visit Purpose.",
                    emp_name,
                    best_emp_id,
                    best_emp_sim,
                )
                try:
                    self.attendance_repo.record_attendance_event(
                        employee_id=best_emp_id,
                        camera_id=uuid.UUID(req.camera_id) if req.camera_id else None,
                        event_type="CHECK_IN",
                        confidence=best_emp_sim,
                    )
                except Exception as ex:
                    logger.debug("Could not record employee punch: %s", str(ex))

                return VisitorDetectResponse(
                    status="EMPLOYEE_DETECTED",
                    is_employee=True,
                    employee_id=best_emp_id,
                    employee_name=emp_name,
                    is_new=False,
                    message=f"Staff member {emp_name} recognized. Visitor registration suppressed.",
                )

        # ---------------------------------------------------------------------
        # 2. Check Visitor Re-Identification (Across Cameras & Time)
        # ---------------------------------------------------------------------
        matched_person_id = None
        matched_similarity = 0.0

        if embedding is not None:
            # Calibrated operating threshold for ArcFace CCTV feature vectors
            calibrated_threshold = float(getattr(settings, "FACE_SIMILARITY_THRESHOLD", 0.32))
            matched_person_id, matched_similarity = self.repo.find_best_match(
                embedding=embedding,
                similarity_threshold=calibrated_threshold,
            )

        # Extended recency safety net: Match visitors active in the showroom today (within 2 hours)
        if not matched_person_id and embedding is not None:
            recent_identities = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.is_expired.is_(False),
                    PersonIdentity.face_embedding.isnot(None),
                )
                .order_by(PersonIdentity.last_seen_at.desc())
                .limit(20)
                .all()
            )
            norm_e = np.linalg.norm(embedding)
            norm_emb = (embedding / norm_e).astype(np.float32) if norm_e > 0 else embedding
            for rec in recent_identities:
                rec_emb = rec.get_embedding_array()
                if rec_emb is not None:
                    norm_r = np.linalg.norm(rec_emb)
                    rec_norm = (rec_emb / norm_r).astype(np.float32) if norm_r > 0 else rec_emb
                    sim = float(np.dot(norm_emb, rec_norm))
                    time_diff = (now_utc - rec.last_seen_at).total_seconds()
                    # 2-hour active window with relaxed cross-camera CCTV similarity threshold
                    if time_diff < 7200.0 and sim >= 0.28:
                        matched_person_id = rec.person_id
                        matched_similarity = sim
                        break

        # Interior Camera Safety Net: Interior cameras (Floor, Display Area, Service Desk)
        # should NEVER mint new store walk-ins if an active customer is already in the showroom
        is_cam_entrance = getattr(req, "is_entrance", True)
        if not matched_person_id and not is_cam_entrance:
            active_visitor = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.is_expired.is_(False),
                    PersonIdentity.exit_time.is_(None),
                )
                .order_by(PersonIdentity.last_seen_at.desc())
                .first()
            )
            if active_visitor:
                time_since_seen = (now_utc - active_visitor.last_seen_at).total_seconds()
                if time_since_seen < 3600.0:
                    matched_person_id = active_visitor.person_id
                    matched_similarity = 0.85
                    logger.info(
                        "Interior camera %s associated detection with active showroom visitor %s (suppressing duplicate).",
                        req.camera_name,
                        matched_person_id,
                    )

        # Case A: Existing visitor recognized (disappeared & reappeared on camera 1 or camera 2)
        if matched_person_id:
            record = self.repo.get_by_person_id(matched_person_id)
            if record:
                record.last_seen_at = now_utc
                if req.camera_name:
                    record.camera_source = req.camera_name
                if req.zone:
                    record.zone_entered = req.zone
                if face_img is not None and face_img.size > 0:
                    try:
                        os.makedirs("static/uploads/visitors", exist_ok=True)
                        filename = f"{record.person_id}.jpg"
                        filepath = os.path.join("static", "uploads", "visitors", filename)
                        cv2.imwrite(filepath, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        record.avatar_url = f"/static/uploads/visitors/{filename}"
                    except Exception as e:
                        logger.error("Failed to update visitor snapshot: %s", str(e))
                self.db.add(record)
                self.db.commit()
                self.db.refresh(record)

                logger.info(
                    "Visitor RE-IDENTIFIED: %s across cameras (similarity %.3f). Maintained same ID.",
                    matched_person_id,
                    matched_similarity,
                )
                return VisitorDetectResponse(
                    status="VISITOR_REIDENTIFIED",
                    is_employee=False,
                    person_id=matched_person_id,
                    is_new=False,
                    visitor=self._to_visitor_response(record),
                    message=f"Visitor {matched_person_id} re-identified across camera feeds. ID preserved.",
                )

        # ---------------------------------------------------------------------
        # 3. New Customer Arrival -> Mint ID, Save Crisp Snapshot, Add to Visit Purpose
        # ---------------------------------------------------------------------
        total_count = self.db.query(PersonIdentity).count()
        new_person_id = f"VIS-2026-{1040 + total_count + 1}"

        # Save full-body portrait snapshot to disk with high quality (quality 95)
        avatar_url = None
        if face_img is not None and face_img.size > 0:
            try:
                os.makedirs("static/uploads/visitors", exist_ok=True)
                filename = f"{new_person_id}.jpg"
                filepath = os.path.join("static", "uploads", "visitors", filename)
                cv2.imwrite(filepath, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                avatar_url = f"/static/uploads/visitors/{filename}"
            except Exception as e:
                logger.error("Failed to write visitor body snapshot: %s", str(e))

        # If no face image was written, leave avatar_url as None so frontend displays a clean placeholder
        if not avatar_url:
            avatar_url = None

        note_text = f"AI Face Detection walk-in (Tracker #{req.tracker_id or 'auto'})"
        if getattr(req, "appearance", None):
            note_text = f"Appearance: {req.appearance} | Tracker #{req.tracker_id or 'auto'}"

        new_record = self.repo.create_or_update(
            person_id=new_person_id,
            embedding=embedding,
            timestamp=now_utc,
            avatar_url=avatar_url,
            camera_source=req.camera_name or "Entrance Camera",
            zone_entered=req.zone or "Main Showroom",
            notes=note_text,
        )

        if getattr(req, "zone_dwell", None):
            new_record.zone_dwell = json.dumps(req.zone_dwell)
        if getattr(req, "exit_time", None):
            new_record.exit_time = req.exit_time
        if getattr(req, "zone_dwell", None) or getattr(req, "exit_time", None):
            self.db.add(new_record)
            self.db.commit()
            self.db.refresh(new_record)

        logger.info(
            "New Customer Registered: %s with face snapshot saved to %s.",
            new_person_id,
            avatar_url,
        )

        return VisitorDetectResponse(
            status="VISITOR_CREATED",
            is_employee=False,
            person_id=new_person_id,
            is_new=True,
            visitor=self._to_visitor_response(new_record),
            message=f"New visitor {new_person_id} registered and added to Visit Purpose queue.",
        )

