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
from app.db.models.footfall_event import FootfallEvent
from app.db.models.visitor_track import VisitorTrack
from app.db.models.processing_run import ProcessingRun
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

        if not exit_time_val and created_ist:
            # If the visit is from a previous calendar date (in IST), departure was when last seen
            if created_ist.date() < now_ist.date():
                exit_time_val = last_seen_str or first_seen_str
            elif last_seen_ist:
                # If visitor hasn't been seen in > 15 minutes today, mark departed
                if (now_ist - last_seen_ist).total_seconds() > 900:
                    exit_time_val = last_seen_str

        # Calculate duration in minutes (till exit if departed, or till now if active inside)
        duration = 0
        if created_ist:
            end_time = None
            if exit_time_val:
                try:
                    # Parse exit time string into today's datetime in IST
                    t_struct = datetime.strptime(exit_time_val.strip(), "%I:%M %p").time()
                    end_time = datetime.combine(created_ist.date(), t_struct, tzinfo=IST)
                    if end_time < created_ist:
                        # If exit was within the same minute as entry (e.g. 5 seconds later)
                        if exit_time_val == first_seen_str:
                            end_time = created_ist
                        elif last_seen_ist and last_seen_ist >= created_ist:
                            end_time = last_seen_ist
                except Exception:
                    end_time = last_seen_ist or now_ist
            elif last_seen_ist:
                # Active inside if seen within 15 mins, otherwise departed when last seen
                is_active = (now_ist - last_seen_ist).total_seconds() <= 900
                end_time = now_ist if is_active else last_seen_ist
            else:
                end_time = now_ist

            delta_sec = max(0.0, (end_time - created_ist).total_seconds())
            duration = int(round(delta_sec / 60.0))

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

        # Dynamic cache-busting timestamp on avatarUrl so browsers NEVER serve stale cached images
        avatar_url_res = record.avatar_url
        if avatar_url_res:
            mtime = 0
            try:
                clean_path = avatar_url_res.lstrip("/").split("?")[0]
                if os.path.exists(clean_path):
                    mtime = int(os.path.getmtime(clean_path))
            except Exception:
                pass
            if not mtime:
                ts_dt = record.last_seen_at or record.created_at
                mtime = int(ts_dt.timestamp()) if ts_dt else int(datetime.now(timezone.utc).timestamp())
            base_url = avatar_url_res.split("?")[0]
            avatar_url_res = f"{base_url}?t={mtime}"

        return VisitorResponse(
            id=str(record.id),
            visitorId=record.person_id,
            avatarUrl=avatar_url_res,
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
        now_utc = datetime.now(timezone.utc)
        if not record and payload.tracker_id:
            cand = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.notes.like(f"%Tracker #{payload.tracker_id}%"),
                    PersonIdentity.is_expired.is_(False),
                    PersonIdentity.exit_time.is_(None),
                )
                .order_by(PersonIdentity.last_seen_at.desc())
                .first()
            )
            if cand and cand.last_seen_at:
                time_diff = (now_utc - cand.last_seen_at).total_seconds()
                if time_diff < 45.0:
                    record = cand

        # Fallback for store exit: if tracker ID was reset on 180-deg turnaround (e.g. #2 became #6)
        if not record and payload.exit_time:
            active_cand = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.is_expired.is_(False),
                    PersonIdentity.exit_time.is_(None),
                )
                .order_by(PersonIdentity.last_seen_at.desc())
                .first()
            )
            if active_cand and active_cand.last_seen_at:
                # Any active customer who entered today (within 12 hours)
                time_diff = (now_utc - active_cand.last_seen_at).total_seconds()
                if time_diff < 43200.0:
                    record = active_cand
                    logger.info(
                        "Resolved entrance exit event to active showroom visitor %s (Tracker swapped on turnaround).",
                        record.person_id,
                    )

        if not record:
            return None
        record.last_seen_at = now_utc
        if payload.tracker_id and f"Tracker #{payload.tracker_id}" not in (record.notes or ""):
            record.notes = f"{record.notes} | Tracker #{payload.tracker_id}" if record.notes else f"Tracker #{payload.tracker_id}"
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
        """Deletes all visitor records, footfall events, tracks, and cleans up stored visitor snapshots on disk."""
        footfall_count = self.db.query(FootfallEvent).delete()
        track_count = self.db.query(VisitorTrack).delete()
        run_count = self.db.query(ProcessingRun).delete()
        count = self.db.query(PersonIdentity).delete()
        self.db.commit()

        # Clear edge telemetry in-memory cache
        try:
            from app.services.camera_service import _edge_telemetry
            _edge_telemetry.clear()
        except Exception:
            pass

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
            "deleted_tracks": track_count,
            "deleted_footfall": footfall_count,
            "deleted_files": deleted_files,
            "message": "All visitor records, footfall events, tracks, and face snapshots cleared successfully.",
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
            cand = self.repo.get_by_person_id(req.person_id)
            if cand and cand.exit_time is None and not cand.is_expired:
                existing_rec = cand
        if not existing_rec and req.tracker_id:
            # Only match the exact same track if it was active within the last 30 seconds and hasn't departed
            cand = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.notes.like(f"%Tracker #{req.tracker_id}%"),
                    PersonIdentity.is_expired.is_(False),
                    PersonIdentity.exit_time.is_(None),
                )
                .order_by(PersonIdentity.last_seen_at.desc())
                .first()
            )
            if cand and cand.last_seen_at:
                time_diff = (now_utc - cand.last_seen_at).total_seconds()
                if time_diff < 30.0:
                    existing_rec = cand

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
                    now_epoch = int(datetime.now(timezone.utc).timestamp())
                    filename = f"{existing_rec.person_id}_{now_epoch}.jpg"
                    if existing_rec.avatar_url:
                        old_f = os.path.join(".", existing_rec.avatar_url.lstrip("/").split("?")[0])
                        if os.path.isfile(old_f):
                            try:
                                os.remove(old_f)
                            except Exception:
                                pass
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
        is_active_session = False
        is_returning_customer = False
        past_record = None

        now_ist = datetime.now(IST)

        if embedding is not None:
            visitor_reid_thresh = float(getattr(settings, "VISITOR_REID_THRESHOLD", 0.55))

            # Step 2A: Check for an ongoing active visit inside the showroom TODAY
            active_pid, active_sim = self.repo.find_best_match(
                embedding=embedding,
                similarity_threshold=visitor_reid_thresh,
                only_active=True,
            )
            if active_pid:
                cand = self.repo.get_by_person_id(active_pid)
                if cand and not cand.is_expired and cand.exit_time is None:
                    cand_ist = to_ist(cand.created_at)
                    if cand_ist and cand_ist.date() == now_ist.date():
                        matched_person_id = active_pid
                        matched_similarity = active_sim
                        is_active_session = True
                    else:
                        # Stale visit from a previous calendar day: close it out
                        c_last_ist = to_ist(cand.last_seen_at) or cand_ist
                        cand.exit_time = c_last_ist.strftime("%I:%M %p") if c_last_ist else "Exited"
                        self.db.add(cand)
                        self.db.commit()

            # Step 2B: If not currently active today, check 14-day history for returning customer
            if not is_active_session:
                history_pid, history_sim = self.repo.find_best_match(
                    embedding=embedding,
                    similarity_threshold=visitor_reid_thresh,
                    only_active=False,
                )
                if history_pid:
                    history_cand = self.repo.get_by_person_id(history_pid)
                    if history_cand and not history_cand.is_expired:
                        hist_ist = to_ist(history_cand.created_at)
                        # A visit from a previous day OR with an exit_time marked indicates a returning customer
                        if (hist_ist and hist_ist.date() < now_ist.date()) or history_cand.exit_time is not None:
                            is_returning_customer = True
                            past_record = history_cand
                            if not history_cand.exit_time:
                                h_last_ist = to_ist(history_cand.last_seen_at) or hist_ist
                                history_cand.exit_time = h_last_ist.strftime("%I:%M %p") if h_last_ist else "Exited"
                                self.db.add(history_cand)
                                self.db.commit()
                        else:
                            matched_person_id = history_pid
                            matched_similarity = history_sim
                            is_active_session = True

        # Extended recency safety net: Match visitors active in the showroom today (within 2 hours)
        # ONLY if biometric match is genuinely high (sim >= 0.55)
        if not is_active_session and not is_returning_customer and embedding is not None:
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
                    if time_diff < 7200.0 and sim >= 0.55:
                        rec_ist = to_ist(rec.created_at)
                        if rec_ist and rec_ist.date() == now_ist.date() and rec.exit_time is None:
                            matched_person_id = rec.person_id
                            matched_similarity = sim
                            is_active_session = True
                        elif (rec_ist and rec_ist.date() < now_ist.date()) or rec.exit_time is not None:
                            is_returning_customer = True
                            past_record = rec
                        break

        # Interior Camera Safety Net: Interior cameras (Floor, Display Area, Service Desk)
        # should NEVER mint new store walk-ins if an active customer is already in the showroom
        is_cam_entrance = getattr(req, "is_entrance", True)
        if not is_active_session and not is_returning_customer and not is_cam_entrance:
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
                act_ist = to_ist(active_visitor.created_at)
                time_since_seen = (now_utc - active_visitor.last_seen_at).total_seconds()
                if act_ist and act_ist.date() == now_ist.date() and time_since_seen < 3600.0:
                    matched_person_id = active_visitor.person_id
                    matched_similarity = 0.85
                    is_active_session = True
                    logger.info(
                        "Interior camera %s associated detection with active showroom visitor %s (suppressing duplicate).",
                        req.camera_name,
                        matched_person_id,
                    )

        # Case A: Existing visitor recognized active in showroom (disappeared & reappeared on camera 1 or camera 2)
        if is_active_session and matched_person_id:
            record = self.repo.get_by_person_id(matched_person_id)
            if record and not record.is_expired:
                record.last_seen_at = now_utc
                if req.camera_name:
                    record.camera_source = req.camera_name
                if req.zone:
                    record.zone_entered = req.zone
                if req.tracker_id and f"Tracker #{req.tracker_id}" not in (record.notes or ""):
                    record.notes = f"{record.notes} | Tracker #{req.tracker_id}" if record.notes else f"Tracker #{req.tracker_id}"
                if face_img is not None and face_img.size > 0:
                    try:
                        os.makedirs("static/uploads/visitors", exist_ok=True)
                        now_epoch = int(datetime.now(timezone.utc).timestamp())
                        filename = f"{record.person_id}_{now_epoch}.jpg"
                        if record.avatar_url:
                            old_f = os.path.join(".", record.avatar_url.lstrip("/").split("?")[0])
                            if os.path.isfile(old_f):
                                try:
                                    os.remove(old_f)
                                except Exception:
                                    pass
                        filepath = os.path.join("static", "uploads", "visitors", filename)
                        cv2.imwrite(filepath, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        record.avatar_url = f"/static/uploads/visitors/{filename}"
                    except Exception as e:
                        logger.error("Failed to update visitor snapshot: %s", str(e))
                self.db.add(record)
                self.db.commit()
                self.db.refresh(record)

                logger.info(
                    "Visitor RE-IDENTIFIED (Active): %s across cameras (similarity %.3f). Maintained active visit.",
                    matched_person_id,
                    matched_similarity,
                )
                return VisitorDetectResponse(
                    status="VISITOR_REIDENTIFIED",
                    is_employee=False,
                    person_id=matched_person_id,
                    is_new=False,
                    visitor=self._to_visitor_response(record),
                    message=f"Visitor {matched_person_id} active in showroom. Presence updated.",
                )

        # ---------------------------------------------------------------------
        # 3. New or Returning Customer Arrival -> Mint ID, Save Crisp Snapshot, Add to Visit Purpose
        # ---------------------------------------------------------------------
        total_count = self.db.query(PersonIdentity).count()
        new_person_id = f"VIS-2026-{1040 + total_count + 1}"

        # Save full-body portrait snapshot to disk with high quality (quality 95)
        avatar_url = None
        if face_img is not None and face_img.size > 0:
            try:
                os.makedirs("static/uploads/visitors", exist_ok=True)
                now_epoch = int(datetime.now(timezone.utc).timestamp())
                filename = f"{new_person_id}_{now_epoch}.jpg"
                filepath = os.path.join("static", "uploads", "visitors", filename)
                cv2.imwrite(filepath, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                avatar_url = f"/static/uploads/visitors/{filename}"
            except Exception as e:
                logger.error("Failed to write visitor body snapshot: %s", str(e))

        # If no face image was written, leave avatar_url as None so frontend displays a clean placeholder
        if not avatar_url:
            avatar_url = None

        note_text = f"AI Face Detection walk-in (Tracker #{req.tracker_id or 'auto'})"
        if is_returning_customer and past_record:
            note_text = f"Returning customer (previously {past_record.person_id}) | Tracker #{req.tracker_id or 'auto'}"
        elif getattr(req, "appearance", None):
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
        new_record.visitor_type = "Returning" if is_returning_customer else "New"
        new_record.exit_time = None

        if getattr(req, "zone_dwell", None):
            new_record.zone_dwell = json.dumps(req.zone_dwell)
        if getattr(req, "exit_time", None):
            new_record.exit_time = req.exit_time
        if getattr(req, "zone_dwell", None) or getattr(req, "exit_time", None):
            self.db.add(new_record)
            self.db.commit()
            self.db.refresh(new_record)

        if is_returning_customer:
            logger.info(
                "Returning Customer Registered: %s (previously %s) with face snapshot saved to %s.",
                new_person_id,
                past_record.person_id if past_record else "known",
                avatar_url,
            )
            return VisitorDetectResponse(
                status="VISITOR_CREATED",
                is_employee=False,
                person_id=new_person_id,
                is_new=False,
                visitor=self._to_visitor_response(new_record),
                message=f"Returning visitor {new_person_id} registered and added to Visit Purpose queue.",
            )
        else:
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

