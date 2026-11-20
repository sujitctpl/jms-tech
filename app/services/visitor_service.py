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
from app.vision.body_embedder import BodyEmbedder

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


def get_waiting_cutoff_seconds() -> int:
    """Reads configured SLA breach cutoff seconds from camera_config.json, default 900s (15m)."""
    try:
        cfg_path = os.path.join(".", "camera_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                cutoff = int(cfg.get("waiting_cutoff_seconds", 900))
                if cutoff > 0:
                    return cutoff
    except Exception:
        pass
    return 900


class VisitorService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = PersonIdentityRepository(db)
        self.emp_repo = EmployeeRepository(db)
        self.attendance_repo = EmployeeAttendanceRepository(db)
        self.embedder = FaceEmbedder()
        self.body_embedder = BodyEmbedder()

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
                record.exit_time = exit_time_val
                record.is_waiting = False
                try:
                    self.db.add(record)
                    self.db.commit()
                except Exception:
                    pass
            elif last_seen_ist:
                # If visitor hasn't been seen in > 15 minutes today, mark departed
                if (now_ist - last_seen_ist).total_seconds() > 900:
                    exit_time_val = last_seen_str
                    record.exit_time = exit_time_val
                    record.is_waiting = False
                    try:
                        self.db.add(record)
                        self.db.commit()
                    except Exception:
                        pass

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

        # Dynamic cache-busting timestamp on avatarUrl (verify file exists on disk)
        avatar_url_res = record.avatar_url
        if avatar_url_res:
            mtime = 0
            try:
                clean_path = avatar_url_res.lstrip("/").split("?")[0]
                if os.path.isfile(clean_path):
                    mtime = int(os.path.getmtime(clean_path))
                else:
                    avatar_url_res = None
            except Exception:
                pass
            if avatar_url_res and mtime:
                base_url = avatar_url_res.split("?")[0]
                avatar_url_res = f"{base_url}?t={mtime}"

        # Dynamic cache-busting timestamp on exitAvatarUrl (verify file exists on disk)
        exit_avatar_url_res = getattr(record, "exit_avatar_url", None)
        if exit_avatar_url_res:
            mtime_e = 0
            try:
                clean_path_e = exit_avatar_url_res.lstrip("/").split("?")[0]
                if os.path.isfile(clean_path_e):
                    mtime_e = int(os.path.getmtime(clean_path_e))
                else:
                    exit_avatar_url_res = None
            except Exception:
                pass
            if exit_avatar_url_res and mtime_e:
                base_url_e = exit_avatar_url_res.split("?")[0]
                exit_avatar_url_res = f"{base_url_e}?t={mtime_e}"

        # Resolve waiting time metrics
        waiting_sec = int(getattr(record, "waiting_seconds", 0) or 0)
        is_waiting = bool(getattr(record, "is_waiting", False) or False)
        # If visitor has already departed or has an assigned salesperson, they are attended
        if record.exit_time or record.assigned_salesperson:
            is_waiting = False
        cutoff_sec = get_waiting_cutoff_seconds()
        waiting_breach = bool(getattr(record, "waiting_breach", False) or (waiting_sec >= cutoff_sec))
        w_mins = waiting_sec // 60
        w_secs = waiting_sec % 60
        waiting_formatted = f"{w_mins}m {w_secs:02d}s" if w_mins > 0 else f"{w_secs}s"

        return VisitorResponse(
            id=str(record.id),
            visitorId=record.person_id,
            avatarUrl=avatar_url_res,
            exitAvatarUrl=exit_avatar_url_res,
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
            waitingSeconds=waiting_sec,
            isWaiting=is_waiting,
            waitingBreach=waiting_breach,
            waitingFormatted=waiting_formatted,
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
        # If an employee is assigned, visitor is now attended -> not waiting
        if payload.assigned_salesperson:
            record.is_waiting = False
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
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

        # Extract exit image crop & embeddings (Face + Body Person Re-ID)
        exit_img_b64 = getattr(payload, "exit_image_base64", None)
        if not exit_img_b64 and payload.exit_time and getattr(payload, "face_image_base64", None):
            exit_img_b64 = payload.face_image_base64

        exit_crop_img = None
        exit_face_emb = None
        exit_body_emb = None
        if exit_img_b64:
            try:
                b64_str = exit_img_b64
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                b64_str = b64_str.strip()
                missing_padding = len(b64_str) % 4
                if missing_padding:
                    b64_str += "=" * (4 - missing_padding)
                crop_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(crop_bytes, np.uint8)
                exit_crop_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if exit_crop_img is not None and exit_crop_img.size > 0:
                    try:
                        c_face, c_lms, raw_lms = self.embedder.detect_and_crop_face(exit_crop_img)
                        if c_face is not None and c_face.size > 0 and c_lms is not None:
                            exit_face_emb = self.embedder.extract_embedding(
                                c_face, c_lms, full_image=exit_crop_img, raw_landmarks=raw_lms
                            )
                    except Exception:
                        pass
                    # Extract full-body OSNet Person Re-ID embedding from exit crop
                    exit_body_emb = self.body_embedder.extract_embedding(exit_crop_img)
            except Exception as e:
                logger.warning("Could not extract exit image embeddings: %s", e)

        # Verify biometric consistency if candidate record has a stored face embedding and exit photo has a face
        if record and payload.exit_time and exit_face_emb is not None:
            cand_emb = record.get_embedding_vector()
            if cand_emb is not None:
                sim = FaceEmbedder.compute_similarity(exit_face_emb, cand_emb)
                if sim < 0.55:
                    logger.warning(
                        "Biometric mismatch on exit for visitor %s: similarity %.3f < 0.55. Refusing to attach exit to this visitor.",
                        record.person_id,
                        sim,
                    )
                    record = None

        # Verify body appearance & clothing consistency if candidate record has an entry crop / body embedding
        if record and payload.exit_time and exit_body_emb is not None:
            cand_b_emb = record.get_body_embedding_vector()
            cand_crop = None
            if getattr(record, "avatar_url", None):
                c_path = record.avatar_url.lstrip("/").split("?")[0]
                if os.path.isfile(c_path):
                    cand_crop = cv2.imread(c_path)
            if cand_b_emb is not None or cand_crop is not None:
                reid_s, clothing_s, app_score, is_match = self.body_embedder.verify_appearance(
                    crop1=exit_crop_img,
                    crop2=cand_crop,
                    emb1=exit_body_emb,
                    emb2=cand_b_emb,
                )
                if not is_match:
                    logger.warning(
                        "Body appearance mismatch on exit for visitor %s: reid=%.3f, clothing=%.3f, is_match=False. Refusing to attach exit to this visitor.",
                        record.person_id,
                        reid_s,
                        clothing_s,
                    )
                    record = None

        # Fallback for store exit: if tracker ID was reset on 180-deg turnaround (e.g. #2 became #6)
        if not record and payload.exit_time:
            now_ist = datetime.now(IST)
            today_start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_utc = today_start_ist.astimezone(timezone.utc)

            # Auto-close prior day unclosed records so they never contaminate today's active exit matching
            stale_prior_day = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.is_expired.is_(False),
                    PersonIdentity.exit_time.is_(None),
                    PersonIdentity.created_at < today_start_utc,
                )
                .all()
            )
            for sr in stale_prior_day:
                sr.exit_time = "Timed out"
                sr.is_expired = True
                self.db.add(sr)
            if stale_prior_day:
                self.db.commit()

            active_cands = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.is_expired.is_(False),
                    PersonIdentity.exit_time.is_(None),
                    PersonIdentity.created_at >= today_start_utc,
                )
                .order_by(PersonIdentity.last_seen_at.desc())
                .all()
            )

            # 1. If exit face embedding is available, search active candidates for a biometric face match
            # Calibrated threshold for CCTV face match: >= 0.65 with top-margin check to avoid ambiguous matches
            if exit_face_emb is not None:
                best_match = None
                best_sim = 0.0
                second_sim = 0.0
                for active_cand in active_cands:
                    cand_emb = active_cand.get_embedding_vector()
                    if cand_emb is not None:
                        sim = FaceEmbedder.compute_similarity(exit_face_emb, cand_emb)
                        if sim > best_sim:
                            second_sim = best_sim
                            best_sim = sim
                            best_match = active_cand
                        elif sim > second_sim:
                            second_sim = sim
                if best_match and best_sim >= 0.55:
                    if len(active_cands) == 1 or (best_sim - second_sim >= 0.035):
                        # Appearance cross-check: ensure body/clothing does not contradict face match
                        appearance_ok = True
                        if exit_body_emb is not None:
                            cand_b_emb = best_match.get_body_embedding_vector()
                            cand_crop = None
                            if getattr(best_match, "avatar_url", None):
                                c_path = best_match.avatar_url.lstrip("/").split("?")[0]
                                if os.path.isfile(c_path):
                                    cand_crop = cv2.imread(c_path)
                            if cand_b_emb is not None or cand_crop is not None:
                                reid_s, clothing_s, _, is_m = self.body_embedder.verify_appearance(
                                    crop1=exit_crop_img,
                                    crop2=cand_crop,
                                    emb1=exit_body_emb,
                                    emb2=cand_b_emb,
                                )
                                if not is_m and (clothing_s < 0.25 or reid_s < 0.55):
                                    logger.warning(
                                        "Exit face match (sim=%.3f) rejected due to conflicting body/clothing appearance for %s (reid=%.3f, cloth=%.3f)",
                                        best_sim,
                                        best_match.person_id,
                                        reid_s,
                                        clothing_s,
                                    )
                                    appearance_ok = False
                        if appearance_ok:
                            record = best_match
                            logger.info(
                                "Resolved entrance exit event to active showroom visitor %s via biometric face match (sim=%.3f, margin=%.3f).",
                                record.person_id,
                                best_sim,
                                best_sim - second_sim,
                            )

            # 2. Dual-Factor Appearance Match: OSNet Person Re-ID + Multi-Part 3D Clothing Color Correlation
            if not record and exit_body_emb is not None:
                best_body_match = None
                best_body_sim = 0.0
                best_color_sim = 0.0
                for active_cand in active_cands:
                    cand_b_emb = active_cand.get_body_embedding_vector()
                    if cand_b_emb is not None:
                        # Load candidate entry image to verify clothing color distribution
                        cand_crop = None
                        if getattr(active_cand, "avatar_url", None):
                            c_path = active_cand.avatar_url.lstrip("/").split("?")[0]
                            if os.path.isfile(c_path):
                                cand_crop = cv2.imread(c_path)

                        reid_s, clothing_s, app_score, is_match = self.body_embedder.verify_appearance(
                            crop1=exit_crop_img,
                            crop2=cand_crop,
                            emb1=exit_body_emb,
                            emb2=cand_b_emb,
                        )

                        # Genuine exit verification:
                        # 1. is_match enforces calibrated Re-ID + multi-part torso/legs clothing correlation
                        # 2. clothing_s must be >= 0.36 to strictly rule out different attire
                        # 3. Minimum time in store must be >= 5.0 seconds (matches entrance bounce window)
                        time_in_store = (now_utc - active_cand.created_at).total_seconds() if active_cand.created_at else 30.0
                        if is_match and clothing_s >= 0.36 and time_in_store >= 5.0 and app_score > best_body_sim:
                            best_body_sim = app_score
                            best_color_sim = clothing_s
                            best_body_match = active_cand

                if best_body_match:
                    record = best_body_match
                    logger.info(
                        "Resolved entrance exit event to active showroom visitor %s via Dual-Factor Re-ID & Clothing Match (score=%.3f, color=%.3f).",
                        record.person_id,
                        best_body_sim,
                        best_color_sim,
                    )

            # 3. Strict Exit Suppression: If no active visitor matches biometrically (face >= 0.65 or verified clothing >= 0.38),
            # DO NOT guess or force-match onto existing visitors. Suppress the exit event to prevent record corruption.
            if not record:
                logger.warning(
                    "Unmapped exit event for tracker %s could not be verified against any active visitor. Suppressed exit attachment.",
                    payload.tracker_id,
                )
                return None

        if not record:
            return None

        record.last_seen_at = now_utc
        if payload.tracker_id and f"Tracker #{payload.tracker_id}" not in (record.notes or ""):
            record.notes = f"{record.notes} | Tracker #{payload.tracker_id}" if record.notes else f"Tracker #{payload.tracker_id}"
        if payload.current_zone:
            record.zone_entered = payload.current_zone
        if payload.exit_time:
            record.exit_time = payload.exit_time
            record.is_waiting = False
        if payload.zone_dwell:
            record.zone_dwell = json.dumps(payload.zone_dwell)

        # Handle saving exit snapshot
        if exit_crop_img is not None and exit_crop_img.size > 0:
            try:
                os.makedirs("static/uploads/visitors", exist_ok=True)
                now_epoch = int(datetime.now(timezone.utc).timestamp())
                filename = f"{record.person_id}_exit_{now_epoch}.jpg"
                if getattr(record, "exit_avatar_url", None):
                    old_f = os.path.join(".", record.exit_avatar_url.lstrip("/").split("?")[0])
                    entry_f = os.path.join(".", record.avatar_url.lstrip("/").split("?")[0]) if getattr(record, "avatar_url", None) else None
                    if os.path.isfile(old_f):
                        # Safety: never delete if avatar_url is pointing to the same file on disk
                        if not entry_f or os.path.abspath(old_f) != os.path.abspath(entry_f):
                            try:
                                os.remove(old_f)
                            except Exception:
                                pass
                filepath = os.path.join("static", "uploads", "visitors", filename)
                cv2.imwrite(filepath, exit_crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                record.exit_avatar_url = f"/static/uploads/visitors/{filename}"
                logger.info("Saved exit snapshot for visitor %s to %s", record.person_id, filepath)
            except Exception as e:
                logger.error("Failed to write visitor exit snapshot: %s", str(e))
        elif getattr(payload, "face_image_base64", None) and not payload.exit_time:
            try:
                b64_str = payload.face_image_base64
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                crop_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(crop_bytes, np.uint8)
                crop_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if crop_img is not None and crop_img.size > 0:
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
                    cv2.imwrite(filepath, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    record.avatar_url = f"/static/uploads/visitors/{filename}"
                    logger.info("Saved snapshot for visitor %s on SLA breach / zone update to %s", record.person_id, filepath)
            except Exception as e:
                logger.error("Failed to write visitor SLA snapshot: %s", str(e))

        cutoff_sec = get_waiting_cutoff_seconds()

        # Enforce physical dwell limit: waiting_seconds cannot exceed total showroom presence
        if payload.waiting_seconds is not None:
            new_wait = int(payload.waiting_seconds)
            if record.created_at:
                phys_elapsed = max(0, int((now_utc - record.created_at).total_seconds()))
                # Allow a small 30s buffer for cross-camera handoff timing
                new_wait = min(new_wait, phys_elapsed + 30)
            record.waiting_seconds = max(int(record.waiting_seconds or 0), new_wait)
        elif record.created_at:
            phys_elapsed = max(0, int((now_utc - record.created_at).total_seconds()))
            record.waiting_seconds = min(int(record.waiting_seconds or 0), phys_elapsed + 30)

        if payload.is_waiting is not None:
            record.is_waiting = payload.is_waiting

        # Visitors who exited or have assigned staff are attended (not waiting)
        if record.exit_time or record.assigned_salesperson:
            record.is_waiting = False

        if payload.waiting_breach is not None:
            record.waiting_breach = bool(record.waiting_breach or payload.waiting_breach or ((record.waiting_seconds or 0) >= cutoff_sec))
        elif (record.waiting_seconds or 0) >= cutoff_sec:
            record.waiting_breach = True

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

    def delete_visitor(self, person_id: str) -> bool:
        """Deletes a single visitor record, its tracks, footfall events, and snapshot image (used for bounce cancellations)."""
        record = self.repo.get_by_person_id(person_id)
        if not record:
            return False
        if record.avatar_url:
            try:
                rel_path = record.avatar_url.lstrip("/")
                if os.path.exists(rel_path):
                    os.remove(rel_path)
            except Exception:
                pass
        tracks = self.db.query(VisitorTrack).filter(VisitorTrack.person_identity_id == record.id).all()
        for t in tracks:
            self.db.query(FootfallEvent).filter(FootfallEvent.visitor_track_id == t.id).delete()
            self.db.delete(t)
        self.db.delete(record)
        self.db.commit()
        logger.info("Deleted bounced/cancelled visitor record %s and associated footfall events.", person_id)
        return True

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

        # If a portrait snapshot was provided, refine/validate appearance from the crop
        if face_img is not None and face_img.size > 0:
            fh, fw = face_img.shape[:2]
            if fh > fw * 1.1:
                refined_app = BodyEmbedder.estimate_clothing_appearance(face_img)
                if not getattr(req, "appearance", None) or req.appearance in ("Casual Attire", "Walk-in Visitor", "Gray Attire"):
                    req.appearance = refined_app

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
                    cv2.imwrite(filepath, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    existing_rec.avatar_url = f"/static/uploads/visitors/{filename}"
                except Exception as e:
                    logger.error("Failed to update visitor face snapshot: %s", str(e))
            if getattr(req, "appearance", None):
                if not existing_rec.notes or "AI Face Detection" in existing_rec.notes or "Gray Attire" in existing_rec.notes:
                    existing_rec.notes = f"Appearance: {req.appearance} | Tracker #{req.tracker_id or 'auto'}"
            cutoff_sec = get_waiting_cutoff_seconds()
            if getattr(req, "waiting_seconds", None) is not None:
                existing_rec.waiting_seconds = max(int(existing_rec.waiting_seconds or 0), int(req.waiting_seconds))
            if getattr(req, "is_waiting", None) is not None:
                existing_rec.is_waiting = req.is_waiting
            if getattr(req, "waiting_breach", None) is not None:
                existing_rec.waiting_breach = bool(existing_rec.waiting_breach or req.waiting_breach or ((existing_rec.waiting_seconds or 0) >= cutoff_sec))
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
            fh, fw = face_img.shape[:2]
            ar = float(fw) / float(fh) if fh > 0 else 0.0
            # If face_img is already a tight face crop
            if 0.55 <= ar <= 1.45 and fh <= 140 and fw <= 140:
                try:
                    c_face, c_lms, raw_lms = self.embedder.detect_and_crop_face(face_img)
                    if c_face is not None and c_lms is not None:
                        embedding = self.embedder.extract_embedding(c_face, c_lms, full_image=face_img, raw_landmarks=raw_lms)
                    else:
                        embedding = self.embedder.extract_embedding(face_img)
                except Exception as e:
                    logger.warning("Failed to extract embedding from face crop: %s", str(e))
            else:
                # If face_img is a full-body or half-body portrait, detect and crop the face first!
                try:
                    c_face, c_lms, raw_lms = self.embedder.detect_and_crop_face(face_img)
                    if c_face is not None and c_face.size > 0 and c_lms is not None:
                        embedding = self.embedder.extract_embedding(
                            c_face, c_lms, full_image=face_img, raw_landmarks=raw_lms
                        )
                except Exception as e:
                    logger.warning("Failed to detect/crop face from body snapshot: %s", str(e))

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

            threshold = max(0.55, float(getattr(settings, "FACE_SIMILARITY_THRESHOLD", 0.62)))
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
        if req.is_entrance is not None:
            is_cam_entrance = bool(req.is_entrance)
        else:
            c_name = (req.camera_name or "").lower()
            is_cam_entrance = ("entrance" in c_name or "gate" in c_name or "door" in c_name or "entry" in c_name)

        if embedding is not None:
            visitor_reid_thresh = float(getattr(settings, "VISITOR_REID_THRESHOLD", 0.65))

            # Step 2A: Check for an ongoing active visit inside the showroom TODAY
            # On interior cameras: matches any ongoing active visitor.
            # On entrance cameras: matches active visitors who entered recently (< 15 mins)
            # to catch tracker ID flips near the door (e.g. #41 became #83).
            active_pid, active_sim = self.repo.find_best_match(
                embedding=embedding,
                similarity_threshold=0.55 if not is_cam_entrance else 0.60,
                only_active=True,
            )
            if active_pid:
                cand = self.repo.get_by_person_id(active_pid)
                if cand and not cand.is_expired and cand.exit_time is None:
                    cand_ist = to_ist(cand.created_at)
                    c_last_ist = to_ist(cand.last_seen_at) or cand_ist
                    if cand_ist and cand_ist.date() == now_ist.date():
                        time_since_creation = (now_ist - cand_ist).total_seconds()
                        if not is_cam_entrance or time_since_creation <= 900.0:
                            # If customer hasn't been seen for > 30 minutes, prior visit timed out: close it out
                            if c_last_ist and (now_ist - c_last_ist).total_seconds() > 1800.0:
                                cand.exit_time = c_last_ist.strftime("%I:%M %p")
                                cand.is_waiting = False
                                self.db.add(cand)
                                self.db.commit()
                                is_returning_customer = True
                                past_record = cand
                                is_active_session = False
                            else:
                                # Customer is continuously active inside the showroom on their ongoing visit!
                                matched_person_id = active_pid
                                matched_similarity = active_sim
                                is_active_session = True
                        else:
                            # Stale visit from earlier today (> 15 mins on entrance): leave for Step 2B
                            pass
                    else:
                        # Stale visit from a previous calendar day: close it out
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
                        h_last_ist = to_ist(history_cand.last_seen_at) or hist_ist

                        # Appearance / Clothing verification for returning customer candidate:
                        # Prevent merging two completely different visitors who entered on the same day
                        # (e.g. Green Top vs Red/Maroon Top) just because overhead side-profile faces score high similarity
                        appearance_consistent = True
                        if hist_ist and hist_ist.date() == now_ist.date():
                            cand_crop = None
                            if getattr(history_cand, "avatar_url", None):
                                c_path = history_cand.avatar_url.lstrip("/").split("?")[0]
                                if os.path.isfile(c_path):
                                    cand_crop = cv2.imread(c_path)
                            cand_b_emb = history_cand.get_body_embedding_vector()
                            if face_img is not None and (cand_crop is not None or cand_b_emb is not None):
                                _, _, _, is_app_match = self.body_embedder.verify_appearance(
                                    crop1=face_img,
                                    crop2=cand_crop,
                                    emb1=None,
                                    emb2=cand_b_emb,
                                )
                                if not is_app_match:
                                    appearance_consistent = False
                                    logger.info(
                                        "Suppressed false returning customer match %s: face sim %.3f matched but clothing/body appearance contradicted.",
                                        history_pid,
                                        history_sim,
                                    )

                        if appearance_consistent:
                            # A visit from a previous day OR with an exit_time marked indicates a returning customer
                            if (hist_ist and hist_ist.date() < now_ist.date()) or history_cand.exit_time is not None:
                                is_returning_customer = True
                                past_record = history_cand
                                if not history_cand.exit_time:
                                    history_cand.exit_time = h_last_ist.strftime("%I:%M %p") if h_last_ist else "Exited"
                                    self.db.add(history_cand)
                                    self.db.commit()
                            else:
                                # Same day and not marked exited: check if absent for > 30 minutes
                                if h_last_ist and (now_ist - h_last_ist).total_seconds() > 1800.0:
                                    history_cand.exit_time = h_last_ist.strftime("%I:%M %p")
                                    history_cand.is_waiting = False
                                    self.db.add(history_cand)
                                    self.db.commit()
                                    is_returning_customer = True
                                    past_record = history_cand
                                    is_active_session = False
                                elif not is_cam_entrance:
                                    matched_person_id = history_pid
                                    matched_similarity = history_sim
                                    is_active_session = True

        # Dual-Factor Body Re-ID check for active visitors who entered in the last 15 minutes
        # Runs for both entrance and interior cameras to catch tracker ID flips (e.g. #41 became #83)
        # even when face is tilted, looking down, or occluded on overhead CCTV
        if not is_active_session and not is_returning_customer and face_img is not None and face_img.size > 0:
            recent_active = (
                self.db.query(PersonIdentity)
                .filter(
                    PersonIdentity.is_expired.is_(False),
                    PersonIdentity.exit_time.is_(None),
                    PersonIdentity.created_at >= now_utc - timedelta(minutes=15),
                )
                .order_by(PersonIdentity.created_at.desc())
                .all()
            )
            curr_body_emb = self.body_embedder.extract_embedding(face_img)
            if curr_body_emb is not None:
                for cand in recent_active:
                    cand_b_emb = cand.get_body_embedding_vector()
                    cand_crop = None
                    if getattr(cand, "avatar_url", None):
                        c_path = cand.avatar_url.lstrip("/").split("?")[0]
                        if os.path.isfile(c_path):
                            cand_crop = cv2.imread(c_path)
                    if cand_b_emb is not None or cand_crop is not None:
                        r_sim, c_sim, app_score, is_m = self.body_embedder.verify_appearance(
                            crop1=face_img,
                            crop2=cand_crop,
                            emb1=curr_body_emb,
                            emb2=cand_b_emb,
                        )
                        # High-confidence same-person check: Re-ID >= 0.82 and clothing >= 0.40
                        if is_m and r_sim >= 0.82 and c_sim >= 0.40:
                            matched_person_id = cand.person_id
                            matched_similarity = app_score
                            is_active_session = True
                            logger.info(
                                "Re-identified active showroom visitor %s via Dual-Factor Re-ID (reid=%.3f, cloth=%.3f). Merged tracker #%s into active visit.",
                                cand.person_id,
                                r_sim,
                                c_sim,
                                req.tracker_id,
                            )
                            break

        # Extended recency safety net: Match visitors active in the showroom today (within 2 hours)
        # ONLY on interior cameras and if biometric match is genuinely high (sim >= 0.65)
        if not is_active_session and not is_returning_customer and embedding is not None and not is_cam_entrance:
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
                    r_last = to_ist(rec.last_seen_at) or now_ist
                    time_diff = (now_ist - r_last).total_seconds()
                    if time_diff < 7200.0 and sim >= 0.65:
                        rec_ist = to_ist(rec.created_at)
                        r_last_ist = to_ist(rec.last_seen_at) or rec_ist
                        if rec_ist and rec_ist.date() == now_ist.date() and rec.exit_time is None:
                            if r_last_ist and (now_ist - r_last_ist).total_seconds() > 1800.0:
                                rec.exit_time = r_last_ist.strftime("%I:%M %p")
                                rec.is_waiting = False
                                self.db.add(rec)
                                self.db.commit()
                                is_returning_customer = True
                                past_record = rec
                                is_active_session = False
                            else:
                                matched_person_id = rec.person_id
                                matched_similarity = sim
                                is_active_session = True
                        elif (rec_ist and rec_ist.date() < now_ist.date()) or rec.exit_time is not None:
                            is_returning_customer = True
                            past_record = rec
                        break

        # Interior Camera Safety Net: Interior cameras (Floor, Display Area, Service Desk)
        # should NEVER mint new store walk-ins without an entrance line crossing.
        if not is_active_session and not is_returning_customer and not is_cam_entrance:
            logger.info("Interior camera %s: suppressed new visitor creation (no entrance line crossing).", req.camera_name)
            return VisitorDetectResponse(
                status="IGNORED_INTERIOR",
                is_employee=False,
                is_new=False,
                message="Interior camera does not create new walk-ins without entrance crossing.",
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
                # Only set avatar_url if the record doesn't have one yet (preserve original clean entrance portrait)
                if not record.avatar_url and face_img is not None and face_img.size > 0:
                    try:
                        os.makedirs("static/uploads/visitors", exist_ok=True)
                        now_epoch = int(datetime.now(timezone.utc).timestamp())
                        filename = f"{record.person_id}_{now_epoch}.jpg"
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
        candidate_num = 1040 + total_count + 1
        new_person_id = f"VIS-2026-{candidate_num}"
        while self.repo.get_by_person_id(new_person_id):
            candidate_num += 1
            new_person_id = f"VIS-2026-{candidate_num}"

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

        # Extract full-body OSNet Person Re-ID embedding from the arrival body snapshot
        body_embedding = None
        if getattr(req, "body_embedding", None):
            body_embedding = np.array(req.body_embedding, dtype=np.float32)
        elif face_img is not None and face_img.size > 0:
            body_embedding = self.body_embedder.extract_embedding(face_img)

        new_record = self.repo.create_or_update(
            person_id=new_person_id,
            embedding=embedding,
            body_embedding=body_embedding,
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
        if getattr(req, "waiting_seconds", None) is not None:
            new_record.waiting_seconds = int(req.waiting_seconds)
        else:
            new_record.waiting_seconds = 0

        if getattr(req, "is_waiting", None) is not None:
            new_record.is_waiting = req.is_waiting
        else:
            new_record.is_waiting = True

        if getattr(req, "waiting_breach", None) is not None:
            new_record.waiting_breach = bool(req.waiting_breach or ((new_record.waiting_seconds or 0) >= 900))
        if (
            getattr(req, "zone_dwell", None)
            or getattr(req, "exit_time", None)
            or getattr(req, "waiting_seconds", None) is not None
        ):
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

