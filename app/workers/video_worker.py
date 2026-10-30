import threading
import time
import uuid
from typing import Any, Dict, Optional
import cv2
from app.core.config import settings
from app.core.logging import logger, sanitize_url
from app.db.models.camera import Camera
from app.db.session import SessionLocal
from app.repositories.detection_event_repo import DetectionEventRepository
from app.repositories.footfall_event_repo import FootfallEventRepository
from app.repositories.processing_run_repo import ProcessingRunRepository
from app.repositories.visitor_track_repo import VisitorTrackRepository
from app.vision.entry_counter import EntryLineCounter
from app.vision.persistent_tracker import PersistentCameraTracker
from app.vision.stream_reader import StreamReader
from app.vision.tracker import PersonTracker


class VideoWorker(threading.Thread):
    """
    Background worker thread running video stream ingestion, person tracking,
    and virtual entry-line crossing detection for a single camera.
    """

    def __init__(
        self,
        camera_id: uuid.UUID,
        camera_name: str,
        stream_url: str,
        entry_line_config: Optional[Dict[str, Any]],
    ):
        super().__init__(name=f"Worker-Cam-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.stream_url = stream_url
        self.entry_line_config = entry_line_config or {}
        self.stop_event = threading.Event()

        self.run_id: Optional[uuid.UUID] = None
        self.frames_processed: int = 0
        self.last_error: Optional[str] = None
        self.is_running: bool = False
        self.latest_frame_jpeg: Optional[bytes] = None

    def stop(self) -> None:
        """Signals the worker thread to stop cleanly."""
        logger.info("Signaling worker for camera %s (%s) to stop...", self.camera_name, self.camera_id)
        self.stop_event.set()

    def run(self) -> None:
        self.is_running = True
        masked_url = sanitize_url(self.stream_url)
        logger.info("Starting video worker for camera '%s' (%s) on stream: %s", self.camera_name, self.camera_id, masked_url)

        db = SessionLocal()
        run_repo = ProcessingRunRepository(db)
        track_repo = VisitorTrackRepository(db)
        footfall_repo = FootfallEventRepository(db)
        det_repo = DetectionEventRepository(db)

        # Initialize processing run in database
        try:
            processing_run = run_repo.create(self.camera_id)
            self.run_id = processing_run.id
            logger.info("Created processing run %s for camera %s", self.run_id, self.camera_id)
        except Exception as e:
            self.last_error = f"Failed to create processing run: {e}"
            logger.error("Error creating processing run for camera %s: %s", self.camera_id, str(e), exc_info=True)
            self.is_running = False
            db.close()
            return

        stream_reader = StreamReader(self.stream_url)
        tracker = PersonTracker()
        entry_counter = EntryLineCounter(self.entry_line_config)
        persistent_tracker = PersistentCameraTracker(ttl_seconds=5.0)

        # Persistent Identity Recovery & pgvector Similarity Search (14-Day Retention)
        from datetime import datetime, timezone
        import numpy as np
        from app.repositories.employee_attendance_repo import EmployeeAttendanceRepository
        from app.repositories.employee_repo import EmployeeRepository
        from app.repositories.person_identity_repo import PersonIdentityRepository
        from app.vision.employee_recognizer import EmployeeRecognitionManager
        from app.vision.face_embedder import FaceEmbedder
        from app.vision.identity_tracker import IdentityRecoveryManager
        from app.vision.vector_store import PgVectorStoreManager

        identity_repo = PersonIdentityRepository(db)
        emp_repo = EmployeeRepository(db)
        emp_attendance_repo = EmployeeAttendanceRepository(db)
        face_embedder = FaceEmbedder()
        visitor_vector_store = PgVectorStoreManager(db=db, similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD)
        employee_vector_store = PgVectorStoreManager(db=db, similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD)

        # Dedicated Employee Recognition Engine backed by pgvector
        employee_recognizer = EmployeeRecognitionManager(
            embedder=face_embedder,
            vector_store=employee_vector_store,
            similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD,
            min_confirm_frames=settings.EMPLOYEE_MIN_CONFIRM_FRAMES,
            vote_window=settings.EMPLOYEE_VOTE_WINDOW,
        )

        def handle_matched(pid: str, ts: float):
            try:
                dt_ts = datetime.fromtimestamp(ts, tz=timezone.utc)
                identity_repo.update_last_seen(pid, dt_ts)
            except Exception as ex:
                logger.error("Failed to update last_seen_at for %s: %s", pid, str(ex))

        def handle_created(pid: str, emb: np.ndarray, ts: float):
            try:
                dt_ts = datetime.fromtimestamp(ts, tz=timezone.utc)
                identity_repo.create_or_update(pid, emb, dt_ts)
            except Exception as ex:
                logger.error("Failed to persist new identity %s: %s", pid, str(ex))

        identity_manager = IdentityRecoveryManager(
            embedder=face_embedder,
            vector_store=visitor_vector_store,
            similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD,
            on_matched_callback=handle_matched,
            on_created_callback=handle_created,
        )

        # Connect to stream
        if not stream_reader.connect():
            logger.warning("Initial connection failed for camera %s. Will retry in loop.", self.camera_id)

        try:
            last_db_sync = time.time()
            frame_counter = 0

            while not self.stop_event.is_set():
                success, frame = stream_reader.read_frame()

                if not success or frame is None:
                    # Give thread breathing room if stream is waiting/reconnecting
                    time.sleep(0.1)
                    continue

                frame_counter += 1
                self.frames_processed = frame_counter

                # Encode latest frame for live preview snapshot/streaming
                try:
                    ret_enc, enc_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret_enc:
                        self.latest_frame_jpeg = enc_buf.tobytes()
                except Exception:
                    pass

                # Optional frame skip for high-FPS RTSP feeds
                if settings.FRAME_SKIP > 0 and (frame_counter % (settings.FRAME_SKIP + 1)) != 0:
                    continue

                # Run detection and tracking
                try:
                    tracked_persons = tracker.track_frame(frame)
                except Exception as e:
                    logger.error("Inference exception on camera %s: %s", self.camera_id, str(e))
                    continue

                # Encode annotated frame with persistent detected persons (face & body)
                try:
                    annotated = frame.copy()
                    raw_dets = [(p.box, p.confidence) for p in tracked_persons]
                    active_tracks = persistent_tracker.update(raw_dets)

                    h_img, w_img = annotated.shape[:2]
                    for trk in active_tracks:
                        x1, y1, x2, y2 = map(int, trk.box)
                        x1 = max(0, min(x1, w_img - 1))
                        y1 = max(0, min(y1, h_img - 1))
                        x2 = max(0, min(x2, w_img - 1))
                        y2 = max(0, min(y2, h_img - 1))

                        # Format live tracking duration timer
                        now_trk = time.time()
                        dwell_sec = max(0.0, now_trk - (trk.first_seen if trk.first_seen > 0 else now_trk))
                        tot_s = int(dwell_sec)
                        tot_h = tot_s // 3600
                        tot_m = (tot_s % 3600) // 60
                        tot_s_rem = tot_s % 60
                        timer_str = f"{tot_h}:{tot_m:02d}:{tot_s_rem:02d}" if tot_h > 0 else f"{tot_m:02d}:{tot_s_rem:02d}"

                        # Body bounding box (green)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 115), 2)
                        label = f"Person #{trk.track_id} | {timer_str}"
                        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 115), 2)

                        # Face / Head bounding box (cyan)
                        fx1, fy1, fx2, fy2 = trk.face_box
                        fx1 = max(0, min(fx1, w_img - 1))
                        fy1 = max(0, min(fy1, h_img - 1))
                        fx2 = max(0, min(fx2, w_img - 1))
                        fy2 = max(0, min(fy2, h_img - 1))
                        if fy2 > fy1 and fx2 > fx1:
                            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 180, 0), 2)
                            cv2.putText(annotated, "Face", (fx1, max(14, fy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

                    ret_enc, enc_buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret_enc:
                        self.latest_frame_jpeg = enc_buf.tobytes()
                except Exception:
                    pass

                # Track lifecycle cleanup for lost tracks
                current_tids = {str(p.tracker_id) for p in tracked_persons}
                for old_tid in list(employee_recognizer.track_to_employee.keys()):
                    if old_tid not in current_tids:
                        employee_recognizer.on_track_lost(old_tid)
                for old_tid in list(identity_manager.track_to_person.keys()):
                    if old_tid not in current_tids:
                        identity_manager.on_track_lost(old_tid)

                # Process tracks in database with persistent identity recovery
                track_model_map = {}
                for person in tracked_persons:
                    try:
                        str_tid = str(person.tracker_id)
                        # Attempt face identity recovery if face region is available
                        px1, py1, px2, py2 = map(int, person.box)
                        h_box = py2 - py1
                        w_box = px2 - px1
                        if h_box > 40 and w_box > 40:
                            fy2 = py1 + int(h_box * 0.28)
                            fx1 = max(0, px1 + int(w_box * 0.15))
                            fx2 = min(frame.shape[1], px2 - int(w_box * 0.15))
                            fy1 = max(0, py1)
                            if fy2 > fy1 + 20 and fx2 > fx1 + 20:
                                face_crop = frame[fy1:fy2, fx1:fx2]
                                # 1. Check if recognized as showroom employee first
                                emp_match = employee_recognizer.evaluate_track_face(
                                    tracker_id=str_tid,
                                    face_crop=face_crop,
                                    face_conf=person.confidence,
                                )
                                if emp_match:
                                    emp_id, emp_conf = emp_match
                                    try:
                                        emp_attendance_repo.record_attendance_event(
                                            employee_id=emp_id,
                                            camera_id=self.camera_id,
                                            event_type="CHECK_IN",
                                            confidence=emp_conf,
                                        )
                                    except Exception as ex:
                                        logger.error("Failed to record employee punch for %s: %s", emp_id, str(ex))
                                else:
                                    # 2. Otherwise process as customer visitor identity
                                    identity_manager.process_track_face(
                                        tracker_id=person.tracker_id,
                                        face_crop=face_crop,
                                        face_conf=person.confidence,
                                    )

                        person.person_id = identity_manager.get_person_id(person.tracker_id)

                        vt = track_repo.get_or_create(
                            camera_id=self.camera_id,
                            processing_run_id=self.run_id,
                            tracker_id=person.tracker_id,
                            person_id=person.person_id,
                        )
                        track_model_map[person.tracker_id] = vt

                        # Optional detection persistence
                        if settings.SAVE_DETECTION_EVENTS:
                            det_repo.create(
                                camera_id=self.camera_id,
                                processing_run_id=self.run_id,
                                visitor_track_id=vt.id,
                                frame_number=frame_counter,
                                confidence=person.confidence,
                                bbox_x1=person.box[0],
                                bbox_y1=person.box[1],
                                bbox_x2=person.box[2],
                                bbox_y2=person.box[3],
                            )
                    except Exception as e:
                        logger.error("Failed to persist track %s: %s", person.tracker_id, str(e))

                # Evaluate entry line crossing
                new_entries = entry_counter.process_frame(tracked_persons)
                for entry_person in new_entries:
                    # Suppress visitor entry count if the person is a recognized showroom employee
                    if employee_recognizer.is_track_employee(str(entry_person.tracker_id)):
                        emp_id = employee_recognizer.get_employee_id(str(entry_person.tracker_id))
                        logger.info(
                            "Staff member %s crossed entry line on camera %s - visitor footfall count suppressed.",
                            emp_id,
                            self.camera_id,
                        )
                        continue

                    vt = track_model_map.get(entry_person.tracker_id)
                    if vt:
                        try:
                            footfall_repo.record_entry(
                                camera_id=self.camera_id,
                                visitor_track_id=vt.id,
                            )
                            logger.info(
                                "Recorded entry footfall for camera %s, track %s",
                                self.camera_id,
                                entry_person.tracker_id,
                            )
                        except Exception as e:
                            logger.error("Failed to record footfall event: %s", str(e))

                # Periodically sync frame count to DB (e.g. every 5 seconds or 50 frames)
                if time.time() - last_db_sync > 5.0:
                    try:
                        run_repo.update_frame_count(self.run_id, self.frames_processed)
                        last_db_sync = time.time()
                    except Exception as e:
                        logger.warning("Failed to sync frame count to DB: %s", str(e))

        except Exception as e:
            self.last_error = str(e)
            logger.error("Worker fatal exception on camera %s: %s", self.camera_id, str(e), exc_info=True)
        finally:
            stream_reader.release()
            final_status = "FAILED" if self.last_error else "STOPPED"
            try:
                run_repo.finish_run(
                    run_id=self.run_id,
                    status=final_status,
                    frames_processed=self.frames_processed,
                    error_message=self.last_error,
                )
                logger.info("Finalized processing run %s with status %s", self.run_id, final_status)
            except Exception as e:
                logger.error("Failed to finalize processing run %s: %s", self.run_id, str(e))

            db.close()
            self.is_running = False
            logger.info("Video worker stopped for camera %s", self.camera_id)


class VideoWorkerManager:
    """
    Manages camera worker threads. Ensures only one worker runs per camera.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._workers: Dict[uuid.UUID, VideoWorker] = {}

    def start_worker(self, camera: Camera) -> bool:
        """
        Starts a background worker for the given camera.
        Returns True if started, False if already running.
        """
        with self._lock:
            existing = self._workers.get(camera.id)
            if existing and existing.is_alive():
                logger.warning("Worker already active for camera %s", camera.id)
                return False

            worker = VideoWorker(
                camera_id=camera.id,
                camera_name=camera.name,
                stream_url=camera.stream_url,
                entry_line_config=camera.entry_line_config,
            )
            self._workers[camera.id] = worker
            worker.start()
            return True

    def stop_worker(self, camera_id: uuid.UUID, wait: bool = False) -> bool:
        """
        Stops the worker for the given camera.
        Returns True if stopped, False if not running.
        """
        with self._lock:
            worker = self._workers.pop(camera_id, None)
            if worker and worker.is_alive():
                worker.stop()
                if wait:
                    worker.join(timeout=2.0)
                return True
            return False

    def get_status(self, camera_id: uuid.UUID) -> Dict[str, Any]:
        """Returns processing status for a camera."""
        with self._lock:
            worker = self._workers.get(camera_id)
            if worker and worker.is_alive():
                return {
                    "is_processing": True,
                    "active_run_id": worker.run_id,
                    "frames_processed": worker.frames_processed,
                    "last_error": worker.last_error,
                }
            elif worker:
                return {
                    "is_processing": False,
                    "active_run_id": None,
                    "frames_processed": worker.frames_processed,
                    "last_error": worker.last_error,
                }
            else:
                return {
                    "is_processing": False,
                    "active_run_id": None,
                    "frames_processed": 0,
                    "last_error": None,
                }

    def get_latest_frame(self, camera_id: uuid.UUID) -> Optional[bytes]:
        """Returns the latest encoded JPEG frame for a camera if active."""
        with self._lock:
            worker = self._workers.get(camera_id)
            if worker and worker.is_alive():
                return worker.latest_frame_jpeg
            return None

    def stop_all(self) -> None:
        """Stops all running workers."""
        with self._lock:
            for cid, worker in self._workers.items():
                if worker.is_alive():
                    worker.stop()


# Singleton instance
worker_manager = VideoWorkerManager()
