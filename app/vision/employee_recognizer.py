from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from app.core.config import settings
from app.core.logging import logger
from app.vision.face_embedder import FaceEmbedder
from app.vision.vector_store import PgVectorStoreManager


class EmployeeRecognitionManager:
    """
    Dedicated employee face recognition engine utilizing ArcFace and PostgreSQL pgvector.
    Completely isolated from visitor tracking and retention policies.
    Maintains employee IDs across temporary tracking IDs, enforces multi-frame voting
    for zero false positives, and provides exclusion signals to prevent employees
    from being counted in visitor footfall analytics.
    """

    def __init__(
        self,
        embedder: Optional[FaceEmbedder] = None,
        vector_store: Optional[Any] = None,
        similarity_threshold: Optional[float] = None,
        min_confirm_frames: Optional[int] = None,
        vote_window: Optional[int] = None,
        faiss_index: Optional[Any] = None,
    ):
        self.embedder = embedder or FaceEmbedder()
        if vector_store is not None:
            self.vector_store = vector_store
        elif faiss_index is not None:
            self.vector_store = faiss_index
        else:
            self.vector_store = PgVectorStoreManager(
                similarity_threshold=similarity_threshold or settings.FACE_SIMILARITY_THRESHOLD
            )
        self.faiss_index = self.vector_store
        self.similarity_threshold = (
            similarity_threshold or settings.FACE_SIMILARITY_THRESHOLD
        )
        conf_frames = min_confirm_frames if min_confirm_frames is not None else 1
        v_win = vote_window if vote_window is not None else getattr(settings, "EMPLOYEE_VOTE_WINDOW", 5)
        self.min_confirm_frames = max(1, conf_frames)
        self.vote_window = max(self.min_confirm_frames, v_win)

        # Temporary tracker_id -> confirmed employee_id mapping
        self.track_to_employee: Dict[str, str] = {}
        # Multi-frame voting buffer: tracker_id -> list of (emp_id, score, weight) recent observations
        self.track_votes: Dict[str, List[Tuple[str, float, float]]] = {}
        # Multi-attempt counter per track
        self.track_attempts: Dict[str, int] = {}
        # Set of active employee IDs currently in the camera frame
        self.active_detected_employees: Set[str] = set()
        # Track recent loss timestamp for re-acquisition tolerance across short gaps: emp_id -> timestamp
        self._recently_lost_employees: Dict[str, float] = {}

    def load_active_employees(self, employee_records: List[Tuple[str, np.ndarray]]) -> int:
        """
        Populates the pgvector store cache with registered active employee face vectors.
        Supports multi-angle gallery embeddings.
        """
        if hasattr(self.vector_store, "clear"):
            self.vector_store.clear()
        count = 0
        for emp_id, emb in employee_records:
            if hasattr(self.vector_store, "add"):
                self.vector_store.add(emp_id, emb, replace=False)
            count += 1
        logger.info("EmployeeRecognitionManager loaded %d employee face embeddings into pgvector cache.", count)
        return count

    def register_employee(
        self,
        employee_id: str,
        embedding: Optional[np.ndarray] = None,
        gallery_embeddings: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Adds or updates an active employee in pgvector recognition cache."""
        eid = str(employee_id)
        if hasattr(self.vector_store, "remove"):
            self.vector_store.remove(eid)

        angles = gallery_embeddings if gallery_embeddings else ([embedding] if embedding is not None else [])
        for ang in angles:
            if hasattr(self.vector_store, "add"):
                self.vector_store.add(eid, ang, replace=False)

        logger.info("Registered employee %s (%d photos) into pgvector recognition cache.", eid, len(angles))

    def remove_employee(self, employee_id: str) -> bool:
        """Removes a deactivated or removed employee from the recognition cache."""
        eid = str(employee_id)
        removed = False
        if hasattr(self.vector_store, "remove"):
            removed = self.vector_store.remove(eid)
        # Clear any active track mapping
        for tid, e_id in list(self.track_to_employee.items()):
            if e_id == eid:
                del self.track_to_employee[tid]
        self.active_detected_employees.discard(eid)
        return removed

    def is_track_employee(self, tracker_id: str) -> bool:
        """Returns True if the temporary track has been identified as a showroom employee."""
        return str(tracker_id) in self.track_to_employee

    def get_employee_id(self, tracker_id: str) -> Optional[str]:
        """Returns the employee_id for this track, or None if visitor / unresolved."""
        return self.track_to_employee.get(str(tracker_id))

    def evaluate_track_face(
        self,
        tracker_id: str,
        face_crop: Optional[np.ndarray],
        face_conf: float,
        landmarks: Optional[List[Tuple[int, int]]] = None,
        full_image: Optional[np.ndarray] = None,
        raw_landmarks: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Attempts to recognize whether a detected face belongs to a registered showroom employee
        using PostgreSQL pgvector and multi-frame temporal voting.
        Returns (employee_id, similarity) if confirmed across required consecutive frames, else None.
        """
        track_id = str(tracker_id)
        tid = track_id

        # 1. If this track is already confirmed as an employee, return cached match
        if tid in self.track_to_employee:
            return self.track_to_employee[tid], 1.0

        # Increment attempts counter
        self.track_attempts[track_id] = self.track_attempts.get(track_id, 0) + 1
        attempts = self.track_attempts[track_id]

        face_confidence = float(face_conf)
        threshold = float(self.similarity_threshold)
        similarity = 0.0
        employee_id = None
        embedding_generated = False
        final_status = "UNKNOWN"

        # Compute face size and blur score
        if face_crop is not None and face_crop.size > 0:
            h, w = face_crop.shape[:2]
            face_size = f"{w}x{h}"
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        else:
            face_size = "0x0"
            blur_score = 0.0

        # Compute eye distance
        eye_distance = 0.0
        lms = raw_landmarks if (raw_landmarks and len(raw_landmarks) >= 2) else landmarks
        if lms and len(lms) >= 2:
            lx, ly = lms[0]
            rx, ry = lms[1]
            eye_distance = float(np.hypot(rx - lx, ry - ly))

        def _log_attempt(status_str: str, emp_val: Optional[str] = None, sim_val: Optional[float] = None):
            logger.info(
                "FACE_RECOGNITION | "
                "track_id=%s | "
                "face_size=%s | "
                "face_confidence=%.3f | "
                "blur_score=%.2f | "
                "eye_distance=%.2f | "
                "quality_passed=%s | "
                "embedding_generated=%s | "
                "best_employee=%s | "
                "similarity=%.4f | "
                "threshold=%.4f | "
                "attempts=%s | "
                "final_status=%s",
                track_id,
                face_size,
                face_confidence,
                blur_score,
                eye_distance,
                quality_passed,
                embedding_generated,
                emp_val if emp_val is not None else employee_id,
                sim_val if sim_val is not None else similarity,
                threshold,
                attempts,
                status_str
            )

        # 2. Strict face quality check
        is_reliable, reason = self.embedder.is_face_reliable(face_crop, face_conf, landmarks)
        quality_passed = bool(is_reliable)
        if not is_reliable or face_crop is None:
            final_status = f"REJECTED_QUALITY_{reason.upper()}"
            _log_attempt(final_status)
            return None

        # 3. Extract embedding and query dedicated Employee recognition store via pgvector
        try:
            if full_image is not None and raw_landmarks is not None and len(raw_landmarks) == 5:
                embedding = self.embedder.extract_embedding(
                    face_crop=face_crop,
                    landmarks=landmarks,
                    full_image=full_image,
                    raw_landmarks=raw_landmarks,
                )
            else:
                embedding = self.embedder.extract_embedding(face_crop, landmarks)
            embedding_generated = True
        except Exception as e:
            logger.warning("Error extracting face embedding for employee check: %s", str(e))
            final_status = "EXTRACTION_FAILED"
            _log_attempt(final_status)
            return None

        best_emp_id = None
        best_score = 0.0

        try:
            if hasattr(self.vector_store, "find_best_employee_match"):
                best_emp_id, best_score = self.vector_store.find_best_employee_match(
                    embedding, similarity_threshold=self.similarity_threshold
                )
            elif hasattr(self.vector_store, "find_best_match"):
                best_emp_id, best_score = self.vector_store.find_best_match(embedding)
        except Exception as e:
            logger.warning("Error querying pgvector store: %s", str(e))
            best_emp_id, best_score = None, 0.0

        employee_id = best_emp_id
        similarity = float(best_score)

        if best_emp_id is not None:
            # Enforce single active physical body: an employee cannot be in two locations at the same time
            for existing_tid, existing_eid in list(self.track_to_employee.items()):
                if existing_eid == best_emp_id and existing_tid != tid:
                    logger.debug(
                        "Employee %s already actively tracked on track %s. Rejecting duplicate match for track %s",
                        best_emp_id,
                        existing_tid,
                        tid,
                    )
                    final_status = f"REJECTED_DUPLICATE_BODY_{existing_tid}"
                    _log_attempt(final_status)
                    return None

            # Multi-Frame Temporal Confirmation (Quality-Weighted Voting Buffer)
            quality_weight = float(
                min(1.0, max(0.1, face_confidence)) *
                min(1.0, max(0.2, blur_score / 20.0))
            )

            if self.min_confirm_frames > 1:
                votes = self.track_votes.setdefault(tid, [])
                votes.append((best_emp_id, best_score, quality_weight))
                if len(votes) > self.vote_window:
                    votes.pop(0)

                matching_obs = [(score, weight) for (eid, score, weight) in votes if eid == best_emp_id]
                other_obs = [(eid, score, weight) for (eid, score, weight) in votes if eid != best_emp_id]

                if len(matching_obs) < self.min_confirm_frames:
                    logger.debug(
                        "Track %s tentative match %s (%d/%d frames confirmed, sim: %.3f)",
                        tid,
                        best_emp_id,
                        len(matching_obs),
                        self.min_confirm_frames,
                        best_score,
                    )
                    final_status = f"PENDING_VOTING_CONFIRMATION_{len(matching_obs)}/{self.min_confirm_frames}"
                    _log_attempt(final_status)
                    return None

                # Discard conflicting assignments if multiple employees compete in voting window
                if other_obs:
                    other_weights = sum(w for (_, _, w) in other_obs)
                    matching_weights = sum(w for (_, w) in matching_obs)
                    if matching_weights < 1.5 * other_weights:
                        logger.debug(
                            "Track %s voting conflict: %s (weight: %.2f) vs others (weight: %.2f) - inconclusive",
                            tid,
                            best_emp_id,
                            matching_weights,
                            other_weights,
                        )
                        final_status = "REJECTED_VOTING_CONFLICT"
                        _log_attempt(final_status)
                        return None

                # Quality-weighted average similarity across confirming observations
                weights_sum = sum(w for (_, w) in matching_obs)
                if weights_sum > 0:
                    best_score = float(sum(s * w for (s, w) in matching_obs) / weights_sum)
                else:
                    best_score = float(np.mean([s for (s, _) in matching_obs]))
                similarity = best_score

            # Verified Employee Recognition!
            self.track_to_employee[tid] = best_emp_id
            self.active_detected_employees.add(best_emp_id)
            final_status = "CONFIRMED_MATCH"
            _log_attempt(final_status)
            return best_emp_id, best_score

        final_status = "NO_MATCH_BELOW_THRESHOLD"
        _log_attempt(final_status)
        return None

    def on_track_lost(self, tracker_id: str) -> None:
        """Cleans up temporary tracking and voting buffer when track disappears."""
        import time
        tid = str(tracker_id)
        self.track_votes.pop(tid, None)
        self.track_attempts.pop(tid, None)
        emp_id = self.track_to_employee.pop(tid, None)
        if emp_id:
            self._recently_lost_employees[emp_id] = time.time()
            if emp_id not in self.track_to_employee.values():
                self.active_detected_employees.discard(emp_id)

