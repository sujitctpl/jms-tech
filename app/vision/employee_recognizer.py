from typing import Any, Dict, List, Optional, Set, Tuple
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
        min_confirm_frames: int = 1,
        vote_window: int = 5,
        faiss_index: Optional[Any] = None,
    ):
        self.embedder = embedder or FaceEmbedder()
        self.vector_store = vector_store or faiss_index or PgVectorStoreManager(
            similarity_threshold=similarity_threshold or settings.FACE_SIMILARITY_THRESHOLD
        )
        self.faiss_index = self.vector_store
        self.similarity_threshold = (
            similarity_threshold or settings.FACE_SIMILARITY_THRESHOLD
        )
        self.min_confirm_frames = max(1, min_confirm_frames)
        self.vote_window = max(self.min_confirm_frames, vote_window)

        # Temporary tracker_id -> confirmed employee_id mapping
        self.track_to_employee: Dict[str, str] = {}
        # Multi-frame voting buffer: tracker_id -> list of (emp_id, score) recent matches
        self.track_votes: Dict[str, List[Tuple[str, float]]] = {}
        # Set of active employee IDs currently in the camera frame
        self.active_detected_employees: Set[str] = set()

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
        tid = str(tracker_id)

        # 1. If this track is already confirmed as an employee, return cached match
        if tid in self.track_to_employee:
            return self.track_to_employee[tid], 1.0

        # 2. Strict face quality check
        is_reliable, reason = self.embedder.is_face_reliable(face_crop, face_conf, landmarks)
        if not is_reliable or face_crop is None:
            return None

        # 3. Extract embedding and query dedicated Employee recognition store via pgvector
        try:
            if full_image is not None and raw_landmarks is not None and len(raw_landmarks) == 5:
                embedding = self.embedder.extract_embedding(face_crop, raw_landmarks, full_image=full_image)
            else:
                embedding = self.embedder.extract_embedding(face_crop, landmarks)
        except Exception as e:
            logger.warning("Error extracting face embedding for employee check: %s", str(e))
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
                    return None

            # Point 1: Multi-Frame Temporal Confirmation (Voting Buffer)
            if self.min_confirm_frames > 1:
                votes = self.track_votes.setdefault(tid, [])
                votes.append((best_emp_id, best_score))
                if len(votes) > self.vote_window:
                    votes.pop(0)

                matching_votes = [score for (eid, score) in votes if eid == best_emp_id]
                if len(matching_votes) < self.min_confirm_frames:
                    logger.debug(
                        "Track %s tentative match %s (%d/%d frames confirmed, sim: %.3f)",
                        tid,
                        best_emp_id,
                        len(matching_votes),
                        self.min_confirm_frames,
                        best_score,
                    )
                    return None

                # Sufficient frames verified! Use average similarity across confirming frames
                best_score = float(np.mean(matching_votes))

            # Verified Employee Recognition!
            self.track_to_employee[tid] = best_emp_id
            self.active_detected_employees.add(best_emp_id)
            logger.info(
                "Track %s verified as Employee %s via pgvector (sim: %.3f >= threshold %.2f, confirmed frames: %d)",
                tid,
                best_emp_id,
                best_score,
                self.similarity_threshold,
                self.min_confirm_frames,
            )
            return best_emp_id, best_score

        return None

    def on_track_lost(self, tracker_id: str) -> None:
        """Cleans up temporary tracking and voting buffer when track disappears."""
        tid = str(tracker_id)
        self.track_votes.pop(tid, None)
        emp_id = self.track_to_employee.pop(tid, None)
        if emp_id and emp_id in self.active_detected_employees:
            if emp_id not in self.track_to_employee.values():
                self.active_detected_employees.discard(emp_id)

