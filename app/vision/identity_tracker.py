from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid
import numpy as np
from app.core.config import settings
from app.core.logging import logger
from app.vision.face_embedder import FaceEmbedder
from app.vision.faiss_index import FaissIndexManager
from app.vision.vector_store import PgVectorStoreManager


class IdentityMatchStatus(str, Enum):
    MATCHED = "MATCHED"
    CREATED = "CREATED"
    UNRESOLVED = "UNRESOLVED"
    UNRELIABLE_FACE = "UNRELIABLE_FACE"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class IdentityMatchResult:
    tracker_id: str
    person_id: Optional[str]
    similarity: float
    status: IdentityMatchStatus
    message: str


class IdentityRecoveryManager:
    """
    Coordinates face-based visitor identity recovery when tracking is lost or new tracks appear.
    Maintains a strict separation between temporary ByteTrack tracking IDs and persistent person IDs.
    Applies quality checks, caching, cooldowns, and controlled recognition attempts to prevent
    unnecessary inference and avoid duplicate entry counts.
    """

    def __init__(
        self,
        embedder: Optional[FaceEmbedder] = None,
        vector_store: Optional[Any] = None,
        faiss_index: Optional[Any] = None,
        similarity_threshold: Optional[float] = None,
        max_attempts: Optional[int] = None,
        cooldown_frames: Optional[int] = None,
        on_matched_callback: Optional[Callable[[str, float], None]] = None,
        on_created_callback: Optional[Callable[[str, np.ndarray, float], None]] = None,
    ):
        self.embedder = embedder or FaceEmbedder()
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.FACE_SIMILARITY_THRESHOLD
        )
        self.vector_store = vector_store or faiss_index or PgVectorStoreManager(similarity_threshold=self.similarity_threshold)
        # Backward compatibility alias
        self.faiss_index = self.vector_store
        self.max_attempts = (
            max_attempts if max_attempts is not None else settings.FACE_MAX_RECOGNITION_ATTEMPTS
        )
        self.cooldown_frames = (
            cooldown_frames
            if cooldown_frames is not None
            else settings.FACE_RECOGNITION_COOLDOWN_FRAMES
        )

        # Callbacks for persistence layer synchronization
        self.on_matched_callback = on_matched_callback
        self.on_created_callback = on_created_callback

        # Separation of temporary tracker_id <-> persistent person_id
        self.track_to_person: Dict[str, str] = {}
        self.person_to_tracks: Dict[str, Set[str]] = {}

        # Track inference controls & caching
        self.track_attempts: Dict[str, int] = {}
        self.track_cooldown: Dict[str, int] = {}
        self.track_embeddings: Dict[str, np.ndarray] = {}
        self.track_reliable_frames: Dict[str, int] = {}

        # Duplicate entry prevention sets
        self.counted_persons: Set[str] = set()
        self.counted_tracks: Set[str] = set()

    def get_person_id(self, tracker_id: str) -> Optional[str]:
        """Returns the persistent person_id associated with a temporary tracker_id, or None if unresolved."""
        return self.track_to_person.get(str(tracker_id))

    def associate_identity(self, tracker_id: str, person_id: str) -> None:
        """Explicitly links a temporary tracking ID to a persistent person ID."""
        tid = str(tracker_id)
        pid = str(person_id)
        self.track_to_person[tid] = pid
        self.person_to_tracks.setdefault(pid, set()).add(tid)

        # If this temporary track already triggered an entry event before resolution,
        # carry forward the entry status to the persistent person ID to prevent double counting
        if tid in self.counted_tracks:
            self.counted_persons.add(pid)

    def process_track_face(
        self,
        tracker_id: str,
        face_crop: Optional[np.ndarray],
        face_conf: float,
        landmarks: Optional[List[Tuple[int, int]]] = None,
        timestamp: Optional[float] = None,
    ) -> IdentityMatchResult:
        """
        Evaluates a detected face for identity recovery.
        Executes face embedding generation and FAISS search only when quality criteria are met.
        Avoids redundant inference if the track is already reliably identified.
        """
        tid = str(tracker_id)
        ts = timestamp or time.time()

        # 1. If already resolved to a persistent person ID, skip redundant inference
        if tid in self.track_to_person:
            existing_pid = self.track_to_person[tid]
            return IdentityMatchResult(
                tracker_id=tid,
                person_id=existing_pid,
                similarity=1.0,
                status=IdentityMatchStatus.MATCHED,
                message="Already identified from previous frames.",
            )

        # 2. Cooldown check to prevent inference on every frame
        cooldown = self.track_cooldown.get(tid, 0)
        if cooldown > 0:
            self.track_cooldown[tid] = cooldown - 1
            return IdentityMatchResult(
                tracker_id=tid,
                person_id=None,
                similarity=0.0,
                status=IdentityMatchStatus.UNRESOLVED,
                message=f"Inference in cooldown ({cooldown} frames remaining).",
            )

        # 3. Controlled attempts limit: if track reached max attempts without match, back off
        attempts = self.track_attempts.get(tid, 0)
        if attempts >= self.max_attempts:
            # Back off for 20 frames before retrying
            self.track_cooldown[tid] = 20
            return IdentityMatchResult(
                tracker_id=tid,
                person_id=None,
                similarity=0.0,
                status=IdentityMatchStatus.UNRESOLVED,
                message="Max recognition attempts reached for this track.",
            )

        # 4. Strict Face Quality Check
        is_reliable, reason = self.embedder.is_face_reliable(face_crop, face_conf, landmarks)
        if not is_reliable:
            # Leave persistent identity unresolved; do not guess on low-quality face
            return IdentityMatchResult(
                tracker_id=tid,
                person_id=None,
                similarity=0.0,
                status=IdentityMatchStatus.UNRELIABLE_FACE,
                message=f"Face quality insufficient: {reason}.",
            )

        # 5. Extract Face Embedding
        self.track_attempts[tid] = attempts + 1
        self.track_cooldown[tid] = self.cooldown_frames

        try:
            embedding = self.embedder.extract_embedding(face_crop, landmarks)
        except Exception as e:
            logger.warning("Failed to extract face embedding for track %s: %s", tid, str(e))
            return IdentityMatchResult(
                tracker_id=tid,
                person_id=None,
                similarity=0.0,
                status=IdentityMatchStatus.UNRESOLVED,
                message=f"Embedding extraction error: {e}",
            )

        self.track_embeddings[tid] = embedding
        self.track_reliable_frames[tid] = self.track_reliable_frames.get(tid, 0) + 1

        # 6. Similarity Search against Stored Identities via pgvector / vector store
        matched_pid, best_score = None, 0.0
        if self.vector_store is not None:
            if hasattr(self.vector_store, "find_best_visitor_match"):
                matched_pid, best_score = self.vector_store.find_best_visitor_match(
                    embedding, similarity_threshold=self.similarity_threshold
                )
            elif hasattr(self.vector_store, "find_best_match"):
                matched_pid, best_score = self.vector_store.find_best_match(embedding)

        if matched_pid is not None:
            # Match exceeds calibrated similarity threshold -> Associate track with existing persistent person ID
            self.associate_identity(tid, matched_pid)
            logger.info(
                "Identity recovered for track %s -> %s (similarity: %.3f >= threshold %.2f)",
                tid,
                matched_pid,
                best_score,
                self.similarity_threshold,
            )

            # Only a reliable identity match triggers the callback to update last_seen_at
            if self.on_matched_callback:
                try:
                    self.on_matched_callback(matched_pid, ts)
                except Exception as e:
                    logger.error("Error in on_matched_callback for %s: %s", matched_pid, str(e))

            return IdentityMatchResult(
                tracker_id=tid,
                person_id=matched_pid,
                similarity=best_score,
                status=IdentityMatchStatus.MATCHED,
                message=f"Reliably matched existing identity {matched_pid} (sim: {best_score:.3f}).",
            )

        # 7. No Match Exceeding Threshold
        # Avoid assuming the visitor is definitely a new person immediately on a single frame or border-line score
        # Calibrated logic:
        # If best_score is in uncertain zone (e.g. 0.38 - threshold), leave unresolved!
        if best_score >= 0.38:
            return IdentityMatchResult(
                tracker_id=tid,
                person_id=None,
                similarity=best_score,
                status=IdentityMatchStatus.UNCERTAIN,
                message=f"Uncertain similarity ({best_score:.3f}); leaving unresolved to prevent false association.",
            )

        # Require at least 2 reliable face observations before creating a new persistent identity
        # to prevent temporary noise / partial profile poster reflections from creating duplicate identities
        if self.track_reliable_frames[tid] < 2:
            return IdentityMatchResult(
                tracker_id=tid,
                person_id=None,
                similarity=best_score,
                status=IdentityMatchStatus.UNRESOLVED,
                message="Awaiting second reliable frame before creating new persistent identity.",
            )

        # Create New Persistent Identity
        new_pid = f"PID_{uuid.uuid4().hex[:8].upper()}"
        if self.vector_store is not None and hasattr(self.vector_store, "add"):
            try:
                self.vector_store.add(new_pid, embedding)
            except Exception:
                pass
        self.associate_identity(tid, new_pid)
        logger.info(
            "Created new persistent identity %s for track %s (max_sim: %.3f)",
            new_pid,
            tid,
            best_score,
        )

        if self.on_created_callback:
            try:
                self.on_created_callback(new_pid, embedding, ts)
            except Exception as e:
                logger.error("Error in on_created_callback for %s: %s", new_pid, str(e))

        return IdentityMatchResult(
            tracker_id=tid,
            person_id=new_pid,
            similarity=best_score,
            status=IdentityMatchStatus.CREATED,
            message=f"Created new persistent identity {new_pid}.",
        )

    def should_count_entry(self, tracker_id: str) -> bool:
        """
        Determines whether a crossing event should count as a new visitor entry.
        Strictly prevents duplicate entry counts if the persistent person has already
        been counted during their visit session.
        """
        tid = str(tracker_id)
        pid = self.track_to_person.get(tid)

        if pid is not None:
            # Persistent identity is known
            if pid in self.counted_persons:
                logger.info(
                    "Duplicate entry suppressed: persistent visitor %s (track %s) already counted.",
                    pid,
                    tid,
                )
                return False
            self.counted_persons.add(pid)
            self.counted_tracks.add(tid)
            return True
        else:
            # Persistent identity unresolved; check temporary track ID
            if tid in self.counted_tracks:
                return False
            self.counted_tracks.add(tid)
            return True

    def on_track_lost(self, tracker_id: str) -> None:
        """
        Cleans up per-track frame cache when a track is lost,
        while maintaining persistent identity mappings and FAISS index.
        """
        tid = str(tracker_id)
        self.track_attempts.pop(tid, None)
        self.track_cooldown.pop(tid, None)
        self.track_embeddings.pop(tid, None)
        self.track_reliable_frames.pop(tid, None)

    def clear_active_session(self) -> None:
        """Resets active session tracking structures."""
        self.track_to_person.clear()
        self.person_to_tracks.clear()
        self.track_attempts.clear()
        self.track_cooldown.clear()
        self.track_embeddings.clear()
        self.track_reliable_frames.clear()
        self.counted_persons.clear()
        self.counted_tracks.clear()

