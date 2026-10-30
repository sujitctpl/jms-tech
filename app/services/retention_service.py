from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import glob
import os
import threading
import time
from typing import Any, List, Optional
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.db.session import SessionLocal
from app.repositories.person_identity_repo import PersonIdentityRepository
from app.vision.vector_store import PgVectorStoreManager


@dataclass
class RetentionCleanupReport:
    success: bool
    expired_count: int
    retention_days: int
    cutoff_timestamp: datetime
    execution_time_seconds: float
    message: str


class RetentionService:
    """
    Implements a 14-day rolling retention policy for biometric face embeddings.
    Embeddings not reliably matched within 14 days are deleted from the database
    and vector store, while preserving historical footfall and analytics.
    """

    def __init__(
        self,
        db: Optional[Session] = None,
        vector_store: Optional[Any] = None,
        retention_days: Optional[int] = None,
        db_session: Optional[Session] = None,
        faiss_index: Optional[Any] = None,
    ):
        self._db = db or db_session
        self.vector_store = vector_store if vector_store is not None else faiss_index
        # Backward compatibility alias
        self.faiss_index = self.vector_store
        self.retention_days = (
            retention_days if retention_days is not None else settings.FACE_RETENTION_DAYS
        )

    def cleanup_expired_embeddings(
        self,
        db: Optional[Session] = None,
        retention_days: Optional[int] = None,
        now: Optional[datetime] = None,
        face_crops_dir: Optional[str] = None,
    ) -> RetentionCleanupReport:
        """
        Executes an idempotent biometric cleanup job.
        Deletes face embeddings older than retention_days (default 14 days from last_seen_at)
        from both the database and vector store.
        """
        start_time = time.time()
        active_db = db or self._db
        close_on_exit = False

        if active_db is None:
            active_db = SessionLocal()
            close_on_exit = True

        days = retention_days if retention_days is not None else self.retention_days
        current_time = now or datetime.now(timezone.utc)
        cutoff = current_time - timedelta(days=days)

        try:
            repo = PersonIdentityRepository(active_db)
            expired_records = repo.get_expired_identities(cutoff)

            if not expired_records:
                elapsed = time.time() - start_time
                logger.info(
                    "Retention cleanup: 0 expired embeddings found (cutoff: %s, policy: %d days).",
                    cutoff.isoformat(),
                    days,
                )
                return RetentionCleanupReport(
                    success=True,
                    expired_count=0,
                    retention_days=days,
                    cutoff_timestamp=cutoff,
                    execution_time_seconds=round(elapsed, 4),
                    message="No expired face embeddings found.",
                )

            expired_person_ids: List[str] = [r.person_id for r in expired_records]

            # 1. Remove from vector index if an external in-memory index is attached
            if self.vector_store is not None:
                if hasattr(self.vector_store, "remove_ids"):
                    try:
                        deleted_count = self.vector_store.remove_ids(expired_person_ids)
                        logger.info("Removed %d expired vectors from vector store.", deleted_count)
                    except Exception as e:
                        logger.error("Failed to remove expired IDs from vector store: %s", str(e))
                elif hasattr(self.vector_store, "remove"):
                    for pid in expired_person_ids:
                        try:
                            self.vector_store.remove(pid)
                        except Exception:
                            pass

            # 2. Purge associated biometric files (face crops / cached embeddings on disk)
            if face_crops_dir and os.path.exists(face_crops_dir):
                self._purge_face_crops(face_crops_dir, expired_person_ids)

            # 3. Purge biometric embeddings from database (sets face_embedding = None, is_expired = True)
            purged_count = repo.expire_identities(expired_person_ids, expired_at=current_time)

            elapsed = time.time() - start_time
            logger.info(
                "Biometric retention cleanup completed successfully: purged %d expired embeddings (retention: %d days, cutoff: %s).",
                purged_count,
                days,
                cutoff.isoformat(),
            )

            return RetentionCleanupReport(
                success=True,
                expired_count=purged_count,
                retention_days=days,
                cutoff_timestamp=cutoff,
                execution_time_seconds=round(elapsed, 4),
                message=f"Purged {purged_count} expired face embeddings safely.",
            )

        except Exception as e:
            active_db.rollback()
            elapsed = time.time() - start_time
            logger.error(
                "Retention cleanup failed during database transaction: %s",
                str(e),
                exc_info=True,
            )
            return RetentionCleanupReport(
                success=False,
                expired_count=0,
                retention_days=days,
                cutoff_timestamp=cutoff,
                execution_time_seconds=round(elapsed, 4),
                message=f"Cleanup error: {e}",
            )
        finally:
            if close_on_exit:
                active_db.close()

    @staticmethod
    def _purge_face_crops(directory: str, person_ids: List[str]) -> int:
        """Removes local cached face crops associated with expired identities."""
        deleted = 0
        for pid in person_ids:
            pattern = os.path.join(directory, f"*{pid}*")
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    deleted += 1
                except Exception:
                    pass
        return deleted


class RetentionScheduler:
    """
    Background daemon running daily automated biometric retention cleanup.
    """

    def __init__(
        self,
        service: Optional[RetentionService] = None,
        interval_hours: Optional[int] = None,
    ):
        self.service = service or RetentionService()
        self.interval_seconds = (
            (interval_hours or settings.RETENTION_CLEANUP_INTERVAL_HOURS) * 3600
        )
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="RetentionScheduler")
        self.thread.start()
        logger.info(
            "RetentionScheduler started (interval: %d hours).",
            self.interval_seconds // 3600,
        )

    def stop(self) -> None:
        if self.thread:
            self.stop_event.set()
            self.thread.join(timeout=2.0)
            logger.info("RetentionScheduler stopped.")

    def _run_loop(self) -> None:
        # Initial run shortly after startup (10s delay)
        if not self.stop_event.wait(10.0):
            try:
                self.service.cleanup_expired_embeddings()
            except Exception as e:
                logger.error("Error during initial retention cleanup: %s", str(e))

        # Recurring daily cycle
        while not self.stop_event.is_set():
            if self.stop_event.wait(self.interval_seconds):
                break
            try:
                self.service.cleanup_expired_embeddings()
            except Exception as e:
                logger.error("Error during scheduled retention cleanup: %s", str(e))


# Global singleton scheduler instance
retention_scheduler = RetentionScheduler()

