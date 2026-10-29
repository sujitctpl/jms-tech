from datetime import datetime, timezone
from typing import List, Optional, Tuple
import uuid
import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.db.models.person_identity import PersonIdentity
from app.vision.vector_store import PgVectorStoreManager


class PersonIdentityRepository:
    def __init__(self, db: Session):
        self.db = db
        self.vector_manager = PgVectorStoreManager(db)

    def get_by_person_id(self, person_id: str) -> Optional[PersonIdentity]:
        return (
            self.db.query(PersonIdentity)
            .filter(PersonIdentity.person_id == str(person_id))
            .first()
        )

    def get_by_id(self, identity_id: uuid.UUID) -> Optional[PersonIdentity]:
        return (
            self.db.query(PersonIdentity)
            .filter(PersonIdentity.id == identity_id)
            .first()
        )

    def find_best_match(
        self,
        embedding: np.ndarray,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """Queries pgvector with HNSW index for the most similar active visitor face."""
        return self.vector_manager.find_best_visitor_match(
            embedding=embedding,
            similarity_threshold=similarity_threshold,
            db=self.db,
        )

    def create_or_update(
        self,
        person_id: str,
        embedding: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> PersonIdentity:
        ts = timestamp or datetime.now(timezone.utc)
        vec_data = embedding.astype(np.float32).tolist()

        try:
            record = self.get_by_person_id(person_id)
            if record:
                record.face_embedding = vec_data
                record.last_seen_at = ts
                record.is_expired = False
                record.expired_at = None
            else:
                record = PersonIdentity(
                    person_id=str(person_id),
                    face_embedding=vec_data,
                    created_at=ts,
                    last_seen_at=ts,
                    is_expired=False,
                )
                self.db.add(record)

            self.db.commit()
            self.db.refresh(record)
            return record
        except Exception:
            self.db.rollback()
            record = self.get_by_person_id(person_id)
            if record:
                record.face_embedding = vec_data
                record.last_seen_at = ts
                record.is_expired = False
                record.expired_at = None
                self.db.add(record)
                self.db.commit()
                self.db.refresh(record)
                return record
            raise

    def update_last_seen(
        self,
        person_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[PersonIdentity]:
        """
        Updates last_seen_at for a reliably recognized visitor.
        Only reliable face matches should invoke this method.
        """
        ts = timestamp or datetime.now(timezone.utc)
        try:
            record = self.get_by_person_id(person_id)
            if record:
                record.last_seen_at = ts
                self.db.add(record)
                self.db.commit()
                self.db.refresh(record)
            return record
        except Exception:
            self.db.rollback()
            raise

    def get_active_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """
        Retrieves all unexpired identities and their decoded float32 embeddings.
        """
        records = (
            self.db.query(PersonIdentity)
            .filter(
                PersonIdentity.is_expired.is_(False),
                PersonIdentity.face_embedding.isnot(None),
            )
            .all()
        )

        results = []
        for r in records:
            arr = r.get_embedding_array()
            if arr is not None:
                results.append((r.person_id, arr))
        return results

    def get_expired_identities(self, cutoff: datetime) -> List[PersonIdentity]:
        """
        Returns identities whose last_seen_at is older than the cutoff timestamp
        and whose embeddings have not yet been purged.
        """
        return (
            self.db.query(PersonIdentity)
            .filter(
                PersonIdentity.last_seen_at < cutoff,
                PersonIdentity.face_embedding.isnot(None),
            )
            .all()
        )

    def expire_identities(
        self,
        person_ids: List[str],
        expired_at: Optional[datetime] = None,
    ) -> int:
        """
        Permanently purges face embeddings for specified person IDs.
        Sets face_embedding = None and is_expired = True while preserving person records
        and associated historical tracking events.
        """
        if not person_ids:
            return 0

        ts = expired_at or datetime.now(timezone.utc)
        records = (
            self.db.query(PersonIdentity)
            .filter(PersonIdentity.person_id.in_(person_ids))
            .all()
        )

        count = 0
        for r in records:
            r.face_embedding = None
            r.is_expired = True
            r.expired_at = ts
            count += 1

        self.db.commit()
        return count

