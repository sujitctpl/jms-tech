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
        rec = (
            self.db.query(PersonIdentity)
            .filter(PersonIdentity.person_id == str(person_id))
            .first()
        )
        if not rec:
            try:
                uid = uuid.UUID(str(person_id))
                rec = self.db.query(PersonIdentity).filter(PersonIdentity.id == uid).first()
            except (ValueError, TypeError, AttributeError):
                pass
        return rec

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
        only_active: bool = False,
    ) -> Tuple[Optional[str], float]:
        """Queries pgvector with HNSW index for the most similar visitor face."""
        return self.vector_manager.find_best_visitor_match(
            embedding=embedding,
            similarity_threshold=similarity_threshold,
            db=self.db,
            only_active=only_active,
        )

    def create_or_update(
        self,
        person_id: str,
        embedding: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
        avatar_url: Optional[str] = None,
        camera_source: Optional[str] = None,
        zone_entered: Optional[str] = None,
        notes: Optional[str] = None,
        visit_purpose: Optional[str] = None,
        assigned_salesperson: Optional[str] = None,
    ) -> PersonIdentity:
        ts = timestamp or datetime.now(timezone.utc)
        vec_data = embedding.astype(np.float32).tolist() if embedding is not None else None

        try:
            record = self.get_by_person_id(person_id)
            if record:
                if vec_data is not None:
                    record.face_embedding = vec_data
                record.last_seen_at = ts
                record.is_expired = False
                record.expired_at = None
                if avatar_url and not record.avatar_url:
                    record.avatar_url = avatar_url
                if camera_source:
                    record.camera_source = camera_source
                if zone_entered:
                    record.zone_entered = zone_entered
                if visit_purpose:
                    record.visit_purpose = visit_purpose
                    record.tagging_status = "Tagged"
                if assigned_salesperson:
                    record.assigned_salesperson = assigned_salesperson
                if notes:
                    record.notes = notes
            else:
                tag_status = "Tagged" if visit_purpose else "Untagged"
                record = PersonIdentity(
                    person_id=str(person_id),
                    face_embedding=vec_data,
                    created_at=ts,
                    last_seen_at=ts,
                    is_expired=False,
                    avatar_url=avatar_url,
                    camera_source=camera_source or "Entrance Camera",
                    zone_entered=zone_entered or "Main Showroom",
                    notes=notes or "Walk-in visitor tracked via CCTV AI Detection Line",
                    visit_purpose=visit_purpose,
                    assigned_salesperson=assigned_salesperson,
                    tagging_status=tag_status,
                    visitor_type="New",
                )
                self.db.add(record)

            self.db.commit()
            self.db.refresh(record)
            return record
        except Exception:
            self.db.rollback()
            record = self.get_by_person_id(person_id)
            if record:
                if vec_data is not None:
                    record.face_embedding = vec_data
                record.last_seen_at = ts
                record.is_expired = False
                record.expired_at = None
                if avatar_url and not record.avatar_url:
                    record.avatar_url = avatar_url
                self.db.add(record)
                self.db.commit()
                self.db.refresh(record)
                return record
            raise

    def list_visitors(
        self,
        tagging_status: Optional[str] = None,
        visit_purpose: Optional[str] = None,
        limit: int = 200,
    ) -> List[PersonIdentity]:
        query = self.db.query(PersonIdentity)
        if tagging_status and tagging_status.lower() != "all":
            query = query.filter(PersonIdentity.tagging_status == tagging_status)
        if visit_purpose and visit_purpose.lower() != "all":
            query = query.filter(PersonIdentity.visit_purpose == visit_purpose)
        return query.order_by(PersonIdentity.last_seen_at.desc()).limit(limit).all()

    def update_visitor_tag(
        self,
        person_id: str,
        visit_purpose: str,
        assigned_salesperson: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[PersonIdentity]:
        record = self.get_by_person_id(person_id)
        if not record:
            return None
        record.visit_purpose = visit_purpose
        if assigned_salesperson is not None:
            record.assigned_salesperson = assigned_salesperson
        if notes is not None:
            record.notes = notes
        record.tagging_status = "Tagged"
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

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
                PersonIdentity.exit_time.is_(None),
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

