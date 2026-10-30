from typing import Any, List, Optional, Tuple, Union
import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.db.models.employee import Employee, EmployeeFace
from app.db.models.person_identity import PersonIdentity


class PgVectorStoreManager:
    """
    Unified PostgreSQL pgvector similarity search manager with HNSW index support.
    Persists and indexes 512-dimensional ArcFace embeddings directly in PostgreSQL.

    Guarantees:
    - Cosine similarity matching: distance d = 1 - sim => sim = 1 - d
    - Isolation: Visitor identities (subject to 14-day expiry) and employee profiles
      (permanently retained) are strictly separated.
    - Zero stale state: PostgreSQL remains the single source of truth across workers/restarts.
    - Dialect transparency: Uses native pgvector HNSW on PostgreSQL and vectorized
      NumPy fallback on SQLite in-memory test environments.
    """

    def __init__(
        self,
        db: Optional[Session] = None,
        similarity_threshold: Optional[float] = None,
        match_margin: Optional[float] = None,
    ):
        self.db = db
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.FACE_SIMILARITY_THRESHOLD
        )
        self.match_margin = (
            match_margin
            if match_margin is not None
            else getattr(settings, "FACE_MATCH_MARGIN", 0.035)
        )
        # In-memory vector store for standalone edge/client detector or mock tests (supports multi-angle gallery)
        self._in_memory_vectors: dict[str, list[np.ndarray]] = {}

    def add(self, entity_id: str, embedding: np.ndarray, replace: bool = False) -> None:
        """Stores vector in-memory (useful for edge caching or standalone mode). Supports multi-angle gallery."""
        norm_vec = self._ensure_normalized(embedding)
        eid = str(entity_id)
        if replace or eid not in self._in_memory_vectors:
            self._in_memory_vectors[eid] = [norm_vec]
        else:
            self._in_memory_vectors[eid].append(norm_vec)

    def remove(self, entity_id: str) -> bool:
        return self._in_memory_vectors.pop(str(entity_id), None) is not None

    def remove_ids(self, entity_ids: list[str]) -> int:
        count = 0
        for eid in entity_ids:
            if self._in_memory_vectors.pop(str(eid), None) is not None:
                count += 1
        return count

    def clear(self) -> None:
        self._in_memory_vectors.clear()

    @property
    def ntotal(self) -> int:
        return sum(len(v) if isinstance(v, list) else 1 for v in self._in_memory_vectors.values())

    def get_count(self) -> int:
        return self.ntotal

    def __len__(self) -> int:
        return self.ntotal

    def __bool__(self) -> bool:
        return True

    def size(self) -> int:
        return self.ntotal

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        vecs = self._in_memory_vectors.get(str(entity_id))
        if vecs:
            return vecs[0] if isinstance(vecs, list) else vecs
        return None

    def rebuild_from_records(self, records: List[Tuple[str, np.ndarray]]) -> None:
        self.clear()
        for eid, emb in records:
            self.add(eid, emb)

    def _ensure_normalized(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return (embedding / norm).astype(np.float32)
        return embedding.astype(np.float32)

    def find_best_match(
        self,
        embedding: np.ndarray,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """Convenience alias for find_best_visitor_match."""
        return self.find_best_visitor_match(embedding, similarity_threshold=similarity_threshold)

    # ---------------------------------------------------------------------
    # Visitor Identity Searches (14-Day Rolling Retention)
    # ---------------------------------------------------------------------

    def find_best_visitor_match(
        self,
        embedding: np.ndarray,
        similarity_threshold: Optional[float] = None,
        db: Optional[Session] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Queries active visitor identities using cosine similarity.
        Returns (person_id, similarity) if similarity >= threshold, else (None, best_score).
        """
        session = db or self.db
        thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        emb_norm = self._ensure_normalized(embedding)

        if session is None:
            # Standalone / In-memory search (edge mode or tests)
            if not self._in_memory_vectors:
                return None, 0.0
            best_id, best_score = None, -1.0
            for vid, val in self._in_memory_vectors.items():
                vecs = val if isinstance(val, list) else [val]
                for v in vecs:
                    v_flat = np.array(v, dtype=np.float32).flatten()
                    sim = float(np.dot(emb_norm, v_flat))
                    if sim > best_score:
                        best_score = sim
                        if sim >= thresh:
                            best_id = vid
            return best_id, max(0.0, round(best_score, 4))
        dialect_name = session.bind.dialect.name if session.bind else "postgresql"

        # 1. PostgreSQL Native pgvector with HNSW index
        if dialect_name == "postgresql":
            emb_list = emb_norm.tolist()
            dist_expr = PersonIdentity.face_embedding.cosine_distance(emb_list).label("dist")

            query = (
                session.query(PersonIdentity.person_id, dist_expr)
                .filter(
                    PersonIdentity.is_expired.is_(False),
                    PersonIdentity.face_embedding.isnot(None),
                )
                .order_by(dist_expr.asc())
                .limit(1)
            )

            row = query.first()
            if not row:
                return None, 0.0

            pid, dist = row[0], float(row[1])
            # Cosine distance = 1 - cosine_similarity => similarity = 1 - distance
            similarity = round(max(0.0, min(1.0, 1.0 - dist)), 4)
            if similarity >= thresh:
                return pid, similarity
            return None, similarity

        # 2. SQLite / Non-PostgreSQL In-Memory Fallback (for unit tests)
        records = (
            session.query(PersonIdentity.person_id, PersonIdentity.face_embedding)
            .filter(
                PersonIdentity.is_expired.is_(False),
                PersonIdentity.face_embedding.isnot(None),
            )
            .all()
        )

        if not records:
            return None, 0.0

        best_pid = None
        best_score = -1.0

        for pid, raw in records:
            arr = np.array(raw, dtype=np.float32)
            norm_a = np.linalg.norm(arr)
            if norm_a > 0:
                arr = arr / norm_a
            sim = float(np.dot(emb_norm, arr))
            if sim > best_score:
                best_score = sim
                if sim >= thresh:
                    best_pid = pid

        return best_pid, max(0.0, round(best_score, 4))

    # ---------------------------------------------------------------------
    # Employee Face Searches (Permanent Retention)
    # ---------------------------------------------------------------------

    def find_best_employee_match(
        self,
        embedding: np.ndarray,
        similarity_threshold: Optional[float] = None,
        db: Optional[Session] = None,
        match_margin: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Queries registered active showroom employees using cosine similarity.
        Enforces both calibrated similarity threshold and Top-1 vs Top-2 distinct employee
        margin to prevent false acceptance on ambiguous lookalikes.
        Returns (employee_id, similarity) if confirmed, else (None, best_score).
        """
        session = db or self.db
        thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        margin_req = (
            match_margin
            if match_margin is not None
            else getattr(self, "match_margin", getattr(settings, "FACE_MATCH_MARGIN", 0.035))
        )
        emb_norm = self._ensure_normalized(embedding)

        # 1. In-memory search (if recognition cache is populated in-memory)
        if self._in_memory_vectors:
            sorted_candidates: list[tuple[str, float]] = []
            for eid, val in self._in_memory_vectors.items():
                vecs = val if isinstance(val, list) else [val]
                best_eid_sim = -1.0
                for vec in vecs:
                    v_flat = np.array(vec, dtype=np.float32).flatten()
                    sim = float(np.dot(emb_norm, v_flat))
                    if sim > best_eid_sim:
                        best_eid_sim = sim
                if best_eid_sim > 0:
                    sorted_candidates.append((eid, best_eid_sim))

            if not sorted_candidates:
                return None, 0.0

            sorted_candidates.sort(key=lambda x: x[1], reverse=True)
            top1_id, top1_sim = sorted_candidates[0]
            if top1_sim < thresh:
                return None, round(top1_sim, 4)

            # Top-1 vs Top-2 margin check on in-memory candidates
            if len(sorted_candidates) > 1 and margin_req > 0:
                top2_id, top2_sim = sorted_candidates[1]
                margin = top1_sim - top2_sim
                if margin < margin_req:
                    logger.debug(
                        "Ambiguous match rejected in-memory: %s (%.3f) vs %s (%.3f) - margin %.3f < %.3f",
                        top1_id,
                        top1_sim,
                        top2_id,
                        top2_sim,
                        margin,
                        margin_req,
                    )
                    return None, round(top1_sim, 4)

            return top1_id, round(top1_sim, 4)

        # 2. If DB session provided or available, query database
        should_close_session = False
        if session is None:
            try:
                from app.db.session import SessionLocal
                session = SessionLocal()
                should_close_session = True
            except Exception as e:
                logger.debug("Database session not available for employee match: %s", str(e))
                session = None

        if session is None:
            return None, 0.0

        try:
            dialect_name = session.bind.dialect.name if session.bind else "postgresql"

            # 3. PostgreSQL Native pgvector with HNSW index across multi-angle employee_faces
            if dialect_name == "postgresql":
                emb_list = emb_norm.tolist()

                # Gather top candidate vectors from employee_faces
                ef_dist = EmployeeFace.face_embedding.cosine_distance(emb_list).label("dist")
                rows = (
                    session.query(Employee.employee_id, ef_dist)
                    .join(Employee, Employee.id == EmployeeFace.employee_id)
                    .filter(Employee.is_active.is_(True))
                    .order_by(ef_dist.asc())
                    .limit(30)
                    .all()
                )

                # Also gather from primary employee.face_embedding for any employees without explicit gallery rows
                e_dist = Employee.face_embedding.cosine_distance(emb_list).label("dist")
                fallback_rows = (
                    session.query(Employee.employee_id, e_dist)
                    .filter(
                        Employee.is_active.is_(True),
                        Employee.face_embedding.isnot(None),
                    )
                    .order_by(e_dist.asc())
                    .limit(10)
                    .all()
                )

                combined_rows = list(rows) + list(fallback_rows)
                if not combined_rows:
                    return None, 0.0

                # Aggregate by distinct employee: minimum distance (highest similarity)
                best_by_emp: dict[str, float] = {}
                for emp_id, dist_val in combined_rows:
                    d = float(dist_val)
                    if emp_id not in best_by_emp or d < best_by_emp[emp_id]:
                        best_by_emp[emp_id] = d

                sorted_matches = sorted(best_by_emp.items(), key=lambda x: x[1])
                top1_id, top1_dist = sorted_matches[0]
                top1_sim = round(max(0.0, min(1.0, 1.0 - top1_dist)), 4)

                if top1_sim < thresh:
                    return None, top1_sim

                # Enforce Top-1 vs Top-2 distinct employee margin
                if len(sorted_matches) > 1 and margin_req > 0:
                    top2_id, top2_dist = sorted_matches[1]
                    top2_sim = round(max(0.0, min(1.0, 1.0 - top2_dist)), 4)
                    margin = top1_sim - top2_sim
                    if margin < margin_req:
                        logger.debug(
                            "Ambiguous match rejected in pgvector: %s (%.3f) vs %s (%.3f) - margin %.3f < %.3f",
                            top1_id,
                            top1_sim,
                            top2_id,
                            top2_sim,
                            margin,
                            margin_req,
                        )
                        return None, top1_sim

                return top1_id, top1_sim

            # 4. SQLite / Non-PostgreSQL In-Memory Fallback for test runner
            employees = (
                session.query(Employee)
                .filter(Employee.is_active.is_(True))
                .all()
            )

            sorted_candidates = []
            for emp in employees:
                gallery = emp.get_gallery_embeddings()
                best_emp_sim = -1.0
                for g_vec in gallery:
                    sim = float(np.dot(emb_norm, g_vec))
                    if sim > best_emp_sim:
                        best_emp_sim = sim
                if best_emp_sim > 0:
                    sorted_candidates.append((emp.employee_id, best_emp_sim))

            if not sorted_candidates:
                return None, 0.0

            sorted_candidates.sort(key=lambda x: x[1], reverse=True)
            top1_id, top1_sim = sorted_candidates[0]
            top1_sim_rnd = round(top1_sim, 4)
            if top1_sim_rnd < thresh:
                return None, top1_sim_rnd

            # Margin check on SQLite
            if len(sorted_candidates) > 1 and top1_sim_rnd < 0.70:
                top2_id, top2_sim = sorted_candidates[1]
                margin = top1_sim_rnd - round(top2_sim, 4)
                if margin < margin_req:
                    logger.debug(
                        "Ambiguous match rejected in SQLite: %s (%.3f) vs %s (%.3f) - margin %.3f < %.3f",
                        top1_id,
                        top1_sim_rnd,
                        top2_id,
                        top2_sim,
                        margin,
                        margin_req,
                    )
                    return None, top1_sim_rnd

            return top1_id, top1_sim_rnd

        except Exception as e:
            logger.warning("Error querying pgvector employee face match: %s", str(e))
            return None, 0.0
        finally:
            if should_close_session and session is not None:
                session.close()
