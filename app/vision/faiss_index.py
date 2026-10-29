import threading
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from app.core.config import settings
from app.core.logging import logger

try:
    import faiss
except ImportError:
    faiss = None


class FaissIndexManager:
    """
    FAISS similarity search and index lifecycle manager.
    Stores 512-dimensional L2-normalized ArcFace embeddings.
    Uses Inner Product (Cosine Similarity) metric.
    Safely handles deletion and index rebuilding across index types.
    """

    def __init__(
        self,
        dimension: int = 512,
        similarity_threshold: Optional[float] = None,
    ):
        self.dimension = dimension
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.FACE_SIMILARITY_THRESHOLD
        )
        self.lock = threading.RLock()

        # Bidirectional ID mappings: int_id <-> person_id
        self._next_id: int = 1
        self._int_to_person: Dict[int, str] = {}
        self._person_to_int: Dict[str, int] = {}
        self._person_to_ints: Dict[str, Set[int]] = {}
        # In-memory vector store for safe index rebuilding and fast retrieval
        self._vectors: Dict[int, np.ndarray] = {}

        # Underlying FAISS index
        self._index = None
        self._using_native_faiss: bool = False
        self._init_index()

    def _init_index(self) -> None:
        """Initializes native FAISS IndexFlatIP + IndexIDMap2 or fallback."""
        if faiss is not None:
            try:
                base_index = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIDMap2(base_index)
                self._using_native_faiss = True
                logger.info("Native FAISS IndexIDMap2(IndexFlatIP) initialized successfully.")
                return
            except Exception as e:
                logger.warning("Failed to initialize native FAISS: %s. Using NumPy vector fallback.", str(e))

        self._using_native_faiss = False
        self._index = None

    @property
    def is_native_faiss(self) -> bool:
        return self._using_native_faiss

    def size(self) -> int:
        with self.lock:
            return len(self._vectors)

    def add(self, person_id: str, embedding: np.ndarray, replace: bool = True) -> int:
        """
        Adds an L2-normalized embedding for a persistent person_id.
        If replace is True (default for single-photo enrollment), removes previous vectors for this person.
        If replace is False (for multi-angle gallery), appends the vector to the person's gallery.
        Returns the assigned integer ID in the index.
        """
        with self.lock:
            vec = embedding.flatten().astype(np.float32)
            # Ensure unit length
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            # If replace is requested and person already exists, remove previous vector(s)
            if replace and (person_id in self._person_to_int or person_id in self._person_to_ints):
                self.remove(person_id)

            int_id = self._next_id
            self._next_id += 1

            self._int_to_person[int_id] = person_id
            self._person_to_int[person_id] = int_id
            self._person_to_ints.setdefault(person_id, set()).add(int_id)
            self._vectors[int_id] = vec

            if self._using_native_faiss and self._index is not None:
                try:
                    vec_2d = np.ascontiguousarray(np.expand_dims(vec, axis=0))
                    ids_array = np.array([int_id], dtype=np.int64)
                    self._index.add_with_ids(vec_2d, ids_array)
                except Exception as e:
                    logger.warning("Error adding to native FAISS index: %s. Rebuilding index.", str(e))
                    self._rebuild_native_index()

            return int_id

    def add_gallery(self, person_id: str, embeddings: List[np.ndarray]) -> List[int]:
        """
        Adds multiple gallery angle embeddings for an employee.
        Removes any previously stored embeddings for this employee first.
        """
        with self.lock:
            self.remove(person_id)
            ids = []
            for emb in embeddings:
                ids.append(self.add(person_id, emb, replace=False))
            return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Performs inner-product (cosine similarity) search for the query embedding.
        Returns a list of (person_id, similarity_score) sorted in descending order.
        """
        with self.lock:
            if not self._vectors:
                return []

            vec = query_embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            if self._using_native_faiss and self._index is not None and self._index.ntotal > 0:
                try:
                    query_2d = np.ascontiguousarray(np.expand_dims(vec, axis=0))
                    # Retrieve more internal candidates to allow gallery deduplication
                    search_k = min(max(top_k * 4, 10), self._index.ntotal)
                    distances, indices = self._index.search(query_2d, search_k)

                    seen_scores: Dict[str, float] = {}
                    for sim, idx in zip(distances[0], indices[0]):
                        int_id = int(idx)
                        if int_id in self._int_to_person:
                            pid = self._int_to_person[int_id]
                            score_f = float(sim)
                            if pid not in seen_scores or score_f > seen_scores[pid]:
                                seen_scores[pid] = score_f

                    results = sorted(seen_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                    return results
                except Exception as e:
                    logger.warning("Error during native FAISS search: %s. Falling back to NumPy search.", str(e))

            # NumPy-based high-performance inner-product matrix multiplication
            int_ids = list(self._vectors.keys())
            matrix = np.stack([self._vectors[i] for i in int_ids])  # (N, 512)
            scores = np.dot(matrix, vec)  # (N,)

            # Deduplicate by person_id keeping maximum similarity across multi-angle gallery
            seen_scores_np: Dict[str, float] = {}
            for int_id, score in zip(int_ids, scores):
                pid = self._int_to_person[int_id]
                score_f = float(score)
                if pid not in seen_scores_np or score_f > seen_scores_np[pid]:
                    seen_scores_np[pid] = score_f

            results = sorted(seen_scores_np.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return results

    def find_best_match(
        self,
        query_embedding: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Finds the single best match exceeding the calibrated similarity threshold.
        Returns (person_id, similarity) if match exceeds threshold, otherwise (None, best_similarity).
        """
        results = self.search(query_embedding, top_k=1)
        if not results:
            return None, 0.0

        best_person_id, best_score = results[0]
        if best_score >= self.similarity_threshold:
            return best_person_id, best_score
        return None, best_score

    def remove(self, person_id: str) -> bool:
        """
        Safely deletes all embeddings for a person (including multi-angle gallery)
        from the FAISS index and mappings.
        """
        with self.lock:
            int_ids = set()
            if person_id in self._person_to_ints:
                int_ids.update(self._person_to_ints.pop(person_id))
            single_id = self._person_to_int.pop(person_id, None)
            if single_id is not None:
                int_ids.add(single_id)

            if not int_ids:
                return False

            for int_id in int_ids:
                self._int_to_person.pop(int_id, None)
                self._vectors.pop(int_id, None)

            if self._using_native_faiss and self._index is not None:
                try:
                    # Native remove_ids using IDSelector
                    id_selector = faiss.IDSelectorBatch(list(int_ids))
                    self._index.remove_ids(id_selector)
                except Exception as e:
                    logger.info("Direct remove_ids not supported or failed (%s); safely rebuilding index.", str(e))
                    self._rebuild_native_index()

            return True

    def remove_ids(self, person_ids: List[str]) -> int:
        """
        Batch deletes expired embeddings from the FAISS index.
        Returns the number of embeddings successfully deleted.
        """
        with self.lock:
            deleted_count = 0
            int_ids_to_remove = []

            for pid in person_ids:
                if pid in self._person_to_ints:
                    p_ids = self._person_to_ints.pop(pid)
                    int_ids_to_remove.extend(p_ids)
                single_id = self._person_to_int.pop(pid, None)
                if single_id is not None and single_id not in int_ids_to_remove:
                    int_ids_to_remove.append(single_id)

            for int_id in int_ids_to_remove:
                if self._int_to_person.pop(int_id, None) is not None:
                    deleted_count += 1
                self._vectors.pop(int_id, None)

            if not int_ids_to_remove:
                return 0

            if self._using_native_faiss and self._index is not None:
                try:
                    id_selector = faiss.IDSelectorBatch(int_ids_to_remove)
                    self._index.remove_ids(id_selector)
                except Exception:
                    self._rebuild_native_index()

            return deleted_count

    def _rebuild_native_index(self) -> None:
        """Safely reconstructs the native FAISS index from the active vector store."""
        if faiss is None:
            return

        try:
            base_index = faiss.IndexFlatIP(self.dimension)
            new_index = faiss.IndexIDMap2(base_index)

            if self._vectors:
                int_ids = np.array(list(self._vectors.keys()), dtype=np.int64)
                vectors = np.stack(list(self._vectors.values())).astype(np.float32)
                vectors = np.ascontiguousarray(vectors)
                new_index.add_with_ids(vectors, int_ids)

            self._index = new_index
            logger.info("Rebuilt native FAISS index successfully with %d vectors.", len(self._vectors))
        except Exception as e:
            logger.error("Failed to rebuild native FAISS index: %s", str(e), exc_info=True)

    def rebuild_from_records(self, records: List[Tuple[str, np.ndarray]]) -> None:
        """
        Replaces index contents with provided (person_id, embedding) records.
        """
        with self.lock:
            self.clear()
            for pid, emb in records:
                self.add(pid, emb)

    def clear(self) -> None:
        """Clears all embeddings and ID mappings."""
        with self.lock:
            self._vectors.clear()
            self._int_to_person.clear()
            self._person_to_int.clear()
            self._person_to_ints.clear()
            self._next_id = 1
            self._init_index()

    def get_embedding(self, person_id: str) -> Optional[np.ndarray]:
        with self.lock:
            int_id = self._person_to_int.get(person_id)
            if int_id is not None:
                return self._vectors.get(int_id)
            return None

    @property
    def ntotal(self) -> int:
        with self.lock:
            return len(self._vectors)

    def get_count(self) -> int:
        return self.ntotal

    def __len__(self) -> int:
        return self.ntotal

