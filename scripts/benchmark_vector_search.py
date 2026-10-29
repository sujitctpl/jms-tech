import time
import numpy as np
import psycopg
from app.vision.faiss_index import FaissIndexManager

DB_URL = "postgresql://postgres:06nov2014@localhost:5432/jms_live"


def generate_normalized_vectors(n: int, dim: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def run_benchmark():
    n_embeddings = 1000
    n_queries = 100
    dim = 512

    print(f"================================================================")
    print(f"  BENCHMARK: FAISS (FlatIP) vs PostgreSQL pgvector (HNSW)      ")
    print(f"  Vectors: {n_embeddings} (512-dim) | Test Queries: {n_queries}               ")
    print(f"================================================================\n")

    # 1. Generate representative normalized embeddings
    print("[1/5] Generating 1,000 synthetic normalized ArcFace 512-d embeddings...")
    stored_vectors = generate_normalized_vectors(n_embeddings, dim=dim, seed=123)
    pids = [f"PID_{i:04d}" for i in range(n_embeddings)]

    # 2. Benchmark FAISS Index Creation & Insertion
    print("[2/5] Benchmarking FAISS index population...")
    t0 = time.perf_counter()
    faiss_mgr = FaissIndexManager(similarity_threshold=0.50)
    for pid, vec in zip(pids, stored_vectors):
        faiss_mgr.add(pid, vec)
    faiss_build_time = time.perf_counter() - t0
    print(f"      -> FAISS index created in {faiss_build_time*1000:.2f} ms")

    # 3. Benchmark PostgreSQL pgvector Insertion & HNSW Index Creation
    print("[3/5] Benchmarking PostgreSQL pgvector insertion & HNSW indexing...")
    conn = psycopg.connect(DB_URL, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS benchmark_vectors;")
        cur.execute("""
            CREATE TABLE benchmark_vectors (
                id SERIAL PRIMARY KEY,
                person_id VARCHAR(50) NOT NULL UNIQUE,
                embedding vector(512) NOT NULL
            );
        """)

        t0 = time.perf_counter()
        # Batch insert
        records = [(pid, vec.tolist()) for pid, vec in zip(pids, stored_vectors)]
        with cur.copy("COPY benchmark_vectors (person_id, embedding) FROM STDIN") as copy:
            for pid, vec_list in records:
                vec_str = "[" + ",".join(f"{x:.6f}" for x in vec_list) + "]"
                copy.write_row((pid, vec_str))
        pgvector_insert_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        cur.execute("""
            CREATE INDEX idx_benchmark_hnsw 
            ON benchmark_vectors 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        hnsw_build_time = time.perf_counter() - t0
        print(f"      -> pgvector batch insert: {pgvector_insert_time*1000:.2f} ms")
        print(f"      -> HNSW index build time: {hnsw_build_time*1000:.2f} ms")

    # 4. Generate 100 test queries
    print("[4/5] Preparing 100 test queries (exact matches, noisy variants, novel faces)...")
    queries = []
    # 50 near-exact probes
    for i in range(50):
        noise = np.random.randn(dim).astype(np.float32) * 0.03
        q = stored_vectors[i] + noise
        q /= np.linalg.norm(q)
        queries.append(q)
    # 50 random novel visitor probes
    novel = generate_normalized_vectors(50, dim=dim, seed=999)
    for q in novel:
        queries.append(q)

    # 5. Measure Latency & Search Accuracy
    print("[5/5] Executing 100 similarity searches on FAISS and pgvector...")
    faiss_latencies = []
    faiss_results = []
    for q in queries:
        t0 = time.perf_counter()
        res = faiss_mgr.find_best_match(q)
        dt = (time.perf_counter() - t0) * 1000.0
        faiss_latencies.append(dt)
        faiss_results.append(res)

    pgvector_latencies = []
    pgvector_results = []
    with conn.cursor() as cur:
        for q in queries:
            q_list = q.tolist()
            t0 = time.perf_counter()
            cur.execute("""
                SELECT person_id, (embedding <=> %s::vector(512)) AS dist
                FROM benchmark_vectors
                ORDER BY dist ASC
                LIMIT 1;
            """, (q_list,))
            row = cur.fetchone()
            dt = (time.perf_counter() - t0) * 1000.0
            pgvector_latencies.append(dt)
            if row:
                dist = float(row[1])
                sim = round(max(0.0, min(1.0, 1.0 - dist)), 4)
                pid = row[0] if sim >= 0.50 else None
                pgvector_results.append((pid, sim))
            else:
                pgvector_results.append((None, 0.0))

    # Calculate agreement / accuracy
    match_agreement = 0
    total = len(queries)
    for (f_id, f_score), (p_id, p_score) in zip(faiss_results, pgvector_results):
        if f_id == p_id:
            match_agreement += 1

    agreement_pct = (match_agreement / total) * 100.0

    # Clean up benchmark table
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS benchmark_vectors;")
    conn.close()

    # Print Results Table
    print("\n================================================================")
    print("                    BENCHMARK RESULTS SUMMARY                   ")
    print("================================================================")
    print(f"{'Metric':<32} | {'FAISS':<14} | {'pgvector (HNSW)':<16}")
    print("-" * 68)
    print(f"{'Mean Search Latency':<32} | {np.mean(faiss_latencies):>10.3f} ms | {np.mean(pgvector_latencies):>12.3f} ms")
    print(f"{'Median (P50) Latency':<32} | {np.median(faiss_latencies):>10.3f} ms | {np.median(pgvector_latencies):>12.3f} ms")
    print(f"{'95th Percentile (P95)':<32} | {np.percentile(faiss_latencies, 95):>10.3f} ms | {np.percentile(pgvector_latencies, 95):>12.3f} ms")
    print(f"{'99th Percentile (P99)':<32} | {np.percentile(faiss_latencies, 99):>10.3f} ms | {np.percentile(pgvector_latencies, 99):>12.3f} ms")
    print(f"{'Index Build Time (1k vecs)':<32} | {faiss_build_time*1000:>10.2f} ms | {hnsw_build_time*1000:>12.2f} ms")
    print(f"{'Top-1 Match Agreement':<32} | {'100.0% (ref)':<14} | {agreement_pct:>12.1f} %")
    print("================================================================\n")


if __name__ == "__main__":
    run_benchmark()

