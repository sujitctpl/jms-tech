"""Migrate face embeddings from BYTEA/FAISS to pgvector with HNSW index

Revision ID: 0004_pgvector_migration
Revises: 0003_employee_attendance
Create Date: 2026-10-29 12:45:00.000000

"""
import os
from typing import Sequence, Union
from alembic import op
import numpy as np
import sqlalchemy as sa
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision: str = "0004_pgvector_migration"
down_revision: Union[str, None] = "0003_employee_attendance"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ensure_pgvector_installed(conn):
    """Ensures vector type and HNSW access method are registered in PostgreSQL."""
    # Check if vector type already exists
    res = conn.execute(text("SELECT typname FROM pg_type WHERE typname = 'vector';")).fetchall()
    if res:
        return

    # Attempt standard CREATE EXTENSION first
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        return
    except Exception:
        pass

    # If extension control file is not in system directory, load vector SQL definitions
    temp_dir = os.environ.get("TEMP", "C:/Users/capermint/AppData/Local/Temp")
    vector_sql_path = os.path.join(temp_dir, "vector_pg18_clean", "share", "extension", "vector--0.8.7.sql")
    if os.path.exists(vector_sql_path):
        with open(vector_sql_path, "r", encoding="utf-8") as f:
            sql_content = f.read()
        lines = [l for l in sql_content.splitlines() if not l.startswith("\\")]
        cleaned = "\n".join(lines).replace("MODULE_PATHNAME", "C:/Program Files/PostgreSQL/18/data/vector.dll")
        conn.execute(text(cleaned))


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        _ensure_pgvector_installed(bind)

        # 1. Migrate person_identities.face_embedding -> vector(512)
        # Add new vector column
        op.execute(text("ALTER TABLE person_identities ADD COLUMN IF NOT EXISTS face_embedding_vec vector(512);"))

        # Convert existing rows from bytea to vector if any exist
        existing_rows = bind.execute(
            text("SELECT id, face_embedding FROM person_identities WHERE face_embedding IS NOT NULL;")
        ).fetchall()
        for row_id, raw_bytes in existing_rows:
            if raw_bytes:
                arr = np.frombuffer(raw_bytes, dtype=np.float32)
                vec_str = "[" + ",".join(str(float(x)) for x in arr) + "]"
                bind.execute(
                    text("UPDATE person_identities SET face_embedding_vec = :v::vector WHERE id = :id"),
                    {"v": vec_str, "id": row_id},
                )

        # Swap columns
        op.execute(text("ALTER TABLE person_identities DROP COLUMN face_embedding;"))
        op.execute(text("ALTER TABLE person_identities RENAME COLUMN face_embedding_vec TO face_embedding;"))

        # Create HNSW index for high-performance sub-millisecond cosine similarity search
        op.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_person_identities_hnsw ON person_identities "
            "USING hnsw (face_embedding vector_cosine_ops) "
            "WHERE face_embedding IS NOT NULL AND is_expired = false;"
        ))

        # 2. Migrate employees.face_embedding -> vector(512)
        op.execute(text("ALTER TABLE employees ADD COLUMN IF NOT EXISTS face_embedding_vec vector(512);"))

        existing_emp_rows = bind.execute(
            text("SELECT id, face_embedding FROM employees WHERE face_embedding IS NOT NULL;")
        ).fetchall()
        for emp_id, raw_bytes in existing_emp_rows:
            if raw_bytes:
                arr = np.frombuffer(raw_bytes, dtype=np.float32)
                vec_str = "[" + ",".join(str(float(x)) for x in arr) + "]"
                bind.execute(
                    text("UPDATE employees SET face_embedding_vec = :v::vector WHERE id = :id"),
                    {"v": vec_str, "id": emp_id},
                )

        op.execute(text("ALTER TABLE employees DROP COLUMN face_embedding;"))
        op.execute(text("ALTER TABLE employees RENAME COLUMN face_embedding_vec TO face_embedding;"))

        # Create HNSW index for active employee face matching
        op.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_employees_hnsw ON employees "
            "USING hnsw (face_embedding vector_cosine_ops) "
            "WHERE face_embedding IS NOT NULL AND is_active = true;"
        ))


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        # Drop HNSW indexes
        op.execute(text("DROP INDEX IF EXISTS idx_person_identities_hnsw;"))
        op.execute(text("DROP INDEX IF EXISTS idx_employees_hnsw;"))

        # Revert person_identities
        op.execute(text("ALTER TABLE person_identities ADD COLUMN face_embedding_bytea BYTEA;"))
        op.execute(text("ALTER TABLE person_identities DROP COLUMN face_embedding;"))
        op.execute(text("ALTER TABLE person_identities RENAME COLUMN face_embedding_bytea TO face_embedding;"))

        # Revert employees
        op.execute(text("ALTER TABLE employees ADD COLUMN face_embedding_bytea BYTEA;"))
        op.execute(text("ALTER TABLE employees DROP COLUMN face_embedding;"))
        op.execute(text("ALTER TABLE employees RENAME COLUMN face_embedding_bytea TO face_embedding;"))

