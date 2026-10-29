"""Add employee_faces table for multi-angle photo gallery in pgvector

Revision ID: 0005_employee_face_gallery
Revises: 0004_pgvector_migration
Create Date: 2026-10-29 17:28:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0005_employee_face_gallery"
down_revision: Union[str, None] = "0004_pgvector_migration"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS employee_faces (
            id UUID PRIMARY KEY,
            employee_id UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
            angle_label VARCHAR(50),
            face_embedding vector(512) NOT NULL,
            photo_url VARCHAR(500),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_employee_faces_hnsw ON employee_faces
        USING hnsw (face_embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS idx_employee_faces_employee_id ON employee_faces(employee_id);
        """))
    else:
        # SQLite / in-memory fallback for test environments
        op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS employee_faces (
            id VARCHAR(36) PRIMARY KEY,
            employee_id VARCHAR(36) NOT NULL,
            angle_label VARCHAR(50),
            face_embedding JSON,
            photo_url VARCHAR(500),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """))


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    op.execute(sa.text("DROP TABLE IF EXISTS employee_faces CASCADE;"))
