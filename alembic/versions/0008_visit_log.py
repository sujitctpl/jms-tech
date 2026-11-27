"""Add person_identities.visit_log (JSON list of entry/exit segments).

Idempotent: the column is only added when missing.

Revision ID: 0008_visit_log
Revises: 0007_visit_sessions_and_face_gallery
Create Date: 2026-11-27 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0008_visit_log"
down_revision: Union[str, None] = "0007_visit_sessions_and_face_gallery"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {c["name"] for c in inspector.get_columns("person_identities")}
    if "visit_log" not in existing:
        op.add_column("person_identities", sa.Column("visit_log", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("person_identities", "visit_log")
