"""Placeholder for a revision that some databases were already stamped with.

Some databases carry `0007_visit_sessions_and_face_gallery` in alembic_version, but the original
file was never committed. The employee face-gallery columns it referred to are created by 0006,
and no visit_sessions table exists in the models, so this revision intentionally does nothing.
It only keeps the migration chain continuous for those databases.

Revision ID: 0007_visit_sessions_and_face_gallery
Revises: 0006_visitor_profile_and_departments
Create Date: 2026-11-27 11:00:00.000000

"""
from typing import Sequence, Union

revision: str = "0007_visit_sessions_and_face_gallery"
down_revision: Union[str, None] = "0006_visitor_profile_and_departments"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
