"""Add departments table, visitor profile / journey columns, body Re-ID vector,
and employee gallery columns that were previously only created via create_all().

Idempotent: every table/column is only created when missing, so this is safe on
databases that were already patched by the app's startup `create_all()`.

Revision ID: 0006_visitor_profile_and_departments
Revises: 0005_employee_face_gallery
Create Date: 2026-11-25 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0006_visitor_profile_and_departments"
down_revision: Union[str, None] = "0005_employee_face_gallery"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _vector_type(dialect: str):
    if dialect == "postgresql":
        from pgvector.sqlalchemy import Vector

        return Vector(512)
    return sa.JSON()


def _person_identity_columns(dialect: str):
    return [
        sa.Column("body_embedding", _vector_type(dialect), nullable=True),
        sa.Column("avatar_url", sa.String(length=500), nullable=True),
        sa.Column("exit_avatar_url", sa.String(length=500), nullable=True),
        sa.Column("visit_purpose", sa.String(length=50), nullable=True),
        sa.Column("assigned_salesperson", sa.String(length=100), nullable=True),
        sa.Column("notes", sa.String(length=500), nullable=True),
        sa.Column("visitor_type", sa.String(length=20), nullable=False, server_default="New"),
        sa.Column("tagging_status", sa.String(length=20), nullable=False, server_default="Untagged"),
        sa.Column("gender_estimated", sa.String(length=20), nullable=True),
        sa.Column("age_group_estimated", sa.String(length=20), nullable=True),
        sa.Column("zone_entered", sa.String(length=100), nullable=True),
        sa.Column("camera_source", sa.String(length=100), nullable=True),
        sa.Column("exit_time", sa.String(length=50), nullable=True),
        sa.Column("zone_dwell", sa.String(length=500), nullable=True),
        sa.Column("waiting_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_waiting", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("waiting_breach", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    ]


def _employee_columns():
    return [
        sa.Column("face_gallery", sa.JSON(), nullable=True),
        sa.Column("photo_url", sa.String(length=500), nullable=True),
    ]


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    # 1. departments table
    if "departments" not in tables:
        op.create_table(
            "departments",
            sa.Column("id", sa.String(length=50), nullable=False),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("color", sa.String(length=20), nullable=False, server_default="#00D7FF"),
            sa.Column("camera_id", sa.Uuid(), nullable=True),
            sa.Column("box", sa.JSON(), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    # 2. person_identities profile / journey / Re-ID columns
    existing = {c["name"] for c in inspector.get_columns("person_identities")}
    for col in _person_identity_columns(dialect):
        if col.name not in existing:
            op.add_column("person_identities", col)

    # HNSW needs the pgvector operator classes; skip (rather than abort the whole upgrade)
    # when the extension is not registered in this database.
    if dialect == "postgresql" and bind.execute(
        sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    ).scalar():
        op.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS idx_person_identities_body_hnsw ON person_identities "
            "USING hnsw (body_embedding vector_cosine_ops);"
        ))

    # 3. employees gallery columns
    existing = {c["name"] for c in inspector.get_columns("employees")}
    for col in _employee_columns():
        if col.name not in existing:
            op.add_column("employees", col)


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "postgresql":
        op.execute(sa.text("DROP INDEX IF EXISTS idx_person_identities_body_hnsw;"))
    for col in _employee_columns():
        op.drop_column("employees", col.name)
    for col in _person_identity_columns(dialect):
        op.drop_column("person_identities", col.name)
    op.drop_table("departments")
