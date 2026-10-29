"""Persistent Identity and 14-day Biometric Retention Schema

Revision ID: 0002_identity_retention
Revises: 0001_initial_phase1
Create Date: 2026-10-29 11:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0002_identity_retention"
down_revision: Union[str, None] = "0001_initial_phase1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. person_identities table
    op.create_table(
        "person_identities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("person_id", sa.String(length=100), nullable=False),
        sa.Column("face_embedding", sa.LargeBinary(), nullable=True),
        sa.Column("is_expired", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expired_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("person_id", name="uq_person_identities_person_id"),
    )
    op.create_index(op.f("ix_person_identities_person_id"), "person_identities", ["person_id"], unique=True)
    op.create_index(op.f("ix_person_identities_is_expired"), "person_identities", ["is_expired"], unique=False)
    op.create_index(op.f("ix_person_identities_last_seen_at"), "person_identities", ["last_seen_at"], unique=False)

    # 2. Add person_identity_id and person_id columns to visitor_tracks table
    op.add_column("visitor_tracks", sa.Column("person_identity_id", sa.Uuid(), nullable=True))
    op.add_column("visitor_tracks", sa.Column("person_id", sa.String(length=100), nullable=True))
    op.create_index(op.f("ix_visitor_tracks_person_identity_id"), "visitor_tracks", ["person_identity_id"], unique=False)
    op.create_index(op.f("ix_visitor_tracks_person_id"), "visitor_tracks", ["person_id"], unique=False)
    op.create_foreign_key(
        "fk_visitor_tracks_person_identities",
        "visitor_tracks",
        "person_identities",
        ["person_identity_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_visitor_tracks_person_identities", "visitor_tracks", type_="foreignkey")
    op.drop_index(op.f("ix_visitor_tracks_person_id"), table_name="visitor_tracks")
    op.drop_index(op.f("ix_visitor_tracks_person_identity_id"), table_name="visitor_tracks")
    op.drop_column("visitor_tracks", "person_id")
    op.drop_column("visitor_tracks", "person_identity_id")
    op.drop_table("person_identities")

