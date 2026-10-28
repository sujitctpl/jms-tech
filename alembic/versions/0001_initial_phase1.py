"""Initial Phase 1 schema

Revision ID: 0001_initial_phase1
Revises:
Create Date: 2026-10-27 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_phase1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. cameras table
    op.create_table(
        "cameras",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("stream_url", sa.Text(), nullable=False),
        sa.Column("location", sa.String(length=150), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "entry_line_config",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_cameras_name"), "cameras", ["name"], unique=False)

    # 2. processing_runs table
    op.create_table(
        "processing_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("camera_id", sa.Uuid(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="RUNNING"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("frames_processed", sa.BigInteger(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_processing_runs_camera_id"), "processing_runs", ["camera_id"], unique=False)
    op.create_index(op.f("ix_processing_runs_status"), "processing_runs", ["status"], unique=False)

    # 3. visitor_tracks table
    op.create_table(
        "visitor_tracks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("camera_id", sa.Uuid(), nullable=False),
        sa.Column("processing_run_id", sa.Uuid(), nullable=False),
        sa.Column("tracker_id", sa.String(length=100), nullable=False),
        sa.Column(
            "first_seen_at",
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
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ACTIVE"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["processing_run_id"], ["processing_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("processing_run_id", "tracker_id", name="uq_processing_run_tracker"),
    )
    op.create_index(op.f("ix_visitor_tracks_camera_id"), "visitor_tracks", ["camera_id"], unique=False)
    op.create_index(op.f("ix_visitor_tracks_processing_run_id"), "visitor_tracks", ["processing_run_id"], unique=False)
    op.create_index(op.f("ix_visitor_tracks_tracker_id"), "visitor_tracks", ["tracker_id"], unique=False)
    op.create_index(op.f("ix_visitor_tracks_status"), "visitor_tracks", ["status"], unique=False)

    # 4. detection_events table
    op.create_table(
        "detection_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("camera_id", sa.Uuid(), nullable=False),
        sa.Column("processing_run_id", sa.Uuid(), nullable=False),
        sa.Column("visitor_track_id", sa.Uuid(), nullable=False),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("frame_number", sa.BigInteger(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox_x1", sa.Float(), nullable=False),
        sa.Column("bbox_y1", sa.Float(), nullable=False),
        sa.Column("bbox_x2", sa.Float(), nullable=False),
        sa.Column("bbox_y2", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["processing_run_id"], ["processing_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["visitor_track_id"], ["visitor_tracks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_detection_events_camera_id"), "detection_events", ["camera_id"], unique=False)
    op.create_index(op.f("ix_detection_events_detected_at"), "detection_events", ["detected_at"], unique=False)
    op.create_index(op.f("ix_detection_events_processing_run_id"), "detection_events", ["processing_run_id"], unique=False)
    op.create_index(op.f("ix_detection_events_visitor_track_id"), "detection_events", ["visitor_track_id"], unique=False)

    # 5. footfall_events table
    op.create_table(
        "footfall_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("camera_id", sa.Uuid(), nullable=False),
        sa.Column("visitor_track_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=10), nullable=False, server_default="entry"),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("event_type = 'entry'", name="ck_footfall_event_type_entry_only"),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["visitor_track_id"], ["visitor_tracks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("visitor_track_id", name="uq_footfall_visitor_track"),
    )
    op.create_index(op.f("ix_footfall_events_camera_id"), "footfall_events", ["camera_id"], unique=False)
    op.create_index(op.f("ix_footfall_events_occurred_at"), "footfall_events", ["occurred_at"], unique=False)
    op.create_index(op.f("ix_footfall_events_visitor_track_id"), "footfall_events", ["visitor_track_id"], unique=False)


def downgrade() -> None:
    op.drop_table("footfall_events")
    op.drop_table("detection_events")
    op.drop_table("visitor_tracks")
    op.drop_table("processing_runs")
    op.drop_table("cameras")
