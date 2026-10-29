"""Employee Management and Daily Attendance Schema

Revision ID: 0003_employee_attendance
Revises: 0002_identity_retention
Create Date: 2026-10-29 11:55:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0003_employee_attendance"
down_revision: Union[str, None] = "0002_identity_retention"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. employees table
    op.create_table(
        "employees",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("employee_id", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("department", sa.String(length=100), nullable=False),
        sa.Column("designation", sa.String(length=100), nullable=False),
        sa.Column("assigned_camera_id", sa.Uuid(), nullable=True),
        sa.Column("assigned_location", sa.String(length=150), nullable=True),
        sa.Column("shift_start_time", sa.Time(), nullable=True, server_default="09:30:00"),
        sa.Column("shift_end_time", sa.Time(), nullable=True, server_default="18:30:00"),
        sa.Column("late_grace_minutes", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("early_departure_grace_minutes", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("face_embedding", sa.LargeBinary(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
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
        sa.ForeignKeyConstraint(["assigned_camera_id"], ["cameras.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("employee_id", name="uq_employees_employee_id"),
    )
    op.create_index(op.f("ix_employees_employee_id"), "employees", ["employee_id"], unique=True)
    op.create_index(op.f("ix_employees_name"), "employees", ["name"], unique=False)
    op.create_index(op.f("ix_employees_department"), "employees", ["department"], unique=False)
    op.create_index(op.f("ix_employees_is_active"), "employees", ["is_active"], unique=False)

    # 2. employee_attendance_logs table
    op.create_table(
        "employee_attendance_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("employee_id", sa.Uuid(), nullable=False),
        sa.Column("camera_id", sa.Uuid(), nullable=True),
        sa.Column("event_type", sa.String(length=20), nullable=False, server_default="CHECK_IN"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.95"),
        sa.Column(
            "timestamp",
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
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_employee_attendance_logs_employee_id"), "employee_attendance_logs", ["employee_id"], unique=False)
    op.create_index(op.f("ix_employee_attendance_logs_camera_id"), "employee_attendance_logs", ["camera_id"], unique=False)
    op.create_index(op.f("ix_employee_attendance_logs_timestamp"), "employee_attendance_logs", ["timestamp"], unique=False)

    # 3. employee_daily_attendance table
    op.create_table(
        "employee_daily_attendance",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("employee_id", sa.Uuid(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("first_check_in", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_check_out", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_work_hours", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("is_late", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("late_minutes", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("is_early_departure", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("early_departure_minutes", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="PRESENT"),
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
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("employee_id", "date", name="uq_employee_daily_attendance"),
    )
    op.create_index(op.f("ix_employee_daily_attendance_employee_id"), "employee_daily_attendance", ["employee_id"], unique=False)
    op.create_index(op.f("ix_employee_daily_attendance_date"), "employee_daily_attendance", ["date"], unique=False)
    op.create_index(op.f("ix_employee_daily_attendance_status"), "employee_daily_attendance", ["status"], unique=False)


def downgrade() -> None:
    op.drop_table("employee_daily_attendance")
    op.drop_table("employee_attendance_logs")
    op.drop_table("employees")

