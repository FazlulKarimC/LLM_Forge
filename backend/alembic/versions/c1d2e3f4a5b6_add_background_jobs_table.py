"""Add durable background jobs table

Revision ID: c1d2e3f4a5b6
Revises: b9f8a7c6d5e4
Create Date: 2026-04-04
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, None] = "b9f8a7c6d5e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "background_jobs",
        sa.Column("job_id", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("job_metadata", sa.JSON(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index(op.f("ix_background_jobs_kind"), "background_jobs", ["kind"], unique=False)
    op.create_index(op.f("ix_background_jobs_status"), "background_jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_background_jobs_status"), table_name="background_jobs")
    op.drop_index(op.f("ix_background_jobs_kind"), table_name="background_jobs")
    op.drop_table("background_jobs")
