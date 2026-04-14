"""add_worker_heartbeats_table

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-04-14 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d2e3f4a5b6c7"
down_revision: Union[str, None] = "c1d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the worker_heartbeats table."""
    op.create_table(
        "worker_heartbeats",
        sa.Column("worker_id", sa.String(128), primary_key=True),
        sa.Column("backend", sa.String(32), nullable=False),
        sa.Column("queue_name", sa.String(64), nullable=False),
        sa.Column("hostname", sa.String(256), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_worker_heartbeats_backend_queue",
        "worker_heartbeats",
        ["backend", "queue_name"],
    )
    op.create_index(
        "ix_worker_heartbeats_updated_at",
        "worker_heartbeats",
        ["updated_at"],
    )


def downgrade() -> None:
    """Drop the worker_heartbeats table."""
    op.drop_index("ix_worker_heartbeats_updated_at", table_name="worker_heartbeats")
    op.drop_index("ix_worker_heartbeats_backend_queue", table_name="worker_heartbeats")
    op.drop_table("worker_heartbeats")
