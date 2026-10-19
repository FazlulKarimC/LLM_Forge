"""Bound hosted benchmark admission without a queue or additional service.

Revision ID: q0r1s2t3u4v5
Revises: p9q0r1s2t3u4
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "q0r1s2t3u4v5"
down_revision = "p9q0r1s2t3u4"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "benchmark_reservations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("resource_id", sa.String(64), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=True),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_benchmark_reservations_resource_id",
        "benchmark_reservations",
        ["resource_id"],
    )
    op.create_index(
        "ix_benchmark_reservation_active",
        "benchmark_reservations",
        ["released_at", "expires_at"],
    )


def downgrade():
    if op.get_bind().scalar(
        sa.text(
            "SELECT count(*) FROM benchmark_reservations WHERE released_at IS NULL AND expires_at > CURRENT_TIMESTAMP"
        )
    ):
        raise RuntimeError(
            "Active benchmark reservations exist; stop execution before downgrade"
        )
    op.drop_table("benchmark_reservations")
