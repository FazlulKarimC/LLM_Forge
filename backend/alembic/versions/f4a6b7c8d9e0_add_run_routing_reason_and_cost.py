"""add run routing reason and cost

Revision ID: f4a6b7c8d9e0
Revises: d2e3f4a5b6c7
Create Date: 2026-04-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f4a6b7c8d9e0"
down_revision: Union[str, None] = "d2e3f4a5b6c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("routing_reason", sa.String(length=64), nullable=True))
    op.add_column("runs", sa.Column("cost_usd", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "cost_usd")
    op.drop_column("runs", "routing_reason")
