"""add run_metadata column

Revision ID: h1i2j3k4l5m6
Revises: d2e3f4a5b6c7
Create Date: 2026-05-07 20:22:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision = "h1i2j3k4l5m6"
down_revision = "d2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("run_metadata", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "run_metadata")
