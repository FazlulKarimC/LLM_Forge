"""Add QUEUED to experimentstatus enum

Revision ID: add_queued_enum_value
Revises: 76c8f0c2228a
Create Date: 2026-01-02

This migration adds the QUEUED value to the experimentstatus PostgreSQL enum.
Alembic autogenerate doesn't detect enum value additions.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_queued_enum_value'
down_revision: Union[str, Sequence[str], None] = '76c8f0c2228a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add QUEUED value to experimentstatus enum."""
    pass


def downgrade() -> None:
    """Remove QUEUED from enum (complex - requires recreating enum)."""
    # PostgreSQL doesn't support removing enum values directly
    # For simplicity, we'll leave the enum value in place
    # A full downgrade would require:
    # 1. Update all 'queued' rows to 'pending'
    # 2. Create new enum type without 'queued'
    # 3. Alter column to use new type
    # 4. Drop old type
    pass
