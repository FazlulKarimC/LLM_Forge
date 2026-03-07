"""add_prompt_versions_table

Revision ID: c8002bf009a7
Revises: de6743478a11
Create Date: 2026-03-08 01:27:15.393137

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c8002bf009a7'
down_revision: Union[str, Sequence[str], None] = 'de6743478a11'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the prompt_versions table."""
    op.create_table(
        'prompt_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('template_text', sa.Text(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('sha256_hash', sa.String(64), nullable=False, index=True),
        sa.Column('parent_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('prompt_versions.id'), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
    )


def downgrade() -> None:
    """Drop the prompt_versions table."""
    op.drop_table('prompt_versions')
