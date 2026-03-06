"""add_tags_and_run_manifest_to_experiments

Revision ID: de6743478a11
Revises: ff53d769bd7c
Create Date: 2026-03-07 02:26:14.495350

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'de6743478a11'
down_revision: Union[str, Sequence[str], None] = 'ff53d769bd7c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('experiments', sa.Column('tags', sa.JSON(), nullable=True))
    op.add_column('experiments', sa.Column('run_manifest', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('experiments', 'run_manifest')
    op.drop_column('experiments', 'tags')

