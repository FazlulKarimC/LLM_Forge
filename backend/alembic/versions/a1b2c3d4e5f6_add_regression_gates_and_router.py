"""Add regression gates and adaptive router columns

New columns:
- Experiment: is_baseline, baseline_id, pinned_attempt, prompt_version_id, regression_passed
- Run: grader_results (JSONB), served_provider

Revision ID: a1b2c3d4e5f6
Revises: 8b1c3d5e7f90
Create Date: 2026-03-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '8b1c3d5e7f90'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Experiment table: regression gates ---
    op.add_column('experiments', sa.Column('is_baseline', sa.Boolean(),
                  server_default='false', nullable=False))
    op.add_column('experiments', sa.Column('baseline_id',
                  postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('experiments', sa.Column('pinned_attempt',
                  sa.Integer(), nullable=True))
    op.add_column('experiments', sa.Column('prompt_version_id',
                  postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('experiments', sa.Column('regression_passed',
                  sa.Boolean(), nullable=True))

    # Self-referential FK for baseline_id
    op.create_foreign_key(
        'fk_experiments_baseline_id',
        'experiments', 'experiments',
        ['baseline_id'], ['id'],
    )

    # --- Run table: grader verdicts + routing telemetry ---
    op.add_column('runs', sa.Column('grader_results',
                  postgresql.JSONB(), nullable=True))
    op.add_column('runs', sa.Column('served_provider',
                  sa.String(64), nullable=True))


def downgrade() -> None:
    # --- Run table ---
    op.drop_column('runs', 'served_provider')
    op.drop_column('runs', 'grader_results')

    # --- Experiment table ---
    op.drop_constraint('fk_experiments_baseline_id', 'experiments', type_='foreignkey')
    op.drop_column('experiments', 'regression_passed')
    op.drop_column('experiments', 'prompt_version_id')
    op.drop_column('experiments', 'pinned_attempt')
    op.drop_column('experiments', 'baseline_id')
    op.drop_column('experiments', 'is_baseline')
