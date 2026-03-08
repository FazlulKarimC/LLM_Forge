"""make_runs_unique_per_attempt

Revision ID: 8b1c3d5e7f90
Revises: 41509c4a1e21
Create Date: 2026-03-08 12:40:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '8b1c3d5e7f90'
down_revision: Union[str, Sequence[str], None] = '41509c4a1e21'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint('uq_runs_experiment_example', 'runs', type_='unique')
    op.create_unique_constraint(
        'uq_runs_experiment_example_attempt',
        'runs',
        ['experiment_id', 'example_id', 'attempt'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('uq_runs_experiment_example_attempt', 'runs', type_='unique')
    op.create_unique_constraint('uq_runs_experiment_example', 'runs', ['experiment_id', 'example_id'])
