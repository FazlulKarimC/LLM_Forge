"""Add explicit regression_status column to experiments

Revision ID: b9f8a7c6d5e4
Revises: a1b2c3d4e5f6
Create Date: 2026-03-30
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision: str = "b9f8a7c6d5e4"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "experiments",
        sa.Column(
            "regression_status",
            sa.String(length=32),
            server_default="not_checked",
            nullable=False,
        ),
    )

    op.execute(
        """
        UPDATE experiments AS e
        SET regression_status = CASE
            WHEN e.regression_passed IS TRUE THEN 'pass'
            WHEN e.regression_passed IS FALSE THEN 'fail'
            WHEN EXISTS (
                SELECT 1
                FROM results AS r
                WHERE r.experiment_id = e.id
                  AND (r.raw_metrics ? 'regression')
                  AND (r.raw_metrics -> 'regression' ->> 'passed') IS NULL
            ) THEN 'inconclusive'
            ELSE 'not_checked'
        END
        """
    )


def downgrade() -> None:
    op.drop_column("experiments", "regression_status")
