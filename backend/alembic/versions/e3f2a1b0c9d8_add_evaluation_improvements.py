"""Add evaluation improvements P0-P2

New columns for improved metric tracking:
- Run: is_exact_match, is_substring_match, parsed_answer, match_alias,
       semantic_similarity, context_relevance_score, attempt
- Result: semantic_similarity
- Experiment: dataset_hash, sample_ids, current_attempt

Revision ID: e3f2a1b0c9d8
Revises: d191e27e1915
Create Date: 2026-02-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = 'e3f2a1b0c9d8'
down_revision: Union[str, None] = 'd191e27e1915'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Run table: evaluation detail columns ---
    op.add_column('runs', sa.Column('is_exact_match', sa.Boolean(), nullable=True))
    op.add_column('runs', sa.Column('is_substring_match', sa.Boolean(), nullable=True))
    op.add_column('runs', sa.Column('parsed_answer', sa.Text(), nullable=True))
    op.add_column('runs', sa.Column('match_alias', sa.String(500), nullable=True))
    op.add_column('runs', sa.Column('semantic_similarity', sa.Float(), nullable=True))
    op.add_column('runs', sa.Column('context_relevance_score', sa.Float(), nullable=True))
    op.add_column('runs', sa.Column('attempt', sa.Integer(), server_default='1', nullable=False))

    # --- Result table: new aggregate metric ---
    op.add_column('results', sa.Column('semantic_similarity', sa.Float(), nullable=True))

    # --- Experiment table: reproducibility metadata ---
    op.add_column('experiments', sa.Column('dataset_hash', sa.String(64), nullable=True))
    op.add_column('experiments', sa.Column('sample_ids', sa.JSON(), nullable=True))
    op.add_column('experiments', sa.Column('current_attempt', sa.Integer(), server_default='1', nullable=False))


def downgrade() -> None:
    # --- Experiment table ---
    op.drop_column('experiments', 'current_attempt')
    op.drop_column('experiments', 'sample_ids')
    op.drop_column('experiments', 'dataset_hash')

    # --- Result table ---
    op.drop_column('results', 'semantic_similarity')

    # --- Run table ---
    op.drop_column('runs', 'attempt')
    op.drop_column('runs', 'context_relevance_score')
    op.drop_column('runs', 'semantic_similarity')
    op.drop_column('runs', 'match_alias')
    op.drop_column('runs', 'parsed_answer')
    op.drop_column('runs', 'is_substring_match')
    op.drop_column('runs', 'is_exact_match')
