"""Allow content-free local evaluation reports; preserve hosted revisions.

Revision ID: p9q0r1s2t3u4
Revises: o8p9q0r1s2t3
"""
from alembic import op
import sqlalchemy as sa

revision = "p9q0r1s2t3u4"
down_revision = "o8p9q0r1s2t3"
branch_labels = None
depends_on = None

def upgrade():
    op.alter_column("evaluation_runs", "dataset_revision_id", nullable=True)

def downgrade():
    if op.get_bind().scalar(sa.text("SELECT count(*) FROM evaluation_runs WHERE dataset_revision_id IS NULL")):
        raise RuntimeError("Local reports exist; downgrade would discard report identity")
    op.alter_column("evaluation_runs", "dataset_revision_id", nullable=False)
