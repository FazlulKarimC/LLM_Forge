"""Explicit evaluation capability for newly created SDK keys."""
from alembic import op
import sqlalchemy as sa

revision = "l5m6n7o8p9q0"
down_revision = "k4l5m6n7o8p9"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("project_api_keys", sa.Column("scopes", sa.JSON(), nullable=False, server_default='["prompts:read"]'))


def downgrade():
    op.drop_column("project_api_keys", "scopes")
