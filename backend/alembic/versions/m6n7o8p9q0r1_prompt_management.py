"""Chat prompts, versioned configuration, tags and custom labels."""
from alembic import op
import sqlalchemy as sa

revision = "m6n7o8p9q0r1"
down_revision = "l5m6n7o8p9q0"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("prompts", sa.Column("prompt_type", sa.String(8), nullable=False, server_default="text"))
    op.add_column("prompts", sa.Column("tags", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("prompt_versions", sa.Column("prompt_type", sa.String(8), nullable=False, server_default="text"))
    op.add_column("prompt_versions", sa.Column("messages", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("prompt_versions", sa.Column("config", sa.JSON(), nullable=False, server_default="{}"))
    op.add_column("prompt_versions", sa.Column("created_by", sa.String(16), nullable=False, server_default="UI"))
    op.drop_constraint("ck_prompt_label", "prompt_labels", type_="check")
    op.alter_column("prompt_labels", "label", type_=sa.String(64), existing_type=sa.String(16))
    op.create_check_constraint("ck_prompt_label_length", "prompt_labels", "length(label) BETWEEN 1 AND 64")
    # latest is derived from prompts.latest_version; existing prompts need no backfill.


def downgrade():
    # Refuse destructive downgrades when newer prompt capabilities are in use.
    connection = op.get_bind()
    used = connection.execute(sa.text("""SELECT EXISTS (
        SELECT 1 FROM prompt_versions WHERE prompt_type = 'chat' OR config::text <> '{}' OR created_by <> 'UI'
        UNION ALL SELECT 1 FROM prompts WHERE tags::text <> '[]' OR name LIKE '%/%'
        UNION ALL SELECT 1 FROM prompt_labels WHERE label NOT IN ('staging', 'production')
    )""")).scalar()
    if used:
        raise RuntimeError("Cannot downgrade while chat prompts, config, tags or custom labels are in use")
    op.drop_constraint("ck_prompt_label_length", "prompt_labels", type_="check")
    op.alter_column("prompt_labels", "label", type_=sa.String(16), existing_type=sa.String(64))
    op.create_check_constraint("ck_prompt_label", "prompt_labels", "label IN ('staging', 'production')")
    for field in ("created_by", "config", "messages", "prompt_type"):
        op.drop_column("prompt_versions", field)
    op.drop_column("prompts", "tags")
    op.drop_column("prompts", "prompt_type")
