"""Stable prompts, immutable versions, release labels and hashed SDK keys."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "j3k4l5m6n7o8"
down_revision = "i2j3k4l5m6n7"
branch_labels = None
depends_on = None


def upgrade():
    uuid = UUID(as_uuid=True)
    op.create_table("prompts",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("project_id", uuid, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("latest_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "name", name="uq_prompt_project_name"))
    op.create_index("ix_prompts_project_id", "prompts", ["project_id"])
    op.add_column("prompt_versions", sa.Column("prompt_id", uuid, nullable=True))
    op.add_column("prompt_versions", sa.Column("template_format", sa.String(16), nullable=False, server_default="fstring"))
    # Preserve existing benchmark snapshots and IDs. Resolve any old parallel
    # version numbering while keeping content and lineage intact.
    op.execute("""INSERT INTO prompts (id, project_id, name, description, archived, latest_version, created_at, updated_at)
        SELECT gen_random_uuid(), project_id, name, '', false, 0, min(created_at), max(created_at)
        FROM prompt_versions GROUP BY project_id, name""")
    op.execute("""UPDATE prompt_versions v SET prompt_id = p.id FROM prompts p
        WHERE v.project_id = p.project_id AND v.name = p.name""")
    op.execute("""WITH numbered AS (SELECT id, row_number() OVER
        (PARTITION BY prompt_id ORDER BY version, created_at, id) AS number FROM prompt_versions)
        UPDATE prompt_versions v SET version = n.number FROM numbered n WHERE v.id = n.id""")
    op.execute("""UPDATE prompts p SET latest_version = (SELECT max(v.version) FROM prompt_versions v WHERE v.prompt_id = p.id)""")
    op.alter_column("prompt_versions", "prompt_id", nullable=False)
    op.create_foreign_key("fk_prompt_version_prompt", "prompt_versions", "prompts", ["prompt_id"], ["id"], ondelete="CASCADE")
    op.create_index("ix_prompt_versions_prompt_id", "prompt_versions", ["prompt_id"])
    op.create_unique_constraint("uq_prompt_version_number", "prompt_versions", ["prompt_id", "version"])
    # Legacy duplicate contents, if any, must stay addressable by snapshot ID;
    # duplicate-content prevention is enforced by the service for new writes.
    op.create_check_constraint("ck_prompt_template_format", "prompt_versions", "template_format IN ('mustache', 'fstring')")
    op.create_table("prompt_labels",
        sa.Column("prompt_id", uuid, sa.ForeignKey("prompts.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("label", sa.String(16), primary_key=True),
        sa.Column("project_id", uuid, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_id", uuid, sa.ForeignKey("prompt_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("label IN ('staging', 'production')", name="ck_prompt_label"))
    op.create_index("ix_prompt_labels_project_id", "prompt_labels", ["project_id"])
    op.create_table("project_api_keys",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("project_id", uuid, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("prefix", sa.String(20), nullable=False),
        sa.Column("secret_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_project_api_keys_project_id", "project_api_keys", ["project_id"])


def downgrade():
    op.drop_table("project_api_keys")
    op.drop_table("prompt_labels")
    op.drop_constraint("ck_prompt_template_format", "prompt_versions", type_="check")
    op.drop_constraint("uq_prompt_version_number", "prompt_versions", type_="unique")
    op.drop_index("ix_prompt_versions_prompt_id", table_name="prompt_versions")
    op.drop_constraint("fk_prompt_version_prompt", "prompt_versions", type_="foreignkey")
    op.drop_column("prompt_versions", "template_format")
    op.drop_column("prompt_versions", "prompt_id")
    op.drop_table("prompts")
