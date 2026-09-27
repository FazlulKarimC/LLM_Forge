"""Add workspaces. Intentionally clears personal benchmark/prompt/job data."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "i2j3k4l5m6n7"
down_revision = "h1i2j3k4l5m6"
branch_labels = None
depends_on = None


def upgrade():
    uuid = postgresql.UUID(as_uuid=True)
    op.create_table("users",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("auth_subject", sa.String(255), nullable=False, unique=True),
        sa.Column("display_name", sa.String(120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("organizations",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("slug", sa.String(160), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("organization_memberships",
        sa.Column("organization_id", uuid, sa.ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", uuid, sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.String(16), nullable=False),
        sa.CheckConstraint("role IN ('owner', 'member')", name="ck_membership_role"))
    op.create_table("projects",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("organization_id", uuid, sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("slug", sa.String(160), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("organization_id", "slug", name="uq_project_org_slug"))
    op.create_index("ix_projects_organization_id", "projects", ["organization_id"])
    op.execute("TRUNCATE TABLE experiments, prompt_versions, background_jobs CASCADE")
    for table in ("experiments", "prompt_versions", "background_jobs"):
        op.add_column(table, sa.Column("project_id", uuid, nullable=False))
        op.create_foreign_key(f"fk_{table}_project", table, "projects", ["project_id"], ["id"], ondelete="CASCADE")
        op.create_index(f"ix_{table}_project_id", table, ["project_id"])


def downgrade():
    for table in ("background_jobs", "prompt_versions", "experiments"):
        op.drop_index(f"ix_{table}_project_id", table_name=table)
        op.drop_constraint(f"fk_{table}_project", table, type_="foreignkey")
        op.drop_column(table, "project_id")
    op.drop_table("projects")
    op.drop_table("organization_memberships")
    op.drop_table("organizations")
    op.drop_table("users")
