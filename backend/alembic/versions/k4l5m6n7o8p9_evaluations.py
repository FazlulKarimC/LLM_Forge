"""Versioned datasets and evaluation runs."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "k4l5m6n7o8p9"
down_revision = "j3k4l5m6n7o8"
branch_labels = None
depends_on = None


def upgrade():
    uuid = UUID(as_uuid=True)
    def identity():
        return [sa.Column("id", uuid, primary_key=True),
                sa.Column("project_id", uuid, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)]
    op.create_table("evaluation_datasets", *identity(),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("latest_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "name", name="uq_dataset_project_name"))
    op.create_table("dataset_revisions", *identity(),
        sa.Column("dataset_id", uuid, sa.ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("cases", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("dataset_id", "version", name="uq_dataset_revision"))
    op.create_table("evaluation_runs", *identity(),
        sa.Column("prompt_version_id", uuid, sa.ForeignKey("prompt_versions.id"), nullable=False),
        sa.Column("dataset_revision_id", uuid, sa.ForeignKey("dataset_revisions.id"), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("completed", sa.Integer(), nullable=False),
        sa.Column("passed", sa.Integer(), nullable=False),
        sa.Column("errors", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('queued','running','completed','failed','cancelled')", name="ck_evaluation_status"))
    op.create_table("evaluation_case_results", *identity(),
        sa.Column("run_id", uuid, sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("case_index", sa.Integer(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.UniqueConstraint("run_id", "case_index", name="uq_evaluation_case"))
    for table in ("evaluation_datasets", "dataset_revisions", "evaluation_runs", "evaluation_case_results"):
        op.create_index(f"ix_{table}_project_id", table, ["project_id"])


def downgrade():
    for table in ("evaluation_case_results", "evaluation_runs", "dataset_revisions", "evaluation_datasets"):
        op.drop_table(table)
