"""Reusable evaluator versions, scores and immutable score-only runs."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "n7o8p9q0r1s2"
down_revision = "m6n7o8p9q0r1"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "source_run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("evaluation_runs.id"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_evaluation_runs_source_run_id", "evaluation_runs", ["source_run_id"]
    )
    op.create_table(
        "evaluators",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("latest_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "name", name="uq_evaluator_name"),
    )
    op.create_table(
        "evaluator_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "evaluator_id",
            UUID(as_uuid=True),
            sa.ForeignKey("evaluators.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("definition", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("evaluator_id", "version", name="uq_evaluator_version"),
    )
    op.create_table(
        "evaluation_scores",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "evaluator_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("evaluator_versions.id"),
        ),
        sa.Column("case_index", sa.Integer(), nullable=False),
        sa.Column("assignment", sa.String(80), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("data_type", sa.String(16), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("required", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "run_id", "case_index", "assignment", "name", name="uq_evaluation_score"
        ),
    )
    for table in ("evaluators", "evaluator_versions", "evaluation_scores"):
        op.create_index(f"ix_{table}_project_id", table, ["project_id"])
    op.create_index("ix_evaluation_scores_run_id", "evaluation_scores", ["run_id"])


def downgrade():
    if (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT EXISTS (SELECT 1 FROM evaluators UNION ALL SELECT 1 FROM evaluation_scores UNION ALL SELECT 1 FROM evaluation_runs WHERE source_run_id IS NOT NULL)"
            )
        )
        .scalar()
    ):
        raise RuntimeError(
            "Cannot downgrade while evaluators, scores or scoring passes are in use"
        )
    for table in ("evaluation_scores", "evaluator_versions", "evaluators"):
        op.drop_table(table)
    op.drop_index("ix_evaluation_runs_source_run_id", "evaluation_runs")
    op.drop_column("evaluation_runs", "source_run_id")
