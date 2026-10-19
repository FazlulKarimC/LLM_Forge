"""Preserve attempt provenance and aggregates without guessing legacy identity.

Revision ID: r1s2t3u4v5w6
Revises: q0r1s2t3u4v5
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "r1s2t3u4v5w6"
down_revision = "q0r1s2t3u4v5"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "experiment_attempts",
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("experiments.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("number", sa.Integer(), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("storage_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("generation_calls", sa.Integer(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("run_manifest", sa.JSON(), nullable=True),
        sa.Column("dataset_hash", sa.String(64), nullable=True),
        sa.Column("sample_ids", sa.JSON(), nullable=True),
        sa.Column("provenance_status", sa.String(24), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_experiment_attempts_project_id", "experiment_attempts", ["project_id"])
    op.add_column("results", sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"))
    op.drop_index("ix_results_experiment_id", table_name="results")
    op.create_index("ix_results_experiment_id", "results", ["experiment_id"])
    op.create_unique_constraint("uq_results_experiment_attempt", "results", ["experiment_id", "attempt"])
    op.create_index("ix_runs_attempt_page", "runs", ["experiment_id", "attempt", "example_id", "id"])
    op.execute("""INSERT INTO experiment_attempts
        (experiment_id, number, project_id, status, provenance_status, created_at)
        SELECT e.id, r.attempt, e.project_id, 'legacy', 'unknown', coalesce(min(r.created_at), now())
        FROM runs r JOIN experiments e ON e.id=r.experiment_id
        GROUP BY e.id, r.attempt, e.project_id""")
    op.execute("""INSERT INTO experiment_attempts
        (experiment_id, number, project_id, status, provenance_status, created_at)
        SELECT e.id, 0, e.project_id, 'legacy', 'unknown', coalesce(r.computed_at, now())
        FROM results r JOIN experiments e ON e.id=r.experiment_id WHERE true ON CONFLICT DO NOTHING""")
    op.execute("""UPDATE experiment_attempts a SET storage_bytes = s.bytes FROM
        (SELECT experiment_id, attempt, sum(octet_length(coalesce(prompt, '')) + octet_length(coalesce(raw_output, '')) +
        octet_length(coalesce(agent_trace::text, '')) + octet_length(coalesce(retrieved_chunks::text, '')) +
        octet_length(coalesce(expected_output, '')) + octet_length(coalesce(parsed_answer, '')) +
        octet_length(coalesce(run_metadata::text, '')))::integer AS bytes FROM runs GROUP BY experiment_id, attempt) s
        WHERE a.experiment_id=s.experiment_id AND a.number=s.attempt""")


def downgrade():
    if op.get_bind().scalar(sa.text("SELECT count(*) FROM experiment_attempts WHERE provenance_status = 'recorded'")):
        raise RuntimeError("Recorded attempt provenance exists; retain the additive schema when rolling back code")
    if op.get_bind().scalar(sa.text("SELECT count(*) FROM results WHERE attempt <> 0")):
        raise RuntimeError("Attempt results exist; retain the additive schema when rolling back code")
    op.drop_index("ix_runs_attempt_page", table_name="runs")
    op.drop_constraint("uq_results_experiment_attempt", "results", type_="unique")
    op.drop_index("ix_results_experiment_id", table_name="results")
    op.create_index("ix_results_experiment_id", "results", ["experiment_id"], unique=True)
    op.drop_column("results", "attempt")
    op.drop_table("experiment_attempts")
