"""Durable evaluation attempts, checkpoints and idempotent dispatch."""

from alembic import op
import sqlalchemy as sa

revision = "o8p9q0r1s2t3"
down_revision = "n7o8p9q0r1s2"
branch_labels = None
depends_on = None


def project():
    return sa.Column(
        "project_id",
        sa.Uuid(),
        sa.ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )


def run():
    return sa.Column(
        "run_id",
        sa.Uuid(),
        sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade():
    op.create_table(
        "evaluation_capacity", sa.Column("id", sa.Integer(), primary_key=True)
    )
    op.execute("INSERT INTO evaluation_capacity (id) VALUES (1)")
    op.create_table(
        "evaluation_executions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        project(),
        run(),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("owner", sa.String(36)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("request_id", sa.String(36)),
        sa.Column("failure_code", sa.String(40)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "attempt", name="uq_evaluation_attempt"),
    )
    op.create_table(
        "evaluation_checkpoints",
        sa.Column("id", sa.Uuid(), primary_key=True),
        project(),
        run(),
        sa.Column("case_index", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(80), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "run_id", "case_index", "stage", name="uq_evaluation_checkpoint"
        ),
    )
    op.create_table(
        "evaluation_idempotency",
        sa.Column("id", sa.Uuid(), primary_key=True),
        project(),
        run(),
        sa.Column("actor", sa.String(80), nullable=False),
        sa.Column("operation", sa.String(100), nullable=False),
        sa.Column("key", sa.String(128), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "project_id", "actor", "operation", "key", name="uq_evaluation_idempotency"
        ),
    )
    for table in (
        "evaluation_executions",
        "evaluation_checkpoints",
        "evaluation_idempotency",
    ):
        op.create_index(f"ix_{table}_project_id", table, ["project_id"])
    for table in ("evaluation_executions", "evaluation_checkpoints"):
        op.create_index(f"ix_{table}_run_id", table, ["run_id"])
    op.create_index(
        "ix_evaluation_executions_status", "evaluation_executions", ["status"]
    )
    op.create_index(
        "ix_evaluation_claim", "evaluation_executions", ["status", "lease_until"]
    )
    op.create_index(
        "ix_evaluation_history", "evaluation_runs", ["project_id", "created_at", "id"]
    )


def downgrade():
    if (
        op.get_bind()
        .execute(sa.text("SELECT EXISTS (SELECT 1 FROM evaluation_executions)"))
        .scalar()
    ):
        raise RuntimeError(
            "Export execution history before downgrading durable evaluations"
        )
    op.drop_index("ix_evaluation_history", "evaluation_runs")
    for table in (
        "evaluation_idempotency",
        "evaluation_checkpoints",
        "evaluation_executions",
        "evaluation_capacity",
    ):
        op.drop_table(table)
