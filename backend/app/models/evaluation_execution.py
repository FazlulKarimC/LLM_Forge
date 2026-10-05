"""Durable execution ownership and stage checkpoints; never provider credentials."""

from uuid import uuid4
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base
from app.models.prompt import now


class EvaluationExecution(Base):
    __tablename__ = "evaluation_executions"
    __table_args__ = (
        UniqueConstraint("run_id", "attempt", name="uq_evaluation_attempt"),
        Index("ix_evaluation_claim", "status", "lease_until"),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="queued", index=True)
    owner = Column(String(36), nullable=True)
    lease_until = Column(DateTime(timezone=True), nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    request_id = Column(String(36), nullable=True)
    failure_code = Column(String(40), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=now)


class EvaluationCheckpoint(Base):
    __tablename__ = "evaluation_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "case_index", "stage", name="uq_evaluation_checkpoint"
        ),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    case_index = Column(Integer, nullable=False)
    stage = Column(String(80), nullable=False)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)


class EvaluationIdempotency(Base):
    __tablename__ = "evaluation_idempotency"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "actor", "operation", "key", name="uq_evaluation_idempotency"
        ),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    actor = Column(String(80), nullable=False)
    operation = Column(String(100), nullable=False)
    key = Column(String(128), nullable=False)
    fingerprint = Column(String(64), nullable=False)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    expires_at = Column(DateTime(timezone=True), nullable=False)


class EvaluationCapacity(Base):
    """A single row serializes global claim admission without long-held locks."""

    __tablename__ = "evaluation_capacity"
    id = Column(Integer, primary_key=True)
