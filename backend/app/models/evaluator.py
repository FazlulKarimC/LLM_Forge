"""Project-scoped evaluator definitions and immutable typed scores."""

from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.models.prompt import now


class Evaluator(Base):
    __tablename__ = "evaluators"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_evaluator_name"),)
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(120), nullable=False)
    description = Column(Text, nullable=False, default="")
    archived = Column(Boolean, nullable=False, default=False)
    latest_version = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)


class EvaluatorVersion(Base):
    __tablename__ = "evaluator_versions"
    __table_args__ = (
        UniqueConstraint("evaluator_id", "version", name="uq_evaluator_version"),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    evaluator_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluators.id", ondelete="CASCADE"),
        nullable=False,
    )
    version = Column(Integer, nullable=False)
    definition = Column(JSON, nullable=False)
    notes = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)


class EvaluationScore(Base):
    __tablename__ = "evaluation_scores"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "case_index", "assignment", "name", name="uq_evaluation_score"
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
    evaluator_version_id = Column(
        UUID(as_uuid=True), ForeignKey("evaluator_versions.id"), nullable=True
    )
    case_index = Column(Integer, nullable=False)
    assignment = Column(String(80), nullable=False)
    name = Column(String(120), nullable=False)
    data_type = Column(String(16), nullable=False)
    value = Column(JSON, nullable=False)
    passed = Column(Boolean, nullable=False)
    required = Column(Boolean, nullable=False, default=True)
    source = Column(String(16), nullable=False)
    reason = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)
