"""Versioned test data and reproducible evaluation records (no credentials)."""

from uuid import uuid4
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Boolean,
    ForeignKey,
    DateTime,
    JSON,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base
from app.models.prompt import now


class Dataset(Base):
    __tablename__ = "evaluation_datasets"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_dataset_project_name"),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False, default="")
    archived = Column(Boolean, nullable=False, default=False)
    latest_version = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)


class DatasetRevision(Base):
    __tablename__ = "dataset_revisions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_dataset_revision"),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    version = Column(Integer, nullable=False)
    cases = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_evaluation_status",
        ),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prompt_version_id = Column(
        UUID(as_uuid=True), ForeignKey("prompt_versions.id"), nullable=False
    )
    dataset_revision_id = Column(
        UUID(as_uuid=True), ForeignKey("dataset_revisions.id"), nullable=False
    )
    source_run_id = Column(
        UUID(as_uuid=True), ForeignKey("evaluation_runs.id"), nullable=True, index=True
    )
    # Snapshot names/settings so history remains useful after renaming/archiving.
    config = Column(JSON, nullable=False)
    status = Column(String(16), nullable=False, default="queued")
    total = Column(Integer, nullable=False)
    completed = Column(Integer, nullable=False, default=0)
    passed = Column(Integer, nullable=False, default=0)
    errors = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=now, onupdate=now
    )


class EvaluationResult(Base):
    __tablename__ = "evaluation_case_results"
    __table_args__ = (
        UniqueConstraint("run_id", "case_index", name="uq_evaluation_case"),
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
    )
    case_index = Column(Integer, nullable=False)
    data = Column(JSON, nullable=False)
