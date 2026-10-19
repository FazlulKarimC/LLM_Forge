"""Attempt identity and frozen execution provenance, including legacy unknowns."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.models.prompt import now


class ExperimentAttempt(Base):
    __tablename__ = "experiment_attempts"

    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), primary_key=True)
    number = Column(Integer, primary_key=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(24), nullable=False)
    storage_bytes = Column(Integer, nullable=False, default=0, server_default="0")
    generation_calls = Column(Integer, nullable=True)
    config = Column(JSON, nullable=True)
    run_manifest = Column(JSON, nullable=True)
    dataset_hash = Column(String(64), nullable=True)
    sample_ids = Column(JSON, nullable=True)
    provenance_status = Column(String(24), nullable=False, default="recorded")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)
