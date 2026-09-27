"""Stable prompt identities, movable releases and read-only SDK credentials."""
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, String, Text, Integer, Boolean, ForeignKey, DateTime, JSON, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base


def now():
    return datetime.now(timezone.utc)


class Prompt(Base):
    __tablename__ = "prompts"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_prompt_project_name"),)
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False, default="")
    archived = Column(Boolean, nullable=False, default=False)
    latest_version = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=now, onupdate=now)


class PromptLabel(Base):
    __tablename__ = "prompt_labels"
    __table_args__ = (CheckConstraint("label IN ('staging', 'production')", name="ck_prompt_label"),)
    prompt_id = Column(UUID(as_uuid=True), ForeignKey("prompts.id", ondelete="CASCADE"), primary_key=True)
    label = Column(String(16), primary_key=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id = Column(UUID(as_uuid=True), ForeignKey("prompt_versions.id", ondelete="CASCADE"), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=now, onupdate=now)


class ProjectAPIKey(Base):
    __tablename__ = "project_api_keys"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    prefix = Column(String(20), nullable=False)
    secret_hash = Column(String(64), nullable=False, unique=True)
    scopes = Column(JSON, nullable=False, default=lambda: ["prompts:read"])
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
