"""
Prompt Version Model

Stores versioned prompt templates for reproducibility and comparison.
Each version is an immutable snapshot with a SHA-256 hash and an
optional link to its parent version, forming a version history chain.
"""

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, String, Text, Integer, ForeignKey, DateTime, UniqueConstraint, CheckConstraint, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from app.core.database import Base


class PromptVersion(Base):
    """
    Immutable prompt template version.

    Each row is a snapshot of a prompt template.
    Versions are linked via parent_id to form a history chain.
    """
    __tablename__ = "prompt_versions"
    __table_args__ = (
        UniqueConstraint("prompt_id", "version", name="uq_prompt_version_number"),
        CheckConstraint("template_format IN ('mustache', 'fstring')", name="ck_prompt_template_format"),
    )

    prompt_id = Column(PG_UUID(as_uuid=True), ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False, index=True)
    template_format = Column(String(16), nullable=False, default="fstring")

    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    template_text = Column(Text, nullable=False)
    prompt_type = Column(String(8), nullable=False, default="text", server_default="text")
    messages = Column(JSON, nullable=False, default=list, server_default="[]")
    config = Column(JSON, nullable=False, default=dict, server_default="{}")
    created_by = Column(String(16), nullable=False, default="UI", server_default="UI")
    version = Column(Integer, nullable=False, default=1)
    sha256_hash = Column(String(64), nullable=False, index=True)
    parent_id = Column(PG_UUID(as_uuid=True), ForeignKey("prompt_versions.id"), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    @staticmethod
    def compute_hash(template_text: str) -> str:
        """Compute SHA-256 hash of the template text."""
        return hashlib.sha256(template_text.encode("utf-8")).hexdigest()
