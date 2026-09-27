"""Application-owned workspaces; Clerk owns authentication only."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    auth_subject: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Organization(Base):
    __tablename__ = "organizations"
    id = mapped_column(Uuid, primary_key=True, default=uuid4)
    name = mapped_column(String(120), nullable=False)
    slug = mapped_column(String(160), unique=True, nullable=False)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


class OrganizationMembership(Base):
    __tablename__ = "organization_memberships"
    __table_args__ = (CheckConstraint("role IN ('owner', 'member')", name="ck_membership_role"),)
    organization_id = mapped_column(Uuid, ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True)
    user_id = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role = mapped_column(String(16), nullable=False, default="member")


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("organization_id", "slug", name="uq_project_org_slug"),)
    id = mapped_column(Uuid, primary_key=True, default=uuid4)
    organization_id = mapped_column(Uuid, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = mapped_column(String(120), nullable=False)
    slug = mapped_column(String(160), nullable=False)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
