"""
SQLAlchemy ORM models. Only accounts live in Postgres -- see the module
docstring in db/postgres.py for the reasoning.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import ARRAY, Boolean, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)

    username: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    display_name: Mapped[str] = mapped_column(String(64), nullable=False)
    bio: Mapped[str] = mapped_column(Text, default="", server_default="")
    location: Mapped[str] = mapped_column(String(120), default="", server_default="")
    avatar_color: Mapped[str] = mapped_column(String(16), default="#D97757", server_default="#D97757")

    # ISO 639-1 codes, e.g. ["en", "hi"] -- matches the frontend's mockData shape.
    languages: Mapped[list] = mapped_column(ARRAY(String(8)), default=list)

    platform_verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    follower_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    following_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # User-level settings mirrored from the Settings page (Home -> per-post
    # preferences, not global app state, live here rather than localStorage).
    auto_analyze: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    disputed_threshold: Mapped[int] = mapped_column(Integer, default=65, server_default="65")
    default_post_language: Mapped[str] = mapped_column(String(8), default="en", server_default="en")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def to_public_dict(self) -> dict:
        """Shape matching the frontend's `seedUsers` / `getUserByUsername` contract."""
        return {
            "id": self.id,
            "username": self.username,
            "displayName": self.display_name,
            "bio": self.bio,
            "location": self.location,
            "avatarColor": self.avatar_color,
            "joinedAt": self.created_at.isoformat() if self.created_at else None,
            "platformVerified": self.platform_verified,
            "languages": self.languages or [],
            "followerCount": self.follower_count,
            "followingCount": self.following_count,
        }
