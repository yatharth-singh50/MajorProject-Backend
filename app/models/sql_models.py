"""
SQLAlchemy ORM models. Only accounts live in Postgres -- see the module
docstring in db/postgres.py for the reasoning.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import ARRAY, Boolean, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import get_settings
from app.db.postgres import Base

VALID_TIERS = ("gold", "news", "government", "company")


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

    # Optional profile images, stored as data: URIs (base64) -- same
    # "images as byte code" approach used for post media. Nullable: a user
    # with neither still falls back to the initials/avatar_color circle and
    # a plain gradient banner on the frontend.
    # NOTE: typed as `Mapped[str]`, not `Mapped[Optional[str]]`, on purpose --
    # SQLAlchemy 2.0's annotation resolution for Optional/Union-typed mapped
    # columns is broken under Python 3.14 (a CPython 3.14 typing.Union change
    # breaks SQLAlchemy's de_stringify_union_elements/make_union_type: "descriptor
    # '__getitem__' requires a 'typing.Union' object but received a 'tuple'").
    # `nullable=True` below is what actually makes the DB column nullable --
    # the Python type hint is just slightly imprecise as a tradeoff to avoid
    # that crash. Safe to switch back to Mapped[Optional[str]] once you're on
    # a SQLAlchemy version with a confirmed fix for this (2.0.41 began Python
    # 3.14 support, but 2.0.37 is still confirmed broken -- check the 2.0.x
    # changelog for the specific de_stringify/annotation fix before assuming
    # any particular later version is safe, then test before reverting this).
    avatar_image: Mapped[str] = mapped_column(Text, nullable=True, default=None)
    banner_image: Mapped[str] = mapped_column(Text, nullable=True, default=None)

    # ISO 639-1 codes, e.g. ["en", "hi"] -- matches the frontend's mockData shape.
    languages: Mapped[list] = mapped_column(ARRAY(String(8)), default=list)

    platform_verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    # Verification tier granted by an admin: "gold" (verified person, like a
    # blue tick), "news" (red -- news channel), "government" (green --
    # official handle), "company" (black/white -- established company).
    # NULL = unverified. (Typed Mapped[str] rather than Optional on purpose --
    # see the Python 3.14 note above.)
    verification_tier: Mapped[str] = mapped_column(String(16), nullable=True, default=None)

    follower_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    following_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # User-level settings mirrored from the Settings page (Home -> per-post
    # preferences, not global app state, live here rather than localStorage).
    auto_analyze: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    disputed_threshold: Mapped[int] = mapped_column(Integer, default=65, server_default="65")
    default_post_language: Mapped[str] = mapped_column(String(8), default="en", server_default="en")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    @property
    def is_admin(self) -> bool:
        return (self.username or "").lower() in get_settings().admin_usernames_set

    @property
    def effective_tier(self):
        """What the UI shows: admins always display as "admin" (gold tick +
        Sathi icon); everyone else shows whatever tier an admin granted."""
        if self.is_admin:
            return "admin"
        return self.verification_tier

    def to_public_dict(self) -> dict:
        """Shape matching the frontend's `seedUsers` / `getUserByUsername` contract."""
        return {
            "id": self.id,
            "username": self.username,
            "displayName": self.display_name,
            "bio": self.bio,
            "location": self.location,
            "avatarColor": self.avatar_color,
            "avatarImage": self.avatar_image,
            "bannerImage": self.banner_image,
            "joinedAt": self.created_at.isoformat() if self.created_at else None,
            "platformVerified": bool(self.effective_tier) or bool(self.platform_verified),
            "verificationTier": self.effective_tier,
            "isAdmin": self.is_admin,
            "languages": self.languages or [],
            "followerCount": self.follower_count,
            "followingCount": self.following_count,
        }
