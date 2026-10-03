"""CRUD helpers for the User table (Postgres)."""

from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.hashing import hash_password, verify_password
from app.models.sql_models import User


async def get_user_by_username(session: AsyncSession, username: str) -> Optional[User]:
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: str) -> Optional[User]:
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_users_by_ids(session: AsyncSession, user_ids: list[str]) -> dict[str, User]:
    """Batch-fetch users for hydrating a list of posts (avoids N+1 lookups
    across the Mongo/Postgres boundary)."""
    if not user_ids:
        return {}
    result = await session.execute(select(User).where(User.id.in_(set(user_ids))))
    return {u.id: u for u in result.scalars().all()}


async def create_user(
    session: AsyncSession,
    *,
    username: str,
    email: str,
    password: str,
    display_name: Optional[str] = None,
) -> User:
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        display_name=display_name or username,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def get_user_by_identifier(session: AsyncSession, identifier: str) -> Optional[User]:
    """Looks up a user by username OR email -- used at sign-in, where the
    person may type either. Registration still checks the two separately
    (get_user_by_username / get_user_by_email) since both must be unique."""
    result = await session.execute(
        select(User).where(or_(User.username == identifier, User.email == identifier))
    )
    return result.scalar_one_or_none()


async def authenticate_user(session: AsyncSession, identifier: str, password: str) -> Optional[User]:
    user = await get_user_by_identifier(session, identifier)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


async def update_user(session: AsyncSession, user: User, patch: dict) -> User:
    for field, value in patch.items():
        if value is not None and hasattr(user, field):
            setattr(user, field, value)
    await session.commit()
    await session.refresh(user)
    return user
