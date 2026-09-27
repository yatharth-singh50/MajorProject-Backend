"""
Async Postgres engine/session (works with a local Postgres or Supabase).

Users and auth live here because they benefit from real relational
constraints (unique usernames/emails, foreign keys) that a document store
gives up more easily. Everything else (posts, comments, notifications)
lives in MongoDB -- see db/mongodb.py.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.POSTGRES_URI, echo=False, pool_pre_ping=True, future=True)

AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


async def init_postgres() -> None:
    """Create tables if they don't exist yet.

    For a real production rollout, prefer Alembic migrations over this; this
    is here so the project runs with zero extra setup steps.
    """
    # Import models so they're registered on Base.metadata before create_all.
    from app.models import sql_models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_postgres() -> None:
    await engine.dispose()


async def get_session_dep() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
