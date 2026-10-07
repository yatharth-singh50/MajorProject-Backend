"""
Async Postgres engine/session (works with a local Postgres or Supabase).

Users and auth live here because they benefit from real relational
constraints (unique usernames/emails, foreign keys) that a document store
gives up more easily. Everything else (posts, comments, notifications)
lives in MongoDB -- see db/mongodb.py.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.POSTGRES_URI, echo=False, pool_pre_ping=True, future=True)

AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


def _add_missing_columns(sync_conn) -> None:
    """Lightweight auto-migration, run after create_all() on every startup.

    `create_all()` only creates TABLES that don't exist yet -- it does
    nothing for a table that already exists but is now missing a column a
    model gained since the table was first created (exactly what happened
    when avatar_image/banner_image were added to User: the live Supabase
    table predated those fields, so `create_all` silently no-op'd on it and
    every query touching the table 500'd with "column ... does not exist").

    This compares each model's declared columns against the live table and
    issues `ALTER TABLE ... ADD COLUMN` for anything missing, so the next
    restart self-heals instead of requiring a manual SQL step. It only ever
    ADDS columns -- it never drops/renames/retypes anything, so it's safe
    to leave running. For renames, type changes, or anything non-additive,
    use a real migration (Alembic) instead.
    """
    inspector = inspect(sync_conn)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue  # brand-new table -- create_all already handled it

        existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_columns:
                continue

            ddl_type = column.type.compile(dialect=sync_conn.dialect)
            clause = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}'

            if column.server_default is not None:
                default_sql = getattr(column.server_default.arg, "text", column.server_default.arg)
                clause += f" DEFAULT {default_sql}"
            elif not column.nullable:
                # Adding a NOT NULL column with no default would fail outright
                # on a non-empty table -- skip and make noise rather than
                # guess at a value. Add it manually (with a DEFAULT, or via a
                # real migration) in this case.
                logger.warning(
                    "Auto-migration: skipping NOT NULL column %s.%s with no server_default -- "
                    "add it manually (needs a DEFAULT on a non-empty table).",
                    table.name,
                    column.name,
                )
                continue

            sync_conn.execute(text(clause))
            logger.info("Auto-migration: added missing column %s.%s", table.name, column.name)


async def init_postgres() -> None:
    """Create any missing tables, then self-heal any missing columns on
    tables that already existed -- see _add_missing_columns' docstring.

    For a real production rollout, prefer Alembic migrations over this; this
    is here so the project runs with zero extra manual-migration steps
    during active development.
    """
    # Import models so they're registered on Base.metadata before create_all.
    from app.models import sql_models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


async def close_postgres() -> None:
    await engine.dispose()


async def get_session_dep() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
