"""
MongoDB connection (via Motor, the async driver) for everything that isn't
user accounts: posts, replies, likes/reposts, notifications, and each post's
ML analysis payload (including matched-claims/pipeline stage data).

MongoDB is a good fit here because posts are read far more than written,
have a naturally nested/variable shape (media, analysis, matchedClaims,
pipeline stages), and the feed/thread access patterns are simple
lookups + sorts rather than complex joins.
"""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

settings = get_settings()

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB has not been initialized yet. Call init_mongo() at startup.")
    return _db


async def init_mongo() -> None:
    global _client, _db
    _client = AsyncIOMotorClient(settings.MONGODB_URI)
    _db = _client[settings.MONGODB_DB_NAME]

    # Indexes -- created idempotently on startup.
    await _db.posts.create_index([("parentId", 1), ("createdAt", -1)])
    await _db.posts.create_index([("authorId", 1), ("createdAt", -1)])
    await _db.posts.create_index([("content", "text"), ("translation", "text")])
    await _db.notifications.create_index([("userId", 1), ("createdAt", -1)])


async def close_mongo() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def posts_collection():
    return get_db().posts


def notifications_collection():
    return get_db().notifications


def trending_collection():
    return get_db().trending
