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
from pymongo.errors import OperationFailure

from app.core.config import get_settings

settings = get_settings()

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None

_TEXT_INDEX_NAME = "posts_text_search"


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
    await _ensure_text_index()
    await _db.notifications.create_index([("userId", 1), ("createdAt", -1)])


async def _ensure_text_index() -> None:
    """Text index used by search (`crud/posts.py::search_posts_docs`).

    IMPORTANT: MongoDB text indexes reserve a field (named "language" by
    default) on each document to override which stemming/stopword language
    to use for that document. Our post documents already have a field
    called `language` -- but it holds `{code, name}` (e.g. the post's
    written language for the UI), not a Mongo text-search language string,
    so the default override collides with it and every insert fails with:

        WriteError: found language override field in document with
        non-string type

    Fixed by pointing `language_override` at an unused field name, and by
    setting `default_language="none"` -- posts are multilingual (English,
    Hindi, Tamil, Bengali, ...) and English-specific stemming/stopwords
    wouldn't meaningfully help most of that content anyway, so plain
    tokenization is the safer default here.

    Self-heals on startup: MongoDB allows only one text index per
    collection, auto-named from its fields if you don't pass `name`
    yourself (e.g. the original "content_text_translation_text" this
    project started with). So rather than assume our own index's name, we
    look at whatever text index already exists on `posts` -- if its options
    don't match what we want, we drop *that one* (by its real name,
    whatever it is) and recreate it correctly.
    """
    desired_kwargs = dict(
        default_language="none",
        language_override="textIndexLanguage",
        name=_TEXT_INDEX_NAME,
    )

    existing = await _db.posts.index_information()
    existing_text_index_name = None
    needs_fix = False
    for index_name, info in existing.items():
        if any(v == "text" for _, v in info.get("key", [])):
            existing_text_index_name = index_name
            is_correct = (
                index_name == _TEXT_INDEX_NAME
                and info.get("default_language") == "none"
                and info.get("language_override") == "textIndexLanguage"
            )
            needs_fix = not is_correct
            break

    if existing_text_index_name and needs_fix:
        await _db.posts.drop_index(existing_text_index_name)
        existing_text_index_name = None

    if existing_text_index_name is None:
        try:
            await _db.posts.create_index([("content", "text"), ("translation", "text")], **desired_kwargs)
        except OperationFailure:
            # Another process/worker won the race and already (re)created it
            # between our check and this call -- fine, nothing more to do.
            pass


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
