"""Storage for uploaded post media (images/videos) in MongoDB.

Each file is its own document holding the raw bytes (BSON Binary) -- not
base64 -- so a 12MB video fits comfortably under Mongo's 16MB document
limit, and posts only carry a small `{id, kind, mimeType, url}` reference.
Files are served by routers/files.py. For a bigger deployment, swap this
module for object storage (S3 / Supabase Storage) and keep the same
`url` contract.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from bson import Binary


async def save_media(mongo_db, *, owner_id: str, kind: str, mime_type: str, data: bytes) -> dict:
    doc = {
        "_id": f"m_{uuid.uuid4().hex[:14]}",
        "ownerId": owner_id,
        "kind": kind,
        "mimeType": mime_type,
        "size": len(data),
        "data": Binary(data),
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    await mongo_db.media.insert_one(doc)
    return doc


async def get_media(mongo_db, media_id: str) -> Optional[dict]:
    return await mongo_db.media.find_one({"_id": media_id})


async def get_media_many(mongo_db, media_ids: list[str]) -> dict[str, dict]:
    if not media_ids:
        return {}
    cursor = mongo_db.media.find({"_id": {"$in": media_ids}})
    return {doc["_id"]: doc async for doc in cursor}


async def delete_media(mongo_db, media_ids: list[str]) -> None:
    if media_ids:
        await mongo_db.media.delete_many({"_id": {"$in": media_ids}})


def attachment_ref(doc: dict) -> dict:
    """The small reference stored on a post (never the bytes themselves)."""
    return {
        "id": doc["_id"],
        "kind": doc["kind"],
        "mimeType": doc["mimeType"],
        "url": f"/media/{doc['_id']}",
    }
