"""
Notifications, generated server-side when someone likes/reposts/replies to
your post, or when your news post's fact-check finishes. Read by the
frontend's Notifications page and the unread badge in the sidebar; a
lightweight `notification` event is pushed over the WebSocket so an open
tab can refresh instantly instead of polling.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from app.websockets.manager import manager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def snippet_of(content: str, limit: int = 110) -> str:
    content = " ".join((content or "").split())
    return content if len(content) <= limit else content[: limit - 1].rstrip() + "…"


async def create_notification(
    mongo_db,
    *,
    user_id: str,
    notif_type: str,
    message: str,
    actor_id: Optional[str] = None,
    post_id: Optional[str] = None,
    snippet: Optional[str] = None,
) -> None:
    if actor_id == user_id:
        return  # don't notify people about their own actions

    # Like/repost are toggles: liking, unliking and re-liking shouldn't stack
    # up a pile of identical notifications while the first is still unread.
    if notif_type in ("like", "repost"):
        existing = await mongo_db.notifications.find_one(
            {"userId": user_id, "type": notif_type, "actorId": actor_id, "postId": post_id, "read": False}
        )
        if existing:
            return

    await mongo_db.notifications.insert_one(
        {
            "_id": f"n_{uuid.uuid4().hex[:10]}",
            "userId": user_id,
            "type": notif_type,
            "actorId": actor_id,
            "postId": post_id,
            "snippet": snippet,
            "message": message,
            "read": False,
            "createdAt": _now_iso(),
        }
    )
    await manager.broadcast({"type": "notification", "userId": user_id})


async def get_notifications_for_user(mongo_db, user_id: str, limit: int = 60) -> list[dict]:
    cursor = mongo_db.notifications.find({"userId": user_id}).sort("createdAt", -1).limit(limit)
    return [doc async for doc in cursor]


async def count_unread(mongo_db, user_id: str) -> int:
    return await mongo_db.notifications.count_documents({"userId": user_id, "read": False})


async def mark_notification_read(mongo_db, notification_id: str, user_id: str) -> bool:
    result = await mongo_db.notifications.update_one(
        {"_id": notification_id, "userId": user_id}, {"$set": {"read": True}}
    )
    return result.matched_count > 0


async def mark_all_read(mongo_db, user_id: str) -> None:
    await mongo_db.notifications.update_many({"userId": user_id, "read": False}, {"$set": {"read": True}})
