"""
Lightweight notifications, generated server-side when someone likes/reposts/
replies to your post, or when a post's analysis finishes. Not part of the
frontend's documented API contract (its Notifications page is currently
static/hardcoded), but a natural extension of "whatever the frontend covers"
-- swap the frontend's hardcoded NOTIFICATIONS array for `GET /notifications`
whenever that page is wired up.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_notification(
    mongo_db,
    *,
    user_id: str,
    notif_type: str,
    message: str,
    actor_id: Optional[str] = None,
    post_id: Optional[str] = None,
) -> None:
    if actor_id == user_id:
        return  # don't notify people about their own actions

    await mongo_db.notifications.insert_one(
        {
            "_id": f"n_{uuid.uuid4().hex[:10]}",
            "userId": user_id,
            "type": notif_type,
            "actorId": actor_id,
            "postId": post_id,
            "message": message,
            "read": False,
            "createdAt": _now_iso(),
        }
    )


async def get_notifications_for_user(mongo_db, user_id: str, limit: int = 50) -> list[dict]:
    cursor = mongo_db.notifications.find({"userId": user_id}).sort("createdAt", -1).limit(limit)
    return [doc async for doc in cursor]


async def mark_notification_read(mongo_db, notification_id: str, user_id: str) -> bool:
    result = await mongo_db.notifications.update_one(
        {"_id": notification_id, "userId": user_id}, {"$set": {"read": True}}
    )
    return result.modified_count > 0
