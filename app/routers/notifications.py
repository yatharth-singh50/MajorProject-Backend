from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import get_current_user
from app.crud import notifications as notifications_crud
from app.db.mongodb import get_db
from app.models.schemas import NotificationOut
from app.models.sql_models import User

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=list[NotificationOut])
async def list_notifications(current_user: User = Depends(get_current_user)):
    mongo_db = get_db()
    docs = await notifications_crud.get_notifications_for_user(mongo_db, current_user.id)
    return [
        NotificationOut(
            id=doc["_id"],
            type=doc["type"],
            actorId=doc.get("actorId"),
            actor=None,  # left for the caller to resolve if needed; kept light here
            postId=doc.get("postId"),
            message=doc["message"],
            read=doc.get("read", False),
            createdAt=doc["createdAt"],
        )
        for doc in docs
    ]


@router.post("/{notification_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_read(notification_id: str, current_user: User = Depends(get_current_user)):
    mongo_db = get_db()
    ok = await notifications_crud.mark_notification_read(mongo_db, notification_id, current_user.id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Notification not found")
