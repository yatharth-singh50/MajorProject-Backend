from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.crud import notifications as notifications_crud
from app.crud import users as users_crud
from app.db.mongodb import get_db
from app.db.postgres import get_session_dep
from app.models.schemas import NotificationOut, UserOut
from app.models.sql_models import User

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=list[NotificationOut])
async def list_notifications(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session_dep),
):
    mongo_db = get_db()
    docs = await notifications_crud.get_notifications_for_user(mongo_db, current_user.id)

    # Resolve each notification's actor (who liked/reposted/replied) in one batch.
    actors = await users_crud.get_users_by_ids(session, [d["actorId"] for d in docs if d.get("actorId")])
    return [
        NotificationOut(
            id=doc["_id"],
            type=doc["type"],
            actorId=doc.get("actorId"),
            actor=UserOut(**actors[doc["actorId"]].to_public_dict()) if doc.get("actorId") in actors else None,
            postId=doc.get("postId"),
            snippet=doc.get("snippet"),
            message=doc["message"],
            read=doc.get("read", False),
            createdAt=doc["createdAt"],
        )
        for doc in docs
    ]


@router.get("/unread-count")
async def unread_count(current_user: User = Depends(get_current_user)):
    return {"count": await notifications_crud.count_unread(get_db(), current_user.id)}


@router.post("/read-all", status_code=status.HTTP_204_NO_CONTENT)
async def read_all(current_user: User = Depends(get_current_user)):
    await notifications_crud.mark_all_read(get_db(), current_user.id)


@router.post("/{notification_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_read(notification_id: str, current_user: User = Depends(get_current_user)):
    ok = await notifications_crud.mark_notification_read(get_db(), notification_id, current_user.id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Notification not found")
