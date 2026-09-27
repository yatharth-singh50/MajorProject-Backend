from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import get_current_user_optional
from app.crud import posts as posts_crud
from app.db.mongodb import get_db
from app.db.postgres import get_session_dep
from app.models.schemas import PostOut
from app.models.sql_models import User

settings = get_settings()
router = APIRouter(tags=["feed"])


@router.get("/feed", response_model=list[PostOut])
async def get_feed(
    limit: int = Query(default=settings.DEFAULT_FEED_LIMIT, ge=1, le=settings.MAX_FEED_LIMIT),
    session: AsyncSession = Depends(get_session_dep),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    mongo_db = get_db()
    docs = await posts_crud.get_feed_docs(mongo_db, limit)
    current_user_id = current_user.id if current_user else None
    return await posts_crud.hydrate_posts(session, mongo_db, docs, current_user_id)
