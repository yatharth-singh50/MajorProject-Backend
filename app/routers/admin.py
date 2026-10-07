from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_admin
from app.crud import users as users_crud
from app.crud.posts import user_public
from app.db.mongodb import get_db
from app.db.postgres import get_session_dep
from app.models.schemas import UserOut, VerificationUpdate
from app.models.sql_models import VALID_TIERS, User

router = APIRouter(prefix="/admin", tags=["admin"])


@router.put("/users/{username}/verification", response_model=UserOut)
async def set_verification(
    username: str,
    body: VerificationUpdate,
    admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session_dep),
):
    """Grant or revoke a verification tier:
        gold       -- verified person (the "blue tick" equivalent, shown yellow)
        news       -- news channel (red tick); posts skip the fact-check pipeline
        government -- official government handle (green tick); skips the pipeline
        company    -- established company (black/white tick)
        null       -- remove verification
    """
    if body.tier is not None and body.tier not in VALID_TIERS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"tier must be one of {list(VALID_TIERS)} or null")

    target = await users_crud.get_user_by_username(session, username)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="User not found")
    if target.is_admin:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Admins are always verified; their tier can't be changed")

    target.verification_tier = body.tier
    await session.commit()
    await session.refresh(target)
    return UserOut(**await user_public(get_db(), target))
