from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user, get_current_user_optional
from app.crud import posts as posts_crud
from app.crud import users as users_crud
from app.db.mongodb import get_db
from app.db.postgres import get_session_dep
from app.models.schemas import PostOut, UserOut, UserUpdate
from app.models.sql_models import User
from app.utils.media import decode_and_validate_image

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{username}", response_model=UserOut)
async def get_user(username: str, session: AsyncSession = Depends(get_session_dep)):
    user = await users_crud.get_user_by_username(session, username)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserOut(**await posts_crud.user_public(get_db(), user))


@router.patch("/{username}", response_model=UserOut)
async def patch_user(
    username: str,
    patch: UserUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session_dep),
):
    if current_user.username != username:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="You can only edit your own profile")

    # Profile photo / banner: client sends a data: URI (already cropped/
    # resized client-side). Re-validated and re-normalized here exactly like
    # post images (utils/media.py), then stored as a clean data: URI so the
    # frontend can drop it straight into an <img src="..."> with no further
    # processing. Uses MAX_IMAGE_SIZE_BYTES, same limit as post images.
    avatar_image_value = None
    if patch.avatar_image is not None:
        decoded = decode_and_validate_image(patch.avatar_image, None)
        avatar_image_value = f"data:{decoded.mime_type};base64,{decoded.base64_str}"

    banner_image_value = None
    if patch.banner_image is not None:
        decoded = decode_and_validate_image(patch.banner_image, None)
        banner_image_value = f"data:{decoded.mime_type};base64,{decoded.base64_str}"

    updated = await users_crud.update_user(
        session,
        current_user,
        {
            "display_name": patch.display_name,
            "bio": patch.bio,
            "location": patch.location,
            "avatar_color": patch.avatar_color,
            "avatar_image": avatar_image_value,
            "banner_image": banner_image_value,
            "languages": patch.languages,
            "auto_analyze": patch.auto_analyze,
            "disputed_threshold": patch.disputed_threshold,
            "default_post_language": patch.default_post_language,
        },
    )
    return UserOut(**await posts_crud.user_public(get_db(), updated))


@router.get("/{username}/posts", response_model=list[PostOut])
async def get_user_posts(
    username: str,
    tab: str = Query(default="posts", pattern="^(posts|replies|likes)$"),
    session: AsyncSession = Depends(get_session_dep),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    user = await users_crud.get_user_by_username(session, username)
    if user is None:
        return []
    mongo_db = get_db()
    docs = await posts_crud.get_user_posts_docs(mongo_db, user.id, tab)
    current_user_id = current_user.id if current_user else None
    return await posts_crud.hydrate_posts(session, mongo_db, docs, current_user_id)
