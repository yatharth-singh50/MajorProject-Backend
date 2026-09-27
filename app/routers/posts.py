import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import get_current_user, get_current_user_optional
from app.crud import notifications as notifications_crud
from app.crud import posts as posts_crud
from app.db.mongodb import get_db
from app.db.postgres import AsyncSessionLocal, get_session_dep
from app.models.schemas import PostCreate, PostOut
from app.models.sql_models import User
from app.services.language_id import KNOWN_LANGUAGES
from app.services.ml_pipeline import run_full_pipeline
from app.services.text_classifier import classifier_backend_name
from app.utils.media import decode_and_validate_image
from app.websockets.manager import manager

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["posts"])


async def _run_pipeline_and_persist(post_id: str, content: str, raw_image_bytes: Optional[bytes]) -> None:
    """Runs in the background so `POST /posts` returns immediately with a
    'processing' post, matching the frontend's optimistic-then-analyzed flow."""

    mongo_db = get_db()
    try:
        result = await run_full_pipeline(content, raw_image_bytes)
        updated_doc = await posts_crud.apply_analysis_result(mongo_db, post_id, result)
    except Exception:  # noqa: BLE001
        logger.exception("ML pipeline failed for post %s", post_id)
        await posts_crud.mark_analysis_failed(mongo_db, post_id, "Analysis failed unexpectedly.")
        updated_doc = await posts_crud.get_post_doc(mongo_db, post_id)

    if updated_doc is None:
        return

    verdict = updated_doc.get("analysis", {}).get("verdict")
    if verdict in ("fake", "uncertain"):
        label = "flagged as disputed" if verdict == "fake" else "flagged as needing more context"
        await notifications_crud.create_notification(
            mongo_db,
            user_id=updated_doc["authorId"],
            notif_type="verdict_ready",
            message=f"Your post was analyzed and {label}",
            post_id=post_id,
        )

    # Hydrate with a fresh Postgres session since this runs outside any request scope.
    async with AsyncSessionLocal() as session:
        hydrated = await posts_crud.hydrate_post(session, mongo_db, updated_doc, current_user_id=None)
    await manager.broadcast_post_updated(hydrated)


@router.post("/posts", response_model=PostOut, status_code=status.HTTP_201_CREATED)
async def create_post(
    payload: PostCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session_dep),
):
    mongo_db = get_db()

    if payload.parentId:
        parent = await posts_crud.get_post_doc(mongo_db, payload.parentId)
        if parent is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Parent post not found")

    media = None
    raw_image_bytes = None
    if payload.image_base64:
        decoded = decode_and_validate_image(payload.image_base64, payload.image_mime_type)
        raw_image_bytes = decoded.raw_bytes
        media = {"mimeType": decoded.mime_type, "dataBase64": decoded.base64_str, "url": None}

    if payload.languageCode:
        language = {"code": payload.languageCode, "name": KNOWN_LANGUAGES.get(payload.languageCode, payload.languageCode)}
    else:
        language = {"code": "auto", "name": "Detecting…"}

    model_label = "MuRIL-FND" if classifier_backend_name() == "muril" else "MuRIL-FND (heuristic fallback)"

    doc = await posts_crud.create_post(
        mongo_db,
        author_id=current_user.id,
        content=payload.content,
        language=language,
        parent_id=payload.parentId,
        media=media,
        model_label=model_label,
    )

    hydrated = await posts_crud.hydrate_post(session, mongo_db, doc, current_user_id=current_user.id)
    await manager.broadcast_post_created(hydrated)

    if payload.parentId and parent is not None and parent["authorId"] != current_user.id:
        await notifications_crud.create_notification(
            mongo_db,
            user_id=parent["authorId"],
            notif_type="reply",
            message=f"{current_user.display_name} replied to your post",
            actor_id=current_user.id,
            post_id=payload.parentId,
        )

    background_tasks.add_task(_run_pipeline_and_persist, doc["_id"], payload.content, raw_image_bytes)

    return hydrated


@router.get("/posts/{post_id}", response_model=PostOut)
async def get_post(
    post_id: str,
    session: AsyncSession = Depends(get_session_dep),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    mongo_db = get_db()
    doc = await posts_crud.get_post_doc(mongo_db, post_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Post not found")
    current_user_id = current_user.id if current_user else None
    return await posts_crud.hydrate_post(session, mongo_db, doc, current_user_id)


@router.get("/posts/{post_id}/replies", response_model=list[PostOut])
async def get_replies(
    post_id: str,
    session: AsyncSession = Depends(get_session_dep),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    mongo_db = get_db()
    docs = await posts_crud.get_replies_docs(mongo_db, post_id)
    current_user_id = current_user.id if current_user else None
    return await posts_crud.hydrate_posts(session, mongo_db, docs, current_user_id)


@router.post("/posts/{post_id}/like", response_model=PostOut)
async def like_post(
    post_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session_dep),
):
    mongo_db = get_db()
    doc = await posts_crud.toggle_like(mongo_db, post_id, current_user.id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Post not found")
    if current_user.id in doc.get("likedBy", []):
        await notifications_crud.create_notification(
            mongo_db,
            user_id=doc["authorId"],
            notif_type="like",
            message=f"{current_user.display_name} liked your post",
            actor_id=current_user.id,
            post_id=post_id,
        )
    hydrated = await posts_crud.hydrate_post(session, mongo_db, doc, current_user_id=current_user.id)
    await manager.broadcast_post_updated(hydrated)
    return hydrated


@router.post("/posts/{post_id}/repost", response_model=PostOut)
async def repost_post(
    post_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session_dep),
):
    mongo_db = get_db()
    doc = await posts_crud.toggle_repost(mongo_db, post_id, current_user.id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Post not found")
    if current_user.id in doc.get("repostedBy", []):
        await notifications_crud.create_notification(
            mongo_db,
            user_id=doc["authorId"],
            notif_type="repost",
            message=f"{current_user.display_name} reposted your post",
            actor_id=current_user.id,
            post_id=post_id,
        )
    hydrated = await posts_crud.hydrate_post(session, mongo_db, doc, current_user_id=current_user.id)
    await manager.broadcast_post_updated(hydrated)
    return hydrated
