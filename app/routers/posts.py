import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import get_current_user, get_current_user_optional
from app.crud import media as media_crud
from app.crud import notifications as notifications_crud
from app.crud import posts as posts_crud
from app.db.mongodb import get_db
from app.db.postgres import AsyncSessionLocal, get_session_dep
from app.models.schemas import PostCreate, PostOut, ThreadOut
from app.models.sql_models import User
from app.services.language_id import KNOWN_LANGUAGES
from app.services.news_detect import has_news_keywords
from app.services.ml_pipeline import run_full_pipeline
from app.services.text_classifier import classifier_backend_name
from app.utils.media import decode_and_validate_image, sniff_media
from app.websockets.manager import manager

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["posts"])


_VERDICT_NOTICE = {
    "fake": "Your news post was fact-checked and flagged as likely fake",
    "uncertain": "Your news post couldn't be verified -- it's marked unverified",
    "real": "Your news post was fact-checked and looks real",
}


async def _run_pipeline_and_persist(post_id: str, content: str, images: list[tuple[bytes, str]]) -> None:
    """Runs in the background so `POST /posts` returns immediately with a
    'processing' post, matching the frontend's optimistic-then-analyzed flow."""

    mongo_db = get_db()
    try:
        result = await run_full_pipeline(content, images)
        updated_doc = await posts_crud.apply_analysis_result(mongo_db, post_id, result)
    except Exception:  # noqa: BLE001
        logger.exception("ML pipeline failed for post %s", post_id)
        await posts_crud.mark_analysis_failed(mongo_db, post_id, "Analysis failed unexpectedly.")
        updated_doc = await posts_crud.get_post_doc(mongo_db, post_id)

    if updated_doc is None:
        return

    overall = (updated_doc.get("analysis") or {}).get("overallAssessment") or {}
    if overall.get("label") in _VERDICT_NOTICE:
        await notifications_crud.create_notification(
            mongo_db,
            user_id=updated_doc["authorId"],
            notif_type="verdict_ready",
            message=_VERDICT_NOTICE[overall["label"]],
            post_id=post_id,
            snippet=notifications_crud.snippet_of(updated_doc["content"]),
        )

    # Hydrate with a fresh Postgres session since this runs outside any request scope.
    async with AsyncSessionLocal() as session:
        hydrated = await posts_crud.hydrate_post(session, mongo_db, updated_doc, current_user_id=None)
    await manager.broadcast_post_updated(hydrated)


async def _collect_attachments(payload: PostCreate, current_user: User, mongo_db):
    """Resolves everything attached to a post into (attachment refs, images
    for the vision model). Only still images are returned for analysis --
    GIFs and videos are shown but never read by the models."""

    attachments: list[dict] = []
    images: list[tuple[bytes, str]] = []

    # Files uploaded through POST /media/upload.
    if payload.mediaIds:
        found = await media_crud.get_media_many(mongo_db, payload.mediaIds)
        for mid in payload.mediaIds:
            doc = found.get(mid)
            if doc is None:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Unknown media id {mid}")
            if doc["ownerId"] != current_user.id:
                raise HTTPException(status.HTTP_403_FORBIDDEN, detail="You can only attach your own uploads")
            attachments.append(media_crud.attachment_ref(doc))
            if doc["kind"] == "image":
                images.append((bytes(doc["data"]), doc["mimeType"]))

    # Legacy single base64 image (older clients / API users).
    if payload.image_base64:
        decoded = decode_and_validate_image(payload.image_base64, payload.image_mime_type)
        kind = (sniff_media(decoded.raw_bytes) or ("image", decoded.mime_type))[0]
        doc = await media_crud.save_media(
            mongo_db, owner_id=current_user.id, kind=kind, mime_type=decoded.mime_type, data=decoded.raw_bytes
        )
        attachments.append(media_crud.attachment_ref(doc))
        if kind == "image":
            images.append((decoded.raw_bytes, decoded.mime_type))

    # GIFs picked from the GIF search are just external URLs.
    for gif_url in [*payload.gifUrls, *([payload.gif_url] if payload.gif_url else [])]:
        if not gif_url.lower().startswith(("https://", "http://")):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="GIF links must be http(s) URLs")
        attachments.append({"id": None, "kind": "gif", "mimeType": "image/gif", "url": gif_url})

    if len(attachments) > settings.MAX_ATTACHMENTS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail=f"A post can have at most {settings.MAX_ATTACHMENTS} attachments"
        )
    return attachments, images


@router.post("/posts", response_model=PostOut, status_code=status.HTTP_201_CREATED)
async def create_post(
    payload: PostCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session_dep),
):
    mongo_db = get_db()

    parent = None
    if payload.parentId:
        parent = await posts_crud.get_post_doc(mongo_db, payload.parentId)
        if parent is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Parent post not found")

    attachments, images = await _collect_attachments(payload, current_user, mongo_db)

    if payload.languageCode:
        language = {"code": payload.languageCode, "name": KNOWN_LANGUAGES.get(payload.languageCode, payload.languageCode)}
    else:
        language = {"code": "auto", "name": "Detecting…"}

    model_label = "MuRIL-FND" if classifier_backend_name() == "muril" else "MuRIL-FND (heuristic fallback)"

    # A post counts as news if the author ticked the News tag OR it carries an
    # obvious news marker (#breaking, #news, "breaking news"). Replies are
    # never news-checked -- they're conversation, not claims.
    is_news = (not payload.parentId) and (payload.isNews or has_news_keywords(payload.content))

    doc = await posts_crud.create_post(
        mongo_db,
        author_id=current_user.id,
        content=payload.content,
        language=language,
        parent_id=payload.parentId,
        attachments=attachments,
        is_news=is_news,
        model_label=model_label,
    )

    # Decide whether this post goes through the fact-check pipeline at all.
    # Each skip is terminal ("skipped", never a perpetual "Analyzing...") and
    # never schedules the background task.
    run_pipeline = False
    tier = (current_user.verification_tier or "").lower()
    if payload.parentId:
        doc = await posts_crud.mark_reply_not_analyzed(mongo_db, doc["_id"])
    elif tier in settings.bypass_pipeline_tiers_set:
        label = {"news": "news", "government": "government"}.get(tier, tier)
        doc = await posts_crud.mark_not_analyzed(
            mongo_db,
            doc["_id"],
            model=f"N/A — verified {label} account",
            explanation=f"Posted by a verified {label} account, so it isn't run through the fact-check pipeline.",
        )
    elif not is_news:
        doc = await posts_crud.mark_not_analyzed(
            mongo_db,
            doc["_id"],
            model="N/A — not a news post",
            explanation="Only posts tagged as news (or marked #breaking / #news) are fact-checked.",
        )
    else:
        run_pipeline = True

    hydrated = await posts_crud.hydrate_post(session, mongo_db, doc, current_user_id=current_user.id)
    await manager.broadcast_post_created(hydrated)

    if payload.parentId and parent is not None and parent["authorId"] != current_user.id:
        await notifications_crud.create_notification(
            mongo_db,
            user_id=parent["authorId"],
            notif_type="reply",
            message=f"{current_user.display_name} replied to your post",
            actor_id=current_user.id,
            post_id=doc["_id"],  # link to the reply itself so the thread opens at it
            snippet=notifications_crud.snippet_of(payload.content),
        )

    if run_pipeline:
        background_tasks.add_task(_run_pipeline_and_persist, doc["_id"], payload.content, images)

    return hydrated


@router.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: str,
    current_user: User = Depends(get_current_user),
):
    mongo_db = get_db()
    doc = await posts_crud.get_post_doc(mongo_db, post_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Post not found")
    if doc["authorId"] != current_user.id and not current_user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="You can only delete your own posts")

    deleted_ids = await posts_crud.delete_post_cascade(mongo_db, post_id)
    await manager.broadcast_post_deleted(deleted_ids)


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


@router.get("/posts/{post_id}/thread", response_model=ThreadOut)
async def get_thread(
    post_id: str,
    session: AsyncSession = Depends(get_session_dep),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """The conversation around a post: its ancestors (root first) and every
    reply beneath it at any depth, so a reply-to-a-reply (e.g. the original
    poster answering a commenter) is shown in context."""
    mongo_db = get_db()
    if await posts_crud.get_post_doc(mongo_db, post_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Post not found")

    ancestors, descendants = await posts_crud.get_thread_docs(mongo_db, post_id)
    current_user_id = current_user.id if current_user else None
    hydrated = await posts_crud.hydrate_posts(session, mongo_db, [*ancestors, *descendants], current_user_id)
    return {"ancestors": hydrated[: len(ancestors)], "replies": hydrated[len(ancestors) :]}


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
            snippet=notifications_crud.snippet_of(doc["content"]),
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
            snippet=notifications_crud.snippet_of(doc["content"]),
        )
    hydrated = await posts_crud.hydrate_post(session, mongo_db, doc, current_user_id=current_user.id)
    await manager.broadcast_post_updated(hydrated)
    return hydrated
