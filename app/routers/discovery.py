import re
from collections import Counter
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user_optional
from app.crud import posts as posts_crud
from app.db.mongodb import get_db
from app.db.postgres import get_session_dep
from app.models.schemas import SearchResponse, TrendingTag, UserOut
from app.models.sql_models import User

router = APIRouter(tags=["discovery"])

_HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)


@router.get("/trending", response_model=list[TrendingTag])
async def get_trending(limit: int = Query(default=5, ge=1, le=20)):
    """Computed live from recent top-level posts' hashtags, rather than a
    static seed list -- counts occurrences of `#tag` and reports the most
    common language among posts that used it."""

    mongo_db = get_db()
    docs = await posts_crud.get_feed_docs(mongo_db, limit=500)

    tag_counts: Counter = Counter()
    tag_languages: dict[str, Counter] = {}

    for doc in docs:
        tags = set(_HASHTAG_RE.findall(doc.get("content", "")))
        for tag in tags:
            tag_counts[tag] += 1
            tag_languages.setdefault(tag, Counter())[doc.get("language", {}).get("name", "Unknown")] += 1

    if not tag_counts:
        return []

    top = tag_counts.most_common(limit)
    return [
        TrendingTag(
            tag=tag,
            posts=count,
            language=tag_languages[tag].most_common(1)[0][0] if len(tag_languages[tag]) == 1 else "Multiple",
        )
        for tag, count in top
    ]


@router.get("/search", response_model=SearchResponse)
async def search_all(
    q: str = Query(min_length=1),
    session: AsyncSession = Depends(get_session_dep),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    mongo_db = get_db()
    post_docs = await posts_crud.search_posts_docs(mongo_db, q)
    current_user_id = current_user.id if current_user else None
    posts = await posts_crud.hydrate_posts(session, mongo_db, post_docs, current_user_id)

    like_pattern = f"%{q}%"
    result = await session.execute(
        select(User).where(or_(User.username.ilike(like_pattern), User.display_name.ilike(like_pattern))).limit(25)
    )
    users = result.scalars().all()
    user_outs = []
    for user in users:
        trust_score = await posts_crud.trust_score_for(mongo_db, user.id)
        user_outs.append(UserOut(**user.to_public_dict(), trustScore=trust_score))

    return SearchResponse(posts=posts, users=user_outs)
