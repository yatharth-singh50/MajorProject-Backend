"""
CRUD + "hydration" for posts/replies (MongoDB), with author info joined in
at the application layer from Postgres (see crud/users.py).

Mirrors the shape and behavior of the frontend's mock `api.js` closely
enough that the frontend's function-by-function "replace the body with a
fetch()" migration path (per its README) lines up field-for-field.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud import users as users_crud
from app.models.sql_models import User


def _new_post_id(is_reply: bool) -> str:
    prefix = "c" if is_reply else "p"
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_analysis(model_label: str) -> dict:
    return {
        "status": "processing",
        "verdict": None,
        "confidence": None,
        "model": model_label,
        "explanation": "",
        "matchedClaims": [],
        "pipeline": [],
        "extractedClaim": None,
        "imageUnderstanding": None,
        "verificationStatus": "unavailable",
        "aiMode": None,
        "overallAssessment": None,
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

async def create_post(
    mongo_db,
    *,
    author_id: str,
    content: str,
    language: dict,
    parent_id: Optional[str],
    attachments: list[dict],
    is_news: bool,
    model_label: str,
) -> dict:
    doc = {
        "_id": _new_post_id(is_reply=bool(parent_id)),
        "authorId": author_id,
        "parentId": parent_id,
        "language": language,
        "content": content,
        "translation": None,
        "media": None,  # legacy single-media field; new posts use `attachments`
        "attachments": attachments,
        "isNews": is_news,
        "createdAt": _now_iso(),
        "stats": {"likes": 0, "reposts": 0, "comments": 0, "views": 1},
        "likedBy": [],
        "repostedBy": [],
        "analysis": _default_analysis(model_label),
    }
    await mongo_db.posts.insert_one(doc)
    return doc


async def apply_analysis_result(mongo_db, post_id: str, result: dict) -> Optional[dict]:
    """Called once the background ML pipeline finishes for a post."""
    analysis = {
        "status": "analyzed",
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "model": result["model"],
        "explanation": result["explanation"],
        "matchedClaims": result["matchedClaims"],
        "pipeline": result["pipeline"],
        "extractedClaim": result.get("extractedClaim"),
        "imageUnderstanding": result.get("imageUnderstanding"),
        "verificationStatus": result.get("verificationStatus", "unavailable"),
        "aiMode": result.get("aiMode"),
        "overallAssessment": result.get("overallAssessment"),
    }
    await mongo_db.posts.update_one(
        {"_id": post_id},
        {"$set": {"analysis": analysis, "language": result["language"]}},
    )
    return await mongo_db.posts.find_one({"_id": post_id})


async def mark_not_analyzed(mongo_db, post_id: str, *, model: str, explanation: str) -> dict:
    """Sets a terminal, non-"processing" analysis state for a post that
    deliberately skips the fact-check pipeline (replies, posts not tagged as
    news, posts from verified news/government accounts), so the UI never
    shows an indefinite "Analyzing..." spinner on it."""

    analysis = {
        "status": "skipped",
        "verdict": None,
        "confidence": None,
        "model": model,
        "explanation": explanation,
        "matchedClaims": [],
        "pipeline": [],
        "extractedClaim": None,
        "imageUnderstanding": None,
        "verificationStatus": "unavailable",
        "aiMode": None,
        "overallAssessment": None,
    }
    await mongo_db.posts.update_one({"_id": post_id}, {"$set": {"analysis": analysis}})
    return await mongo_db.posts.find_one({"_id": post_id})


async def mark_reply_not_analyzed(mongo_db, post_id: str) -> dict:
    return await mark_not_analyzed(
        mongo_db,
        post_id,
        model="N/A — replies aren't analyzed",
        explanation="Replies aren't run through the fact-checking pipeline -- only top-level news posts are.",
    )


async def mark_analysis_failed(mongo_db, post_id: str, reason: str) -> None:
    await mongo_db.posts.update_one(
        {"_id": post_id},
        {"$set": {"analysis.status": "failed", "analysis.explanation": reason}},
    )


async def delete_post_cascade(mongo_db, post_id: str) -> list[str]:
    """Deletes a post and every reply descending from it (replies to
    replies included), so deleting a post never leaves orphaned children
    dangling with a parentId that no longer resolves. Returns every id
    actually deleted, root first, so the caller can broadcast all of them
    (e.g. so another open tab removes them from a feed it's showing)."""

    to_delete = [post_id]
    frontier = [post_id]
    while frontier:
        cursor = mongo_db.posts.find({"parentId": {"$in": frontier}}, {"_id": 1})
        children = [doc["_id"] async for doc in cursor]
        to_delete.extend(children)
        frontier = children

    # Free the uploaded media these posts referenced (images/videos live in
    # their own collection, so deleting the post alone would orphan them).
    media_ids: list[str] = []
    async for doc in mongo_db.posts.find({"_id": {"$in": to_delete}}, {"attachments": 1}):
        media_ids.extend(a["id"] for a in (doc.get("attachments") or []) if a.get("id"))
    if media_ids:
        await mongo_db.media.delete_many({"_id": {"$in": media_ids}})

    await mongo_db.posts.delete_many({"_id": {"$in": to_delete}})
    return to_delete


async def toggle_like(mongo_db, post_id: str, user_id: str) -> Optional[dict]:
    doc = await mongo_db.posts.find_one({"_id": post_id})
    if doc is None:
        return None
    liked_by = set(doc.get("likedBy", []))
    if user_id in liked_by:
        liked_by.discard(user_id)
    else:
        liked_by.add(user_id)
    await mongo_db.posts.update_one(
        {"_id": post_id},
        {"$set": {"likedBy": list(liked_by), "stats.likes": len(liked_by)}},
    )
    return await mongo_db.posts.find_one({"_id": post_id})


async def toggle_repost(mongo_db, post_id: str, user_id: str) -> Optional[dict]:
    doc = await mongo_db.posts.find_one({"_id": post_id})
    if doc is None:
        return None
    reposted_by = set(doc.get("repostedBy", []))
    if user_id in reposted_by:
        reposted_by.discard(user_id)
    else:
        reposted_by.add(user_id)
    await mongo_db.posts.update_one(
        {"_id": post_id},
        {"$set": {"repostedBy": list(reposted_by), "stats.reposts": len(reposted_by)}},
    )
    return await mongo_db.posts.find_one({"_id": post_id})


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

async def get_post_doc(mongo_db, post_id: str) -> Optional[dict]:
    return await mongo_db.posts.find_one({"_id": post_id})


async def get_feed_docs(mongo_db, limit: int) -> list[dict]:
    cursor = mongo_db.posts.find({"parentId": None}).sort("createdAt", -1).limit(limit)
    return [doc async for doc in cursor]


async def get_replies_docs(mongo_db, post_id: str) -> list[dict]:
    cursor = mongo_db.posts.find({"parentId": post_id}).sort("createdAt", 1)
    return [doc async for doc in cursor]


async def get_thread_docs(mongo_db, post_id: str, max_depth: int = 12) -> tuple[list[dict], list[dict]]:
    """Returns (ancestors root-first, descendants oldest-first) for a post.
    Walks up through parentId, and down breadth-first, so a reply-to-a-reply
    can be shown in the context of the whole conversation."""

    ancestors: list[dict] = []
    current = await mongo_db.posts.find_one({"_id": post_id})
    hops = 0
    while current and current.get("parentId") and hops < max_depth:
        parent = await mongo_db.posts.find_one({"_id": current["parentId"]})
        if parent is None:
            break
        ancestors.append(parent)
        current = parent
        hops += 1
    ancestors.reverse()

    descendants: list[dict] = []
    frontier = [post_id]
    depth = 0
    while frontier and depth < max_depth:
        cursor = mongo_db.posts.find({"parentId": {"$in": frontier}})
        children = [d async for d in cursor]
        descendants.extend(children)
        frontier = [c["_id"] for c in children]
        depth += 1
    descendants.sort(key=lambda d: d["createdAt"])
    return ancestors, descendants


async def get_user_posts_docs(mongo_db, author_id: str, tab: str) -> list[dict]:
    if tab == "posts":
        query = {"authorId": author_id, "parentId": None}
    elif tab == "replies":
        query = {"authorId": author_id, "parentId": {"$ne": None}}
    elif tab == "likes":
        query = {"likedBy": author_id}
    else:
        return []
    cursor = mongo_db.posts.find(query).sort("createdAt", -1)
    return [doc async for doc in cursor]


async def search_posts_docs(mongo_db, query: str, limit: int = 25) -> list[dict]:
    try:
        cursor = mongo_db.posts.find(
            {"parentId": None, "$text": {"$search": query}}
        ).limit(limit)
        results = [doc async for doc in cursor]
        if results:
            return results
    except Exception:  # noqa: BLE001 - e.g. no text index yet, or an unsupported test double
        pass

    # $text requires whole-word matches (and a text index); fall back to a
    # substring regex scan so short/partial queries (very common while
    # typing) still return something, and so this still works against a
    # backend without the text index created. Fine at prototype scale; swap
    # for Atlas Search / a real search index if the corpus grows.
    import re

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    cursor = mongo_db.posts.find(
        {"parentId": None, "$or": [{"content": pattern}, {"translation": pattern}]}
    ).limit(limit)
    return [doc async for doc in cursor]


async def count_replies_for(mongo_db, post_ids: list[str]) -> dict[str, int]:
    if not post_ids:
        return {}
    pipeline = [
        {"$match": {"parentId": {"$in": post_ids}}},
        {"$group": {"_id": "$parentId", "count": {"$sum": 1}}},
    ]
    counts: dict[str, int] = {}
    async for row in mongo_db.posts.aggregate(pipeline):
        counts[row["_id"]] = row["count"]
    return counts


# ---------------------------------------------------------------------------
# Trust score ("credibility ring")
# ---------------------------------------------------------------------------

async def trust_stats_for(mongo_db, user_ids: list[str]) -> dict[str, tuple[Optional[int], int]]:
    """{user_id: (score 0-100 or None, number of fact-checked posts)} for
    several users in ONE query.

    The score is built from each fact-checked post's *overall assessment*
    (live evidence decides) -- not MuRIL's raw verdict, which only reads
    writing style and is unreliable. Per fact-checked news post:
        likely real  -> 1.0
        unverified/disputed (uncertain) -> 0.5
        likely fake  -> 0.0
    and the score is the average, as a percentage. Posts that never went
    through the pipeline (ordinary posts, replies, posts from verified
    news/government accounts) don't count either way, so someone who
    never posts news simply has no score yet (None) rather than a made-up
    one. Posts analyzed before overall assessments existed are skipped."""

    if not user_ids:
        return {}

    points: dict[str, float] = {}
    counts: dict[str, int] = {}
    cursor = mongo_db.posts.find(
        {"authorId": {"$in": list(user_ids)}, "analysis.status": "analyzed"},
        {"authorId": 1, "analysis.overallAssessment": 1},
    )
    async for doc in cursor:
        label = ((doc.get("analysis") or {}).get("overallAssessment") or {}).get("label")
        weight = {"real": 1.0, "uncertain": 0.5, "fake": 0.0}.get(label)
        if weight is None:
            continue
        uid = doc["authorId"]
        points[uid] = points.get(uid, 0.0) + weight
        counts[uid] = counts.get(uid, 0) + 1

    return {
        uid: ((round(points[uid] / counts[uid] * 100) if counts.get(uid) else None), counts.get(uid, 0))
        for uid in user_ids
    }


async def trust_score_for(mongo_db, user_id: str) -> Optional[int]:
    return (await trust_stats_for(mongo_db, [user_id])).get(user_id, (None, 0))[0]


async def user_public(mongo_db, user: User) -> dict:
    """A user's public dict including their credibility score/sample size --
    the one place every endpoint should build a UserOut payload from."""
    score, count = (await trust_stats_for(mongo_db, [user.id]))[user.id]
    return {**user.to_public_dict(), "trustScore": score, "trustCount": count}


# ---------------------------------------------------------------------------
# Hydration: attach author + reply counts + likedByMe/repostedByMe
# ---------------------------------------------------------------------------

def _attachments_of(doc: dict) -> list[dict]:
    """The post's attachments, converting the legacy single `media` field
    (base64 image or GIF URL, from before multi-attachment posts) on the fly
    so old posts keep rendering."""
    if doc.get("attachments"):
        return doc["attachments"]
    media = doc.get("media")
    if not media:
        return []
    if media.get("dataBase64"):
        return [{"id": None, "kind": "image", "mimeType": media.get("mimeType", "image/jpeg"),
                 "url": "", "dataBase64": media["dataBase64"]}]
    if media.get("url"):
        return [{"id": None, "kind": "gif", "mimeType": media.get("mimeType", "image/gif"), "url": media["url"]}]
    return []


def _doc_to_out(doc: dict, *, author_out: Optional[dict], reply_count: int, current_user_id: Optional[str]) -> dict:
    liked_by = doc.get("likedBy", [])
    reposted_by = doc.get("repostedBy", [])
    return {
        "id": doc["_id"],
        "authorId": doc["authorId"],
        "author": author_out,
        "parentId": doc.get("parentId"),
        "language": doc["language"],
        "content": doc["content"],
        "translation": doc.get("translation"),
        "media": doc.get("media"),
        "attachments": _attachments_of(doc),
        "isNews": doc.get("isNews", doc["analysis"].get("status") != "skipped"),
        "createdAt": doc["createdAt"],
        "stats": {**doc["stats"], "comments": reply_count},
        "likedByMe": bool(current_user_id and current_user_id in liked_by),
        "repostedByMe": bool(current_user_id and current_user_id in reposted_by),
        "analysis": doc["analysis"],
    }


async def hydrate_posts(
    session: AsyncSession,
    mongo_db,
    docs: list[dict],
    current_user_id: Optional[str],
) -> list[dict]:
    if not docs:
        return []

    author_ids = list({doc["authorId"] for doc in docs})
    authors = await users_crud.get_users_by_ids(session, author_ids)
    reply_counts = await count_replies_for(mongo_db, [doc["_id"] for doc in docs])

    trust = await trust_stats_for(mongo_db, author_ids)

    out = []
    for doc in docs:
        author = authors.get(doc["authorId"])
        author_out = None
        if author is not None:
            score, count = trust.get(author.id, (None, 0))
            author_out = {**author.to_public_dict(), "trustScore": score, "trustCount": count}
        out.append(
            _doc_to_out(
                doc,
                author_out=author_out,
                reply_count=reply_counts.get(doc["_id"], 0),
                current_user_id=current_user_id,
            )
        )
    return out


async def hydrate_post(
    session: AsyncSession,
    mongo_db,
    doc: dict,
    current_user_id: Optional[str],
) -> dict:
    hydrated = await hydrate_posts(session, mongo_db, [doc], current_user_id)
    return hydrated[0]
