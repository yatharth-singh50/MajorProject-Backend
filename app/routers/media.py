from fastapi import APIRouter, Query

from app.services import gif_search

router = APIRouter(prefix="/gifs", tags=["media"])


@router.get("/search")
async def search_gifs(q: str = Query(default="")):
    """Powers the compose box's GIF picker. Empty/missing `q` returns
    featured (trending) GIFs, matching Twitter's picker default view.
    `configured: false` lets the frontend show "GIF search isn't set up"
    instead of an empty-results state when TENOR_API_KEY is unset."""

    results = await gif_search.search_gifs(q) if q.strip() else await gif_search.featured_gifs()
    return {"results": results, "configured": gif_search.is_configured()}
