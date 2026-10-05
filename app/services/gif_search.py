"""
GIF search via the Klipy API, used by the compose box's GIF picker.

Previously used Tenor -- switched because Google fully decommissioned the
public Tenor API on June 30, 2026 (new key registrations had already been
frozen since January 2026; see
https://support.google.com/tenor/answer/10455265). Klipy is the commonly
used drop-in replacement and has a free tier.

Deliberately separate from the image-analysis pipeline: a GIF picked here
is stored on a post as a URL only (schemas.py's PostCreate.gif_url), never
downloaded/decoded into raw_image_bytes, so it never reaches MiniCPM-V/ViT/
Gemini -- per the project's own scope, GIFs are decorative and don't
contribute to fake-news/context-aware analysis, only real uploaded images
do (see routers/posts.py::create_post).

Klipy API notes (docs.klipy.com):
- Auth: the API key is a path segment, not a query param or header --
  `https://api.klipy.com/api/v1/{API_KEY}/gifs/search`.
- Pagination is 1-based `page` + `per_page` (clamped 8-50 by the API),
  not an offset/limit pair.
- Response shape: {"result": true, "data": {"data": [...], "has_next": bool}}.
  Each item has `file.{xs,sm,md,hd}.gif.url` -- `xs`/`sm` make good picker
  thumbnails, `md`/`hd` the actual attached GIF. Items can carry
  `"type": "ad"` when ad params are sent (we don't send any, but they're
  filtered out defensively regardless).
"""

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

KLIPY_BASE_URL = "https://api.klipy.com/api/v1"


def is_configured() -> bool:
    return bool(settings.KLIPY_API_KEY)


def _parse_results(data: dict) -> list[dict]:
    items = ((data.get("data") or {}).get("data")) or []
    results = []
    for item in items:
        if item.get("type") == "ad":
            continue

        files = item.get("file") or {}
        full = files.get("md") or files.get("hd") or files.get("sm") or files.get("xs")
        preview = files.get("xs") or files.get("sm") or full
        full_url = (full or {}).get("gif", {}).get("url")
        if not full_url:
            continue

        results.append(
            {
                "id": str(item.get("id") or item.get("slug") or full_url),
                "url": full_url,
                "previewUrl": (preview or {}).get("gif", {}).get("url", full_url),
                "title": item.get("title") or "",
            }
        )
    return results


async def _get(path: str, params: dict) -> list[dict]:
    url = f"{KLIPY_BASE_URL}/{settings.KLIPY_API_KEY}{path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return _parse_results(resp.json())
    except httpx.HTTPError as exc:
        logger.warning("Klipy request to %s failed: %s", path, exc)
        return []


def _clamp_per_page(limit: int) -> int:
    return max(8, min(limit, 50))  # Klipy's documented per_page bounds


async def search_gifs(query: str, limit: int = 24) -> list[dict]:
    """Returns {"id", "url", "previewUrl", "title"} dicts, or [] if Klipy
    isn't configured or the request fails -- a GIF picker with no results
    is a fine, handleable UI state, never worth raising for."""

    if not is_configured() or not query.strip():
        return []

    params = {"q": query, "page": 1, "per_page": _clamp_per_page(limit), "content_filter": "medium"}
    return await _get("/gifs/search", params)


async def featured_gifs(limit: int = 24) -> list[dict]:
    """Trending GIFs shown before the user types anything -- matches
    Twitter's GIF picker default view."""

    if not is_configured():
        return []

    params = {"page": 1, "per_page": _clamp_per_page(limit), "content_filter": "medium"}
    return await _get("/gifs/trending", params)