"""
Real-time evidence retrieval via DuckDuckGo.

Unlike the Ollama/Groq/Gemini clients, this tool needs no API key and isn't
gated by AI_MODE -- it's the exact same search for the local and cloud
reasoning paths alike. A reasoning model (qwen2.5:3b locally, Groq in the
cloud -- see router.run_evidence_analysis) is handed these results and
asked to judge the claim using ONLY what's here, never anything invented.

Uses the `ddgs` package (the current successor to `duckduckgo-search`),
which is synchronous/blocking under the hood, so calls are run in a thread
via `asyncio.to_thread` to avoid blocking the event loop.
"""

import asyncio
import logging

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from app.services.ai.source_policy import domain_of, rank_results

logger = logging.getLogger(__name__)


def _search_sync(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        return ddgs.text(query, max_results=max_results)


async def web_search(query: str, max_results: int = 6) -> list[dict]:
    """Returns a ranked list of {"title", "url", "snippet", "source"} dicts.

    Never raises -- a failed or empty search is valid information (no
    evidence found / search backend unreachable), not a provider outage, so
    callers always get a list back (possibly empty) rather than having to
    catch an exception. An empty result naturally maps to a
    verificationStatus of "insufficient" downstream, not "unavailable"
    (which means the stage didn't run at all)."""

    query = query.strip()
    if not query:
        return []

    try:
        raw = await asyncio.to_thread(_search_sync, query, max_results)
    except (DDGSException, Exception) as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
        return []

    results = [
        {
            "title": (item.get("title") or "").strip(),
            "url": item.get("href") or "",
            "snippet": (item.get("body") or "").strip(),
            "source": domain_of(item.get("href") or ""),
        }
        for item in raw
        if item.get("href")
    ]
    return rank_results(results)
