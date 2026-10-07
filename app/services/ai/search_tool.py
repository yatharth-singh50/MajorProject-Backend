"""
Real-time evidence retrieval via DuckDuckGo.

Unlike the Ollama/Groq/Gemini clients, this tool needs no API key and isn't
gated by AI_MODE -- it's the exact same search for the local and cloud
reasoning paths alike. A reasoning model (qwen2.5:3b locally, Groq in the
cloud -- see router.run_evidence_analysis) is handed these results and
asked to judge the claim using ONLY what's here, never anything invented.

Freshness: a claim about something that just happened won't be well covered
by an all-time web search, so each lookup runs two searches in parallel --
DuckDuckGo *News* limited to the past month (these carry publication
dates), and a general web search limited to the past year -- and falls back
to an unrestricted search only if both come back empty. News results are
listed first.

Uses the `ddgs` package (the current successor to `duckduckgo-search`),
which is synchronous/blocking under the hood, so calls are run in a thread
via `asyncio.to_thread` to avoid blocking the event loop.
"""

import asyncio
import logging
from typing import Optional

from ddgs import DDGS

from app.services.ai.source_policy import domain_of, rank_results

logger = logging.getLogger(__name__)


def _news_sync(query: str, max_results: int, timelimit: Optional[str]) -> list[dict]:
    with DDGS() as ddgs:
        return ddgs.news(query, max_results=max_results, timelimit=timelimit)


def _text_sync(query: str, max_results: int, timelimit: Optional[str]) -> list[dict]:
    with DDGS() as ddgs:
        return ddgs.text(query, max_results=max_results, timelimit=timelimit)


async def _safe(fn, query: str, max_results: int, timelimit: Optional[str]) -> list[dict]:
    """One search that can never raise: a backend failing or returning
    nothing is just "no results from this source"."""
    try:
        return await asyncio.to_thread(fn, query, max_results, timelimit) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo %s failed for %r: %s", fn.__name__, query, exc)
        return []


def _normalize(items: list[dict]) -> list[dict]:
    out = []
    for item in items:
        url = item.get("href") or item.get("url") or ""
        if not url:
            continue
        out.append(
            {
                "title": (item.get("title") or "").strip(),
                "url": url,
                "snippet": (item.get("body") or "").strip(),
                "source": domain_of(url),
                # News results carry an ISO publication date; text results don't.
                "date": (item.get("date") or "")[:10] or None,
            }
        )
    return out


async def web_search(query: str, max_results: int = 6) -> list[dict]:
    """Returns a ranked list of {"title", "url", "snippet", "source", "date"}
    dicts, freshest-first where dates are known.

    Never raises -- a failed or empty search is valid information (no
    evidence found / search backend unreachable), not a provider outage, so
    callers always get a list back (possibly empty). An empty result
    naturally maps to a verificationStatus of "insufficient" downstream, not
    "unavailable" (which means the stage didn't run at all)."""

    query = query.strip()
    if not query:
        return []

    news_raw, text_raw = await asyncio.gather(
        _safe(_news_sync, query, max_results, "m"),
        _safe(_text_sync, query, max_results, "y"),
    )
    if not news_raw and not text_raw:
        text_raw = await _safe(_text_sync, query, max_results, None)

    seen: set[str] = set()
    merged: list[dict] = []
    for item in _normalize(news_raw) + _normalize(text_raw):  # news first
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        merged.append(item)

    return rank_results(merged)[:max_results]
