"""
Groq cloud client -- fast cloud reasoning/evidence fallback.

Used when AI_MODE is "cloud", or in "hybrid" mode when the local Ollama
server isn't reachable (e.g. the app is deployed somewhere without the
developer's GPU). Uses Groq's OpenAI-compatible chat completions API.

IMPORTANT (per project spec): browser/web search and structured JSON output
are treated as SEPARATE stages -- don't assume one request can both search
the web and return clean structured JSON at the same time. `search` and
`chat_json` below are intentionally two different functions for exactly
that reason; a real verification pipeline calls `search` to gather
evidence, then `chat_json` (with the evidence as context) to get a
structured verdict on it.
"""

import json
import logging
from typing import Optional

import httpx

from app.core.config import get_settings
from app.services.ai.errors import ProviderUnavailable

logger = logging.getLogger(__name__)
settings = get_settings()


def is_configured() -> bool:
    return bool(settings.GROQ_API_KEY)


async def _chat(
    messages: list[dict],
    *,
    response_format: Optional[dict] = None,
    tools: Optional[list[dict]] = None,
) -> dict:
    if not is_configured():
        raise ProviderUnavailable("GROQ_API_KEY is not set")

    payload: dict = {"model": settings.GROQ_MODEL, "messages": messages}
    if response_format:
        payload["response_format"] = response_format
    if tools:
        payload["tools"] = tools

    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{settings.GROQ_BASE_URL}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise ProviderUnavailable(f"Groq not reachable: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise ProviderUnavailable(f"Groq returned {exc.response.status_code}: {exc.response.text[:200]}") from exc


async def chat_json(system: str, user: str) -> dict:
    """Structured-output call -- use for claim extraction, evidence
    analysis, or any stage that needs a clean JSON result back."""

    data = await _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    try:
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise ProviderUnavailable(f"Groq did not return valid JSON: {exc}") from exc


async def search(query: str) -> list[dict]:
    """Evidence retrieval via Groq's hosted browser-search tool.

    Deliberately a separate call from `chat_json` -- see module docstring.
    Returns a best-effort list of whatever the tool call surfaces; shape
    depends on Groq's tool-calling response format, which has changed
    before, so treat this as the extension point for real evidence
    retrieval rather than a finished implementation. Check
    https://console.groq.com/docs/agentic-tooling before relying on this.
    """

    data = await _chat([{"role": "user", "content": query}], tools=[{"type": "browser_search"}])
    results: list[dict] = []
    for choice in data.get("choices", []):
        message = choice.get("message", {})
        for item in message.get("tool_calls", []) or []:
            results.append(item)
    return results
