"""
Thin async client for a local Ollama server.

Hardware note: on a single consumer GPU (e.g. an RTX 4050, 6GB VRAM) only
one of qwen2.5:3b / llama3.1:8b / minicpm-v realistically fits at a time,
alongside MuRIL + ViT (already resident, loaded by text_classifier.py /
image_encoder.py). Two things keep this safe without any manual
load/unload bookkeeping on our side:

1. `_LOCAL_MODEL_LOCK` serializes every call into Ollama from this process
   -- we never ask Ollama to run two different local models concurrently,
   which is what would actually blow the VRAM budget.
2. `OLLAMA_KEEP_ALIVE` is kept short (default "30s") so Ollama evicts a
   model from VRAM shortly after a request finishes, freeing room for
   whichever model is needed next, instead of pinning every model used so
   far in memory simultaneously.

This trades a bit of reload latency for staying inside 6GB. If more VRAM
is available, raise OLLAMA_KEEP_ALIVE (e.g. "5m") to avoid reloading qwen
on every single post.
"""

import asyncio
import json
import logging
from typing import Optional

import httpx

from app.core.config import get_settings
from app.services.ai.errors import ProviderUnavailable

logger = logging.getLogger(__name__)
settings = get_settings()

# Only one local model resident on the GPU at a time -- see module docstring.
_LOCAL_MODEL_LOCK = asyncio.Lock()


async def _post(path: str, payload: dict) -> dict:
    url = f"{settings.OLLAMA_BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=settings.OLLAMA_REQUEST_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise ProviderUnavailable(f"Ollama not reachable at {settings.OLLAMA_BASE_URL}: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise ProviderUnavailable(f"Ollama returned {exc.response.status_code}: {exc.response.text[:200]}") from exc


async def generate(
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    json_mode: bool = False,
    images_base64: Optional[list[str]] = None,
) -> str:
    """Runs one generation against a local Ollama model and returns the raw
    text response. Serialized against every other local-model call so we
    never try to hold two models in VRAM at once (see module docstring)."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"
    if images_base64:
        payload["images"] = images_base64

    async with _LOCAL_MODEL_LOCK:
        data = await _post("/api/generate", payload)
    return data.get("response", "")


async def generate_json(
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    images_base64: Optional[list[str]] = None,
) -> dict:
    """Like `generate`, but asks Ollama for JSON output and parses it.

    Raises ProviderUnavailable if the model doesn't return valid JSON --
    callers (router.py) treat that the same as the provider being
    unavailable and fall back to the next one, rather than crashing the
    pipeline on a malformed model response."""

    raw = await generate(model, prompt, system=system, json_mode=True, images_base64=images_base64)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProviderUnavailable(f"Ollama ({model}) did not return valid JSON: {raw[:200]!r}") from exc


async def is_available() -> bool:
    """Quick reachability check -- doesn't take the model lock, since it
    doesn't run a model, just checks the server is up."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
