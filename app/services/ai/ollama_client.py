"""
Thin async client for a local Ollama server.

Hardware note: on a single consumer GPU (e.g. an RTX 4050, 6GB VRAM) only
one of qwen2.5:3b / llama3.1:8b / minicpm-v realistically fits at a time,
alongside MuRIL + ViT (already resident, loaded by text_classifier.py /
image_encoder.py). Two things keep this safe:

1. `_LOCAL_MODEL_LOCK` serializes every call into Ollama from this process
   -- we never ask Ollama to run two different local models concurrently,
   which is what would actually blow the VRAM budget.
2. Before loading a DIFFERENT model than whichever is currently resident,
   we explicitly evict it first (`ollama.generate(model=current, prompt="",
   keep_alive=0)`) rather than just waiting out its keep_alive timeout --
   this is the exact pattern used for the qwen/llama <-> minicpm-v handoff
   in the NovaAI project, carried over here deliberately rather than
   reinvented, since it's a proven approach on the same hardware. Plain
   `keep_alive` alone (still kept short, see OLLAMA_KEEP_ALIVE) remains the
   fallback for whatever's loaded when the process exits.
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

# Tracks whichever model was loaded by the last call, so we know when a
# switch is happening and an explicit eviction of the previous one is worth
# doing. Only ever read/written while holding _LOCAL_MODEL_LOCK.
_current_loaded_model: Optional[str] = None


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


async def _evict(model: str) -> None:
    """Explicitly frees a model's VRAM right away instead of waiting for its
    keep_alive timeout -- same call shape NovaAI uses for the qwen/llama <->
    minicpm-v handoff. Best-effort: if Ollama is unreachable there's nothing
    to evict anyway, so a failure here is silently ignored rather than
    surfaced (the caller's own request will hit the same unreachable-server
    error right after and handle it properly)."""
    try:
        await _post("/api/generate", {"model": model, "prompt": "", "keep_alive": 0})
    except ProviderUnavailable:
        pass


async def generate(
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    json_mode: bool = False,
    images_base64: Optional[list[str]] = None,
    keep_alive: Optional[object] = None,
    use_options: bool = True,
) -> str:
    """Runs one generation against a local Ollama model and returns the raw
    text response. Serialized against every other local-model call so we
    never try to hold two models in VRAM at once (see module docstring).

    `keep_alive` / `use_options` let a caller deviate from the defaults for
    one specific call -- see router.py's image-understanding call, which
    passes keep_alive=0 and use_options=False to match NovaAI's known-working
    minicpm-v invocation exactly (bare generate call, no JSON mode, no
    options dict, unloaded immediately after)."""

    global _current_loaded_model

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": settings.OLLAMA_KEEP_ALIVE if keep_alive is None else keep_alive,
    }
    if use_options:
        # Explicit, bounded context/output size rather than per-model
        # defaults -- see config.py's OLLAMA_NUM_CTX docstring for why.
        payload["options"] = {
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "num_predict": settings.OLLAMA_NUM_PREDICT,
        }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"
    if images_base64:
        payload["images"] = images_base64

    async with _LOCAL_MODEL_LOCK:
        if _current_loaded_model is not None and _current_loaded_model != model:
            logger.info("Switching local model %s -> %s; evicting %s first", _current_loaded_model, model, _current_loaded_model)
            await _evict(_current_loaded_model)
        data = await _post("/api/generate", payload)
        # keep_alive=0 means Ollama already unloaded it the instant this
        # call finished -- don't remember it as "resident", or the next
        # model switch would issue a pointless (harmless, but wasteful) extra
        # eviction call against a model that's already gone.
        _current_loaded_model = None if payload["keep_alive"] == 0 else model
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
