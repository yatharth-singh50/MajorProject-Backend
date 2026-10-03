"""
Gemini cloud client -- vision fallback (image understanding, OCR, image
claim extraction, visual question answering). Used automatically in
HYBRID/CLOUD mode when local MiniCPM-V isn't available or isn't installed.
"""

import base64
import json
import logging

import httpx

from app.core.config import get_settings
from app.services.ai.errors import ProviderUnavailable

logger = logging.getLogger(__name__)
settings = get_settings()


def is_configured() -> bool:
    return bool(settings.GEMINI_API_KEY)


async def analyze_image(prompt: str, image_bytes: bytes, mime_type: str, *, json_mode: bool = False) -> str:
    """Sends an image + prompt to Gemini and returns the text response (or
    a JSON string if json_mode=True -- use `analyze_image_json` to get it
    parsed directly)."""

    if not is_configured():
        raise ProviderUnavailable("GEMINI_API_KEY is not set")

    url = f"{settings.GEMINI_BASE_URL}/models/{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
    body: dict = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}},
                ]
            }
        ]
    }
    if json_mode:
        body["generationConfig"] = {"response_mime_type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise ProviderUnavailable(f"Gemini not reachable: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise ProviderUnavailable(f"Gemini returned {exc.response.status_code}: {exc.response.text[:200]}") from exc

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ProviderUnavailable(f"Gemini returned an unexpected shape: {data}") from exc


async def analyze_image_json(prompt: str, image_bytes: bytes, mime_type: str) -> dict:
    raw = await analyze_image(prompt, image_bytes, mime_type, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderUnavailable(f"Gemini did not return valid JSON: {raw[:200]!r}") from exc
