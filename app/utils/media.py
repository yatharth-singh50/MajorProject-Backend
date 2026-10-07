"""
Image handling: posts may include an image, which the client sends as a
base64 string. We validate it, decode it to raw bytes for anything that
needs actual pixels (the ViT feature extractor), and persist the base64
form directly on the Mongo post document -- i.e. images are stored as
"byte code" inline with the post, as requested for this stage of the
project. This is simple and fine for a prototype; the obvious next step in
production is swapping this for object storage (S3 / Supabase Storage) and
storing a URL instead, which only requires changing this module.
"""

import base64
import binascii
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, status

from app.core.config import get_settings

settings = get_settings()


@dataclass
class DecodedImage:
    raw_bytes: bytes
    mime_type: str
    base64_str: str  # normalized (no data: prefix)


def decode_and_validate_image(image_base64: str, mime_type: Optional[str]) -> DecodedImage:
    """Accepts either a raw base64 string or a `data:<mime>;base64,<data>` URI."""

    b64_payload = image_base64
    detected_mime = mime_type

    if image_base64.startswith("data:"):
        try:
            header, b64_payload = image_base64.split(",", 1)
            detected_mime = header.split(";")[0].replace("data:", "") or mime_type
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed data URI for image")

    if not detected_mime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_mime_type is required when sending raw base64 (or use a data: URI)",
        )

    if detected_mime not in settings.ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type '{detected_mime}'. Allowed: {settings.ALLOWED_IMAGE_MIME_TYPES}",
        )

    try:
        raw = base64.b64decode(b64_payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 image payload")

    if len(raw) > settings.MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds max size of {settings.MAX_IMAGE_SIZE_BYTES} bytes",
        )

    return DecodedImage(raw_bytes=raw, mime_type=detected_mime, base64_str=b64_payload)


# ---------------------------------------------------------------------------
# Multi-attachment uploads (images + videos)
# ---------------------------------------------------------------------------

ALLOWED_VIDEO_MIME_TYPES = ("video/mp4", "video/webm", "video/quicktime")

# ISO-BMFF "ftyp" brands that are still images (HEIC/AVIF), not video.
_IMAGE_FTYP_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif", b"avis"}


def sniff_media(data: bytes):
    """Identify an upload from its actual bytes -- never trust the client's
    Content-Type, because these files get served back from our own origin
    with whatever type we store. Returns (kind, mime) or None if it isn't a
    supported image/video. kind is "image" | "gif" | "video"."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image", "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif", "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image", "image/webp"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video", "video/webm"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in _IMAGE_FTYP_BRANDS:
            return None
        if brand.startswith(b"qt"):
            return "video", "video/quicktime"
        return "video", "video/mp4"
    return None
