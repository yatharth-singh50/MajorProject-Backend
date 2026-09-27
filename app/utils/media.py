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
