import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, status

from app.core.config import get_settings
from app.core.security import get_current_user
from app.crud import media as media_crud
from app.db.mongodb import get_db
from app.models.sql_models import User
from app.utils.media import sniff_media

settings = get_settings()
router = APIRouter(prefix="/media", tags=["media"])

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_media(file: UploadFile, current_user: User = Depends(get_current_user)):
    """Upload one image or video; returns a reference to attach to a post
    via `mediaIds`. The type is determined from the file's actual bytes,
    not the client-supplied Content-Type."""

    # Read one byte past the larger limit so oversize files are detected
    # without buffering an arbitrarily large upload.
    limit = max(settings.MAX_IMAGE_SIZE_BYTES, settings.MAX_VIDEO_SIZE_BYTES)
    data = await file.read(limit + 1)

    sniffed = sniff_media(data)
    if sniffed is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file. Attach a PNG/JPEG/WebP/GIF image or an MP4/WebM/MOV video.",
        )
    kind, mime = sniffed

    max_bytes = settings.MAX_VIDEO_SIZE_BYTES if kind == "video" else settings.MAX_IMAGE_SIZE_BYTES
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{kind.capitalize()} is too large (max {max_bytes // (1024 * 1024)}MB).",
        )

    doc = await media_crud.save_media(get_db(), owner_id=current_user.id, kind=kind, mime_type=mime, data=data)
    return media_crud.attachment_ref(doc)


@router.get("/{media_id}")
async def serve_media(media_id: str, request: Request):
    """Public (an <img>/<video> tag can't send an auth header). Supports
    HTTP Range so browsers can seek inside videos."""

    doc = await media_crud.get_media(get_db(), media_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Media not found")

    data = bytes(doc["data"])
    total = len(data)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=31536000, immutable",
        "X-Content-Type-Options": "nosniff",
    }

    range_header = request.headers.get("range")
    if range_header:
        m = _RANGE_RE.match(range_header.strip())
        if not m or (m.group(1) == "" and m.group(2) == ""):
            raise HTTPException(status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, detail="Bad Range header")
        if m.group(1) == "":  # suffix range: last N bytes
            start = max(total - int(m.group(2)), 0)
            end = total - 1
        else:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else total - 1
        end = min(end, total - 1)
        if start > end or start >= total:
            raise HTTPException(
                status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail="Range not satisfiable",
                headers={"Content-Range": f"bytes */{total}"},
            )
        headers["Content-Range"] = f"bytes {start}-{end}/{total}"
        return Response(
            content=data[start : end + 1],
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            media_type=doc["mimeType"],
            headers=headers,
        )

    return Response(content=data, media_type=doc["mimeType"], headers=headers)
