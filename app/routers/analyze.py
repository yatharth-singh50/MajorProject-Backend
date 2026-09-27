from fastapi import APIRouter

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.ml_pipeline import run_full_pipeline
from app.utils.media import decode_and_validate_image

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    """Standalone entry point into the ML pipeline, independent of posting.

    This is the exact swap point the frontend README describes for
    `src/services/mlService.js`:

        export async function runPipeline(content) {
          const res = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: content }),
          });
          return res.json();
        }
    """

    raw_image_bytes = None
    if payload.image_base64:
        decoded = decode_and_validate_image(payload.image_base64, payload.image_mime_type)
        raw_image_bytes = decoded.raw_bytes

    result = await run_full_pipeline(payload.text, raw_image_bytes)
    return AnalyzeResponse(**result)
