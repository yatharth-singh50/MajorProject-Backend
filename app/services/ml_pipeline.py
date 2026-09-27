"""
Orchestrates the full analysis pipeline described in
MODEL_DOCUMENTATION.txt's "FINAL APPLICATION WORKFLOW":

    User creates a post -> Backend API -> AI moderation
        -> MuRIL (text) + image model -> multimodal analysis (future)
        -> real-time verification (future) -> evidence + verdict -> frontend

Stages actually implemented right now:
  1. lang_id        -- app.services.language_id
  2. small_lm       -- a cheap heuristic that filters out opinions/questions
                       before spending a full classifier pass on them
  3. transformer     -- app.services.text_classifier (real MuRIL checkpoint
                       when available, heuristic fallback otherwise)
  4. image (optional) -- app.services.image_encoder (feature-extraction only;
                       does not affect the verdict yet -- see that module)
  5. verification (optional/stubbed) -- app.services.verification_service

The return shape exactly matches the frontend's `mlService.js.runPipeline()`
contract (verdict/confidence/model/language/explanation/matchedClaims/
pipeline), plus an extra `image_analysis` key the frontend can ignore until
it grows an image-upload UI.
"""

import re
from typing import Optional

from app.core.config import get_settings
from app.services.image_encoder import encode_image
from app.services.language_id import detect_language
from app.services.text_classifier import classifier_backend_name, classify_text
from app.services.verification_service import fetch_evidence

settings = get_settings()

_EXPLANATIONS = {
    "fake": (
        "Language patterns and structure match signals associated with misinformation "
        "in training data (urgency cues, unverifiable calls to forward, etc.)."
    ),
    "real": "No high-risk misinformation markers detected; claim structure resembles verified reporting.",
    "uncertain": (
        "Statement reads as opinion, personal experience, or a question rather than a "
        "specific, checkable factual claim."
    ),
}

_FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|my|we|we're|our)\b", re.IGNORECASE)


def _looks_like_checkable_claim(text: str) -> bool:
    """Cheap pre-filter (the "small LM triage" stage). Not a trained model --
    just enough of a heuristic to avoid running the full classifier (and
    showing a misleading real/fake stamp) on opinions and questions, mirroring
    the frontend's own seed-data behavior for that class of content."""

    stripped = text.strip()
    if len(stripped) < 12:
        return False
    if stripped.endswith("?"):
        return False
    if _FIRST_PERSON_RE.search(stripped) and len(stripped) < 140:
        return False
    return True


async def run_full_pipeline(
    content: str,
    raw_image_bytes: Optional[bytes] = None,
) -> dict:
    pipeline_stages: list[dict] = []

    # --- Stage 1: language ID -------------------------------------------
    language = detect_language(content)
    pipeline_stages.append(
        {"stage": "lang_id", "label": "Language identification", "detail": f"Detected {language['name']}"}
    )

    # --- Stage 2: small-LM triage ----------------------------------------
    checkable = _looks_like_checkable_claim(content)
    pipeline_stages.append(
        {
            "stage": "small_lm",
            "label": "Small LM triage",
            "detail": (
                "Passed to full classifier"
                if checkable
                else "Low newsworthiness — reads as opinion/question, skipping full classifier"
            ),
        }
    )

    backend_name = classifier_backend_name()
    model_label = settings.TEXT_MODEL_NAME if backend_name == "muril" else f"{settings.TEXT_MODEL_NAME} (heuristic fallback — no checkpoint loaded)"

    # --- Stage 3: transformer classification ------------------------------
    if checkable:
        result = classify_text(content)
        confidence = result.confidence
        verdict = "uncertain" if confidence < settings.UNCERTAIN_CONFIDENCE_THRESHOLD else result.label.lower()
        pipeline_stages.append(
            {
                "stage": "transformer",
                "label": "Transformer classification",
                "detail": f"{result.label} · {confidence * 100:.0f}% confidence ({result.backend})",
            }
        )
    else:
        verdict = "uncertain"
        confidence = 0.4
        pipeline_stages.append(
            {"stage": "transformer", "label": "Transformer classification", "detail": "Skipped — not a checkable factual claim"}
        )

    explanation = _EXPLANATIONS[verdict]

    # --- Stage 4: image (optional, feature-extraction only) ---------------
    image_analysis = None
    if raw_image_bytes is not None:
        img_result = encode_image(raw_image_bytes)
        pipeline_stages.append({"stage": "image", "label": "Image analysis", "detail": img_result.note})
        image_analysis = {
            "backend": img_result.backend,
            "featureSize": img_result.feature_size,
            "note": img_result.note,
        }

    # --- Stage 5: real-time verification (stubbed until enabled) ----------
    matched_claims: list[dict] = []
    if checkable:
        matched_claims = await fetch_evidence(content, language["code"])
        pipeline_stages.append(
            {
                "stage": "verification",
                "label": "Real-time verification",
                "detail": (
                    f"{len(matched_claims)} source(s) checked"
                    if settings.ENABLE_REALTIME_VERIFICATION
                    else "Disabled for this deployment — verdict is model-only, not fact-checked against live sources"
                ),
            }
        )

    return {
        "verdict": verdict,
        "confidence": round(float(confidence), 2),
        "model": model_label,
        "language": language,
        "explanation": explanation,
        "matchedClaims": matched_claims,
        "pipeline": pipeline_stages,
        "image_analysis": image_analysis,
    }
