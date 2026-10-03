"""
Orchestrates the full analysis pipeline described in
MODEL_DOCUMENTATION.txt's "FINAL APPLICATION WORKFLOW":

    User creates a post -> Backend API -> AI moderation
        -> MuRIL (text) + image model -> multimodal analysis (future)
        -> real-time verification (future) -> evidence + verdict -> frontend

Stages implemented right now:
  1. lang_id             -- app.services.language_id
  2. small_lm            -- a cheap heuristic that filters out
                             opinions/questions before spending a full
                             classifier pass on them
  3. transformer          -- app.services.text_classifier (real MuRIL
                             checkpoint when available, heuristic fallback
                             otherwise). This produces `verdict`/`confidence`
                             -- the MODEL CLASSIFICATION signal.
  4. claim_extraction     -- app.services.ai.router (qwen2.5:3b locally,
                             Groq in the cloud). Produces `extractedClaim`.
  5. image (optional)      -- app.services.image_encoder (ViT feature
                             extraction only; does not affect the verdict --
                             see that module)
  6. image_understanding   -- app.services.ai.router (MiniCPM-V locally,
                             Gemini in the cloud). Produces
                             `imageUnderstanding` (OCR + claim from the
                             image) -- also not a verdict on the image.
  7. verification          -- app.services.verification_service (currently a
                             stub -- real evidence retrieval + the separate
                             FACTUAL VERIFICATION signal, `verificationStatus`,
                             is the next stage to build out)

IMPORTANT: `verdict`/`confidence` (model classification) and
`verificationStatus` (factual verification) are deliberately independent
fields, populated by different stages. Never derive one from the other --
see AnalysisOut's docstring in app/models/schemas.py.

The return shape preserves the original frontend `mlService.js.runPipeline()`
contract (verdict/confidence/model/language/explanation/matchedClaims/
pipeline) and extends it additively with extractedClaim/imageUnderstanding/
verificationStatus/aiMode.
"""

import re
from typing import Optional

from app.core.config import get_settings
from app.services.ai import router as ai_router
from app.services.image_encoder import encode_image
from app.services.language_id import detect_language
from app.services.text_classifier import classifier_backend_name, classify_text
from app.services import verification_service

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


def _compute_overall_assessment(
    verdict: str,
    confidence: float,
    explanation: str,
    verification_status: str,
    verification_summary: Optional[str],
) -> dict:
    """Combines the model classification and factual verification into one
    derived "final call" -- WITHOUT mutating either underlying signal (both
    are still returned untouched; see AnalysisOut.overallAssessment's
    docstring). Live evidence takes precedence over the text classifier's
    style/pattern judgment, because it's checked against current reality:

    - "contradicted"  -> overall is "fake" at high confidence, regardless of
                          how confident the model was that it looked real.
    - "supported"     -> overall is "real" at high confidence.
    - "mixed"         -> overall drops to "uncertain" -- genuinely conflicting
                          evidence shouldn't resolve to a confident call.
    - "insufficient"/
      "unavailable"   -> nothing to override with; falls back to the model's
                          own verdict, but confidence is capped since it was
                          never actually checked against live evidence.
    """

    if verification_status == "contradicted":
        return {
            "label": "fake",
            "confidence": 0.95,
            "reason": verification_summary or "Live evidence found contradicts this claim.",
        }
    if verification_status == "supported":
        return {
            "label": "real",
            "confidence": 0.9,
            "reason": verification_summary or "Live evidence found supports this claim.",
        }
    if verification_status == "mixed":
        return {
            "label": "uncertain",
            "confidence": 0.5,
            "reason": verification_summary or "Evidence found on this claim is mixed/conflicting.",
        }

    # insufficient or unavailable -- no evidence to weigh in, fall back to
    # the model-only verdict, capped since it's unverified.
    return {
        "label": verdict,
        "confidence": min(confidence, 0.7),
        "reason": explanation,
    }


async def run_full_pipeline(
    content: str,
    raw_image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
) -> dict:
    pipeline_stages: list[dict] = []
    ai_mode = ai_router.current_mode().value

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

    # --- Stage 4: claim extraction (qwen2.5:3b local / Groq cloud) ---------
    extracted_claim = None
    if checkable:
        claim_result = await ai_router.run_claim_extraction(content)
        pipeline_stages.append(
            {
                "stage": "claim_extraction",
                "label": "Claim extraction",
                "detail": (
                    f"Extracted via {claim_result.provider} ({claim_result.mode} mode)"
                    if claim_result.available
                    else f"Unavailable — {claim_result.note}"
                ),
            }
        )
        if claim_result.available and claim_result.data:
            extracted_claim = {
                "claim": claim_result.data.get("claim"),
                "entities": claim_result.data.get("entities") or [],
                "searchQuery": claim_result.data.get("searchQuery"),
                "provider": claim_result.provider,
            }

    # --- Stage 5: image (optional, ViT feature-extraction only) -----------
    image_analysis = None
    if raw_image_bytes is not None:
        img_result = encode_image(raw_image_bytes)
        pipeline_stages.append({"stage": "image", "label": "Image analysis", "detail": img_result.note})
        image_analysis = {
            "backend": img_result.backend,
            "featureSize": img_result.feature_size,
            "note": img_result.note,
        }

    # --- Stage 6: image understanding (MiniCPM-V local / Gemini cloud) ----
    image_understanding = None
    if raw_image_bytes is not None:
        iu_result = await ai_router.run_image_understanding(raw_image_bytes, image_mime_type or "image/jpeg")
        pipeline_stages.append(
            {
                "stage": "image_understanding",
                "label": "Image understanding",
                "detail": (
                    f"Analyzed via {iu_result.provider} ({iu_result.mode} mode)"
                    if iu_result.available
                    else f"Unavailable — {iu_result.note}"
                ),
            }
        )
        if iu_result.available and iu_result.data:
            image_understanding = {
                "ocrText": iu_result.data.get("ocrText"),
                "claim": iu_result.data.get("claim"),
                "provider": iu_result.provider,
            }

    # --- Stage 7: real-time verification (DuckDuckGo evidence + model judgment) --
    matched_claims: list[dict] = []
    verification_status = "unavailable"
    verification_summary = None
    if checkable:
        if settings.ENABLE_REALTIME_VERIFICATION:
            search_query = (extracted_claim or {}).get("searchQuery") or content[:200]
            claim_text = (extracted_claim or {}).get("claim") or content

            verification_result = await verification_service.run_verification(claim_text, search_query)
            verification_status = verification_result["verificationStatus"]
            matched_claims = verification_result["matchedClaims"]
            verification_summary = verification_result["summary"]

            pipeline_stages.append(
                {
                    "stage": "evidence_retrieval",
                    "label": "Evidence retrieval",
                    "detail": verification_result["searchDetail"],
                }
            )
            pipeline_stages.append(
                {
                    "stage": "verification",
                    "label": "Real-time verification",
                    "detail": verification_result["analysisDetail"],
                }
            )
        else:
            pipeline_stages.append(
                {
                    "stage": "verification",
                    "label": "Real-time verification",
                    "detail": (
                        "Disabled for this deployment (ENABLE_REALTIME_VERIFICATION=false) — "
                        "verdict is model-only, not fact-checked against live sources"
                    ),
                }
            )

    overall_assessment = _compute_overall_assessment(
        verdict, float(confidence), explanation, verification_status, verification_summary
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
        "extractedClaim": extracted_claim,
        "imageUnderstanding": image_understanding,
        "overallAssessment": overall_assessment,
        "verificationStatus": verification_status,
        "aiMode": ai_mode,
    }
