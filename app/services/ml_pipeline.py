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
  5. image_understanding   -- app.services.ai.router (MiniCPM-V locally,
                             Gemini in the cloud). Produces
                             `imageUnderstanding` (OCR + claim from the
                             image) -- not a verdict on the image.
                             (The separate ViT feature-extraction pass that
                             used to run alongside this has been removed --
                             it had no trained fusion model to feed, and
                             MiniCPM-V's own description covers the useful
                             part. See image_encoder.py if that changes.)
  6. verification          -- app.services.verification_service: real
                             evidence retrieval via DuckDuckGo + a reasoning
                             model judging the claim against it, producing
                             the separate FACTUAL VERIFICATION signal,
                             `verificationStatus`.

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


def _compute_overall_assessment(verification_status: str, verification_summary: Optional[str]) -> dict:
    """Combines signals into one derived "final call" -- WITHOUT mutating
    either underlying signal (both are still returned untouched; see
    AnalysisOut.overallAssessment's docstring).

    IMPORTANT: this deliberately does NOT fall back to MuRIL's verdict under
    any circumstance, including when no evidence was found at all. MuRIL
    only ever detects writing-style patterns -- it is trivially fooled by a
    calmly, formally worded false claim, which is exactly the failure mode
    this whole verification pipeline exists to catch. Letting its opinion
    stand in as the headline "final call" whenever live evidence is
    inconclusive defeats that purpose: a confident "Likely real" badge next
    to an unverified, sensational claim is actively misleading, not a
    neutral default. So:

    - "contradicted"  -> "fake" at high confidence, regardless of how
                          confident the model was that it looked real.
    - "supported"     -> "real" at high confidence.
    - "mixed"         -> "uncertain" -- genuinely conflicting evidence
                          shouldn't resolve to a confident call either way.
    - "insufficient"  -> "uncertain" at low confidence: evidence was sought
                          and didn't confirm the claim -- lack of
                          confirmation for a checkable claim is itself a
                          reason for caution, not a green light.
    - "unavailable"   -> "uncertain" at low confidence: verification never
                          ran at all (disabled, or no reachable search),
                          so there is genuinely nothing to confirm this
                          claim with.
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
            "confidence": 0.45,
            "reason": (
                "The evidence found is conflicting and nothing clearly confirms this claim. "
                + (verification_summary or "Treat it as unverified.")
            ),
        }
    if verification_status == "insufficient":
        return {
            "label": "uncertain",
            "confidence": 0.35,
            "reason": "Not enough evidence to prove this claim. It may be fake -- treat it with caution."
            + (f" {verification_summary}" if verification_summary else ""),
        }

    # "unavailable" -- verification never ran at all.
    return {
        "label": "uncertain",
        "confidence": 0.3,
        "reason": "This claim hasn't been fact-checked against live sources, so it can't be confirmed. "
        "It may be fake -- treat it with caution.",
    }


async def run_full_pipeline(
    content: str,
    images: Optional[list[tuple[bytes, str]]] = None,
) -> dict:
    """`images` is a list of (raw_bytes, mime_type) for the post's real image
    attachments. GIFs and videos are never passed in -- only still images
    are read by the vision model."""

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

    # --- Stage 5: image understanding (MiniCPM-V local / Gemini cloud) ----
    # The separate ViT feature-extraction pass (image_encoder.py) has been
    # removed from the pipeline: it only ever produced a feature vector with
    # no trained fusion model to consume it, and MiniCPM-V's own OCR/claim
    # description below does the actually-useful part of "look at the
    # image" on its own. image_encoder.py is left in place, unused, as a
    # starting point if a real trained fusion model shows up later.
    image_analysis = None
    image_understanding = None
    if images:
        analyzed = images[: settings.MAX_ANALYZED_IMAGES]
        texts: list[str] = []
        claim_from_image = None
        providers: set[str] = set()
        notes: list[str] = []
        for idx, (img_bytes, img_mime) in enumerate(analyzed, start=1):
            iu = await ai_router.run_image_understanding(img_bytes, img_mime or "image/jpeg")
            if iu.available and iu.data:
                providers.add(iu.provider)
                text = (iu.data.get("ocrText") or "").strip()
                if text:
                    texts.append(f"Image {idx}: {text}" if len(analyzed) > 1 else text)
                claim_from_image = claim_from_image or iu.data.get("claim")
            else:
                notes.append(iu.note)

        if providers:
            detail = f"Read {len(analyzed)} image(s) via {', '.join(sorted(providers))} ({ai_mode} mode)"
            if len(images) > len(analyzed):
                detail += f" — first {len(analyzed)} of {len(images)} analyzed"
            image_understanding = {
                "ocrText": "\n\n".join(texts) or None,
                "claim": claim_from_image,
                "provider": ", ".join(sorted(providers)),
            }
        else:
            detail = f"Unavailable — {notes[0] if notes else 'no vision model reachable'}"
        pipeline_stages.append({"stage": "image_understanding", "label": "Image understanding", "detail": detail})

    # --- Stage 7: real-time verification (DuckDuckGo evidence + model judgment) --
    matched_claims: list[dict] = []
    verification_status = "unavailable"
    verification_summary = None
    if checkable:
        if settings.ENABLE_REALTIME_VERIFICATION:
            search_query = (extracted_claim or {}).get("searchQuery") or content[:200]
            claim_text = (extracted_claim or {}).get("claim") or content
            # Text visible in an attached image (a screenshot of a "headline",
            # say) is part of what's being claimed -- give the evidence judge
            # that context rather than only the caption.
            if image_understanding and image_understanding.get("ocrText"):
                claim_text += f"\n(Text shown in the attached image: {image_understanding['ocrText'][:400]})"

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

    if checkable:
        overall_assessment = _compute_overall_assessment(verification_status, verification_summary)
    else:
        # Opinions/questions never go through verification at all (nothing
        # to override with), so just surface the triage result directly --
        # the "hasn't been checked, treat with caution" framing above is for
        # claims that were worth checking and came back unconfirmed, not for
        # content that was never a factual claim to begin with.
        overall_assessment = {"label": verdict, "confidence": float(confidence), "reason": explanation}

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
