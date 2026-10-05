"""
Stage of the analysis pipeline: real-time verification.

Retrieves live web evidence via DuckDuckGo (app.services.ai.search_tool --
no API key needed, same tool regardless of AI_MODE) and asks a reasoning
model (local qwen2.5:3b, or Groq in the cloud -- app.services.ai.router) to
judge, using ONLY that retrieved evidence, whether the claim is supported,
contradicted, mixed, or there's insufficient evidence either way.

This is a genuinely SEPARATE signal from the MuRIL text classifier's
verdict (see ml_pipeline.py and AnalysisOut in app/models/schemas.py). A
claim can get a confident MuRIL "real" prediction -- a style/pattern
judgment about how the text is written -- while this stage independently
finds the opposite from live evidence (e.g. an RBI/PIB denial of a
recurring "notes are being discontinued" hoax), or vice versa. Never let
one override or silently stand in for the other.

Controlled by ENABLE_REALTIME_VERIFICATION (default True). Set to False to
skip this stage entirely (e.g. a fully offline demo), in which case
verificationStatus stays "unavailable".
"""

import logging

from app.services.ai import router as ai_router
from app.services.ai import search_tool

logger = logging.getLogger(__name__)


def _reconcile_status(model_status: str, matched_claims: list[dict]) -> str:
    """The reasoning model sometimes returns a top-level verificationStatus
    that's inconsistent with the per-citation stances it ALSO returned in
    the same response -- e.g. citing two sources and marking both
    "contradicts", yet still labeling the overall claim "mixed" rather than
    "contradicted". This happens more with smaller local models (qwen2.5:3b)
    than larger cloud ones, but either way: a single free-text category
    label is less reliable than the model's own itemized, structured
    per-source stances, so when those stances unanimously agree, they
    override the top-level label rather than the other way around.

    Only reconciles when every cited stance agrees -- any genuine mix of
    supports/contradicts is left as "mixed" rather than second-guessed, and
    an empty citation list (nothing confidently cited either way) defers
    entirely to the model's own call."""

    stances = {c["stance"] for c in matched_claims}
    if stances == {"contradicts"}:
        return "contradicted"
    if stances == {"supports"}:
        return "supported"
    return model_status


async def run_verification(claim_text: str, search_query: str) -> dict:
    """Returns:
        {
            "verificationStatus": "supported" | "contradicted" | "mixed" | "insufficient",
            "matchedClaims": [{"title", "source", "stance"}, ...],
            "searchDetail": str,   # for the pipeline's "evidence_retrieval" stage
            "analysisDetail": str, # for the pipeline's "verification" stage
        }
    """

    search_results = await search_tool.web_search(search_query)
    search_detail = (
        f'{len(search_results)} result(s) found via DuckDuckGo for "{search_query}"'
        if search_results
        else f'No results found via DuckDuckGo for "{search_query}"'
    )

    if not search_results:
        return {
            "verificationStatus": "insufficient",
            "matchedClaims": [],
            "summary": None,
            "searchDetail": search_detail,
            "analysisDetail": "No evidence found to analyze",
        }

    analysis = await ai_router.run_evidence_analysis(claim_text, search_results)

    if not analysis.available or not analysis.data:
        # Evidence was found, but nothing could interpret it (local and
        # cloud both unavailable). Report "insufficient" rather than
        # guessing at a status -- never silently claim supported/contradicted
        # without a model actually having judged the evidence.
        return {
            "verificationStatus": "insufficient",
            "matchedClaims": [],
            "summary": None,
            "searchDetail": search_detail,
            "analysisDetail": f"Evidence gathered but not analyzed — {analysis.note}",
        }

    status = analysis.data.get("verificationStatus", "insufficient")
    if status not in ("supported", "contradicted", "mixed", "insufficient"):
        status = "insufficient"
    summary = analysis.data.get("summary") or None

    matched_claims = []
    for citation in analysis.data.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        idx = citation.get("index")
        stance = citation.get("stance")
        if isinstance(idx, int) and 1 <= idx <= len(search_results) and stance in ("supports", "contradicts"):
            result = search_results[idx - 1]
            matched_claims.append({"title": result["title"], "source": result["source"], "stance": stance})

    status = _reconcile_status(status, matched_claims)

    return {
        "verificationStatus": status,
        "matchedClaims": matched_claims,
        "summary": summary,
        "searchDetail": search_detail,
        "analysisDetail": f"Analyzed via {analysis.provider} ({analysis.mode} mode) — {status}",
    }