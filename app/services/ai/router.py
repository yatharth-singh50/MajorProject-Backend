"""
Model routing: decides, per AI_MODE, whether a given pipeline stage runs on
the local Ollama models, a cloud provider (Groq/Gemini), or tries local
first and falls back to cloud.

AI_MODE:
  LOCAL  -- only ever use local Ollama models. If Ollama/the model isn't
            available, the stage is reported "unavailable" -- it never
            silently falls through to a cloud API.
  HYBRID -- (default) try local first; if Ollama isn't reachable or the
            specific model call fails, fall back to the matching cloud
            provider automatically. This is the expected mode for running
            on the developer's own machine.
  CLOUD  -- skip local entirely, always use the cloud provider. This is the
            expected mode once the app is deployed somewhere without the
            developer's GPU (or for a quick settings-flip during dev).

Every function here returns a `StageResult` rather than raising, so a
provider being unavailable is always a normal, handleable outcome for the
pipeline -- never an unhandled exception that would take the whole
analysis down.
"""

import base64
import logging
from enum import Enum
from typing import Optional

from app.core.config import get_settings
from app.services.ai import gemini_client, groq_client, ollama_client
from app.services.ai.errors import ProviderUnavailable
from app.services.ai.time_context import now_context, today_label

logger = logging.getLogger(__name__)
settings = get_settings()


class AIMode(str, Enum):
    local = "local"
    hybrid = "hybrid"
    cloud = "cloud"


def current_mode() -> AIMode:
    try:
        return AIMode(settings.AI_MODE.lower())
    except ValueError:
        logger.warning("Unknown AI_MODE %r, defaulting to hybrid", settings.AI_MODE)
        return AIMode.hybrid


class StageResult:
    """Outcome of one routed AI call.

    `provider` is "ollama" | "groq" | "gemini" | "none" (none = nothing
    available/configured, not an error -- the caller decides how to
    represent that, e.g. verificationStatus="unavailable")."""

    def __init__(self, data: Optional[dict], provider: str, mode: str, available: bool, note: str = ""):
        self.data = data
        self.provider = provider
        self.mode = mode
        self.available = available
        self.note = note


_CLAIM_EXTRACTION_BASE = (
    "You are a fact-checking assistant. Given a social media post, extract the core "
    "checkable factual claim, named entities, and a short web search query that would "
    'help verify it. Respond ONLY with JSON: {"claim": str, "entities": [str], "searchQuery": str}. '
    'If the post is an opinion/question with no checkable claim, respond {"claim": null, "entities": [], "searchQuery": null}.'
)

_IMAGE_UNDERSTANDING_PROMPT_LOCAL = (
    # Plain-text, three-task prompt -- deliberately matches NovaAI's own
    # working minicpm-v prompt verbatim rather than asking for JSON (see
    # run_image_understanding's docstring for why).
    "Task 1: Extract all text visible in this image exactly as written (OCR), if any.\n"
    "Task 2: If the text is not in English, provide a direct English translation.\n"
    "Task 3: State what factual claim, if any, the image appears to make."
)

_IMAGE_UNDERSTANDING_PROMPT_CLOUD = (
    # Gemini has no equivalent JSON-mode crash risk, so it keeps the
    # structured-output prompt for a cleanly parsed ocrText/claim split.
    "Look at this image. If it contains a news headline, screenshot, or caption, transcribe the "
    "visible text verbatim (OCR). Then state what factual claim, if any, the image appears to make. "
    'Respond ONLY with JSON: {"ocrText": str | null, "claim": str | null}.'
)

_EVIDENCE_ANALYSIS_BASE = (
    "You are a careful fact-checking assistant. You will be given a CLAIM and a numbered list of web "
    "search results (source domain, title, and snippet). Using ONLY the information in these results "
    "-- never anything else you may already know -- decide whether the evidence supports, contradicts, "
    "or is mixed/insufficient regarding the SPECIFIC claim as stated.\n\n"
    'Respond ONLY with JSON: {"verificationStatus": "supported" | "contradicted" | "mixed" | "insufficient", '
    '"summary": str, "citations": [{"index": int, "stance": "supports" | "contradicts"}]}.\n\n'
    "How to weigh the evidence:\n"
    "- Judge each result strictly against the claim as stated, not a looser or related version of it. "
    "A result about the general topic (e.g. the search for alien life in general) that does NOT confirm "
    "the specific event claimed (e.g. aliens having actually been detected) COUNTS AS CONTRADICTING or "
    "being irrelevant to the claim -- it is not support just because it shares keywords.\n"
    "- A general reference/encyclopedia-style page about a broad topic (e.g. a Wikipedia overview of "
    '"extraterrestrial life" or "exoplanets" in general) almost NEVER supports a specific, unconfirmed '
    "recent event-claim -- by nature it explains background concepts rather than confirming new specific "
    "events, even when a related search term about a hypothetical/candidate finding appears on it. Only "
    "mark \"supports\" when a result itself reports that the specific event in the claim occurred.\n"
    '- "contradicted": the clear majority of relevant, credible results refute the claim, or one '
    "authoritative source (official denial, a fact-check labeling it false/hoax/misinformation, "
    "scientific consensus/absence of confirmation) directly refutes it. Do not water this down to "
    '"mixed" just because one weak or tangential result could be stretched to sound supportive -- a '
    "single marginal result does not offset a clear majority of credible sources against the claim.\n"
    '- "supported": multiple independent, credible results corroborate the SPECIFIC claim.\n'
    '- "mixed": reserve this ONLY for when comparably credible, specific results genuinely conflict '
    "with each other -- not for one strong consensus plus one weak outlier.\n"
    '- "insufficient": results don\'t address this specific claim either way.\n\n'
    "`summary` must be 1-2 plain sentences a reader can act on (e.g. what the evidence actually shows, "
    "and why), written as if explaining the verdict to the person who posted the claim.\n\n"
    "`index` in citations MUST be the 1-based position of a result you actually used from the list "
    "given -- never invent a source, a number outside the list, or a title/URL not shown to you.\n\n"
    "Worked example of a common mistake to AVOID: claim = \"Government reports confirm aliens found in "
    "the Andromeda galaxy.\" A result titled \"Extragalactic planet - Wikipedia\" that explains what "
    "extragalactic planets are in general is NOT support for this claim -- it never says aliens were "
    "found, let alone by a government. That result is irrelevant/contradicting (it doesn't corroborate "
    "the specific event), not supporting, no matter how topically adjacent it looks."
)


async def run_claim_extraction(text: str) -> StageResult:
    """Text-only reasoning stage: claim/entity extraction + search query
    generation. Local: qwen2.5:3b. Cloud fallback: Groq."""

    mode = current_mode()

    if mode in (AIMode.local, AIMode.hybrid):
        try:
            data = await ollama_client.generate_json(settings.OLLAMA_CLAIM_MODEL, text, system=_claim_system())
            return StageResult(data, "ollama", mode.value, True)
        except ProviderUnavailable as exc:
            logger.info("Local claim extraction unavailable (%s)", exc)
            if mode is AIMode.local:
                return StageResult(None, "none", mode.value, False, str(exc))
            # hybrid -> fall through to cloud below

    try:
        data = await groq_client.chat_json(_claim_system(), text)
        return StageResult(data, "groq", mode.value, True)
    except ProviderUnavailable as exc:
        logger.info("Cloud claim extraction unavailable (%s)", exc)
        return StageResult(None, "none", mode.value, False, str(exc))


async def run_image_understanding(image_bytes: bytes, mime_type: str) -> StageResult:
    """Image stage: OCR / claim extraction from an image. Local: MiniCPM-V
    via Ollama. Cloud fallback: Gemini.

    NOT the fake-news verdict for the image -- this only describes what the
    image says/shows.

    The local call deliberately mirrors the NovaAI project's own
    minicpm-v invocation exactly (see its agent.py::chat_stream): a bare
    `generate` call with `images=[...]`, plain-text output (NOT JSON mode),
    and `keep_alive=0` set on the call itself, not just the pre-eviction.
    Earlier versions of this function used generate_json (grammar-
    constrained JSON decoding) with the default keep_alive -- a heavier
    code path than NovaAI's, and a plausible contributor to a
    Windows/CUDA crash (0xc0000409) that doesn't occur in NovaAI's own
    working setup on the same model. Parsing is intentionally simple (the
    whole response goes into ocrText) rather than asking the model to
    self-structure its own output."""

    mode = current_mode()

    if mode in (AIMode.local, AIMode.hybrid):
        try:
            images_b64 = [base64.b64encode(image_bytes).decode()]
            text = await ollama_client.generate(
                settings.OLLAMA_VISION_MODEL,
                _IMAGE_UNDERSTANDING_PROMPT_LOCAL,
                images_base64=images_b64,
                keep_alive=0,
                use_options=False,
            )
            return StageResult({"ocrText": text.strip() or None, "claim": None}, "ollama", mode.value, True)
        except ProviderUnavailable as exc:
            logger.info("Local image understanding unavailable (%s)", exc)
            if mode is AIMode.local:
                return StageResult(None, "none", mode.value, False, str(exc))

    try:
        data = await gemini_client.analyze_image_json(_IMAGE_UNDERSTANDING_PROMPT_CLOUD, image_bytes, mime_type)
        return StageResult(data, "gemini", mode.value, True)
    except ProviderUnavailable as exc:
        logger.info("Cloud image understanding unavailable (%s)", exc)
        return StageResult(None, "none", mode.value, False, str(exc))


async def run_deep_reasoning(prompt: str) -> StageResult:
    """Optional deeper local reasoning (Llama 3.1 8B) -- invoke sparingly,
    only when a caller decides it's actually necessary (e.g. the fast
    claim-extraction pass came back ambiguous). No cloud equivalent is
    wired up for this specific stage yet; route through
    `groq_client.chat_json` directly if/when one is needed."""

    mode = current_mode()
    if mode in (AIMode.local, AIMode.hybrid):
        try:
            text = await ollama_client.generate(settings.OLLAMA_DEEP_MODEL, prompt)
            return StageResult({"text": text}, "ollama", mode.value, True)
        except ProviderUnavailable as exc:
            logger.info("Local deep reasoning unavailable (%s)", exc)
            if mode is AIMode.local:
                return StageResult(None, "none", mode.value, False, str(exc))

    return StageResult(None, "none", mode.value, False, "No cloud deep-reasoning stage configured yet")


async def _evidence_pass(user_prompt: str, mode: AIMode) -> StageResult:
    """One attempt at evidence analysis, local-first with cloud fallback --
    factored out of run_evidence_analysis so the escalation path below can
    call it a second time against a different (larger) local model without
    duplicating the local/cloud fallback logic."""

    if mode in (AIMode.local, AIMode.hybrid):
        try:
            data = await ollama_client.generate_json(
                settings.OLLAMA_CLAIM_MODEL, user_prompt, system=_evidence_system()
            )
            return StageResult(data, "ollama", mode.value, True)
        except ProviderUnavailable as exc:
            logger.info("Local evidence analysis unavailable (%s)", exc)
            if mode is AIMode.local:
                return StageResult(None, "none", mode.value, False, str(exc))

    try:
        data = await groq_client.chat_json(_evidence_system(), user_prompt)
        return StageResult(data, "groq", mode.value, True)
    except ProviderUnavailable as exc:
        logger.info("Cloud evidence analysis unavailable (%s)", exc)
        return StageResult(None, "none", mode.value, False, str(exc))


async def run_evidence_analysis(claim: str, search_results: list[dict]) -> StageResult:
    """Reasoning stage that judges verification status from already-
    retrieved web evidence (see ai.search_tool.web_search -- the search
    itself isn't gated by AI_MODE, only this reasoning-over-results step
    is). Local: qwen2.5:3b, with escalation to llama3.1:8b (see below).
    Cloud fallback: Groq.

    `search_results` must be the exact list the caller will index into for
    `matchedClaims` -- this function never invents sources, it only asks a
    model to judge the ones it's given.

    Escalation: a 3B model is the weakest link in this whole pipeline for
    nuanced stance judgment -- it can mislabel a topically-related but
    non-corroborating source (e.g. a general Wikipedia overview) as
    "supports" even with explicit prompt instructions against exactly that.
    A first-pass "mixed" result is precisely the ambiguous case worth a
    second, more careful opinion before settling -- so when that happens
    (and we're not in CLOUD-only mode), the same evidence is re-judged by
    the larger local model (llama3.1:8b) as a tie-breaker. This is the
    "optional deeper local reasoning, invoked when necessary" case
    run_deep_reasoning's own docstring describes, applied specifically to
    verification rather than as a separate caller-invoked stage."""

    mode = current_mode()

    numbered = "\n".join(
        f"{i + 1}. [{r.get('source', '?')}{', ' + r['date'] if r.get('date') else ''}] "
        f"{r.get('title', '')} — {r.get('snippet', '')}"
        for i, r in enumerate(search_results)
    )
    user_prompt = f"CLAIM: {claim}\n\nSEARCH RESULTS:\n{numbered}"

    result = await _evidence_pass(user_prompt, mode)

    if (
        mode in (AIMode.local, AIMode.hybrid)
        and result.available
        and result.data
        and result.data.get("verificationStatus") == "mixed"
    ):
        try:
            logger.info(
                "First-pass evidence analysis (%s) was 'mixed' -- escalating to %s for a second opinion",
                settings.OLLAMA_CLAIM_MODEL,
                settings.OLLAMA_DEEP_MODEL,
            )
            deep_data = await ollama_client.generate_json(
                settings.OLLAMA_DEEP_MODEL, user_prompt, system=_evidence_system()
            )
            return StageResult(
                deep_data, "ollama", mode.value, True, note=f"escalated {settings.OLLAMA_CLAIM_MODEL} -> {settings.OLLAMA_DEEP_MODEL}"
            )
        except ProviderUnavailable as exc:
            logger.info("Deep-model escalation unavailable (%s) -- keeping first-pass result", exc)

    return result


def _claim_system() -> str:
    """Claim-extraction prompt, stamped with the current date/time so the
    model knows what 'recent' means and writes a search query aimed at
    current coverage (built per call -- a module-level constant would
    freeze the date at import time)."""
    return (
        f"{now_context()}\n\n{_CLAIM_EXTRACTION_BASE}\n"
        "The searchQuery must be written to surface the MOST RECENT coverage: keep it short and "
        f"specific, and if the post describes a recent or ongoing event, include the month and year "
        f"(it is currently {today_label()})."
    )


def _evidence_system() -> str:
    """Evidence-analysis prompt, stamped with the current date/time."""
    return f"{now_context()}\n\n{_EVIDENCE_ANALYSIS_BASE}"
