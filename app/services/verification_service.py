"""
Stage 4 (future) of the analysis pipeline: real-time verification.

Per the model-training repo's docs, this is a "planned backend component"
meant to supplement the local classifier with current external evidence
(news search, fact-check databases, etc.), producing states like VERIFIED /
SUPPORTED / DISPUTED / UNSUPPORTED / NEEDS_CONTEXT / UNVERIFIED.

Nothing here calls an external API yet -- that's a deliberate scope
decision for this stage of the backend (no API keys / evidence sources have
been chosen). This module exists so the pipeline has a clean seam to grow
into: implement `fetch_evidence()` against whatever search/fact-check API
you choose, keep the return shape (a list of MatchedClaim-shaped dicts), and
flip `ENABLE_REALTIME_VERIFICATION` on in config.

Important (per MODEL_DOCUMENTATION.txt): the ML prediction and factual
verification are separate signals. A claim should not be called false
merely because the classifier predicts FAKE, or because this stage finds no
supporting evidence -- callers should treat an empty/disabled result here as
"no external evidence gathered", not as confirmation either way.
"""

from typing import List

from app.core.config import get_settings

settings = get_settings()


async def fetch_evidence(text: str, language_code: str) -> List[dict]:
    """Returns a list of {"title", "source", "stance"} dicts.

    Currently a no-op stub (returns []) unless/until a real evidence source
    is wired in. Kept `async` since any real implementation will be an
    outbound network call.
    """

    if not settings.ENABLE_REALTIME_VERIFICATION:
        return []

    # --- Extension point -------------------------------------------------
    # Example shape for a real implementation:
    #
    #   results = await some_fact_check_client.search(text, lang=language_code)
    #   return [
    #       {"title": r.title, "source": r.source_domain, "stance": r.stance}
    #       for r in results
    #   ]
    return []
