class ProviderUnavailable(Exception):
    """Raised by any provider client (Ollama, Groq, Gemini) when it can't be
    reached, isn't configured, or returns something unusable.

    This is the single signal `router.py` uses to decide whether to fall
    back to the next provider in the chain (local -> cloud) or report a
    stage as unavailable. It is NOT used for "the model ran fine and
    concluded nothing" -- that's a normal, valid result (e.g. verification
    status "insufficient"), not a provider failure.
    """
