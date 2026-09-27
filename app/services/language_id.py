"""
Stage 1 of the analysis pipeline: language identification.

Two layers, cheapest first:

1. Unicode script ranges for the Indic languages the frontend already knows
   about (`LANGUAGES` in mockData.js) -- instant, zero dependencies, and
   guarantees consistent codes/names with the UI for the common case.
2. `langdetect` as a general-purpose fallback for anything else (including
   non-Indic, non-English languages), so the pipeline supports "languages
   beyond English" in the open-ended way the project calls for, rather than
   only the languages MuRIL/the demo dataset happen to have been trained on.

Returns a `{code, name}` pair matching the frontend's `LanguageOut` shape.
"""

import re
from typing import Optional

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # deterministic results
    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _LANGDETECT_AVAILABLE = False


# Matches the frontend's `LANGUAGES` list in mockData.js so codes/names line up.
KNOWN_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "mr": "Marathi",
}

# ISO 639-1 codes langdetect can return that we haven't named above.
# Extend freely -- this only affects the display name, not detection.
_EXTRA_LANGUAGE_NAMES = {
    "ur": "Urdu",
    "pa": "Punjabi",
    "te": "Telugu",
    "kn": "Kannada",
    "or": "Odia",
    "as": "Assamese",
    "ne": "Nepali",
    "si": "Sinhala",
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "de": "German",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "id": "Indonesian",
}

_SCRIPT_RANGES = [
    ("hi", re.compile(r"[\u0900-\u097F]")),  # Devanagari (Hindi, Marathi share this script)
    ("bn", re.compile(r"[\u0980-\u09FF]")),  # Bengali
    ("ta", re.compile(r"[\u0B80-\u0BFF]")),  # Tamil
    ("gu", re.compile(r"[\u0A80-\u0AFF]")),  # Gujarati
    ("ml", re.compile(r"[\u0D00-\u0D7F]")),  # Malayalam
]


def _language_name(code: str) -> str:
    return KNOWN_LANGUAGES.get(code) or _EXTRA_LANGUAGE_NAMES.get(code) or code.upper()


def detect_language(text: str) -> dict:
    """Returns {"code": str, "name": str}."""

    script_hit = next((code for code, pattern in _SCRIPT_RANGES if pattern.search(text)), None)
    if script_hit:
        # Devanagari is shared by Hindi and Marathi; langdetect can help
        # disambiguate when it's available, otherwise default to Hindi
        # (matches the frontend heuristic exactly).
        if script_hit == "hi" and _LANGDETECT_AVAILABLE:
            detected = _try_langdetect(text)
            if detected == "mr":
                return {"code": "mr", "name": _language_name("mr")}
        return {"code": script_hit, "name": _language_name(script_hit)}

    if _LANGDETECT_AVAILABLE:
        detected = _try_langdetect(text)
        if detected:
            return {"code": detected, "name": _language_name(detected)}

    return {"code": "en", "name": "English"}


def _try_langdetect(text: str) -> Optional[str]:
    try:
        return detect(text)
    except LangDetectException:
        return None
