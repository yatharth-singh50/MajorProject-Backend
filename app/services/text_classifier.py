"""
Stage 3 of the analysis pipeline: the transformer text classifier.

This wraps the exact checkpoint produced by the model-training repo
(`major-project-model-training/src/train.py` -> `./models/muril_combined`,
loaded the same way as that repo's own `src/predict.py`):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    ...
    LABELS = {0: "REAL", 1: "FAKE"}

If `torch`/`transformers` aren't installed, or the checkpoint directory
isn't present (e.g. running this API without the multi-GB model files
checked out), we transparently fall back to a lightweight heuristic scorer
so the rest of the backend -- routing, storage, auth, the API contract --
stays fully exercisable. Swap in the real checkpoint by setting
`TEXT_MODEL_PATH` and installing `torch`+`transformers`; no other code
changes needed.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

LABELS = {0: "REAL", 1: "FAKE"}

# Mirrors the heuristic in the frontend's mlService.js so behavior is
# consistent (and demoable) even before the real checkpoint is wired in.
_FAKE_SIGNALS = [
    "breaking",
    "share before it's deleted",
    "govt hiding",
    "forward to everyone",
    "not on the news",
]


class TextClassificationResult:
    def __init__(self, label: str, confidence: float, backend: str):
        self.label = label  # "REAL" | "FAKE"
        self.confidence = confidence  # 0..1
        self.backend = backend  # "muril" | "heuristic"


class _MurilBackend:
    """Lazily-loaded singleton wrapping the real MuRIL checkpoint."""

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = None
        self._ready = False

    def load(self) -> bool:
        if self._ready:
            return True

        model_path = Path(settings.TEXT_MODEL_PATH)
        if not model_path.exists():
            logger.warning(
                "TEXT_MODEL_PATH '%s' does not exist -- falling back to the heuristic "
                "classifier. Point this at the checkpoint from the model-training repo "
                "to enable real MuRIL predictions.",
                model_path,
            )
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            logger.warning(
                "torch/transformers not installed -- falling back to the heuristic "
                "classifier. Install the ML extras (see requirements.txt) to enable "
                "real MuRIL predictions."
            )
            return False

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self._model.to(self._device)
            self._model.eval()
            self._ready = True
            logger.info("Loaded MuRIL checkpoint from %s on %s", model_path, self._device)
        except Exception:  # noqa: BLE001 - any load failure should degrade gracefully
            logger.exception("Failed to load MuRIL checkpoint from %s", model_path)
            self._ready = False

        return self._ready

    def predict(self, text: str) -> TextClassificationResult:
        import torch  # safe: load() already succeeded if we get here

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.TEXT_MODEL_MAX_LENGTH,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        confidence = probabilities[0, prediction].item()
        return TextClassificationResult(LABELS[prediction], confidence, backend="muril")


_muril_backend = _MurilBackend()


def _heuristic_predict(text: str) -> TextClassificationResult:
    lower = text.lower()
    score = 0.5
    for sig in _FAKE_SIGNALS:
        if sig in lower:
            score += 0.14
    if re.search(r"[!?]{2,}", text):
        score += 0.08
    if len(text) < 40:
        score -= 0.05
    score = max(0.05, min(0.97, score + (len(text) % 7) / 100))

    label = "FAKE" if score >= 0.5 else "REAL"
    confidence = score if label == "FAKE" else 1 - score
    return TextClassificationResult(label, max(confidence, 0.5), backend="heuristic")


@lru_cache
def _get_backend_ready() -> bool:
    return _muril_backend.load()


def classify_text(text: str) -> TextClassificationResult:
    """Returns REAL/FAKE + confidence for the given text, using the real
    MuRIL checkpoint when available and a heuristic fallback otherwise."""

    if _get_backend_ready():
        try:
            return _muril_backend.predict(text)
        except Exception:  # noqa: BLE001
            logger.exception("MuRIL inference failed; falling back to heuristic for this request")

    return _heuristic_predict(text)


def classifier_backend_name() -> str:
    return "muril" if _get_backend_ready() else "heuristic"
