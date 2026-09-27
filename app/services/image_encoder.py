"""
Optional image stage of the analysis pipeline.

Mirrors `major-project-model-training/src/demo_multimodal.py`: a pretrained
ViT (vit-tiny) is used purely as a frozen feature extractor. There is no
trained multimodal fusion model yet (see MODEL_DOCUMENTATION.txt --
"Multimodal fusion: Planned / next implementation stage"), so this stage
only reports that an image was encoded and its feature-vector size; it does
NOT influence the verdict yet. That's intentional -- the training repo is
explicit that "no multimodal results should be claimed" before fusion has
actually been trained and evaluated.

This is the seam where the future fusion classifier plugs in: once
`train_multimodal.py` produces a fusion checkpoint, load it here and combine
`extract_image_features()`'s output with the text stage's logits/embedding
instead of just reporting feature shape.
"""

import io
import logging
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ImageEncodingResult:
    def __init__(self, feature_size: Optional[int], backend: str, note: str):
        self.feature_size = feature_size
        self.backend = backend  # "vit" | "unavailable"
        self.note = note


class _ViTBackend:
    def __init__(self):
        self._processor = None
        self._model = None
        self._device = None
        self._ready = False
        self._attempted = False

    def load(self) -> bool:
        if self._attempted:
            return self._ready
        self._attempted = True

        if not settings.ENABLE_IMAGE_ENCODER:
            return False

        try:
            import torch
            from transformers import AutoImageProcessor, ViTModel
        except ImportError:
            logger.warning(
                "torch/transformers not installed -- image analysis stage will be skipped. "
                "Install the ML extras to enable it."
            )
            return False

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoImageProcessor.from_pretrained(settings.IMAGE_MODEL_NAME)
            self._model = ViTModel.from_pretrained(settings.IMAGE_MODEL_NAME)
            for param in self._model.parameters():
                param.requires_grad = False
            self._model.to(self._device)
            self._model.eval()
            self._ready = True
            logger.info("Loaded image encoder %s on %s", settings.IMAGE_MODEL_NAME, self._device)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load image encoder %s", settings.IMAGE_MODEL_NAME)
            self._ready = False

        return self._ready

    def extract(self, raw_image_bytes: bytes) -> int:
        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]

        return int(features.shape[-1])


_vit_backend = _ViTBackend()


def encode_image(raw_image_bytes: bytes) -> ImageEncodingResult:
    if not _vit_backend.load():
        return ImageEncodingResult(
            feature_size=None,
            backend="unavailable",
            note="Image encoder not loaded (missing ML deps/model, or disabled). "
            "Image was stored but not analyzed.",
        )

    try:
        size = _vit_backend.extract(raw_image_bytes)
        return ImageEncodingResult(
            feature_size=size,
            backend="vit",
            note=(
                f"Image encoded via {settings.IMAGE_MODEL_NAME} ({size}-dim feature vector). "
                "Multimodal fusion with the text classifier is a future stage -- this "
                "does not yet change the verdict."
            ),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Image feature extraction failed")
        return ImageEncodingResult(feature_size=None, backend="unavailable", note="Image could not be processed.")
