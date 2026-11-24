# src/gradient/ai/lab/embed.py
"""Image (and text) embeddings for other apps' metrics, e.g. ArtAgents' LoRA test scoring.

Two kinds of model from the `vision` model folder (or the Hub):
* DINO (DINOv3 / DINOv2, the Lab's feature models): image embeddings only, the global
  (CLS) feature. Good for fidelity to training images and drift.
* CLIP / SigLIP (e.g. CLIP ViT-B/32 from the Models tab): image and text embeddings in
  one space, so prompt adherence (CLIP-T) can be measured too.

All embeddings are L2-normalized. Blocking: run with `run_gpu_task`.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from gradient.core.device import get_device
from gradient.model_manager.inventory import find_model_path

from .features import load_feature_model

logger = logging.getLogger(__name__)

TEXT_MODEL_TYPES = ("clip", "siglip", "siglip2")
BATCH = 16
MAX_IMAGE_PIXELS = 64_000_000
_cache: dict[str, ClipEmbedder] = {}


@dataclass
class ClipEmbedder:
    name: str
    model: Any
    processor: Any

    @torch.no_grad()
    def images(self, images: list[Image.Image]) -> torch.Tensor:
        out = []
        device = next(self.model.parameters()).device
        for i in range(0, len(images), BATCH):
            inputs = self.processor(images=images[i : i + BATCH], return_tensors="pt").to(device)
            out.append(_features(self.model.get_image_features(**inputs)))
        return torch.nn.functional.normalize(torch.cat(out).float(), dim=-1)

    @torch.no_grad()
    def texts(self, texts: list[str]) -> torch.Tensor:
        out = []
        device = next(self.model.parameters()).device
        padding = "max_length" if "siglip" in self.model.config.model_type else True
        for i in range(0, len(texts), BATCH):
            inputs = self.processor(
                text=texts[i : i + BATCH], padding=padding, truncation=True, return_tensors="pt"
            ).to(device)
            out.append(_features(self.model.get_text_features(**inputs)))
        return torch.nn.functional.normalize(torch.cat(out).float(), dim=-1)


def _features(output) -> torch.Tensor:
    """get_*_features returns a tensor, or (newer transformers) an output with pooler_output."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    return pooled if pooled is not None else output[0]


def _source(name: str) -> str:
    local = find_model_path(name, "vision")
    return str(local) if local and local.is_dir() else name


def model_type(name: str) -> str | None:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(_source(name)).model_type
    except Exception:
        return None


def load_clip(name: str) -> ClipEmbedder:
    if name in _cache:
        return _cache[name]
    from transformers import AutoModel, AutoProcessor

    source = _source(name)
    try:
        model = AutoModel.from_pretrained(source, dtype=torch.float32).to(get_device()).eval()
        processor = AutoProcessor.from_pretrained(source)
    except Exception as e:
        raise ValueError(f"Could not load {name}: {str(e).splitlines()[0][:200]}") from e
    _cache[name] = ClipEmbedder(name, model, processor)
    return _cache[name]


def decode_image(data: str) -> Image.Image:
    """A base64 image (a data: URL works too), PNG/JPEG/WEBP only."""
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("An image is not valid base64.") from e
    try:
        image = Image.open(io.BytesIO(raw), formats=["PNG", "JPEG", "WEBP"])
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("An image is too large.")
        image.load()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"An image could not be read: {e}") from e
    return image.convert("RGB")


def embed(
    images: list[Image.Image], texts: list[str] | None, feature_model: str | None
) -> dict[str, Any]:
    """{model, kind, dim, image_embeddings, text_embeddings, note}."""
    texts = list(texts or [])
    if feature_model and model_type(feature_model) in TEXT_MODEL_TYPES:
        clip = load_clip(feature_model)
        image_vectors = clip.images(images) if images else None
        text_vectors = clip.texts(texts) if texts else None
        dim = (image_vectors if image_vectors is not None else text_vectors).shape[-1]
        return {
            "model": clip.name,
            "kind": "clip",
            "dim": int(dim),
            "image_embeddings": image_vectors.cpu().tolist() if image_vectors is not None else [],
            "text_embeddings": text_vectors.cpu().tolist() if text_vectors is not None else None,
            "note": None,
        }
    if texts:
        raise ValueError(
            "Text embeddings need a CLIP or SigLIP model (e.g. CLIP ViT-B/32 from the Models tab); "
            f"{feature_model or 'the default DINO model'} has no text encoder."
        )
    if not images:
        raise ValueError("Nothing to embed: send images, output_ids or texts.")
    fm, note = load_feature_model(feature_model)
    vectors = torch.stack([fm.encode(image)[0] for image in images])
    return {
        "model": fm.name,
        "kind": "dino",
        "dim": int(vectors.shape[-1]),
        "image_embeddings": vectors.cpu().tolist(),
        "text_embeddings": None,
        "note": note,
    }
