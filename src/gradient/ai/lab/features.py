# src/gradient/ai/lab/features.py
"""Self-supervised vision features (DINOv3 / DINOv2) for perceptual distances.

Models are looked up in the `vision` model folder first (downloadable from the Models
tab), then on the Hugging Face Hub. DINOv3 weights are gated by Meta; when they are not
accessible we fall back to the ungated DINOv2 and say so in the response.
"""

import json
import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from gradient.core.config import get_setting
from gradient.core.device import get_device
from gradient.model_manager.inventory import find_model_path

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES = ("facebook/dinov3-vits16-pretrain-lvd1689m", "facebook/dinov2-small")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_cache: dict[str, "FeatureModel"] = {}


@dataclass
class FeatureModel:
    name: str
    model: torch.nn.Module
    patch: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    grid: int = 28  # patches per side

    @torch.no_grad()
    def encode(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (global feature (D,), patch features (grid, grid, D)), L2-normalized."""
        size = self.patch * self.grid
        arr = (
            np.asarray(
                image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC),
                dtype=np.float32,
            )
            / 255
        )
        arr = (arr - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        device = next(self.model.parameters()).device
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)
        tokens = self.model(pixel_values=x).last_hidden_state[0].float()
        n = self.grid * self.grid
        patches = tokens[-n:].reshape(self.grid, self.grid, -1)
        cls = tokens[0]
        return torch.nn.functional.normalize(cls, dim=-1), torch.nn.functional.normalize(
            patches, dim=-1
        )


def _read_norm(path_or_id: str) -> tuple[tuple, tuple]:
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(path_or_id)
        return tuple(proc.image_mean), tuple(proc.image_std)
    except Exception:
        try:
            cfg = json.loads(open(f"{path_or_id}/preprocessor_config.json").read())
            return tuple(cfg["image_mean"]), tuple(cfg["image_std"])
        except Exception:
            return _IMAGENET_MEAN, _IMAGENET_STD


def load_feature_model(name: str | None = None) -> tuple[FeatureModel, str | None]:
    """Load (and cache) a feature model. Returns (model, note-about-fallback)."""
    candidates = (
        [name]
        if name
        else [get_setting("lab.feature_model") or DEFAULT_CANDIDATES[0], *DEFAULT_CANDIDATES[1:]]
    )
    candidates = list(dict.fromkeys(c for c in candidates if c))
    errors = []
    for candidate in candidates:
        if candidate in _cache:
            return _cache[candidate], (
                f"Using {candidate}: " + "; ".join(errors)
            ) if errors else None
        local = find_model_path(candidate, "vision")
        source = str(local) if local and local.is_dir() else candidate
        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(source, dtype=torch.float32).to(get_device()).eval()
        except Exception as e:  # gated, missing, offline ...
            reason = str(e).splitlines()[0][:160]
            errors.append(f"{candidate} unavailable ({reason})")
            logger.info("Feature model %s unavailable: %s", candidate, reason)
            continue
        mean, std = _read_norm(source)
        patch = int(getattr(model.config, "patch_size", 16))
        fm = FeatureModel(candidate, model, patch, mean, std)
        _cache[candidate] = fm
        return fm, (f"Using {candidate}: " + "; ".join(errors)) if errors else None
    raise ValueError(
        "No vision feature model available. Download 'DINOv2 small' (or DINOv3, gated) in the Models tab. "
        + " | ".join(errors)
    )


def distances(fm: FeatureModel, a: Image.Image, b: Image.Image) -> tuple[float, np.ndarray]:
    """(global cosine distance, per-patch cosine distance map)."""
    ca, pa = fm.encode(a)
    cb, pb = fm.encode(b)
    global_d = float(1 - (ca * cb).sum())
    patch_d = (1 - (pa * pb).sum(-1)).clamp(min=0).cpu().numpy()
    return global_d, patch_d
