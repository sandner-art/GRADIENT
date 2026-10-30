# gradient-project-root/src/gradient/ai/viz/latent_space.py
"""Latent space projection prototype.

NOTE: this endpoint projects *random Gaussian latents*; i.i.d. noise has no semantic
structure, so the layout is not meaningful yet. It is kept as API scaffolding for
the planned navigator, which will embed generated outputs (SigLIP/DINO features)
instead -- see docs/research/research_review_2026-10.md.
"""

import base64
import logging
import time
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from gradient.ai.diffusion.loading import _load_vae, get_loaded_entry
from gradient.core.device import get_device
from gradient.core.scheduling import run_gpu_task

from .models import LatentPoint, LatentSpaceVizRequest, LatentSpaceVizResponse

logger = logging.getLogger(__name__)

NOTE = "Prototype: projects random Gaussian latents (no semantic structure). See research review."


def decode_latents_batch(vae, latents: torch.Tensor, resolution: int) -> list[Image.Image]:
    scaling = getattr(vae.config, "scaling_factor", 0.18215) or 0.18215
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    images: list[Image.Image] = []
    with torch.inference_mode():
        for i in range(0, latents.shape[0], 16):
            batch = (latents[i : i + 16] / scaling + shift).to(vae.device, dtype=vae.dtype)
            out = (vae.decode(batch).sample / 2 + 0.5).clamp(0, 1)
            for arr in out.cpu().permute(0, 2, 3, 1).float().numpy():
                img = Image.fromarray((arr * 255).astype(np.uint8))
                images.append(img.resize((resolution, resolution), Image.Resampling.LANCZOS))
    return images


def encode_image_to_base64(img: Image.Image, format: str = "WEBP", quality: int = 75) -> str:
    buf = BytesIO()
    img.save(buf, format=format, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _compute(request: LatentSpaceVizRequest) -> tuple[list[LatentPoint], str]:
    import umap

    vae = None
    if request.generate_previews or request.vae_id:
        if request.vae_id:
            vae = _load_vae(
                request.vae_id, torch.float32 if get_device().type == "cpu" else torch.float16
            ).to(get_device())
        else:
            vae = getattr(get_loaded_entry(request.model_id).pipeline, "vae", None)
        if vae is None and request.generate_previews:
            raise ValueError("No VAE available for previews")

    channels = getattr(getattr(vae, "config", None), "latent_channels", 4) if vae else 4
    blocks = getattr(getattr(vae, "config", None), "block_out_channels", None) if vae else None
    scale = 2 ** (len(blocks) - 1) if blocks else 8
    size = request.preview_resolution // scale
    if size <= 0:
        raise ValueError(
            f"preview_resolution {request.preview_resolution} too small for VAE scale {scale}"
        )

    rng = np.random.default_rng(request.seed)
    latents = rng.standard_normal((request.num_samples, channels, size, size)).astype(np.float32)
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=request.seed
    )
    projected = reducer.fit_transform(latents.reshape(request.num_samples, -1))

    previews = (
        decode_latents_batch(vae, torch.from_numpy(latents), request.preview_resolution)
        if (request.generate_previews and vae)
        else None
    )
    points = [
        LatentPoint(
            id=i,
            x=float(projected[i, 0]),
            y=float(projected[i, 1]),
            image_preview_base64=encode_image_to_base64(previews[i], quality=70)
            if previews
            else None,
        )
        for i in range(request.num_samples)
    ]
    return points, "umap"


async def generate_latent_space_visualization(
    request: LatentSpaceVizRequest,
) -> LatentSpaceVizResponse:
    start = time.time()
    points, method = await run_gpu_task(_compute, request)
    return LatentSpaceVizResponse(
        request_params=request,
        points=points,
        projection_method=method,
        performance_ms=(time.time() - start) * 1000,
        note=NOTE,
    )
