# src/gradient/ai/lab/taylor.py
"""Taylor preview: predict the image at guidance w + Δ without re-rendering.

    image(w + Δ) ≈ image(w) + Δ · ∂image/∂w

The tangent ∂image/∂w comes from an exact forward-mode JVP through the whole sampler
(SD 1.x / SDXL) or from central finite differences (any model family). The browser
then previews any Δ instantly by adding Δ·tangent to the base image.
"""

import time

import numpy as np
import torch
from PIL import Image

from gradient.ai.diffusion.loading import get_pipeline
from gradient.ai.diffusion.pipelines import _set_scheduler, generate_images, resolve_seed

from . import unet_sampler
from .imaging import colorize, pil_to_array, to_b64
from .models import TaylorRequest, TaylorResponse


def _encode_tangent(tangent: np.ndarray) -> tuple[Image.Image, float]:
    scale = float(np.percentile(np.abs(tangent), 99.9)) or 1e-6
    q = np.clip(np.round(128 + 127 * tangent / scale), 0, 255).astype(np.uint8)
    return Image.fromarray(q, "RGB"), scale


def taylor_preview(request: TaylorRequest) -> TaylorResponse:
    """Blocking; run through run_gpu_task."""
    start = time.time()
    base = request.base_request.model_copy(deep=True)
    base.seed = resolve_seed(base.seed)
    if not isinstance(base.guidance_scale, (int, float)):
        raise ValueError("Taylor preview needs a constant guidance_scale")
    w = float(base.guidance_scale)

    pipe, entry = get_pipeline(
        base.model_id, vae_id=base.vae_id, loras=base.loras, embeddings=base.embeddings
    )
    method = request.method
    if method == "auto":
        method = "jvp" if unet_sampler.supports(pipe) else "finite_difference"
    if method == "jvp" and not unet_sampler.supports(pipe):
        raise ValueError(
            f"Exact JVP is available for SD 1.x/SDXL; use finite_difference for {entry.pipeline_class}"
        )
    if method == "jvp" and w <= 1.0:
        raise ValueError("JVP needs classifier-free guidance (guidance_scale > 1)")

    if method == "jvp":
        _set_scheduler(pipe, entry, base.scheduler_id)
        image_t, tangent_t = unet_sampler.guidance_jvp(pipe, base, base.seed)
        image = image_t.numpy()
        tangent = tangent_t.numpy()
        base_img = Image.fromarray((image * 255).round().astype(np.uint8), "RGB")
    else:
        delta = request.delta or 0.5
        lo = base.model_copy(update={"guidance_scale": max(0.0, w - delta)})
        hi = base.model_copy(update={"guidance_scale": w + delta})
        base_img = generate_images(base).images[0]
        img_lo = pil_to_array(generate_images(lo).images[0])
        img_hi = pil_to_array(generate_images(hi).images[0])
        tangent = (img_hi - img_lo) / (w + delta - max(0.0, w - delta))

    tangent_img, scale = _encode_tangent(tangent)
    magnitude = np.abs(tangent).mean(axis=-1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return TaylorResponse(
        method=method,
        guidance_scale=w,
        seed=base.seed,
        base_image_base64=to_b64(base_img),
        tangent_png_base64=to_b64(tangent_img),
        tangent_scale=scale,
        magnitude_heatmap_base64=to_b64(colorize(magnitude)),
        mean_abs_tangent=float(np.abs(tangent).mean()),
        performance_ms=(time.time() - start) * 1000,
    )
