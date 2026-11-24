# src/gradient/ai/diffusion/preview.py
"""Live previews of a running render, cheap enough to make on every step.

Each step's latents are turned into an estimate of the final image and mapped to RGB with a
linear latent→RGB projection. The projection is fitted once per loaded model by encoding a
colour chart with the model's own VAE, so it fits any VAE (SD 1.5, SDXL, SD3, FLUX.1, custom
VAEs) without hand-made constants.

The denoised estimate comes from two consecutive steps: for Euler-type samplers (k-diffusion
and flow matching alike) x_t = x0 + σ·d, so d = Δx/Δσ and x0 ≈ x_t − σ_t·d. Samplers without
σ values (DDIM, PNDM) show the current latents instead.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from gradient.core.config import get_setting
from gradient.core.jobs import current_job, report_preview

logger = logging.getLogger(__name__)

MIN_INTERVAL = 0.25  # seconds between previews


def enabled() -> bool:
    return bool(get_setting("performance.live_preview", True)) and current_job() is not None


def _chart(size: int) -> np.ndarray:
    """A colour chart: hue across, saturation and value down, plus a grey ramp and patches."""
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    h = x * 6.0
    s = np.clip(1.4 - y * 1.2, 0, 1)
    v = np.clip(0.25 + y * 0.9, 0, 1)
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    rgb = np.choose(
        i[..., None].repeat(3, -1),
        [
            np.stack([v, t, p], -1),
            np.stack([q, v, p], -1),
            np.stack([p, v, t], -1),
            np.stack([p, q, v], -1),
            np.stack([t, p, v], -1),
            np.stack([v, p, q], -1),
        ],
    )
    band = size // 8
    rgb[:band] = np.linspace(0, 1, size)[None, :, None]  # grey ramp
    rng = np.random.default_rng(0)
    for _ in range(12):  # flat patches (skin, sky, foliage-like colours)
        cx, cy = rng.integers(band, size - band, 2)
        r = int(rng.integers(size // 20, size // 10))
        rgb[cy - r : cy + r, cx - r : cx + r] = rng.uniform(0.05, 0.95, 3)
    return rgb.astype(np.float32)


def _normalize(z: torch.Tensor, vae) -> torch.Tensor:
    """Raw VAE latents → the pipeline's latent space."""
    cfg = vae.config
    scaling = float(getattr(cfg, "scaling_factor", None) or 1.0)
    mean = getattr(cfg, "latents_mean", None)
    std = getattr(cfg, "latents_std", None)
    if mean is not None and std is not None and len(mean) == z.shape[1]:
        m = torch.tensor(mean, dtype=z.dtype, device=z.device).view(1, -1, 1, 1)
        s = torch.tensor(std, dtype=z.dtype, device=z.device).view(1, -1, 1, 1)
        return (z - m) * scaling / s
    shift = float(getattr(cfg, "shift_factor", None) or 0.0)
    return (z - shift) * scaling


def fit_projection(pipe) -> tuple[np.ndarray, np.ndarray] | None:
    """(W (C, 3), b (3,)) mapping pipeline latents to RGB in [0, 1], or None if unsupported."""
    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "encode"):
        return None
    factor = int(getattr(pipe, "vae_scale_factor", 8) or 8)
    size = 512 if factor >= 16 else 256
    chart = _chart(size)
    device = getattr(pipe, "_execution_device", None) or next(vae.parameters()).device
    dtype = next(p.dtype for p in vae.parameters() if p.is_floating_point())
    x = torch.from_numpy(chart).permute(2, 0, 1)[None].to(device=device, dtype=dtype) * 2 - 1
    with torch.no_grad():
        out = vae.encode(x)
    z = getattr(out, "latent_dist", None)
    z = z.mean if z is not None else getattr(out, "latent", None)
    if z is None:
        z = out[0]
    if z.ndim != 4:
        return None  # video-style VAEs (5-D latents)
    z = _normalize(z.float(), vae)[0]  # (C, h, w)
    c, h, w = z.shape
    target = torch.from_numpy(chart).permute(2, 0, 1)[None]
    target = torch.nn.functional.adaptive_avg_pool2d(target, (h, w))[0]  # (3, h, w)
    X = torch.cat([z.reshape(c, -1).T.cpu(), torch.ones(h * w, 1)], 1).double()  # (N, C+1)
    Y = target.reshape(3, -1).T.double()  # (N, 3)
    ridge = 1e-3 * torch.eye(c + 1, dtype=torch.float64)
    coef = torch.linalg.solve(X.T @ X + ridge, X.T @ Y)  # (C+1, 3)
    return coef[:c].float().numpy(), coef[c].float().numpy()


def estimate_x0(x: torch.Tensor, prev: torch.Tensor, s_now: float, s_prev: float) -> torch.Tensor:
    """Denoised estimate from two consecutive latents of an Euler-type sampler (x = x0 + σ·d)."""
    if abs(s_now - s_prev) < 1e-6:
        return x
    d = (x - prev) / (s_now - s_prev)
    return x - s_now * d


def to_rgb(latents: torch.Tensor, projection: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """(C, h, w) latents → (h, w, 3) uint8 with the fitted projection."""
    w, b = projection
    rgb = torch.einsum("chw,ck->hwk", latents.float(), torch.from_numpy(w).to(latents.device))
    rgb = rgb + torch.from_numpy(b).to(latents.device)
    return (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()


class LivePreview:
    """A probe (see gradient.ai.lab.probes) that publishes a JPEG preview of each step."""

    tensor_inputs = ("latents",)

    def __init__(self, height: int, width: int):
        self.height, self.width = height, width
        self.projection: tuple[np.ndarray, np.ndarray] | None = None
        self.prev: torch.Tensor | None = None
        self.prev_index: int | None = None
        self.last = 0.0
        self.count = 0
        self.disabled = False

    def attach(self, pipe, entry, steps: int, kwargs: dict[str, Any]) -> None:
        cached = getattr(entry, "preview_projection", None)
        if cached is None:
            try:
                cached = fit_projection(pipe) or False
            except Exception as e:  # never let previews break a render
                logger.info("No live preview for %s: %s", entry.model_id, e)
                cached = False
            entry.preview_projection = cached
        if cached is False:
            self.disabled = True
        else:
            self.projection = cached

    def detach(self) -> None:
        self.prev = None

    def report(self) -> dict[str, Any]:
        return {}

    def on_step_end(self, pipeline, step: int, timestep, callback_kwargs: dict) -> dict:
        if self.disabled:
            return callback_kwargs
        try:
            self._step(pipeline, step, callback_kwargs)
        except Exception as e:
            logger.info("Live preview disabled for this render: %s", e)
            self.disabled = True
        return callback_kwargs

    def _latents(self, pipeline, latents: torch.Tensor) -> torch.Tensor | None:
        x = latents[:1]
        if x.ndim == 3:  # packed (FLUX-style): unpack to (1, C, h, w)
            unpack = getattr(pipeline, "_unpack_latents", None)
            if unpack is None:
                return None
            x = unpack(x, self.height, self.width, pipeline.vae_scale_factor)
        if x.ndim == 5 and x.shape[2] == 1:
            x = x[:, :, 0]
        return x if x.ndim == 4 else None

    def _step(self, pipeline, step: int, callback_kwargs: dict) -> None:
        latents = callback_kwargs.get("latents")
        if latents is None:
            return
        sched = getattr(pipeline, "scheduler", None)
        sigmas = getattr(sched, "sigmas", None)
        index = getattr(sched, "step_index", None)
        total = getattr(pipeline, "_num_timesteps", None) or 0
        last_step = total and step + 1 >= total
        now = time.monotonic()
        x = self._latents(pipeline, latents)
        if x is None:
            self.disabled = True
            return
        prev, prev_index = self.prev, self.prev_index
        self.prev, self.prev_index = x.detach().float().clone(), index
        if now - self.last < MIN_INTERVAL and not last_step:
            return
        estimate = x.float()
        if (
            prev is not None
            and torch.is_tensor(sigmas)
            and index is not None
            and prev_index is not None
            and index < len(sigmas)
            and prev_index < len(sigmas)
        ):
            estimate = estimate_x0(estimate, prev, float(sigmas[index]), float(sigmas[prev_index]))
        z = estimate[0]
        if z.shape[0] != self.projection[0].shape[0]:
            self.disabled = True
            return
        img = to_rgb(z, self.projection)
        buf = io.BytesIO()
        Image.fromarray(img, "RGB").save(buf, format="JPEG", quality=88)
        self.count += 1
        self.last = now
        report_preview(buf.getvalue(), step + 1)
