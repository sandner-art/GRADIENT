# src/gradient/ai/diffusion/tiling.py
"""Seamless (tileable) images: the left edge continues the right one, the top the bottom.

Two techniques, chosen by the model:

* Circular padding (UNet models: SD 1.x/2.x, SDXL). Every padded convolution of the UNet
  and the VAE wraps around instead of padding with zeros, per axis, so the network sees the
  image as a torus. This is exact: nothing else changes.
* Latent rolling (DiT models: SD3, FLUX.1). Transformers have no padding to wrap, so after
  every step the latent is shifted by a new random offset (wrapping around) and shifted back
  after the last one. The seam lands somewhere else each step and gets denoised like any
  other region. The VAE decoder still wraps (circular padding).

`tiling` is "x" (left-right), "y" (top-bottom) or "xy".
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from gradient.ai.lab.probes import Probe

logger = logging.getLogger(__name__)


def _padded_convs(module: torch.nn.Module | None) -> list[torch.nn.Conv2d]:
    if module is None:
        return []
    return [
        m
        for m in module.modules()
        if isinstance(m, torch.nn.Conv2d)
        and isinstance(m.padding, tuple)
        and any(m.padding)
        and m.padding_mode == "zeros"
    ]


def _wrapping_forward(conv: torch.nn.Conv2d, wrap_x: bool, wrap_y: bool):
    ph, pw = conv.padding

    def forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        x = F.pad(input, (pw, pw, 0, 0), mode="circular" if wrap_x else "constant")
        x = F.pad(x, (0, 0, ph, ph), mode="circular" if wrap_y else "constant")
        return F.conv2d(x, weight, bias, conv.stride, 0, conv.dilation, conv.groups)

    return forward


class TilingProbe(Probe):
    """Makes the render tile seamlessly along `axes` ("x", "y" or "xy")."""

    tensor_inputs = ("latents",)

    def __init__(self, axes: str, seed: int, roll: bool | None = None) -> None:
        self.axes = axes
        self.wrap_x, self.wrap_y = "x" in axes, "y" in axes
        self.generator = torch.Generator().manual_seed(int(seed) % (2**32))
        self.roll = roll  # None: decide by the model (DiT models roll)
        self.patched: list[torch.nn.Conv2d] = []
        self.shift = [0, 0]  # rows, columns (latent pixels) applied so far
        self.total = 1
        self.size: tuple[int, int] | None = None
        self.notes: list[str] = []

    def attach(self, pipe, entry, total_steps, kwargs):
        self.total = max(1, total_steps)
        self.size = (int(kwargs.get("height") or 0), int(kwargs.get("width") or 0))
        unet = getattr(pipe, "unet", None)
        convs = _padded_convs(unet) + _padded_convs(getattr(pipe, "vae", None))
        for conv in convs:
            conv._conv_forward = _wrapping_forward(conv, self.wrap_x, self.wrap_y)
            self.patched.append(conv)
        if self.roll is None:
            self.roll = unet is None  # transformers have no padding to wrap

    def detach(self):
        for conv in self.patched:
            conv.__dict__.pop("_conv_forward", None)
        self.patched.clear()

    # --- latent rolling
    def _grid(self, pipe, latents: torch.Tensor) -> tuple[torch.Tensor, Any] | None:
        """latents as (B, C, H, W) and a function that puts them back in the pipeline's form."""
        if latents.ndim == 4:
            return latents, lambda grid: grid
        unpack = getattr(pipe, "_unpack_latents", None)
        pack = getattr(pipe, "_pack_latents", None)
        if latents.ndim == 3 and callable(unpack) and callable(pack) and self.size:
            try:  # FLUX.1: 2×2 patches packed into a sequence
                height, width = self.size
                grid = unpack(latents, height, width, getattr(pipe, "vae_scale_factor", 8))
                b, c, h, w = grid.shape
                return grid, lambda g: pack(g, b, c, h, w)
            except Exception as e:  # pragma: no cover - other packed layouts
                logger.debug("Cannot unpack latents for rolling: %s", e)
        return None

    def on_step_end(self, pipe, step, timestep, cb_kwargs):
        if not self.roll or not torch.is_tensor(cb_kwargs.get("latents")):
            return cb_kwargs
        found = self._grid(pipe, cb_kwargs["latents"])
        if found is None:
            if not self.notes:
                self.notes.append("latent rolling isn't available for this model")
            return cb_kwargs
        grid, back = found
        h, w = grid.shape[-2:]
        steps = getattr(pipe, "_num_timesteps", None) or self.total  # what the pipeline runs
        if step + 1 >= steps:  # last step: undo every shift
            move = [-self.shift[0], -self.shift[1]]
        else:
            move = [
                int(torch.randint(0, h, (1,), generator=self.generator)) if self.wrap_y else 0,
                int(torch.randint(0, w, (1,), generator=self.generator)) if self.wrap_x else 0,
            ]
            if cb_kwargs["latents"].ndim == 3:
                move = [m - m % 2 for m in move]  # keep packed 2×2 patches whole
        self.shift = [(self.shift[0] + move[0]) % h, (self.shift[1] + move[1]) % w]
        cb_kwargs["latents"] = back(torch.roll(grid, shifts=move, dims=(-2, -1)))
        return cb_kwargs

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tiling_applied": self.axes,
            "tiling_method": "latent rolling" if self.roll else "circular padding",
        }
        if self.notes:
            out["tiling_notes"] = self.notes
        return out
