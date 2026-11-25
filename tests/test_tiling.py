# tests/test_tiling.py
"""Seamless images: circular padding (UNet, VAE) and latent rolling (DiT)."""

import numpy as np
import torch

BASE = {"width": 64, "height": 64, "num_inference_steps": 3, "guidance_scale": 5.0, "seed": 4}


def _render(model, tiling="none", probes=None, overrides=None, **extra):
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import generate_images

    req = Text2ImageRequest(model_id=model, prompt="a red dog", tiling=tiling, **{**BASE, **extra})
    result = generate_images(req, probes=probes, call_overrides=overrides)
    return np.asarray(result.images[0]).astype(np.int16), result.params


def test_circular_padding_makes_the_render_shift_equivariant(tiny_sd):
    from gradient.ai.diffusion.loading import get_pipeline

    pipe, _ = get_pipeline(tiny_sd)
    factor = pipe.vae_scale_factor
    torch.manual_seed(0)
    latents = torch.randn(1, 4, 64 // factor, 64 // factor)
    shifted = torch.roll(latents, (4, 6), dims=(-2, -1))
    a, params = _render(tiny_sd, "xy", overrides={"latents": latents})
    b, _ = _render(tiny_sd, "xy", overrides={"latents": shifted})
    assert params["tiling_applied"] == "xy" and params["tiling_method"] == "circular padding"
    # a torus: shifting the start shifts the image, edges included
    assert np.abs(np.roll(a, (4 * factor, 6 * factor), axis=(0, 1)) - b).max() <= 2
    # without tiling the borders break the symmetry
    c, _ = _render(tiny_sd, overrides={"latents": latents})
    d, _ = _render(tiny_sd, overrides={"latents": shifted})
    assert np.abs(np.roll(c, (4 * factor, 6 * factor), axis=(0, 1)) - d).max() > 10
    # the patch is gone after the render
    assert not any("_conv_forward" in m.__dict__ for m in pipe.unet.modules())


def test_one_axis_only(tiny_sd):
    from gradient.ai.diffusion.loading import get_pipeline

    pipe, _ = get_pipeline(tiny_sd)
    factor = pipe.vae_scale_factor
    torch.manual_seed(1)
    latents = torch.randn(1, 4, 64 // factor, 64 // factor)
    a, _ = _render(tiny_sd, "x", overrides={"latents": latents})
    b, _ = _render(tiny_sd, "x", overrides={"latents": torch.roll(latents, 6, dims=-1)})
    assert np.abs(np.roll(a, 6 * factor, axis=1) - b).max() <= 2  # wraps left-right
    c, _ = _render(tiny_sd, "x", overrides={"latents": torch.roll(latents, 4, dims=-2)})
    assert np.abs(np.roll(a, 4 * factor, axis=0) - c).max() > 10  # not top-bottom


def test_latent_rolling_on_dit(tiny_flux):
    from gradient.ai.lab.probes import Probe

    plain, _ = _render(tiny_flux, guidance_scale=1.0)
    rolled, params = _render(tiny_flux, "xy", guidance_scale=1.0)
    assert params["tiling_method"] == "latent rolling" and "tiling_notes" not in params
    assert rolled.shape == plain.shape and not np.array_equal(rolled, plain)

    class Keeper(Probe):  # like the Explorer: latents must stay where they are
        needs_fixed_latents = True

    _, fixed = _render(tiny_flux, "xy", probes=[Keeper()], guidance_scale=1.0)
    assert fixed["tiling_method"] == "circular padding"
