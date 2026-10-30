# src/gradient/ai/lab/unet_sampler.py
"""A transparent denoising loop for UNet models (SD 1.x / SDXL).

Unlike the pipelines' `__call__`, every quantity here is an ordinary tensor expression,
so it works with PyTorch forward-mode autodiff (dual numbers): passing a dual guidance
scale yields the exact derivative of the final image with respect to guidance (JVP).
The same loop is the base for future step-level instrumentation (x0 previews,
guiders, per-step perturbations).
"""

import contextlib
import logging
from dataclasses import dataclass

import torch

from gradient.ai.diffusion.guidance import parse_guidance_schedule
from gradient.ai.diffusion.models import DiffusionTaskBase

logger = logging.getLogger(__name__)

SUPPORTED = ("StableDiffusionPipeline", "StableDiffusionXLPipeline")


def supports(pipe) -> bool:
    return type(pipe).__name__ in SUPPORTED


@dataclass
class Conditioning:
    embeds: torch.Tensor
    added: dict | None
    cfg: bool


def encode(
    pipe, prompt: str, negative: str | None, height: int, width: int, cfg: bool
) -> Conditioning:
    device = pipe._execution_device
    if type(pipe).__name__ == "StableDiffusionXLPipeline":
        pe, ne, ppe, npe = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=cfg,
            negative_prompt=negative or None,
        )
        proj = (
            pipe.text_encoder_2.config.projection_dim
            if pipe.text_encoder_2 is not None
            else int(ppe.shape[-1])
        )
        time_ids = pipe._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=pe.dtype,
            text_encoder_projection_dim=proj,
        )
        time_ids = time_ids.to(device)
        if cfg:
            return Conditioning(
                torch.cat([ne, pe]),
                {"text_embeds": torch.cat([npe, ppe]), "time_ids": torch.cat([time_ids, time_ids])},
                True,
            )
        return Conditioning(pe, {"text_embeds": ppe, "time_ids": time_ids}, False)
    pe, ne = pipe.encode_prompt(prompt, device, 1, cfg, negative or None)
    return Conditioning(torch.cat([ne, pe]) if cfg else pe, None, cfg)


def make_scheduler(pipe, request: DiffusionTaskBase):
    sched = type(pipe.scheduler).from_config(pipe.scheduler.config)
    device = pipe._execution_device
    if request.custom_sigmas:
        sched.set_timesteps(sigmas=list(request.custom_sigmas), device=device)
    else:
        sched.set_timesteps(request.num_inference_steps, device=device)
    return sched


def initial_noise(pipe, request: DiffusionTaskBase, seed: int) -> torch.Tensor:
    """Unit-variance starting noise, drawn exactly like the pipelines do for this seed."""
    from diffusers.utils.torch_utils import randn_tensor

    device = pipe._execution_device
    gen_device = "cpu" if torch.device(device).type == "mps" else device
    generator = [torch.Generator(device=gen_device).manual_seed(seed)]
    f = pipe.vae_scale_factor
    shape = (1, pipe.unet.config.in_channels, request.height // f, request.width // f)
    return randn_tensor(
        shape, generator=generator, device=torch.device(device), dtype=pipe.unet.dtype
    )


def sample(
    pipe, request: DiffusionTaskBase, seed: int, guidance_offset: torch.Tensor | float = 0.0
) -> torch.Tensor:
    """Denoise and return final latents. `guidance_offset` is added to every step's guidance
    value; pass a dual tensor to differentiate with respect to it."""
    sched = make_scheduler(pipe, request)
    timesteps = sched.timesteps
    schedule = parse_guidance_schedule(request.guidance_scale, len(timesteps))
    cfg = max(schedule) > 1.0 or torch.is_tensor(guidance_offset)
    cond = encode(pipe, request.prompt, request.negative_prompt, request.height, request.width, cfg)
    latents = initial_noise(pipe, request, seed) * sched.init_noise_sigma
    unet = pipe.unet
    for i, t in enumerate(timesteps):
        x = torch.cat([latents] * 2) if cond.cfg else latents
        x = sched.scale_model_input(x, t)
        noise = unet(
            x, t, encoder_hidden_states=cond.embeds, added_cond_kwargs=cond.added, return_dict=False
        )[0]
        if cond.cfg:
            uncond, text = noise.chunk(2)
            noise = uncond + (schedule[i] + guidance_offset) * (text - uncond)
        latents = sched.step(noise, t, latents, return_dict=False)[0]
    return latents


def decode(pipe, latents: torch.Tensor) -> torch.Tensor:
    """Latents -> image tensor in [0, 1], shape (1, 3, H, W) (dual-number safe)."""
    vae = pipe.vae
    scaling = getattr(vae.config, "scaling_factor", 0.18215)
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    upcast = bool(getattr(vae.config, "force_upcast", False)) and vae.dtype == torch.float16
    if upcast:
        vae.to(torch.float32)
    try:
        z = latents / scaling + shift
        image = vae.decode(z.to(vae.dtype), return_dict=False)[0]
    finally:
        if upcast:
            vae.to(torch.float16)
    return (image / 2 + 0.5).clamp(0, 1)


@contextlib.contextmanager
def autodiff_attention():
    """Forward-mode AD needs the plain 'math' attention kernel (fused kernels lack JVP rules)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with contextlib.ExitStack() as stack:
        stack.enter_context(sdpa_kernel([SDPBackend.MATH]))
        try:
            from diffusers.models.attention_dispatch import attention_backend

            stack.enter_context(attention_backend("_native_math"))
        except Exception:  # pragma: no cover - older diffusers
            pass
        yield


def guidance_jvp(pipe, request: DiffusionTaskBase, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (image, d image / d guidance), both (H, W, 3) float32 in [0, 1] image units."""
    import torch.autograd.forward_ad as fwAD

    device = pipe._execution_device
    dtype = pipe.unet.dtype
    with torch.no_grad(), autodiff_attention(), fwAD.dual_level():
        offset = fwAD.make_dual(
            torch.zeros((), device=device, dtype=dtype), torch.ones((), device=device, dtype=dtype)
        )
        latents = sample(pipe, request, seed, guidance_offset=offset)
        image = decode(pipe, latents)
        primal, tangent = fwAD.unpack_dual(image)
    if tangent is None:
        raise RuntimeError("The derivative did not propagate (unsupported operation in the model)")
    return primal[0].permute(1, 2, 0).float().cpu(), tangent[0].permute(1, 2, 0).float().cpu()
