# src/gradient/ai/lab/explorer.py
"""The explorer: travel through a model's denoising trajectory, one region at a time.

A render is recorded as its trajectory: the latents after every step. *Branching* opens a
region again at step k. The model's denoised estimate at k (x0_k, from two consecutive
latents: x = x0 + σ·d) is noised back to step k's noise level with fresh noise, or
denoised under another prompt, and from then on everything outside the region is copied
from the recorded trajectory after each step (blended-latent style), so only the region
moves. Only the remaining steps run, so late branches are cheap. Branches use the model's
own image-to-image pipeline with the latents as input, which keeps the noise level and the
scheduler maths exact for every architecture (SD 1.x/SDXL UNets, FLUX, SD3, ...).

* settling map: branches at several steps measure how much the region can still change,
  relative to a completely new region (step 0). Next to it, a free estimate from the
  render alone (how much the denoised estimate still moves after each step), with their
  agreement, so the free estimate's precision is measured instead of assumed.
* word suggestions: which prompt words act on the region (cross-attention maps,
  ConceptAttention, or ablation for any model), with alternatives from the text
  encoder's vocabulary.
* branches: new seeds, other words, prompt mixes (embedding average) or LoRA weights,
  only in the region.

A path (the branches taken) replays exactly, so exploration can continue from any branch,
also after a restart: recorded trajectories are cached by (settings, path).
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import logging
import math
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from gradient.ai.diffusion import preview
from gradient.ai.diffusion.guidance import parse_guidance_schedule
from gradient.ai.diffusion.loading import get_pipeline
from gradient.ai.diffusion.models import Text2ImageRequest
from gradient.ai.diffusion.pipelines import compatible_schedulers, generate_images, resolve_seed
from gradient.core.config import get_setting
from gradient.core.jobs import check_cancelled, report_progress

from .ablation import content_words, remove_word
from .concept_attention import (
    ConceptAttentionProbe,
    concept_support,
    encode_concepts,
    parse_blocks,
)
from .features import distances, load_feature_model
from .imaging import overlay, thumbnail_b64, to_b64
from .models import (
    ExploreBase,
    ExploreBranch,
    ExploreBranchesRequest,
    ExploreBranchesResponse,
    ExploreFrame,
    ExploreMapPoint,
    ExploreMapRequest,
    ExploreMapResponse,
    ExploreOp,
    ExploreStartRequest,
    ExploreStartResponse,
    ExploreTokensRequest,
    ExploreTokensResponse,
    TokenSuggestion,
)
from .probes import AttentionRecorder, Probe, denoiser_of, word_token_positions
from .unet_sampler import weight_dtype

logger = logging.getLogger(__name__)

#: samplers whose steps map one-to-one to σ values (what branching and x0 estimates need)
SUPPORTED_SCHEDULERS = (
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "DPMSolverMultistepScheduler",
    "UniPCMultistepScheduler",
    "DEISMultistepScheduler",
    "LMSDiscreteScheduler",
    "FlowMatchEulerDiscreteScheduler",
)
FALLBACK_SCHEDULERS = ("EulerDiscreteScheduler", "FlowMatchEulerDiscreteScheduler")
#: suggested steps: where the region still changes this much (relative to a new region)
TARGETS = {"big": 0.75, "variations": 0.45, "details": 0.15}
CACHE_SIZE = 8
SEED_OFFSET = 1000  # default branch seeds: render seed + 1000, + 1001, ...


# --------------------------------------------------------------------------- trajectories
@dataclass
class Trajectory:
    latents: list[torch.Tensor]  # (1, ...) float32 on the CPU, after each step
    sigmas: list[float]  # noise level of each stored state
    space: str  # 'flow' (x = (1-σ)x0 + σε), 've' (x = x0 + σε) or 'vp' (x = (x0 + σε)/√(1+σ²))
    total: int  # steps of the full schedule (an image-to-image base runs only the last ones)
    image: Image.Image
    request: Text2ImageRequest  # the effective base settings (seed and sampler fixed)
    prompt: str  # prompt in effect after the path (what the next branch inherits)
    mix_prompt: str | None = None
    mix_weight: float = 0.5
    lora_weight: float | None = None

    @property
    def steps(self) -> int:
        return len(self.latents)


_cache: OrderedDict[str, Trajectory] = OrderedDict()


def _remember(key: str, t: Trajectory) -> None:
    _cache[key] = t
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)


def clear_cache() -> None:
    _cache.clear()


def noise_space(sched) -> str:
    sigmas = getattr(sched, "sigmas", None)
    if not torch.is_tensor(sigmas) or not hasattr(sched, "step_index"):
        raise ValueError(
            f"The explorer needs a sampler with a σ schedule; {type(sched).__name__} has none "
            "(pick Euler, DPM++ 2M or UniPC)"
        )
    if float(sigmas.max()) <= 1.0001:
        return "flow"
    return "vp" if abs(float(getattr(sched, "init_noise_sigma", 1.0)) - 1.0) < 1e-3 else "ve"


class TrajectoryRecorder(Probe):
    """Keeps the latents after every step (per image of the batch) and their noise level."""

    tensor_inputs = ("latents",)
    needs_fixed_latents = True

    def __init__(self) -> None:
        self.latents: list[list[torch.Tensor]] = []
        self.sigmas: list[float] = []
        self.space: str | None = None

    def on_step_end(self, pipe, step, timestep, cb_kwargs):
        latents = cb_kwargs.get("latents")
        if latents is None:
            return cb_kwargs
        sched = pipe.scheduler
        if self.space is None:
            self.space = noise_space(sched)
        index = sched.step_index
        sigmas = sched.sigmas
        level = float(sigmas[index]) if index is not None and index < len(sigmas) else 0.0
        if not self.latents:
            self.latents = [[] for _ in range(latents.shape[0])]
        for b in range(latents.shape[0]):
            self.latents[b].append(latents[b : b + 1].detach().float().cpu().clone())
        self.sigmas.append(level)
        return cb_kwargs


class RegionLock(Probe):
    """After each step, copy everything outside the region from a recorded trajectory."""

    tensor_inputs = ("latents",)
    needs_fixed_latents = True

    def __init__(self, reference: list[torch.Tensor], start: int, mask: torch.Tensor) -> None:
        self.reference = reference
        self.start = start
        self.mask = mask

    def on_step_end(self, pipe, step, timestep, cb_kwargs):
        latents = cb_kwargs.get("latents")
        i = self.start + step
        if latents is None or i >= len(self.reference):
            return cb_kwargs
        ref = self.reference[i].to(latents.device, latents.dtype)
        m = self.mask.to(latents.device, latents.dtype)
        cb_kwargs["latents"] = latents * m + ref * (1 - m)
        return cb_kwargs


def x0_estimate(t: Trajectory, k: int) -> torch.Tensor:
    """The model's denoised estimate when entering step k (k=0: the one of step 1)."""
    if t.steps < 2:
        return t.latents[-1]
    j = min(max(k, 1), t.steps - 1)
    x, prev = t.latents[j], t.latents[j - 1]
    s, sp = t.sigmas[j], t.sigmas[j - 1]
    if t.space == "vp":  # to σ-space: x·√(1+σ²) = x0 + σε
        x, prev = x * math.sqrt(1 + s * s), prev * math.sqrt(1 + sp * sp)
    return preview.estimate_x0(x, prev, s, sp)


# --------------------------------------------------------------------------- settings
def _effective_request(spec: ExploreBase) -> tuple[Text2ImageRequest, str | None]:
    """The base settings with a fixed seed and a supported sampler (and why it changed)."""
    req = spec.base_request.model_copy(deep=True)
    req.num_outputs = 1
    req.seed = resolve_seed(req.seed)
    if req.custom_sigmas:
        raise ValueError("Custom sigmas aren't supported by the explorer yet")
    notes = []
    if req.modulation:
        req.modulation = None
        notes.append("modulation lanes are left out (branches restart the step count)")
    if spec.init_image_base64 and not spec.strength:
        raise ValueError("Exploring an image-to-image render needs its strength")
    _pipe, entry = get_pipeline(
        req.model_id,
        vae_id=req.vae_id,
        loras=req.loras,
        embeddings=req.embeddings,
        quantization=req.quantization,
    )
    default = (entry.default_scheduler_config or {}).get("_class_name")
    current = req.scheduler_id if req.scheduler_id and req.scheduler_id != "Default" else default
    if current not in SUPPORTED_SCHEDULERS:
        compatible = compatible_schedulers(entry)
        for alt in FALLBACK_SCHEDULERS:
            if alt in compatible:
                req.scheduler_id = alt
                notes.append(f"sampler {current} replaced by {alt} (branching needs σ steps)")
                break
        else:
            raise ValueError(
                f"The explorer needs a σ-based sampler ({', '.join(SUPPORTED_SCHEDULERS)}); "
                f"{entry.pipeline_class} offers none"
            )
    return req, "; ".join(notes) or None


def _key(req: Text2ImageRequest, spec: ExploreBase, path: list[ExploreOp]) -> str:
    init = spec.init_image_base64
    data = {
        "request": req.model_dump(mode="json"),
        "init": hashlib.sha1(init.encode()).hexdigest() if init else None,
        "strength": spec.strength if init else None,
        "path": [op.model_dump(mode="json") for op in path],
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()


def get_trajectory(spec: ExploreBase) -> tuple[Trajectory, Text2ImageRequest, str | None]:
    req, note = _effective_request(spec)
    return _trajectory(req, spec, list(spec.path)), req, note


def _trajectory(req: Text2ImageRequest, spec: ExploreBase, path: list[ExploreOp]) -> Trajectory:
    key = _key(req, spec, path)
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit
    if path:
        t = _branch(_trajectory(req, spec, path[:-1]), [path[-1]])[0]
    else:
        t = _record_base(req, spec)
    _remember(key, t)
    return t


def _record_base(req: Text2ImageRequest, spec: ExploreBase) -> Trajectory:
    rec = TrajectoryRecorder()
    i2i = bool(spec.init_image_base64)
    result = generate_images(
        req,
        task="image2image" if i2i else "text2image",
        init_image_b64=spec.init_image_base64,
        strength=spec.strength if i2i else None,
        probes=[rec],
    )
    if not rec.latents:
        raise ValueError(
            f"{result.params.get('pipeline')} doesn't report its steps; the explorer needs step callbacks"
        )
    latents = rec.latents[0]
    if not i2i and len(latents) != req.num_inference_steps:
        raise ValueError(
            f"Unexpected step count ({len(latents)} for {req.num_inference_steps} steps); "
            "this sampler can't be explored"
        )
    return Trajectory(
        latents=latents,
        sigmas=rec.sigmas,
        space=rec.space or "ve",
        total=req.num_inference_steps,
        image=result.images[0].convert("RGB"),
        request=req,
        prompt=req.prompt,
        lora_weight=req.loras[0].weight if req.loras else None,
    )


# --------------------------------------------------------------------------- masks
def mask_array(mask_b64: str | None, width: int, height: int) -> np.ndarray:
    """(H, W) float32 in [0, 1]; white or opaque = the region; None = the whole image."""
    if not mask_b64:
        return np.ones((height, width), np.float32)
    data = mask_b64.split(",", 1)[1] if mask_b64.startswith("data:") else mask_b64
    try:
        img = Image.open(io.BytesIO(base64.b64decode(data)))
        img.load()
    except Exception as e:
        raise ValueError(f"Could not read the region mask: {e}") from e
    if img.mode in ("RGBA", "LA") and img.getchannel("A").getextrema()[0] < 255:
        gray = img.getchannel("A")
    else:
        gray = img.convert("L")
    arr = np.asarray(gray.resize((width, height), Image.Resampling.BILINEAR), np.float32) / 255
    if arr.max() < 0.5:
        raise ValueError("The region is empty: draw it on the image first")
    return arr


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8), "L")
    return np.asarray(img.resize((shape[1], shape[0]), Image.Resampling.BOX), np.float32) / 255


def _soften(m: torch.Tensor) -> torch.Tensor:
    """Grow the region by one latent pixel and feather its edge (no seams at the border)."""
    grown = F.max_pool2d(m, 3, 1, 1)
    g = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    kernel = (torch.outer(g, g) / 256)[None, None]
    blurred = F.conv2d(F.pad(grown, (2, 2, 2, 2), mode="replicate"), kernel)
    return torch.maximum(m, blurred).clamp(0, 1)


def latent_mask(pipe, like: torch.Tensor, mask_b64: str | None, width: int, height: int):
    """The region in the pipeline's latent layout (packed or not), soft-edged, on the CPU."""
    if not mask_b64:
        return torch.ones_like(like)
    m = torch.from_numpy(mask_array(mask_b64, width, height))[None, None]
    if like.ndim == 3:  # packed tokens (FLUX, Qwen-Image): (1, seq, C*4)
        factor = int(getattr(pipe, "vae_scale_factor", 8))
        c = like.shape[-1] // 4
        h, w = 2 * (height // (factor * 2)), 2 * (width // (factor * 2))
    elif like.ndim == 4:
        c, h, w = like.shape[1:]
    elif like.ndim == 5:
        c, h, w = like.shape[1], like.shape[3], like.shape[4]
    else:
        raise ValueError(f"Unsupported latent layout {tuple(like.shape)}")
    m = _soften(F.adaptive_avg_pool2d(m, (h, w)))
    if like.ndim == 4:
        return m.expand(1, c, h, w).contiguous()
    if like.ndim == 5:
        return m[:, :, None].expand(1, c, like.shape[2], h, w).contiguous()
    pack = getattr(pipe, "_pack_latents", None)
    try:
        packed = pack(m.expand(1, c, h, w).contiguous(), 1, c, h, w)
    except Exception as e:
        raise ValueError(f"Region masks aren't supported for {type(pipe).__name__} yet: {e}") from e
    if packed.shape != like.shape:
        raise ValueError(f"Region mask layout {tuple(packed.shape)} != latents {tuple(like.shape)}")
    return packed


def _unpacked(pipe, x: torch.Tensor, req: Text2ImageRequest) -> torch.Tensor:
    if x.ndim == 3:
        return pipe._unpack_latents(x, req.height, req.width, pipe.vae_scale_factor)
    return x


# --------------------------------------------------------------------------- branching
def _mixed_embeddings(pipe, req: Text2ImageRequest, prompt: str, mix: str, w: float) -> dict:
    """Prompt embeddings blended (1-w)·prompt + w·mix ('average' conditioning)."""
    device = pipe._execution_device
    cfg = parse_guidance_schedule(req.guidance_scale, max(1, req.num_inference_steps))[0] > 1
    negative = req.negative_prompt or None
    name = type(pipe).__name__

    def lerp(a, b):
        return a * (1 - w) + b * w

    with torch.no_grad():
        if name == "StableDiffusionPipeline":
            pa, na = pipe.encode_prompt(prompt, device, 1, cfg, negative)
            pb, _ = pipe.encode_prompt(mix, device, 1, False)
            out = {"prompt_embeds": lerp(pa, pb)}
            if cfg:
                out["negative_prompt_embeds"] = na
            return out
        if name == "StableDiffusionXLPipeline":
            pa, na, ppa, npa = pipe.encode_prompt(
                prompt=prompt, device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=cfg, negative_prompt=negative,
            )  # fmt: skip
            pb, _, ppb, _ = pipe.encode_prompt(
                prompt=mix,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            out = {"prompt_embeds": lerp(pa, pb), "pooled_prompt_embeds": lerp(ppa, ppb)}
            if cfg:
                out.update(negative_prompt_embeds=na, negative_pooled_prompt_embeds=npa)
            return out
        if name == "FluxPipeline":
            pa, ppa, _ = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=device, num_images_per_prompt=1
            )
            pb, ppb, _ = pipe.encode_prompt(
                prompt=mix, prompt_2=None, device=device, num_images_per_prompt=1
            )
            return {"prompt_embeds": lerp(pa, pb), "pooled_prompt_embeds": lerp(ppa, ppb)}
        if name == "StableDiffusion3Pipeline":
            pa, na, ppa, npa = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None, device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=cfg, negative_prompt=negative,
            )  # fmt: skip
            pb, _, ppb, _ = pipe.encode_prompt(
                prompt=mix, prompt_2=None, prompt_3=None, device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=False,
            )  # fmt: skip
            out = {"prompt_embeds": lerp(pa, pb), "pooled_prompt_embeds": lerp(ppa, ppb)}
            if cfg:
                out.update(negative_prompt_embeds=na, negative_pooled_prompt_embeds=npa)
            return out
    raise ValueError(f"Prompt mixing isn't available for {name} yet; use a word branch instead")


def _branch(parent: Trajectory, ops: list[ExploreOp]) -> list[Trajectory]:
    """Run branches that differ only by consecutive seeds as one batch."""
    op = ops[0]
    k, n = op.step, parent.steps
    if not 0 <= k < n:
        raise ValueError(f"Step {k} is outside this render's steps (0–{n - 1})")
    req = parent.request.model_copy(deep=True)
    req.seed = op.seed
    req.prompt = op.prompt if op.prompt is not None else parent.request.prompt
    if op.lora_weight is not None:
        if not req.loras:
            raise ValueError("This render has no LoRA whose weight could change")
        req.loras = [req.loras[0].model_copy(update={"weight": op.lora_weight}), *req.loras[1:]]
    pipe, _entry = get_pipeline(
        req.model_id,
        vae_id=req.vae_id,
        loras=req.loras,
        embeddings=req.embeddings,
        quantization=req.quantization,
    )
    device = pipe._execution_device
    dtype = weight_dtype(denoiser_of(pipe))
    image = _unpacked(pipe, x0_estimate(parent, k), req).to(device, dtype)
    if len(ops) > 1:  # one input per branch (diffusers deprecates duplicating one image)
        image = image.expand(len(ops), *image.shape[1:]).contiguous()
    mask = latent_mask(pipe, parent.latents[0], op.mask_base64, req.width, req.height)
    overrides: dict = {"image": image}
    if op.mix_prompt:
        overrides.update(_mixed_embeddings(pipe, req, req.prompt, op.mix_prompt, op.mix_weight))
    rec = TrajectoryRecorder()
    i2i, _ = get_pipeline(
        req.model_id, task="image2image", vae_id=req.vae_id, loras=req.loras,
        embeddings=req.embeddings, quantization=req.quantization,
    )  # fmt: skip
    result = generate_images(
        req,
        task="image2image",
        strength=branch_strength(n - k, parent.total, start_rounding(i2i)),
        num_outputs=len(ops),
        probes=[RegionLock(parent.latents, k, mask), rec],
        call_overrides=overrides,
    )
    if not rec.latents or len(rec.latents[0]) != n - k:
        got = len(rec.latents[0]) if rec.latents else 0
        raise RuntimeError(f"The branch ran {got} steps instead of {n - k}")
    out = []
    for b, img in enumerate(result.images):
        out.append(
            Trajectory(
                latents=parent.latents[:k] + rec.latents[b],
                sigmas=parent.sigmas[:k] + rec.sigmas,
                space=parent.space,
                total=parent.total,
                image=img.convert("RGB"),
                request=parent.request,
                prompt=req.prompt,
                mix_prompt=op.mix_prompt,
                mix_weight=op.mix_weight,
                lora_weight=op.lora_weight if op.lora_weight is not None else parent.lora_weight,
            )
        )
    return out


def start_rounding(pipe) -> str:
    """How an image-to-image pipeline turns strength into steps: 'steps' (SD/SDXL:
    int(N·s) steps run) or 'start' (FLUX/SD3: the first step is int(N - N·s))."""
    try:
        source = inspect.getsource(type(pipe).get_timesteps).replace(" ", "")
    except (AttributeError, OSError, TypeError):
        return "steps"
    return "steps" if "int(num_inference_steps*strength)" in source else "start"


def branch_strength(steps_left: int, total: int, rounding: str = "steps") -> float:
    """Image-to-image strength that runs exactly `steps_left` of `total` steps.

    N·s is nudged to the side where the pipeline's rounding lands on `steps_left`
    (N·s rarely equals it exactly in floating point)."""
    s = steps_left / total
    up = rounding == "steps"
    for _ in range(16):
        product = s * total
        if product == steps_left or (product > steps_left) == up:
            break
        s = math.nextafter(s, math.inf if up else -math.inf)
    return min(1.0, s)


def _same_but_seed(a: ExploreOp, b: ExploreOp) -> bool:
    skip = {"seed", "label"}
    return a.model_dump(exclude=skip) == b.model_dump(exclude=skip)


def branch_all(parent: Trajectory, ops: list[ExploreOp]) -> list[Trajectory]:
    """Branches in order; ops that differ only by consecutive seeds share one batched call."""
    limit = max(1, int(get_setting("lab.explore_batch", 4) or 1))
    out: list[Trajectory] = []
    i = 0
    while i < len(ops):
        group = [ops[i]]
        while (
            i + len(group) < len(ops)
            and len(group) < limit
            and _same_but_seed(ops[i], ops[i + len(group)])
            and ops[i + len(group)].seed == (ops[i].seed + len(group)) % (2**32)
        ):
            group.append(ops[i + len(group)])
        check_cancelled()
        out += _branch(parent, group)
        i += len(group)
    return out


def inherit(
    t: Trajectory, step: int, mask_b64: str | None, kind: str, seed: int, **change
) -> ExploreOp:
    """An op that keeps the trajectory's prompt, mix and LoRA weight unless changed."""
    fields = {
        "prompt": t.prompt,
        "mix_prompt": t.mix_prompt,
        "mix_weight": t.mix_weight,
        "lora_weight": t.lora_weight if t.lora_weight != _base_lora(t) else None,
    }
    if kind == "prompt":
        fields["mix_prompt"] = None
    fields.update({k: v for k, v in change.items() if v is not None or k == "label"})
    if fields["prompt"] == t.request.prompt:
        fields["prompt"] = None
    return ExploreOp(step=step, mask_base64=mask_b64, kind=kind, seed=seed % (2**32), **fields)


def _base_lora(t: Trajectory) -> float | None:
    return t.request.loras[0].weight if t.request.loras else None


# --------------------------------------------------------------------------- measuring
def _features(name: str | None):
    try:
        return load_feature_model(name)
    except Exception as e:
        if "No vision feature model" in str(e):
            return None, "Measured in pixels: no vision model installed (Models tab → DINOv2 small)"
        return None, f"Measured in pixels ({str(e).splitlines()[0][:160]})"


def region_distance(fm, a: Image.Image, b: Image.Image, mask: np.ndarray):
    """(distance in the region, distance outside it, per-cell map) in DINO patches or pixels."""
    if fm is not None:
        _, cells = distances(fm, a, b)
    else:
        size = (64, 64)
        pa = np.asarray(a.convert("RGB").resize(size, Image.Resampling.BOX), np.float32) / 255
        pb = np.asarray(b.convert("RGB").resize(size, Image.Resampling.BOX), np.float32) / 255
        cells = np.abs(pa - pb).mean(-1)
    m = _resize_mask(mask, cells.shape)
    inside = float((cells * m).sum() / max(m.sum(), 1e-6))
    rest = 1 - m
    outside = float((cells * rest).sum() / rest.sum()) if rest.sum() > 0.5 else 0.0
    return inside, outside, cells


def _projection(pipe, entry):
    cached = getattr(entry, "preview_projection", None)
    if cached is None:
        try:
            cached = preview.fit_projection(pipe) or False
        except Exception as e:
            logger.info("No latent preview for %s: %s", entry.model_id, e)
            cached = False
        entry.preview_projection = cached
    return cached or None


def _rgb(pipe, projection, x: torch.Tensor, req: Text2ImageRequest) -> np.ndarray:
    z = _unpacked(pipe, x, req)
    if z.ndim == 5:
        z = z[:, :, 0]
    return preview.to_rgb(z[0], projection).astype(np.float32) / 255


def still_changing(t: Trajectory, steps: list[int]) -> dict[int, np.ndarray] | None:
    """Per step: how far the denoised estimate still is from the final image (free)."""
    req = t.request
    pipe, entry = get_pipeline(
        req.model_id, vae_id=req.vae_id, loras=req.loras, embeddings=req.embeddings,
        quantization=req.quantization,
    )  # fmt: skip
    projection = _projection(pipe, entry)
    if projection is None:
        return None
    final = _rgb(pipe, projection, t.latents[-1], req)
    return {
        k: np.abs(_rgb(pipe, projection, x0_estimate(t, k), req) - final).mean(-1) for k in steps
    }


def _interpolate_step(steps: list[int], values: list[float], target: float) -> float | None:
    pairs = list(zip(steps, values, strict=True))
    for (s0, v0), (s1, v1) in zip(pairs, pairs[1:], strict=False):
        if v0 >= target >= v1 and v0 != v1:
            return s0 + (s1 - s0) * (v0 - target) / (v0 - v1)
    return None


def default_steps(n: int, count: int) -> list[int]:
    hi = max(1, round(n * 0.85))  # the last steps only polish
    return sorted({int(v) for v in np.round(np.linspace(0, hi, count)) if v < n})


def _sheet(rows: list[tuple[str, list[Image.Image]]], cell: int = 168) -> Image.Image:
    cols = max(1, max(len(images) for _, images in rows))
    label_w = 170
    sheet = Image.new("RGB", (label_w + cols * cell, len(rows) * cell), (24, 24, 27))
    draw = ImageDraw.Draw(sheet)
    for r, (label, images) in enumerate(rows):
        y = r * cell
        draw.multiline_text((10, y + 10), label, fill=(235, 235, 235), spacing=4)
        for c, img in enumerate(images):
            thumb = img.copy()
            thumb.thumbnail((cell - 6, cell - 6))
            x = label_w + c * cell + (cell - thumb.width) // 2
            sheet.paste(thumb, (x, y + (cell - thumb.height) // 2))
    return sheet


# --------------------------------------------------------------------------- tools
def explore_start(request: ExploreStartRequest) -> ExploreStartResponse:
    """Render (or replay) the image and show how it formed. Blocking."""
    started = time.time()
    t, req, note = get_trajectory(request)
    n = t.steps
    first = 1 if n > 2 else 0  # step 0 has no estimate of its own (it shows step 1's)
    frames_at = sorted(
        {int(v) for v in np.round(np.linspace(first, n - 1, min(request.frames, n - first)))}
    )
    maps = still_changing(t, frames_at)
    frames = []
    if maps is None:
        note = "; ".join(filter(None, [note, "no latent preview for this model"]))
    else:
        pipe, entry = get_pipeline(
            req.model_id, vae_id=req.vae_id, loras=req.loras, embeddings=req.embeddings,
            quantization=req.quantization,
        )  # fmt: skip
        projection = _projection(pipe, entry)
        first = maps[frames_at[0]]
        scale = float(np.percentile(first, 99)) or 1.0
        mean0 = float(first.mean()) or 1.0
        for k in frames_at:
            rgb = (_rgb(pipe, projection, x0_estimate(t, k), req) * 255).astype(np.uint8)
            estimate = Image.fromarray(rgb, "RGB").resize(t.image.size, Image.Resampling.BILINEAR)
            frames.append(
                ExploreFrame(
                    step=k,
                    sigma=t.sigmas[k - 1] if k > 0 else t.sigmas[0],
                    image_base64=thumbnail_b64(estimate, 320),
                    still_changing_base64=thumbnail_b64(
                        overlay(t.image, np.clip(maps[k] / scale, 0, 1), absolute=True), 320
                    ),
                    still_changing=float(maps[k].mean()) / mean0,
                )
            )
    return ExploreStartResponse(
        image_base64=to_b64(t.image),
        seed=req.seed,
        steps=n,
        scheduler=req.scheduler_id or "default",
        prompt=t.prompt,
        frames=frames,
        note=note,
        performance_ms=(time.time() - started) * 1000,
    )


def explore_map(request: ExploreMapRequest) -> ExploreMapResponse:
    """When does the region settle? Branches at several steps, measured. Blocking."""
    started = time.time()
    t, req, note = get_trajectory(request)
    n = t.steps
    steps = sorted(
        {s for s in (request.steps or default_steps(n, request.num_steps)) if 0 <= s < n}
    )
    if not steps or steps[0] != 0:
        steps = [0, *steps]  # step 0 (a new region) is the scale
    mask = mask_array(request.mask_base64, req.width, req.height)
    fm, fm_note = _features(request.feature_model)
    seed0 = (request.seed if request.seed is not None else req.seed + SEED_OFFSET) % (2**32)
    quick = still_changing(t, steps)

    measured = []
    for i, k in enumerate(steps):
        report_progress(item=i + 1, items=len(steps), phase=f"branching at step {k}")
        ops = [
            inherit(t, k, request.mask_base64, "seed", seed0 + j) for j in range(request.branches)
        ]
        images = [b.image for b in branch_all(t, ops)]
        rows = [region_distance(fm, t.image, img, mask) for img in images]
        d = np.array([r[0] for r in rows])
        cells = np.mean([r[2] for r in rows], axis=0)
        measured.append((k, images, float(d.mean()), float(d.std()), cells))

    scale = measured[0][2] or 1e-9
    cell_scale = float(np.percentile(measured[0][4], 99)) or 1e-9
    region_cells = _resize_mask(mask, measured[0][4].shape)
    quick_scale = None
    if quick is not None:
        mq = _resize_mask(mask, quick[steps[0]].shape)
        quick_scale = float((quick[steps[0]] * mq).sum() / max(mq.sum(), 1e-6)) or 1e-9

    points, correlations = [], []
    for k, images, dist, spread, cells in measured:
        estimate = None
        if quick is not None:
            mq = _resize_mask(mask, quick[k].shape)
            estimate = float((quick[k] * mq).sum() / max(mq.sum(), 1e-6)) / quick_scale
            if dist / scale > 0.05:  # where something still changes, compare the maps
                q = np.asarray(
                    Image.fromarray(quick[k].astype(np.float32), "F").resize(
                        (cells.shape[1], cells.shape[0]), Image.Resampling.BOX
                    )
                )
                keep = region_cells > 0.5
                if keep.sum() >= 6 and q[keep].std() > 0 and cells[keep].std() > 0:
                    correlations.append(float(np.corrcoef(q[keep], cells[keep])[0, 1]))
        heat = np.clip(cells / cell_scale, 0, 1) * np.clip(region_cells * 1.5, 0.15, 1)
        points.append(
            ExploreMapPoint(
                step=k,
                sigma=t.sigmas[k - 1] if k > 0 else t.sigmas[0],
                change=dist / scale,
                distance=dist,
                spread=spread,
                estimate=estimate,
                images_base64=[thumbnail_b64(img, 256) for img in images],
                heatmap_base64=thumbnail_b64(overlay(t.image, heat, absolute=True), 384),
            )
        )

    changes = [p.change for p in points]
    window = None
    hi, lo = _interpolate_step(steps, changes, 0.8), _interpolate_step(steps, changes, 0.2)
    if hi is not None or lo is not None:
        window = [
            round(hi if hi is not None else 0.0, 2),
            round(lo if lo is not None else n - 1, 2),
        ]
    suggested = {}
    for name, target in TARGETS.items():
        at = _interpolate_step(steps, changes, target)
        if at is None:
            at = steps[-1] if all(c > target for c in changes) else 0
        suggested[name] = int(min(n - 1, max(0, round(at))))
    estimate_error = None
    if quick is not None:
        estimate_error = float(np.mean([abs((p.estimate or 0) - p.change) for p in points]))
    correlation = float(np.mean(correlations)) if correlations else None

    rows = [
        (f"step {p.step}/{n}\nchange {p.change:.2f}", images)
        for p, (_, images, *_rest) in zip(points, measured, strict=True)
    ]
    sheet = _sheet([("this image", [t.image]), *rows])
    summary = _map_summary(points, window, suggested, estimate_error, correlation, n)
    return ExploreMapResponse(
        image_base64=to_b64(sheet),
        reference_image_base64=to_b64(t.image),
        seed=req.seed,
        points=points,
        window=window,
        suggested=suggested,
        estimate_error=estimate_error,
        estimate_correlation=correlation,
        metric="dino" if fm is not None else "pixels",
        feature_model=fm.name if fm is not None else None,
        feature_note="; ".join(filter(None, [note, fm_note])) or None,
        summary=summary,
        performance_ms=(time.time() - started) * 1000,
    )


def _map_summary(points, window, suggested, error, correlation, n) -> str:
    parts = []
    if window:
        parts.append(
            f"The region settles between steps {window[0]:.1f} and {window[1]:.1f} of {n}."
        )
    else:
        parts.append("The region didn't settle within the tested steps.")
    parts.append(
        f"Branch at step {suggested['big']} for big changes, {suggested['variations']} for "
        f"variations, {suggested['details']} for details."
    )
    if error is not None:
        agree = f"The free estimate is off by {error:.2f} on average"
        if correlation is not None:
            agree += f"; its map correlates {correlation:.2f} with the measured one"
        parts.append(agree + ".")
    return " ".join(parts)


# ---- word suggestions
# keyed by id(text encoder); the weak reference tells a new encoder that reuses an id apart
_vocab_cache: dict[int, tuple[Any, list[str], torch.Tensor]] = {}


def _vocabulary(pipe):
    tok, enc = getattr(pipe, "tokenizer", None), getattr(pipe, "text_encoder", None)
    if tok is None or enc is None or not hasattr(enc, "get_input_embeddings"):
        return None
    key = id(enc)
    cached = _vocab_cache.get(key)
    if cached is not None and cached[0]() is enc:
        return cached[1], cached[2]
    weight = enc.get_input_embeddings().weight
    words: dict[str, int] = {}
    for piece, index in tok.get_vocab().items():
        if piece.endswith("</w>"):  # CLIP BPE
            word = piece[:-4]
        elif piece[:1] in ("▁", "Ġ"):  # SentencePiece, byte-level BPE
            word = piece[1:]
        else:
            continue
        word = word.lower()
        if len(word) >= 3 and word.isascii() and word.isalpha() and index < weight.shape[0]:
            words.setdefault(word, index)
        if len(words) >= 60000:
            break
    if not words:
        return None
    names = list(words)
    with torch.no_grad():
        rows = weight[torch.tensor(list(words.values()), device=weight.device)].float()
        matrix = F.normalize(rows, dim=-1).cpu().half()
    _vocab_cache[key] = (weakref.ref(enc), names, matrix)
    return names, matrix


def clear_vocabularies() -> int:
    count = len(_vocab_cache)
    _vocab_cache.clear()
    return count


def vocabulary_alternatives(pipe, word: str, count: int, exclude: set[str]) -> list[str]:
    """Nearest whole words to `word` in the text encoder's token-embedding space."""
    if count <= 0:
        return []
    vocab = _vocabulary(pipe)
    if vocab is None:
        return []
    names, matrix = vocab
    ids = pipe.tokenizer(word, add_special_tokens=False).input_ids
    if not ids:
        return []
    weight = pipe.text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        query = F.normalize(weight[torch.tensor(ids, device=weight.device)].float().mean(0), dim=0)
    sims = matrix.float() @ query.cpu()
    stem = word.lower()[:4]
    out: list[str] = []
    for i in sims.topk(min(len(names), count * 8)).indices.tolist():
        cand = names[i]
        if cand == word.lower() or cand in exclude or cand in out:
            continue
        if len(stem) >= 4 and cand[:4] == stem:  # plurals and other forms of the same word
            continue
        out.append(cand)
        if len(out) >= count:
            break
    return out


def _word_maps(pipe, plain: Text2ImageRequest, words: list[str], method: str, fm):
    """{word: 2-D map} and the image they belong to."""
    if method == "attention":
        if getattr(pipe, "unet", None) is not None:
            recorder = AttentionRecorder("*attn2", 0.0, 1.0)
            plain = plain.model_copy(update={"prompt_syntax": "plain"})
            image = generate_images(plain, probes=[recorder]).images[0]
            f = getattr(pipe, "vae_scale_factor", 8)
            maps = recorder.token_maps(plain.height // f, plain.width // f)
            if maps is None:
                raise ValueError("No attention was recorded")
            max_len = getattr(pipe.tokenizer, "model_max_length", 77)
            positions = {
                w.lower(): p for w, p in word_token_positions(pipe.tokenizer, plain.prompt, max_len)
            }
            out = {}
            for word in words:
                pos = [p for p in positions.get(word.lower(), []) if p < maps.shape[-1]]
                if pos:
                    out[word] = maps[..., pos].sum(-1).numpy()
            return out, image
        support = concept_support(pipe)
        if support is None:
            raise ValueError(f"No attention maps for {type(pipe).__name__}; use ablation")
        kind, n_blocks = support
        concepts = [*words, "background"]
        embeds, positions = encode_concepts(pipe, kind, concepts)
        probe = ConceptAttentionProbe(kind, embeds, positions, parse_blocks(None, n_blocks))
        image = generate_images(plain, probes=[probe]).images[0]
        maps = probe.concept_maps()
        if maps is None:
            raise ValueError("No concept attention was recorded")
        tokens = maps.shape[1]
        gh = max(1, round((tokens * plain.height / plain.width) ** 0.5))
        grid = maps.reshape(len(concepts), gh, tokens // gh).numpy()
        return {w: grid[i] for i, w in enumerate(words)}, image
    if fm is None:
        raise ValueError("Ablation needs a vision feature model (download DINOv2 small in Models)")
    image = generate_images(plain).images[0]
    out = {}
    for i, word in enumerate(words):
        check_cancelled()
        report_progress(item=i + 1, items=len(words), phase=f"without '{word}'")
        ablated = remove_word(plain.prompt, word) or " "
        other = generate_images(plain.model_copy(update={"prompt": ablated})).images[0]
        out[word] = distances(fm, image, other)[1]
    return out, image


def explore_tokens(request: ExploreTokensRequest) -> ExploreTokensResponse:
    """Which words act on the region, and what else could stand there. Blocking."""
    started = time.time()
    t, req, note = get_trajectory(request)
    prompt = t.prompt
    words = content_words(prompt, request.max_words)
    if not words:
        raise ValueError("The prompt has no words to suggest alternatives for")
    pipe, _entry = get_pipeline(
        req.model_id, vae_id=req.vae_id, loras=req.loras, embeddings=req.embeddings,
        quantization=req.quantization,
    )  # fmt: skip
    method = request.method
    if method == "auto":
        attention = getattr(pipe, "unet", None) is not None or concept_support(pipe) is not None
        method = "attention" if attention else "ablation"
    fm = None
    if method == "ablation":
        fm, fm_note = _features(request.feature_model)
        note = "; ".join(filter(None, [note, fm_note if fm is None else None])) or None
    plain = req.model_copy(update={"prompt": prompt})
    maps, image = _word_maps(pipe, plain, words, method, fm)
    if t.prompt != req.prompt or request.path:
        note = "; ".join(filter(None, [note, "measured on a plain render of the current prompt"]))

    mask = mask_array(request.mask_base64, req.width, req.height)
    exclude = {w.lower() for w in content_words(prompt, 60)}
    raw = []
    for word, m in maps.items():
        m = np.clip(np.asarray(m, np.float32), 0, None)
        region = _resize_mask(mask, m.shape)
        area = float(region.mean()) or 1e-6
        inside = float((m * region).sum())
        total = float(m.sum()) or 1e-9
        strength = inside / max(float(region.sum()), 1e-6)
        lift = (inside / total) / area
        raw.append((word, strength, lift, m))
    top = max((s * min(lift, 4.0) for _, s, lift, _ in raw), default=1.0) or 1.0
    suggestions = [
        TokenSuggestion(
            word=word,
            score=strength * min(lift, 4.0) / top,
            strength=strength,
            lift=lift,
            alternatives=vocabulary_alternatives(pipe, word, request.alternatives, exclude),
            overlay_base64=thumbnail_b64(overlay(image, m), 256),
        )
        for word, strength, lift, m in raw
    ]
    suggestions.sort(key=lambda s: s.score, reverse=True)
    return ExploreTokensResponse(
        image_base64=to_b64(t.image),
        prompt=prompt,
        method=method,
        words=suggestions,
        note=note,
        performance_ms=(time.time() - started) * 1000,
    )


def _label(op: ExploreOp, t: Trajectory) -> str:
    if op.label:
        return op.label
    if op.kind == "prompt" and op.prompt is not None:
        before, after = set(t.prompt.lower().split()), (op.prompt or "").split()
        new = [w for w in after if w.lower() not in before]
        return "+ " + " ".join(new) if new else (op.prompt or "")[:40]
    if op.kind == "mix":
        return f"mix '{op.mix_prompt}' {op.mix_weight:.0%}"
    if op.kind == "lora":
        return f"LoRA weight {op.lora_weight:g}"
    return f"seed {op.seed}"


def explore_branches(request: ExploreBranchesRequest) -> ExploreBranchesResponse:
    """Render branches of the region from one step. Blocking."""
    started = time.time()
    t, req, note = get_trajectory(request)
    k = request.step
    if not 0 <= k < t.steps:
        raise ValueError(f"Step {k} is outside this render's steps (0–{t.steps - 1})")
    seed0 = req.seed + SEED_OFFSET
    seeds_used = 0
    ops = []
    for spec in request.branches:
        if spec.kind == "prompt" and not (spec.prompt or "").strip():
            raise ValueError("A word branch needs its prompt")
        if spec.kind == "mix" and not (spec.mix_prompt or "").strip():
            raise ValueError("A mix branch needs the prompt to mix in")
        if spec.kind == "lora" and spec.lora_weight is None:
            raise ValueError("A LoRA branch needs the weight")
        if spec.seed is not None:
            seed = spec.seed
        elif spec.kind == "seed":
            seed, seeds_used = seed0 + seeds_used, seeds_used + 1
        else:
            seed = seed0  # same noise for every word/mix/LoRA branch: only that differs
        change = {"label": spec.label}
        if spec.kind == "prompt":
            change["prompt"] = spec.prompt.strip()
        elif spec.kind == "mix":
            change.update(mix_prompt=spec.mix_prompt.strip(), mix_weight=spec.mix_weight)
            if spec.prompt:
                change["prompt"] = spec.prompt.strip()
        elif spec.kind == "lora":
            change["lora_weight"] = spec.lora_weight
        ops.append(inherit(t, k, request.mask_base64, spec.kind, seed, **change))

    for op in ops:  # the path shows these labels
        op.label = _label(op, t)
    trajectories = branch_all(t, ops)
    mask = mask_array(request.mask_base64, req.width, req.height)
    fm, fm_note = _features(request.feature_model)
    base_params = req.model_dump(mode="json")
    results = []
    for op, branch in zip(ops, trajectories, strict=True):
        path = [*request.path, op]
        _remember(_key(req, request, path), branch)
        inside, outside, _ = region_distance(fm, t.image, branch.image, mask)
        params = {
            **base_params,
            "explore_path": [o.model_dump(mode="json") for o in path],
            "explore_label": _label(op, t),
        }
        if request.init_image_base64:
            params["explore_image_to_image"] = {"strength": request.strength}
        results.append(
            ExploreBranch(
                label=_label(op, t),
                op=op,
                image_base64=to_b64(branch.image),
                parameters=params,
                change=inside,
                outside_change=outside if request.mask_base64 else None,
            )
        )
    return ExploreBranchesResponse(
        reference_image_base64=to_b64(t.image),
        seed=req.seed,
        step=k,
        sigma=t.sigmas[k - 1] if k > 0 else t.sigmas[0],
        branches=results,
        metric="dino" if fm is not None else "pixels",
        feature_note="; ".join(filter(None, [note, fm_note])) or None,
        performance_ms=(time.time() - started) * 1000,
    )
