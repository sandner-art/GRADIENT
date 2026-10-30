# src/gradient/ai/lab/probes.py
"""Step-level instrumentation attached to a diffusers pipeline call.

A *probe* is attached before the pipeline runs, sees every step boundary through
`callback_on_step_end`, and is detached afterwards. Two probes live here:

* ModulationController - structural modulation lanes (guidance, prompt switch/blend,
  attention output scaling on selected layers, LoRA weight) over step ranges.
* AttentionRecorder - records cross-attention probabilities for selected layers and
  steps (UNet models: SD 1.x / SDXL) and turns them into per-word heatmaps.
"""

import fnmatch
import logging
import math
import re
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Probe:
    """Base class: override what you need."""

    #: tensors the probe needs in callback_kwargs (intersected with the pipeline's allowed list)
    tensor_inputs: tuple[str, ...] = ()

    def attach(self, pipe, entry, total_steps: int, kwargs: dict[str, Any]) -> None:  # noqa: B027
        pass

    def on_step_end(self, pipe, step: int, timestep, cb_kwargs: dict[str, Any]) -> dict[str, Any]:
        return cb_kwargs

    def detach(self) -> None:  # noqa: B027
        pass

    def report(self) -> dict[str, Any]:
        return {}


# --------------------------------------------------------------------------- helpers
def denoiser_of(pipe) -> torch.nn.Module:
    for name in ("unet", "transformer"):
        module = getattr(pipe, name, None)
        if module is not None:
            return module
    raise ValueError(f"{type(pipe).__name__} has no unet/transformer")


def is_attention_module(module: torch.nn.Module) -> bool:
    cls = type(module)
    if not cls.__name__.endswith("Attention"):
        return False
    return hasattr(module, "processor") or hasattr(module, "to_q") or hasattr(module, "to_qkv")


def attention_layers(pipe) -> list[dict[str, Any]]:
    """Names and kinds of the attention layers in the denoiser (for UI pickers)."""
    layers = []
    for name, module in denoiser_of(pipe).named_modules():
        if not is_attention_module(module):
            continue
        if name.endswith("attn2") or getattr(module, "is_cross_attention", False):
            kind = "cross"
        elif name.endswith("attn1"):
            kind = "self"
        else:
            kind = "joint"
        layers.append({"name": name, "kind": kind})
    return layers


def match_layers(pipe, pattern: str) -> list[tuple[str, torch.nn.Module]]:
    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    found = [
        (name, module)
        for name, module in denoiser_of(pipe).named_modules()
        if is_attention_module(module) and any(fnmatch.fnmatch(name, p) for p in patterns)
    ]
    if not found:
        raise ValueError(
            f"No attention layers match '{pattern}'. List them with GET /api/v1/lab/layers?model_id=..."
        )
    return found


def step_range(start: float, end: float, total: int) -> tuple[int, int]:
    return int(math.floor(start * total + 1e-9)), int(math.ceil(end * total - 1e-9))


class _CaptureDone(Exception):
    pass


def capture_text_tensors(
    pipe, kwargs: dict[str, Any], prompt: str, keys: list[str]
) -> dict[str, torch.Tensor]:
    """Run one denoising step with `prompt` to read the pipeline's own text tensors.

    This yields embeddings in exactly the format each pipeline family uses internally
    (CFG-concatenated for SD, pooled embeds for SDXL, ...), so prompt lanes work across
    families without per-family encoding code.
    """
    captured: dict[str, torch.Tensor] = {}

    def grab(_pipe, _step, _t, cb_kwargs):
        for key in keys:
            if torch.is_tensor(cb_kwargs.get(key)):
                captured[key] = cb_kwargs[key].detach().clone()
        raise _CaptureDone

    run = dict(kwargs)
    run.update(prompt=prompt, num_inference_steps=1, output_type="latent")
    run.pop("sigmas", None)
    gens = run.get("generator")
    n = len(gens) if isinstance(gens, list) else 1
    run["generator"] = [
        torch.Generator(device="cpu").manual_seed(0) for _ in range(n)
    ]  # keep real RNGs untouched
    run["callback_on_step_end"] = grab
    run["callback_on_step_end_tensor_inputs"] = keys
    try:
        pipe(**run)
    except _CaptureDone:
        pass
    return captured


# --------------------------------------------------------------------------- modulation
class ModulationController(Probe):
    def __init__(self, lanes: list, request) -> None:
        self.lanes = list(lanes)
        self.request = request
        self.hooks: list[Any] = []
        self.scales: dict[str, float] = {}
        self.layer_lanes: list[tuple[Any, list[str]]] = []
        self.text_keys: list[str] = []
        self.prompt_tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.base_prompt = request.prompt
        self.lora_names: list[str] = []
        self.lora_base: list[float] = []
        self.lora_current: list[float] | None = None
        self.notes: list[str] = []
        self.applied: dict[str, bool] = {}
        self.total = 1
        self.cfg_active = True
        self.dynamic_cfg = False

    @property
    def tensor_inputs(self) -> tuple[str, ...]:  # type: ignore[override]
        return tuple(self.text_keys)

    def _active(self, lane, step: int) -> bool:
        lo, hi = step_range(lane.start, lane.end, self.total)
        return lo <= step < hi

    def attach(self, pipe, entry, total_steps, kwargs):
        from gradient.ai.diffusion.loading import DYNAMIC_CFG_FAMILIES

        self.pipe = pipe
        self.total = max(1, total_steps)
        g0 = kwargs.get("guidance_scale", kwargs.get("true_cfg_scale", 1.0)) or 0.0
        self.cfg_active = float(g0) > 1.0
        self.dynamic_cfg = entry.family in DYNAMIC_CFG_FAMILIES and "guidance_scale" in kwargs

        # guidance lanes
        if any(lane.type == "guidance" for lane in self.lanes):
            if not self.dynamic_cfg:
                raise ValueError(
                    f"guidance lanes need a classic-CFG model (SD/SDXL/SD3); {entry.pipeline_class} is not"
                )
            self.applied["guidance"] = True
            for lane in self.lanes:
                if lane.type == "guidance" and self._active(lane, 0):
                    kwargs["guidance_scale"] = self._clamp(lane.value)

        # attention-scale lanes: forward hooks on matching attention modules
        for lane in (lane for lane in self.lanes if lane.type == "attention_scale"):
            names = []
            for name, module in match_layers(pipe, lane.targets):
                names.append(name)
                if name not in self.scales:
                    self.scales[name] = 1.0
                    self.hooks.append(module.register_forward_hook(self._make_hook(name)))
            self.layer_lanes.append((lane, names))
        if self.layer_lanes:
            self.applied["attention_scale"] = True
            self._update_scales(0)

        # LoRA weight lanes
        if any(lane.type == "lora_weight" for lane in self.lanes):
            loras = self.request.loras or []
            active = list(getattr(pipe, "get_active_adapters", lambda: [])())
            self.lora_names = [f"g_adapter_{i}" for i in range(len(loras))]
            self.lora_base = [float(lora.weight) for lora in loras]
            for lane in self.lanes:
                if lane.type == "lora_weight" and (
                    lane.lora_index >= len(loras) or self.lora_names[lane.lora_index] not in active
                ):
                    raise ValueError(
                        f"lora_weight lane refers to LoRA #{lane.lora_index}, which is not loaded"
                    )
            self.applied["lora_weight"] = True
            self._update_loras(0)

        # prompt lanes: capture every prompt's text tensors in the pipeline's own format
        prompt_lanes = [lane for lane in self.lanes if lane.type == "prompt"]
        if prompt_lanes:
            allowed = list(getattr(pipe, "_callback_tensor_inputs", []))
            self.text_keys = [k for k in allowed if k != "latents" and "embeds" in k]
            if "prompt_embeds" not in self.text_keys:
                raise ValueError(
                    f"{type(pipe).__name__} does not expose prompt embeddings to step callbacks"
                )
            initial = self.request.prompt
            for lane in prompt_lanes:
                if lane.mode == "replace" and self._active(lane, 0):
                    initial = lane.prompt
            prompts = {self.request.prompt, *(lane.prompt for lane in prompt_lanes)}
            for prompt in prompts:
                self.prompt_tensors[prompt] = capture_text_tensors(
                    pipe, kwargs, prompt, self.text_keys
                )
            shapes = {tuple(t["prompt_embeds"].shape) for t in self.prompt_tensors.values()}
            if len(shapes) > 1:
                raise ValueError(
                    f"Prompt lanes need equal-length embeddings, but {type(pipe).__name__} produced {sorted(shapes)}. "
                    "Use prompts of similar length or a CLIP/T5-padded model family."
                )
            kwargs["prompt"] = initial
            self.applied["prompt"] = True
            if any(lane.mode == "blend" and self._active(lane, 0) for lane in prompt_lanes):
                self.notes.append("prompt blends active at the first step take effect from step 2")

    def _clamp(self, value: float) -> float:
        # CFG batch layout is fixed at the first step: keep CFG on (>1) or off (<=1)
        return max(value, 1.0001) if self.cfg_active else min(value, 1.0)

    def _make_hook(self, name: str) -> Callable:
        def hook(_module, _args, output):
            scale = self.scales.get(name, 1.0)
            if scale == 1.0:
                return None
            if isinstance(output, tuple):
                return tuple(o * scale if torch.is_tensor(o) else o for o in output)
            return output * scale

        return hook

    def _update_scales(self, step: int) -> None:
        for name in self.scales:
            self.scales[name] = 1.0
        for lane, names in self.layer_lanes:
            if self._active(lane, step):
                for name in names:
                    self.scales[name] *= float(lane.value)

    def _update_loras(self, step: int) -> None:
        weights = list(self.lora_base)
        for lane in self.lanes:
            if lane.type == "lora_weight" and self._active(lane, step):
                weights[lane.lora_index] = float(lane.value)
        if weights != self.lora_current:
            self.pipe.set_adapters(self.lora_names, adapter_weights=weights)
            self.lora_current = weights

    def _prompt_state(self, step: int) -> dict[str, torch.Tensor]:
        state = dict(self.prompt_tensors[self.request.prompt])
        for lane in self.lanes:
            if lane.type != "prompt" or not self._active(lane, step):
                continue
            target = self.prompt_tensors[lane.prompt]
            if lane.mode == "replace":
                state = dict(target)
            else:
                w = 0.5 if lane.value is None else float(lane.value)
                state = {
                    k: torch.lerp(state[k], target[k], w) if k in target else state[k]
                    for k in state
                }
        return state

    def on_step_end(self, pipe, step, timestep, cb_kwargs):
        nxt = step + 1
        if nxt >= self.total:
            return cb_kwargs
        if self.layer_lanes:
            self._update_scales(nxt)
        if self.lora_names:
            self._update_loras(nxt)
        if self.dynamic_cfg:
            for lane in self.lanes:
                if lane.type == "guidance" and self._active(lane, nxt):
                    pipe._guidance_scale = self._clamp(lane.value)
        if self.prompt_tensors:
            for key, value in self._prompt_state(nxt).items():
                if key in cb_kwargs and cb_kwargs[key].shape == value.shape:
                    cb_kwargs[key] = value.to(cb_kwargs[key].device, cb_kwargs[key].dtype)
        return cb_kwargs

    def detach(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        if self.lora_names and self.lora_current != self.lora_base:
            try:
                self.pipe.set_adapters(self.lora_names, adapter_weights=self.lora_base)
            except Exception as e:  # pragma: no cover
                logger.warning("Could not restore LoRA weights: %s", e)

    def report(self):
        out: dict[str, Any] = {"modulation_applied": self.applied}
        if self.layer_lanes:
            out["modulation_layers"] = {
                lane.targets: len(names) for lane, names in self.layer_lanes
            }
        if self.notes:
            out["modulation_notes"] = self.notes
        return out


# --------------------------------------------------------------------------- attention maps
def word_token_positions(
    tokenizer, prompt: str, max_length: int = 77
) -> list[tuple[str, list[int]]]:
    """Split a prompt into words and find each word's token positions (CLIP-style BPE).

    Position 0 is the start token; words are tokenized one by one so multi-piece words
    get all their pieces.
    """
    words = re.findall(r"[\w'-]+|[^\w\s]", prompt)
    cursor = 1
    out: list[tuple[str, list[int]]] = []
    for word in words:
        ids = tokenizer(word, add_special_tokens=False).input_ids
        positions = [p for p in range(cursor, cursor + len(ids)) if p < max_length - 1]
        cursor += len(ids)
        if re.match(r"[\w'-]+", word) and positions:
            out.append((word, positions))
    return out


class _RecordingProcessor:
    """Drop-in for diffusers' AttnProcessor2_0 that also records attention probabilities."""

    def __init__(self, recorder: "AttentionRecorder", name: str, original) -> None:
        self.recorder = recorder
        self.name = name
        self.original = original

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        if not self.recorder.recording or encoder_hidden_states is None:
            return self.original(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)
        batch = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            mask = attn.prepare_attention_mask(attention_mask, key.shape[2], batch)
            scores = scores + mask.view(batch, attn.heads, -1, mask.shape[-1]).float()
        probs = scores.softmax(dim=-1)
        self.recorder.record(self.name, probs)

        hidden_states = torch.matmul(probs.to(value.dtype), value)
        hidden_states = (
            hidden_states.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim).to(query.dtype)
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class AttentionRecorder(Probe):
    def __init__(self, layers: str = "*attn2", start: float = 0.0, end: float = 1.0) -> None:
        self.pattern = layers
        self.start, self.end = start, end
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.originals: dict[str, Any] = {}
        self.modules: dict[str, Any] = {}
        self.step = 0
        self.total = 1
        self.cond_half = True

    @property
    def recording(self) -> bool:
        lo, hi = step_range(self.start, self.end, self.total)
        return lo <= self.step < hi

    def attach(self, pipe, entry, total_steps, kwargs):
        if getattr(pipe, "unet", None) is None:
            raise ValueError(
                f"Cross-attention maps need a UNet model (SD 1.x, SDXL); {entry.pipeline_class} uses joint "
                "attention. Use ConceptAttention instead (POST /api/v1/lab/concept-attention)."
            )
        self.total = max(1, total_steps)
        self.cond_half = float(kwargs.get("guidance_scale", 1.0) or 0) > 1.0
        for name, module in match_layers(pipe, self.pattern):
            if not (name.endswith("attn2") or getattr(module, "is_cross_attention", False)):
                continue
            self.originals[name] = module.processor
            self.modules[name] = module
            module.set_processor(_RecordingProcessor(self, name, module.processor))
        if not self.modules:
            raise ValueError(f"'{self.pattern}' matches no cross-attention layers (try '*attn2')")

    def record(self, name: str, probs: torch.Tensor) -> None:
        if self.cond_half and probs.shape[0] % 2 == 0:
            probs = probs[probs.shape[0] // 2 :]
        mean = (
            probs[:1].mean(dim=1)[0].detach().float().cpu()
        )  # (pixels, tokens) for the first image
        if name in self.sums:
            self.sums[name] += mean
        else:
            self.sums[name] = mean.clone()
        self.counts[name] = self.counts.get(name, 0) + 1

    def on_step_end(self, pipe, step, timestep, cb_kwargs):
        self.step = step + 1
        return cb_kwargs

    def detach(self):
        for name, module in self.modules.items():
            module.set_processor(self.originals[name])

    def token_maps(self, latent_h: int, latent_w: int) -> torch.Tensor | None:
        """Average recorded layers into one (latent_h, latent_w, tokens) map."""
        maps = []
        for name, total in self.sums.items():
            avg = total / max(1, self.counts[name])
            pixels, tokens = avg.shape
            factor = math.sqrt(latent_h * latent_w / pixels)
            h, w = round(latent_h / factor), round(latent_w / factor)
            if h * w != pixels:
                continue
            grid = avg.T.reshape(1, tokens, h, w)
            maps.append(
                F.interpolate(
                    grid, size=(latent_h, latent_w), mode="bilinear", align_corners=False
                )[0]
            )
        if not maps:
            return None
        return torch.stack(maps).mean(0).permute(1, 2, 0)  # (H, W, T)

    def report(self):
        return {
            "attention_layers_recorded": sorted(self.sums),
            "attention_steps": [self.start, self.end],
        }
