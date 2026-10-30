# src/gradient/ai/diffusion/loading.py
"""Loading and caching of diffusion pipelines for any diffusers-supported family.

* Diffusers folders / Hub repos: the pipeline class is read from model_index.json,
  so SD1.x, SDXL, SD3.5, FLUX.1/FLUX.2, Qwen-Image, Z-Image, Chroma, Sana, ... all work.
* Single-file checkpoints (.safetensors): the architecture is detected from the
  tensor names (only the header + a few small tensors are read).
* Image-to-image pipelines are derived with `AutoPipelineForImage2Image.from_pipe`,
  sharing all weights with the cached text-to-image pipeline (no extra VRAM).
* Memory placement follows `performance.memory_mode` (see config/default_settings.yaml).

All functions here are blocking; call them through `gradient.core.scheduling.run_gpu_task`.
"""

import gc
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL, AutoPipelineForImage2Image, DiffusionPipeline

from gradient.core.config import get_setting
from gradient.core.device import default_dtype, empty_cache, free_vram_bytes, get_device
from gradient.core.families import family_from_class  # noqa: F401  (re-exported)
from gradient.core.resource_manager import resource_manager
from gradient.model_manager.inventory import dir_size, find_model_path, get_manifest_meta

from .loading_embeddings import load_textual_inversions
from .loading_loras import apply_loras_to_pipeline, unload_lora_weights

logger = logging.getLogger(__name__)

MEMORY_MODES = ("auto", "gpu", "balanced", "low", "minimal")

# Single-file checkpoint architecture (diffusers' infer_diffusers_model_type) -> pipeline class
SINGLE_FILE_PIPELINES: dict[str, str] = {
    "v1": "StableDiffusionPipeline",
    "v2": "StableDiffusionPipeline",
    "inpainting": "StableDiffusionInpaintPipeline",
    "inpainting_v2": "StableDiffusionInpaintPipeline",
    "xl_base": "StableDiffusionXLPipeline",
    "playground-v2-5": "StableDiffusionXLPipeline",
    "xl_refiner": "StableDiffusionXLImg2ImgPipeline",
    "xl_inpaint": "StableDiffusionXLInpaintPipeline",
    "sd3": "StableDiffusion3Pipeline",
    "sd35_large": "StableDiffusion3Pipeline",
    "sd35_medium": "StableDiffusion3Pipeline",
    "flux-dev": "FluxPipeline",
    "flux-schnell": "FluxPipeline",
    "chroma": "ChromaPipeline",
    "z-image-turbo": "ZImagePipeline",
}

# Families whose pipelines read `self._guidance_scale` on every step (classic CFG),
# so the guidance value can be changed from a step callback.
DYNAMIC_CFG_FAMILIES = {"sd15", "sdxl", "sd3", "pixart", "kolors"}


@dataclass
class LoadedPipeline:
    key: str
    model_id: str
    pipeline: DiffusionPipeline
    pipeline_class: str
    family: str | None
    dtype: torch.dtype
    memory_mode: str
    vae_id: str | None = None
    default_scheduler_config: dict | None = None
    derived: dict[str, DiffusionPipeline] = field(default_factory=dict)
    lora_signature: tuple = ()
    embeddings: set[str] = field(default_factory=set)


# LRU cache of base pipelines, most recently used last
loaded_pipelines: "OrderedDict[str, LoadedPipeline]" = OrderedDict()
_cache_lock = threading.RLock()


# --------------------------------------------------------------------------- resolution
class _LazySafetensors(Mapping):
    """Mapping view of a .safetensors file that only loads tensors on access."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self._file = safe_open(str(path), framework="pt", device="cpu")
        self._keys = set(self._file.keys())
        self._cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._keys:
            raise KeyError(key)
        if key not in self._cache:
            self._cache[key] = self._file.get_tensor(key)
        return self._cache[key]

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def detect_single_file_pipeline(path: Path) -> str:
    """Guess the pipeline class of a single-file checkpoint (defaults to SD1.x)."""
    meta = get_manifest_meta("checkpoints", path.name)
    if meta.get("pipeline"):
        return meta["pipeline"]
    if path.suffix.lower() not in (".safetensors", ".sft"):
        return "StableDiffusionPipeline"
    try:
        from diffusers.loaders.single_file_utils import infer_diffusers_model_type

        model_type = infer_diffusers_model_type(_LazySafetensors(path))
        cls = SINGLE_FILE_PIPELINES.get(model_type)
        logger.info("Single-file checkpoint %s detected as %r -> %s", path.name, model_type, cls)
        if cls is None:
            raise ValueError(
                f"Checkpoint '{path.name}' looks like a '{model_type}' model, which GRADIENT cannot load from a single "
                "file yet. Download the diffusers-format repository instead (Models tab)."
            )
        return cls
    except ValueError:
        raise
    except Exception as e:
        logger.warning("Could not detect architecture of %s (%s); assuming SD1.x", path.name, e)
        return "StableDiffusionPipeline"


@dataclass
class ModelSource:
    model_id: str
    load_path: str
    single_file: bool
    local: bool
    pipeline_class: str | None
    variant: str | None
    manifest: dict[str, Any]
    size_bytes: int | None


def _has_variant_files(folder: Path, variant: str) -> bool:
    return any(folder.rglob(f"*.{variant}.safetensors")) or any(folder.rglob(f"*.{variant}.bin"))


def resolve_model_source(model_id: str) -> ModelSource:
    diffusers_dir = find_model_path(model_id, "diffusers")
    if diffusers_dir and diffusers_dir.is_dir():
        meta = get_manifest_meta("diffusers", diffusers_dir.name)
        pipeline_class = meta.get("pipeline")
        index = diffusers_dir / "model_index.json"
        if index.is_file():
            try:
                pipeline_class = json.loads(index.read_text())["_class_name"]
            except (KeyError, json.JSONDecodeError):
                pass
        variant = meta.get("variant")
        if variant and not _has_variant_files(diffusers_dir, variant):
            variant = None
        if not variant and _has_variant_files(diffusers_dir, "fp16"):
            variant = "fp16"
        return ModelSource(
            model_id,
            str(diffusers_dir),
            False,
            True,
            pipeline_class,
            variant,
            meta,
            dir_size(diffusers_dir),
        )

    ckpt = find_model_path(model_id, "checkpoints")
    if ckpt and ckpt.is_file():
        return ModelSource(
            model_id,
            str(ckpt),
            True,
            True,
            detect_single_file_pipeline(ckpt),
            None,
            get_manifest_meta("checkpoints", ckpt.name),
            ckpt.stat().st_size,
        )

    # Not local: treat as a Hugging Face Hub repo id (downloaded to the HF cache on first use).
    if "/" not in model_id:
        raise ValueError(
            f"Model '{model_id}' not found in your model folders. Download it from the Models tab "
            "or use a Hugging Face repo id like 'stabilityai/sdxl-turbo'."
        )
    logger.info("Model %s not found locally; loading from the Hugging Face Hub cache", model_id)
    pipeline_class = None
    try:
        config = DiffusionPipeline.load_config(model_id)
        pipeline_class = config.get("_class_name")
    except Exception as e:
        logger.warning("Could not read model_index.json for %s: %s", model_id, e)
    return ModelSource(model_id, model_id, False, False, pipeline_class, None, {}, None)


# --------------------------------------------------------------------------- memory modes
def choose_memory_mode(size_bytes: int | None) -> str:
    requested = str(get_setting("performance.memory_mode", "auto")).lower()
    if requested not in MEMORY_MODES:
        logger.warning("Unknown memory_mode %r; using auto", requested)
        requested = "auto"
    device = get_device()
    if device.type != "cuda":
        return "gpu"  # CPU / MPS / XPU: plain placement (unified memory or no accelerator)
    if requested != "auto":
        return requested
    free = free_vram_bytes() or 0
    if not size_bytes:
        return "gpu" if free > 14 * 1024**3 else "balanced"
    if size_bytes * 1.2 < free:
        return "gpu"
    if size_bytes * 0.55 < free:
        return "balanced"
    return "low"


def apply_memory_mode(pipe: DiffusionPipeline, mode: str) -> str:
    device = get_device()
    try:
        if mode == "balanced":
            pipe.enable_model_cpu_offload(device=device)
        elif mode == "low":
            pipe.enable_group_offload(
                onload_device=device,
                offload_type="leaf_level",
                use_stream=device.type == "cuda",
                low_cpu_mem_usage=True,
            )
        elif mode == "minimal":
            pipe.enable_sequential_cpu_offload(device=device)
        else:
            pipe.to(device)
            mode = "gpu"
    except Exception as e:
        logger.warning(
            "Memory mode %r failed (%s); falling back to model CPU offload / full placement",
            mode,
            e,
        )
        try:
            pipe.enable_model_cpu_offload(device=device)
            mode = "balanced"
        except Exception:
            pipe.to(device)
            mode = "gpu"
    return mode


def _apply_attention_backend(pipe: DiffusionPipeline) -> None:
    backend = get_setting("performance.attention_backend")
    if not backend:
        return
    for name in ("transformer", "transformer_2", "unet"):
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "set_attention_backend"):
            try:
                module.set_attention_backend(backend)
                logger.info("Attention backend %r enabled on %s", backend, name)
            except Exception as e:
                logger.warning("Attention backend %r not available for %s: %s", backend, name, e)


# --------------------------------------------------------------------------- loading
def _load_vae(vae_id: str, dtype: torch.dtype) -> AutoencoderKL:
    path = find_model_path(vae_id, "vae")
    if path and path.is_file():
        return AutoencoderKL.from_single_file(str(path), dtype=dtype)
    return AutoencoderKL.from_pretrained(str(path) if path else vae_id, dtype=dtype)


def _pipeline_key(model_id: str, vae_id: str | None) -> str:
    return f"{model_id}|vae={vae_id or ''}"


def _load_base(model_id: str, vae_id: str | None) -> LoadedPipeline:
    import diffusers

    source = resolve_model_source(model_id)
    dtype = default_dtype(source.pipeline_class)
    if source.manifest.get("torch_dtype") in ("float16", "bfloat16") and get_device().type != "cpu":
        dtype = getattr(torch, source.manifest["torch_dtype"])

    max_cached = int(get_setting("performance.max_cached_pipelines", 1))
    while len(loaded_pipelines) >= max(1, max_cached):
        oldest = next(iter(loaded_pipelines))
        _unload_entry(oldest)

    load_args: dict[str, Any] = {"dtype": dtype}
    if vae_id:
        load_args["vae"] = _load_vae(vae_id, dtype)
    logger.info(
        "Loading %s (%s, %s) from %s",
        model_id,
        source.pipeline_class or "auto",
        dtype,
        source.load_path,
    )
    if source.single_file:
        cls = getattr(diffusers, source.pipeline_class or "StableDiffusionPipeline")
        pipe = cls.from_single_file(source.load_path, **load_args)
    else:
        if source.variant:
            load_args["variant"] = source.variant
        pipe = DiffusionPipeline.from_pretrained(source.load_path, **load_args)

    if hasattr(pipe, "safety_checker") and getattr(pipe, "safety_checker", None) is not None:
        pipe.safety_checker = None  # research tool: the user is responsible for outputs
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    mode = apply_memory_mode(pipe, choose_memory_mode(source.size_bytes))
    _apply_attention_backend(pipe)
    pipeline_class = type(pipe).__name__
    key = _pipeline_key(model_id, vae_id)
    entry = LoadedPipeline(
        key=key,
        model_id=model_id,
        pipeline=pipe,
        pipeline_class=pipeline_class,
        family=source.manifest.get("family") or family_from_class(pipeline_class),
        dtype=dtype,
        memory_mode=mode,
        vae_id=vae_id,
        default_scheduler_config=dict(pipe.scheduler.config)
        if getattr(pipe, "scheduler", None)
        else None,
    )
    vram_mb = int(source.size_bytes / 1024**2) if (source.size_bytes and mode == "gpu") else 0
    resource_manager.register(key, f"pipeline:{entry.family}", vram_mb, unload=_async_unloader(key))
    logger.info(
        "Loaded %s as %s (family=%s, memory_mode=%s)", model_id, pipeline_class, entry.family, mode
    )
    return entry


def _async_unloader(key: str):
    async def _unload() -> None:
        from gradient.core.scheduling import run_gpu_task

        await run_gpu_task(_unload_entry, key, False)

    return _unload


def _unload_entry(key: str, release: bool = True) -> None:
    with _cache_lock:
        entry = loaded_pipelines.pop(key, None)
    if entry is None:
        return
    logger.info("Unloading pipeline %s", key)
    for pipe in [*entry.derived.values(), entry.pipeline]:
        try:
            if hasattr(pipe, "remove_all_hooks"):
                pipe.remove_all_hooks()
        except Exception:
            pass
    entry.derived.clear()
    del entry
    gc.collect()
    empty_cache()
    if release:
        resource_manager.release(key)


def unload_all_pipelines() -> None:
    for key in list(loaded_pipelines):
        _unload_entry(key)


def unload_model(model_id: str) -> int:
    """Unload every cached pipeline built from `model_id` (e.g. before deleting it)."""
    keys = [k for k, e in loaded_pipelines.items() if e.model_id == model_id]
    for key in keys:
        _unload_entry(key)
    return len(keys)


def get_loaded_entry(
    model_id: str, vae_id: str | None = None, force_reload: bool = False
) -> LoadedPipeline:
    key = _pipeline_key(model_id, vae_id)
    with _cache_lock:
        if force_reload and key in loaded_pipelines:
            _unload_entry(key)
        entry = loaded_pipelines.get(key)
        if entry is None:
            entry = _load_base(model_id, vae_id)
            loaded_pipelines[key] = entry
        loaded_pipelines.move_to_end(key)
    resource_manager.update_usage(key)
    return entry


def get_pipeline(
    model_id: str,
    task: str = "text2image",
    vae_id: str | None = None,
    loras: list | None = None,
    embeddings: list[str] | None = None,
) -> tuple[DiffusionPipeline, LoadedPipeline]:
    """Return a ready-to-call pipeline for `task` ('text2image' or 'image2image')."""
    entry = get_loaded_entry(model_id, vae_id)
    if task == "text2image":
        pipe = entry.pipeline
    elif task == "image2image":
        pipe = entry.derived.get(task)
        if pipe is None:
            try:
                pipe = AutoPipelineForImage2Image.from_pipe(entry.pipeline)
            except Exception as e:
                raise ValueError(
                    f"Image-to-image is not available for {entry.pipeline_class}: {e}"
                ) from e
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)
            entry.derived[task] = pipe
    else:
        raise ValueError(f"Unknown task {task!r}")

    signature = tuple(
        (getattr(lora, "name", None), float(getattr(lora, "weight", 1.0))) for lora in (loras or [])
    )
    if signature != entry.lora_signature:
        if entry.lora_signature:
            unload_lora_weights(pipe)
        if signature:
            apply_loras_to_pipeline(pipe, loras)
        entry.lora_signature = signature
    if embeddings:
        entry.embeddings |= load_textual_inversions(
            pipe, [e for e in embeddings if e not in entry.embeddings]
        )
    return pipe, entry


def load_diffusion_pipeline(model_path_or_id: str, **_: Any) -> DiffusionPipeline:
    """Backward-compatible helper used by the visualization modules."""
    return get_loaded_entry(model_path_or_id).pipeline


def loaded_summary() -> list[dict[str, Any]]:
    return [
        {
            "model_id": e.model_id,
            "pipeline": e.pipeline_class,
            "family": e.family,
            "dtype": str(e.dtype).removeprefix("torch."),
            "memory_mode": e.memory_mode,
            "vae_id": e.vae_id,
            "loras": [name for name, _ in e.lora_signature],
            "derived": list(e.derived),
        }
        for e in loaded_pipelines.values()
    ]
