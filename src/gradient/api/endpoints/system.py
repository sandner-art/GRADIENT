# src/gradient/api/endpoints/system.py

import asyncio
import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, Field

from gradient.ai.diffusion import loading
from gradient.core import shutdown
from gradient.core.device import device_summary
from gradient.core.resource_manager import resource_manager
from gradient.core.scheduling import gpu_queue_depth, progress_tracker, run_gpu_task
from gradient.model_manager.inventory import list_model_names

logger = logging.getLogger(__name__)
router = APIRouter()

_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "huggingface_hub",
    "accelerate",
    "peft",
    "fastapi",
    "llama_cpp_python",
)


def _pkg_version(name: str) -> str | None:
    try:
        return version(name.replace("_", "-"))
    except PackageNotFoundError:
        return None


# --- Model lists (names usable as model_id / lora name etc.) ---
@router.get("/system/models/diffusion", response_model=list[str], tags=["System Info"])
async def get_diffusion_models():
    """Diffusion models (diffusers folders and single-file checkpoints) in the model folders.

    GGUF files are denoiser weights for quantized loading (GET /system/quantization), not
    models of their own, so they are left out."""
    checkpoints = {n for n in list_model_names("checkpoints") if not n.lower().endswith(".gguf")}
    return sorted(set(list_model_names("diffusers")) | checkpoints)


@router.get("/system/quantization", tags=["System Info"])
async def get_quantization_options(model_id: str | None = None):
    """Precisions a model can be loaded with (request field `quantization`): whether each
    works on this machine (backend installed, GPU) and, with `model_id`, for that model (a GGUF
    for another model family, bitsandbytes on a single-file checkpoint), with the reason."""
    from gradient.ai.diffusion import quantization

    family, single_file = loading.model_profile(model_id) if model_id else (None, False)
    return {
        "model_family": family,
        "single_file": single_file,
        "options": quantization.options(family, single_file),
    }


@router.get("/system/models/loras", response_model=list[str], tags=["System Info"])
async def get_loras():
    return list_model_names("lora")


@router.get("/system/models/embeddings", response_model=list[str], tags=["System Info"])
async def get_embeddings():
    return list_model_names("embeddings")


@router.get("/system/models/embeddings/details", tags=["System Info"])
async def get_embedding_details(model_id: str | None = Query(None)):
    """Installed embeddings with their token, model family and fit for `model_id`.

    `fits` is true/false when both families are known, else null. Write `token` in the
    prompt (or negative prompt) to use an embedding.
    """
    from gradient.ai.diffusion import loading_embeddings as emb

    def details():
        family = loading.model_profile(model_id)[0] if model_id else None
        rows = []
        for name in emb.installed_embeddings():
            info = emb.embedding_info(name)
            rows.append(
                {
                    "name": info.name,
                    "token": info.token,
                    "family": info.family,
                    "family_label": emb.FAMILY_LABELS.get(info.family or ""),
                    "vectors": info.vectors,
                    "fits": emb.fits(info.family, family),
                    "error": info.error,
                }
            )
        return {"model_family": family, "embeddings": rows}

    return await asyncio.to_thread(details)


@router.get("/system/models/vae", response_model=list[str], tags=["System Info"])
async def get_vaes():
    return list_model_names("vae")


@router.get("/system/models/controlnet", response_model=list[str], tags=["System Info"])
async def get_controlnets():
    return list_model_names("controlnet")


@router.get("/system/models/llm", response_model=list[str], tags=["System Info"])
async def get_llms():
    return [n for n in list_model_names("llm") if n.lower().endswith(".gguf")]


# --- Schedulers ---
@router.get("/system/schedulers", response_model=list[str], tags=["System Info"])
async def get_schedulers(
    model_id: str | None = Query(
        None, description="Only list schedulers compatible with this loaded model"
    ),
):
    """Schedulers from diffusers. With model_id (of a loaded model) only compatible ones are listed."""
    if model_id:
        from gradient.ai.diffusion.pipelines import compatible_schedulers

        for entry in loading.loaded_pipelines.values():
            if entry.model_id == model_id:
                return ["Default", *compatible_schedulers(entry)]
    from diffusers.schedulers.scheduling_utils import SchedulerMixin

    module = importlib.import_module("diffusers.schedulers")
    names = set()
    for attr in dir(module):
        if not attr.endswith("Scheduler"):
            continue
        try:
            obj = getattr(module, attr)
        except Exception:
            continue
        if isinstance(obj, type) and issubclass(obj, SchedulerMixin) and obj is not SchedulerMixin:
            names.add(attr)
    return ["Default", *sorted(names)]


# --- Runtime info ---
@router.get("/system/info", tags=["System Info"])
async def get_system_info():
    """Device, library versions and loaded models."""
    from gradient.core.version import app_version, install_is_stale, installed_version

    return {
        "version": app_version(),
        "installed_version": installed_version(),
        "install_is_stale": install_is_stale(),
        "device": device_summary(),
        "versions": {p: _pkg_version(p) for p in _PACKAGES},
        "loaded_pipelines": loading.loaded_summary(),
        "resources": resource_manager.snapshot(),
    }


@router.get("/system/status", tags=["System Info"])
async def get_runtime_status():
    """Current generation progress and queue depth (cheap; poll while generating)."""
    return {"queue_depth": gpu_queue_depth(), "current": progress_tracker.snapshot()}


_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _is_local(request: Request) -> bool:
    return bool(request.client and request.client.host in _LOCAL_HOSTS)


@router.post("/system/shutdown", tags=["System Info"])
async def stop_gradient(request: Request, by: Literal["web", "cli"] = "web"):
    """Stop GRADIENT gracefully, like the first Ctrl+C (only from this computer)."""
    if not _is_local(request):
        raise HTTPException(
            status_code=403, detail="GRADIENT can only be stopped from this computer"
        )
    if not shutdown.can_stop():
        raise HTTPException(
            status_code=409,
            detail="This GRADIENT wasn't started with `gradient serve`; stop it where it runs",
        )
    reason = "stopped from the web UI" if by == "web" else "gradient stop"
    asyncio.get_running_loop().call_later(0.2, shutdown.begin, reason)  # answer first
    return {"stopping": True}


MEMORY_TARGETS = ("diffusion", "vision", "llm", "caches")


def _vram() -> dict | None:
    """Free / total VRAM of the device and what PyTorch holds (CUDA only), in GB."""
    import torch

    from gradient.core.device import get_device

    device = get_device()
    if device.type != "cuda":
        return None
    idx = device.index or 0
    free, total = torch.cuda.mem_get_info(idx)
    gb = 1024**3
    return {
        "total_gb": round(total / gb, 2),
        "free_gb": round(free / gb, 2),
        "allocated_gb": round(torch.cuda.memory_allocated(idx) / gb, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(idx) / gb, 2),
    }


def _memory_report() -> dict:
    from gradient.ai.diffusion import prompt_syntax
    from gradient.ai.lab import embed, explorer, features
    from gradient.ai.llm.loading import loaded_llms

    tracked = {r["id"]: r["vram_mb"] for r in resource_manager.snapshot()}
    diffusion = [
        {
            "key": key,
            "name": entry.model_id,
            "detail": ", ".join(
                x
                for x in (
                    entry.family,
                    entry.quantization if entry.quantization not in (None, "none") else None,
                    entry.memory_mode,
                )
                if x
            ),
            "vram_mb": tracked.get(key),
        }
        for key, entry in loading.loaded_pipelines.items()
    ]
    return {
        "device": str(device_summary().get("device")),
        "vram": _vram(),
        "loaded": {
            "diffusion": diffusion,
            "vision": [{**m, "detail": "metrics (DINO)"} for m in features.loaded_feature_models()]
            + [{**m, "detail": "metrics (CLIP/SigLIP)"} for m in embed.loaded_clip_models()],
            "llm": [{"name": k, "vram_mb": tracked.get(k)} for k in loaded_llms],
        },
        "caches": {
            "explorer_paths": len(explorer._cache),
            "word_vocabularies": len(explorer._vocab_cache),
            "prompt_tokenizers": len(prompt_syntax._tokenizers),
        },
    }


@router.get("/system/memory", tags=["System Info"])
async def memory():
    """What is loaded (image models, vision models for metrics, LLMs) and the VRAM in use."""
    return await asyncio.to_thread(_memory_report)


class UnloadRequest(BaseModel):
    targets: list[Literal["diffusion", "vision", "llm", "caches"]] | None = Field(
        default=None,
        description=(
            "What to free: diffusion (image models), vision (DINO/CLIP/SigLIP models used for "
            "metrics), llm, caches (Explorer paths, word vocabularies, tokenizers; RAM only). "
            "Default: diffusion, vision and llm"
        ),
    )


@router.post("/system/unload", tags=["System Info"])
async def unload_models(body: UnloadRequest | None = Body(default=None)):
    """Unload models and free VRAM: image models, vision models and LLMs by default.

    Waits for the running GPU job, so nothing is unloaded under a render. `unloaded` counts
    the image models (as before); `freed` lists everything by kind.
    """
    from gradient.ai.lab import embed, explorer, features

    targets = set(body.targets if body and body.targets else ("diffusion", "vision", "llm"))
    before = _vram()
    freed: dict[str, list[str] | dict] = {}
    count = len(loading.loaded_pipelines)
    if "diffusion" in targets:
        freed["diffusion"] = sorted({e.model_id for e in loading.loaded_pipelines.values()})
        await run_gpu_task(loading.unload_all_pipelines)
        explorer.clear_vocabularies()  # built from the unloaded text encoders
    if "vision" in targets:

        def drop_vision() -> list[str]:
            return features.unload_feature_models() + embed.unload_clip_models()

        freed["vision"] = await run_gpu_task(drop_vision)
    if "llm" in targets:
        from gradient.ai.llm.loading import loaded_llms, unload_llm

        freed["llm"] = list(loaded_llms)
        for key in list(loaded_llms):
            await unload_llm(key)
    if "caches" in targets:
        from gradient.ai.diffusion import prompt_syntax

        freed["caches"] = {
            "explorer_paths": len(explorer._cache),
            "word_vocabularies": explorer.clear_vocabularies(),
            "prompt_tokenizers": len(prompt_syntax._tokenizers),
        }
        explorer.clear_cache()
        prompt_syntax._tokenizers.clear()
    await run_gpu_task(features.free_memory)
    after = _vram()
    return {
        "unloaded": count if "diffusion" in targets else 0,
        "freed": freed,
        "vram_before": before,
        "vram_after": after,
    }
