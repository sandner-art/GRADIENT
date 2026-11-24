# src/gradient/api/endpoints/system.py

import asyncio
import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

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
async def get_quantization_options():
    """Quantizations a model can be loaded with (request field `quantization`), with whether
    each works on this machine (backend installed, GPU needed) and GGUF denoiser files found."""
    from gradient.ai.diffusion import quantization

    return {"options": quantization.options()}


@router.get("/system/models/loras", response_model=list[str], tags=["System Info"])
async def get_loras():
    return list_model_names("lora")


@router.get("/system/models/embeddings", response_model=list[str], tags=["System Info"])
async def get_embeddings():
    return list_model_names("embeddings")


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


@router.post("/system/unload", tags=["System Info"])
async def unload_models():
    """Unload all cached diffusion pipelines and free VRAM."""
    count = len(loading.loaded_pipelines)
    await run_gpu_task(loading.unload_all_pipelines)
    return {"unloaded": count}
