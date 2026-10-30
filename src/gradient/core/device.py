# gradient-project-root/src/gradient/core/device.py
"""Device / dtype selection shared by all AI modules."""

import functools
import logging

import torch

from gradient.core.config import get_setting

logger = logging.getLogger(__name__)

# Pipelines that were trained/distributed for fp16 and are happy in it.
_FP16_PIPELINE_HINTS = (
    "StableDiffusionPipeline",
    "StableDiffusionImg2ImgPipeline",
    "StableDiffusionInpaintPipeline",
    "StableDiffusionXL",
    "LatentConsistencyModel",
    "PixArt",
    "Kandinsky",
    "Kolors",
    "StableDiffusionControlNet",
)


@functools.cache
def get_device() -> torch.device:
    """Pick the compute device: config override, else cuda > mps > xpu > cpu."""
    override = get_setting("performance.device")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():  # also true for ROCm builds
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def device_summary() -> dict:
    """Human-readable info about the compute device (used by /system/info)."""
    device = get_device()
    info: dict = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        info.update(
            name=props.name,
            total_vram_gb=round(total / 1024**3, 2),
            free_vram_gb=round(free / 1024**3, 2),
            compute_capability=f"{props.major}.{props.minor}",
            cuda=torch.version.cuda,
            hip=getattr(torch.version, "hip", None),
            bf16=torch.cuda.is_bf16_supported(),
        )
    return info


def free_vram_bytes() -> int | None:
    device = get_device()
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device.index or 0)
        return free
    return None


def total_vram_bytes() -> int | None:
    device = get_device()
    if device.type == "cuda":
        _, total = torch.cuda.mem_get_info(device.index or 0)
        return total
    return None


def prefers_fp16(pipeline_class_name: str | None) -> bool:
    if not pipeline_class_name:
        return True
    return any(hint in pipeline_class_name for hint in _FP16_PIPELINE_HINTS)


def default_dtype(pipeline_class_name: str | None = None) -> torch.dtype:
    """Choose a sensible dtype for a pipeline class on the current device.

    UNet-era models (SD1.x/SDXL) use fp16 on accelerators. Modern DiT / flow-matching
    models (FLUX, SD3.5, Qwen-Image, Z-Image, ...) overflow in fp16 and are run in bf16.
    """
    device = get_device()
    fp16_family = prefers_fp16(pipeline_class_name)
    if device.type == "cpu":
        # fp32 is the safe choice on CPU; very large DiT models are kept in bf16 to fit RAM.
        return torch.float32 if fp16_family else torch.bfloat16
    if fp16_family:
        return torch.float16
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        logger.warning("GPU lacks bf16 support; falling back to fp16 for %s", pipeline_class_name)
        return torch.float16
    return torch.bfloat16


def empty_cache() -> None:
    device = get_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
