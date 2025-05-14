# src/gradient/api/endpoints/system.py (NEW FILE)

import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Dict, Any
import pkg_resources # To inspect installed packages like diffusers
import importlib

from gradient.core.config import get_setting, PROJECT_ROOT
from diffusers.schedulers.scheduling_utils import SchedulerMixin # Base class for check

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Helper to scan model directories ---
def scan_model_directory(model_type: str) -> List[str]:
    """Scans configured search paths for models of a specific type."""
    found_models = set()
    search_dirs_config = get_setting("paths.models.search_dirs", ["./data/models"])
    subdir_name = get_setting(f"paths.models.{model_type}", model_type)

    for dir_str in search_dirs_config:
        base_dir = Path(dir_str)
        if not base_dir.is_absolute():
            base_dir = (PROJECT_ROOT / base_dir).resolve()

        model_subdir = base_dir / subdir_name
        if model_subdir.is_dir():
            try:
                for item in model_subdir.iterdir():
                    # For diffusers, add directories.
                    if model_type == "diffusers":
                        if item.is_dir():
                            found_models.add(item.name) # Add folder name
                    # --- MODIFIED FOR LoRA --- VVV
                    elif model_type == "lora":
                        # Add PEFT LoRA directories (must contain adapter_config.json)
                        if item.is_dir() and (item / "adapter_config.json").is_file():
                            found_models.add(item.name) # Add folder name
                        # Add single LoRA files
                        elif item.is_file() and item.suffix.lower() in ['.pt', '.pth', '.safetensors', '.ckpt']:
                             found_models.add(item.name) # Add file name
                    # --- MODIFIED FOR LoRA --- ^^^
                    elif model_type in ["checkpoints", "vae", "embeddings", "controlnet", "llm"]:
                         # Add files with common extensions
                         if item.is_file() and item.suffix.lower() in ['.pt', '.pth', '.safetensors', '.ckpt', '.bin', '.gguf']:
                             found_models.add(item.name) # Add file name
                         # VAEs can also be folders (often full diffusers models)
                         # Let's ensure VAE directories are also captured if they are not already by a diffusers-like check
                         elif model_type == "vae" and item.is_dir() and (item / "config.json").is_file(): # A simple check for a diffusers-like structure
                             found_models.add(item.name)
            except FileNotFoundError:
                logger.warning(f"Model subdirectory not found or inaccessible: {model_subdir}")
            except OSError as e:
                 logger.error(f"Error scanning directory {model_subdir}: {e}")

    return sorted(list(found_models))

# --- API Endpoints ---

@router.get("/system/models/diffusion", response_model=List[str], tags=["System Info"])
async def get_diffusion_models():
    """Lists diffusion models (checkpoints & diffusers) found in configured directories."""
    checkpoints = scan_model_directory("checkpoints")
    diffusers_models = scan_model_directory("diffusers")
    all_models = sorted(list(set(checkpoints + diffusers_models)))
    logger.info(f"Found diffusion models: {all_models}")
    return all_models

@router.get("/system/models/loras", response_model=List[str], tags=["System Info"])
async def get_loras():
    """Lists LoRA models (single files and PEFT directories) found in the configured directory."""
    loras = scan_model_directory("lora")
    logger.info(f"Found LoRAs: {loras}")
    return loras

@router.get("/system/models/embeddings", response_model=List[str], tags=["System Info"])
async def get_embeddings():
    """Lists Textual Inversion embeddings found in the configured directory."""
    embeddings = scan_model_directory("embeddings")
    logger.info(f"Found Embeddings: {embeddings}")
    return embeddings

@router.get("/system/schedulers", response_model=List[str], tags=["System Info"])
async def get_schedulers():
    """Lists available schedulers discoverable within the diffusers library."""
    schedulers = ["Default"]
    try:
        schedulers_module = importlib.import_module("diffusers.schedulers")
        for attr_name in dir(schedulers_module):
            if attr_name.endswith("Scheduler"):
                try:
                    attr = getattr(schedulers_module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, SchedulerMixin) and attr is not SchedulerMixin:
                        schedulers.append(attr_name)
                except Exception:
                    continue
        schedulers = sorted(list(set(schedulers)))
        logger.info(f"Discovered schedulers: {schedulers}")
    except Exception as e:
        logger.error(f"Could not dynamically discover schedulers: {e}", exc_info=True)
        schedulers = [
            "Default", "DDIMScheduler", "DDPMScheduler", "DEISMultistepScheduler",
            "DPMSolverMultistepScheduler", "DPMSolverSinglestepScheduler",
            "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
            "HeunDiscreteScheduler", "IPNDMScheduler", "KDPM2AncestralDiscreteScheduler",
            "KDPM2DiscreteScheduler", "LCMScheduler", "LMSDiscreteScheduler",
            "PNDMScheduler", "UniPCMultistepScheduler"
        ]
    return schedulers