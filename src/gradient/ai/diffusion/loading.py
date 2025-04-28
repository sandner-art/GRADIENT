# gradient-project-root/src/gradient/ai/diffusion/loading.py

import logging
import torch
from torch import dtype # Ensure dtype is imported
from diffusers import DiffusionPipeline, AutoencoderKL, StableDiffusionPipeline # Add more pipeline types later as needed
from pathlib import Path
from typing import Optional, Dict, Any # Ensure Optional, Dict, Any are imported

# Assuming resource_manager is accessible, e.g., imported from core
# Adjust import path as necessary based on your structure if core isn't directly importable
# This might involve setting up __init__.py properly or using relative imports if kept within ai module.
# For now, let's assume direct import works via PYTHONPATH setup in pyproject.toml
from gradient.core.resource_manager import resource_manager

logger = logging.getLogger(__name__)

# Cache for loaded models/pipelines to avoid redundant loading
# key = unique identifier (e.g., model_path_or_id:variant:dtype:device), value = loaded pipeline/component object
loaded_pipelines: Dict[str, Any] = {}
loaded_components: Dict[str, Any] = {} # For VAEs, etc.

# --- Model Path Functions (Placeholder) ---
# TODO: Replace with actual config access later
def get_model_base_path() -> Path:
    """Gets the base path for diffusers models."""
    # from gradient.core.config import get_setting
    # return Path(get_setting("paths.models.diffusers", "./data/models/diffusers"))
    return Path("./data/models/diffusers") # Default for now

def get_checkpoint_base_path() -> Path:
    """Gets the base path for checkpoint models."""
    # from gradient.core.config import get_setting
    # return Path(get_setting("paths.models.checkpoints", "./data/models/checkpoints"))
    return Path("./data/models/checkpoints") # Default for now

def get_vae_base_path() -> Path:
    """Gets the base path for VAE models."""
    # from gradient.core.config import get_setting
    # return Path(get_setting("paths.models.vae", "./data/models/vae"))
    return Path("./data/models/vae") # Default for now

# --- Model Loading Functions ---

async def load_diffusion_pipeline(
    model_path_or_id: str,
    pipeline_type: str = "StableDiffusionPipeline", # e.g., StableDiffusionPipeline, StableDiffusionXLPipeline
    vae_id: Optional[str] = None, # Allow specifying a VAE
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: dtype = torch.float16 if torch.cuda.is_available() else torch.float32, # Use float16 on GPU default
    use_safetensors: bool = True,
    variant: Optional[str] = "fp16" if torch.float16 == dtype else None,
    force_reload: bool = False
) -> Optional[DiffusionPipeline]:
    """
    Loads a diffusion pipeline (e.g., Stable Diffusion) using diffusers.
    Handles diffusers format, single checkpoints, VAE loading, caching, and VRAM allocation.
    """
    # Include VAE in the key if specified
    vae_key_part = f":vae={vae_id}" if vae_id else ""
    pipeline_key = f"{pipeline_type}:{model_path_or_id}:{variant}:{dtype}:{device}{vae_key_part}"

    if not force_reload and pipeline_key in loaded_pipelines:
        logger.info(f"Returning cached pipeline: {pipeline_key}")
        resource_manager.update_usage(pipeline_key)
        return loaded_pipelines[pipeline_key]

    model_load_path: Path | str
    load_from_single_file = False # Flag to know if we are loading a checkpoint

    # --- Determine actual path for main model/pipeline ---
    potential_path_diffusers = get_model_base_path() / model_path_or_id
    potential_path_ckpt = get_checkpoint_base_path() / model_path_or_id

    if potential_path_diffusers.is_dir():
        model_load_path = potential_path_diffusers
        logger.info(f"Loading local diffusers model from: {model_load_path}")
    elif potential_path_ckpt.is_file() or Path(model_path_or_id).is_file():
        model_load_path = potential_path_ckpt if potential_path_ckpt.is_file() else Path(model_path_or_id)
        logger.info(f"Identified single checkpoint file: {model_load_path}")
        load_from_single_file = True # Set the flag
    else:
        # Assume it's a Hugging Face Hub model ID
        model_load_path = model_path_or_id
        logger.info(f"Assuming Hugging Face Hub ID: {model_load_path}")

    # Convert Path object to string for diffusers functions
    model_load_path_str = str(model_load_path)

    # --- Load Optional VAE ---
    vae = None
    vae_load_path_str = None
    if vae_id:
        potential_vae_path_local = get_vae_base_path() / vae_id
        if potential_vae_path_local.is_file() or Path(vae_id).is_file():
            vae_load_path = potential_vae_path_local if potential_vae_path_local.is_file() else Path(vae_id)
            vae_load_path_str = str(vae_load_path)
            logger.info(f"Attempting to load specified VAE from local path: {vae_load_path_str}")
        else:
             vae_load_path_str = vae_id # Assume HF Hub ID for VAE
             logger.info(f"Attempting to load specified VAE from Hub ID: {vae_load_path_str}")

        try:
            # TODO: Add VAE caching and resource management if needed separately
            vae = AutoencoderKL.from_pretrained(
                vae_load_path_str,
                torch_dtype=dtype, # Load VAE in same dtype? Or specific (e.g., float32)? Often float16 is fine.
                # variant=variant # Does VAE have fp16 variant? Check docs if needed
            )
            vae.to(device) # Move VAE to target device
            logger.info(f"Successfully loaded VAE: {vae_id}")
        except Exception as e:
            logger.error(f"Failed to load specified VAE '{vae_id}': {e}", exc_info=True)
            # Decide whether to proceed without VAE or fail entirely
            # For now, let's fail if specific VAE load fails
            return None


    # --- VRAM Estimation & Allocation (Placeholder) ---
    # TODO: Implement more accurate VRAM estimation based on model config/size/VAE
    estimated_vram_mb = 4 * 1024 + (1 * 1024 if vae else 0) # Dummy: 4GB base + 1GB for VAE
    component_type = f"pipeline_{pipeline_type}"

    try:
        await resource_manager.allocate(pipeline_key, estimated_vram_mb, component_type)
    except MemoryError as e:
        logger.error(f"Failed to allocate VRAM for {pipeline_key}: {e}")
        # If VAE was loaded, unload it? Need proper resource tracking for components too.
        del vae # Basic cleanup
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    # --- Load the Main Pipeline ---
    pipeline = None
    try:
        load_args = {
            "torch_dtype": dtype,
            # Pass loaded VAE if available
            **({"vae": vae} if vae else {}),
        }
        if variant:
            load_args["variant"] = variant

        # Dynamically get the pipeline class
        try:
            diffusers_module = __import__('diffusers', fromlist=[pipeline_type])
            pipeline_class = getattr(diffusers_module, pipeline_type)
        except (ImportError, AttributeError) as import_err:
             logger.error(f"Could not find pipeline class '{pipeline_type}' in diffusers: {import_err}")
             await resource_manager.release(pipeline_key)
             return None

        logger.info(f"Loading {pipeline_type} '{model_load_path_str}' with args: {load_args}")

        # Use from_single_file for checkpoints if available and appropriate
        if load_from_single_file and hasattr(pipeline_class, 'from_single_file'):
             logger.info(f"Using {pipeline_type}.from_single_file for checkpoint.")
             # from_single_file often takes similar args, but check specific pipeline docs
             # It usually infers safetensors based on extension
             pipeline = pipeline_class.from_single_file(model_load_path_str, **load_args)
        else:
             # Use standard from_pretrained for directories or Hub IDs
             load_args["use_safetensors"] = use_safetensors # Add back for from_pretrained
             pipeline = pipeline_class.from_pretrained(model_load_path_str, **load_args)

        # Move final pipeline to device (VAE should already be there if loaded)
        pipeline.to(device)

        # Optional: Disable safety checker if desired
        if hasattr(pipeline, "safety_checker") and pipeline.safety_checker is not None:
            logger.warning("Disabling pipeline safety checker.")
            pipeline.safety_checker = None
            # Also potentially remove requires_safety_checker attribute if needed by some pipelines
            # pipeline.requires_safety_checker = False


        logger.info(f"Successfully loaded pipeline: {pipeline_key}")
        loaded_pipelines[pipeline_key] = pipeline
        # Update resource manager timestamp after successful load
        resource_manager.update_usage(pipeline_key)
        return pipeline

    except Exception as e:
        logger.error(f"Error loading diffusion pipeline {pipeline_key}: {e}", exc_info=True)
        # Release allocated VRAM if loading failed
        await resource_manager.release(pipeline_key)
        # Cleanup loaded VAE if main pipeline load failed
        del vae
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

async def unload_pipeline(pipeline_key: str):
    """Unloads a specific pipeline from memory and cache."""
    if pipeline_key in loaded_pipelines:
        logger.info(f"Unloading pipeline: {pipeline_key}")
        # Release VRAM first (manager keeps track of amount)
        await resource_manager.release(pipeline_key)
        pipeline = loaded_pipelines.pop(pipeline_key)
        # Explicitly delete components if possible, especially the VAE if tracked separately
        if hasattr(pipeline, 'vae'): del pipeline.vae
        if hasattr(pipeline, 'unet'): del pipeline.unet
        # ... etc for other large components
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear GPU cache
        logger.info(f"Pipeline {pipeline_key} unloaded.")
    else:
        logger.warning(f"Attempted to unload non-cached pipeline: {pipeline_key}")

# --- TODO: Add functions for loading specific components like ControlNets, LoRAs ---
# async def load_lora(...)

# --- Coordination with ResourceManager ---
# TODO: Implement proper unload callback registration with ResourceManager
# Example: resource_manager.register_unload_handler("pipeline_", unload_pipeline)