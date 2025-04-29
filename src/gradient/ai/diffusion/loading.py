# gradient-project-root/src/gradient/ai/diffusion/loading.py

import logging
import torch
from torch import dtype
from diffusers import DiffusionPipeline, AutoencoderKL, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pathlib import Path
from typing import Optional, Dict, Any, List

from gradient.core.resource_manager import resource_manager
from gradient.core.config import get_setting, PROJECT_ROOT # Import config helpers

logger = logging.getLogger(__name__)

# Cache
loaded_pipelines: Dict[str, Any] = {}
loaded_components: Dict[str, Any] = {}

# --- Path Helper Functions (SHOULD ONLY BE THESE NOW) ---

def get_model_search_dirs() -> List[Path]:
    """Gets the configured list of base directories to search for models."""
    search_dirs_config = get_setting("paths.models.search_dirs", ["./data/models"])
    resolved_dirs = []
    for dir_str in search_dirs_config:
        path = Path(dir_str)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        resolved_dirs.append(path)
    # logger.debug(f"Resolved model search directories: {resolved_dirs}") # Keep debug if useful
    return resolved_dirs

def find_model_path(model_id_or_partial_path: str, model_type: str) -> Optional[Path]:
    """
    Searches for a model/checkpoint/VAE file or directory within configured search paths.
    Handles relative paths within subdirs (diffusers, checkpoints, vae) and absolute paths.
    """
    input_path = Path(model_id_or_partial_path)

    if input_path.is_absolute():
        if input_path.exists():
            # logger.debug(f"Found existing absolute path for '{model_id_or_partial_path}'.")
            return input_path.resolve()
        else:
            logger.warning(f"Provided absolute path '{model_id_or_partial_path}' does not exist.")
            return None

    search_dirs = get_model_search_dirs()
    subdir_name = get_setting(f"paths.models.{model_type}", model_type)

    for base_dir in search_dirs:
        potential_path = (base_dir / subdir_name / model_id_or_partial_path).resolve()
        if potential_path.exists():
            logger.debug(f"Found '{model_id_or_partial_path}' of type '{model_type}' at '{potential_path}'")
            return potential_path

    # logger.debug(f"Model '{model_id_or_partial_path}' of type '{model_type}' not found locally.") # Keep debug if useful
    return None

# --- Model Loading Function ---

async def load_diffusion_pipeline(
    model_path_or_id: str,
    pipeline_type: str = "StableDiffusionPipeline",
    vae_id: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors: bool = True,
    variant: Optional[str] = "fp16" if torch.cuda.is_available() and dtype == torch.float16 else None,
    force_reload: bool = False
) -> Optional[DiffusionPipeline]:
    """
    Loads a diffusion pipeline. Handles local paths (searching configured dirs),
    absolute paths, HF Hub IDs, VAE loading, caching, and basic resource management.
    """
    # --- Key Generation (using dtype name) ---
    vae_key_part = f":vae={vae_id}" if vae_id else ""
    try:
        # Attempt to get a clean string representation like 'float16' or 'bfloat16'
        dtype_str = str(dtype).split('.')[-1]
    except Exception:
        dtype_str = 'unknown_dtype' # Fallback
    pipeline_key = f"{pipeline_type}:{model_path_or_id}:{variant}:{dtype_str}:{device}{vae_key_part}"


    # --- Cache Check ---
    if not force_reload and pipeline_key in loaded_pipelines:
        logger.info(f"Returning cached pipeline: {pipeline_key}")
        resource_manager.update_usage(pipeline_key)
        return loaded_pipelines[pipeline_key]

    # --- Resolve Paths using find_model_path ---
    model_load_path_str: str = ""
    load_from_single_file = False

    resolved_diffusers_path = find_model_path(model_path_or_id, "diffusers")
    resolved_ckpt_path = find_model_path(model_path_or_id, "checkpoints")

    if resolved_diffusers_path and resolved_diffusers_path.is_dir():
        model_load_path_str = str(resolved_diffusers_path)
        logger.info(f"Loading local diffusers model from: {model_load_path_str}")
    elif resolved_ckpt_path and resolved_ckpt_path.is_file():
        model_load_path_str = str(resolved_ckpt_path)
        logger.info(f"Identified single checkpoint file: {model_load_path_str}")
        load_from_single_file = True
    else:
        model_load_path_str = model_path_or_id
        if Path(model_path_or_id).is_absolute():
             logger.warning(f"Absolute path specified but not found: '{model_path_or_id}'. Passing to diffusers.")
        else:
             logger.info(f"Assuming Hugging Face Hub ID or incorrect local path: {model_load_path_str}")

    # --- Load VAE using find_model_path ---
    vae = None
    if vae_id:
        resolved_vae_path = find_model_path(vae_id, "vae")
        if resolved_vae_path and (resolved_vae_path.is_file() or resolved_vae_path.is_dir()):
            vae_load_path_str = str(resolved_vae_path)
            logger.info(f"Attempting to load specified VAE from local path: {vae_load_path_str}")
        else:
             vae_load_path_str = vae_id
             logger.info(f"Assuming Hub ID or incorrect local path for VAE: {vae_load_path_str}")
        try:
            vae = AutoencoderKL.from_pretrained(
                vae_load_path_str, torch_dtype=dtype
            ).to(device)
            logger.info(f"Successfully loaded VAE: {vae_id}")
        except Exception as e:
            logger.error(f"Failed to load specified VAE '{vae_id}': {e}", exc_info=True)
            return None

    # --- VRAM Allocation ---
    estimated_vram_mb = 4 * 1024 + (1 * 1024 if vae else 0) # Dummy
    component_type = f"pipeline_{pipeline_type}"
    try:
        await resource_manager.allocate(pipeline_key, estimated_vram_mb, component_type)
    except MemoryError as e:
        logger.error(f"Failed to allocate VRAM for {pipeline_key}: {e}")
        del vae
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    # --- Load Main Pipeline ---
    pipeline = None
    try:
        load_args = {"torch_dtype": dtype, **({"vae": vae} if vae else {})}
        if variant: load_args["variant"] = variant

        try:
            diffusers_module = __import__('diffusers', fromlist=[pipeline_type])
            pipeline_class = getattr(diffusers_module, pipeline_type)
        except (ImportError, AttributeError) as import_err:
             logger.error(f"Could not find pipeline class '{pipeline_type}' in diffusers: {import_err}")
             await resource_manager.release(pipeline_key); del vae; return None # Added cleanup

        logger.info(f"Loading {pipeline_type} '{model_load_path_str}' with args: {load_args}")

        if load_from_single_file and hasattr(pipeline_class, 'from_single_file'):
             logger.info(f"Using {pipeline_type}.from_single_file for checkpoint.")
             pipeline = pipeline_class.from_single_file(model_load_path_str, **load_args)
        else:
             # Add use_safetensors back for from_pretrained if desired, though often default True
             load_args["use_safetensors"] = use_safetensors
             pipeline = pipeline_class.from_pretrained(model_load_path_str, **load_args)

        pipeline.to(device)

        if hasattr(pipeline, "safety_checker") and pipeline.safety_checker is not None:
            logger.warning("Disabling pipeline safety checker.")
            pipeline.safety_checker = None # Standard way

        logger.info(f"Successfully loaded pipeline: {pipeline_key}")
        loaded_pipelines[pipeline_key] = pipeline
        resource_manager.update_usage(pipeline_key)
        return pipeline

    except Exception as e:
        logger.error(f"Error loading diffusion pipeline {pipeline_key} from '{model_load_path_str}': {e}", exc_info=True)
        await resource_manager.release(pipeline_key)
        del vae
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

# --- Unload Function (Ensure it matches previous version) ---
async def unload_pipeline(pipeline_key: str):
    # ... (Keep the implementation from before) ...
    if pipeline_key in loaded_pipelines:
        logger.info(f"Unloading pipeline: {pipeline_key}")
        await resource_manager.release(pipeline_key) # Release VRAM tracking first
        pipeline = loaded_pipelines.pop(pipeline_key)
        # Attempt to delete known large components
        if hasattr(pipeline, 'vae') and pipeline.vae is not None: del pipeline.vae
        if hasattr(pipeline, 'unet') and pipeline.unet is not None: del pipeline.unet
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None: del pipeline.text_encoder
        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None: del pipeline.text_encoder_2
        del pipeline # Delete the main object
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Attempt to clear GPU cache
        logger.info(f"Pipeline {pipeline_key} unloaded.")
    else:
        logger.warning(f"Attempted to unload non-cached pipeline: {pipeline_key}")