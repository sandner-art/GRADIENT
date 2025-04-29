# src/gradient/ai/diffusion/loading.py

import logging
import torch
from torch import dtype
from diffusers import DiffusionPipeline, AutoencoderKL, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from gradient.core.resource_manager import resource_manager
from gradient.core.config import get_setting, PROJECT_ROOT

# --- NEW: Import helpers from new modules --- vvv
from .loading_loras import apply_loras_to_pipeline, unload_lora_weights
from .loading_embeddings import load_textual_inversions
# --- NEW: Import helpers from new modules --- ^^^


logger = logging.getLogger(__name__)

# Cache (unchanged)
loaded_pipelines: Dict[str, Any] = {}
loaded_components: Dict[str, Any] = {}

# --- Path Helper Functions (Keep find_model_path accessible here or move to core.paths) --- vvv
# (Keep get_model_search_dirs and find_model_path here for now)
def get_model_search_dirs() -> List[Path]:
    # ... (implementation unchanged) ...
    search_dirs_config = get_setting("paths.models.search_dirs", ["./data/models"])
    resolved_dirs = []
    for dir_str in search_dirs_config:
        path = Path(dir_str)
        if not path.is_absolute(): path = (PROJECT_ROOT / path).resolve()
        resolved_dirs.append(path)
    return resolved_dirs

def find_model_path(model_id_or_partial_path: str, model_type: str) -> Optional[Path]:
    # ... (implementation unchanged) ...
    input_path = Path(model_id_or_partial_path)
    if input_path.is_absolute(): return input_path.resolve() if input_path.exists() else None
    search_dirs = get_model_search_dirs()
    subdir_name = get_setting(f"paths.models.{model_type}", model_type)
    for base_dir in search_dirs:
        potential_path = (base_dir / subdir_name / model_id_or_partial_path).resolve()
        if potential_path.exists(): return potential_path
    return None
# --- Path Helper Functions --- ^^^

# --- Main Loading Function ---
async def load_diffusion_pipeline(
    model_path_or_id: str,
    pipeline_type: str = "StableDiffusionPipeline",
    vae_id: Optional[str] = None,
    # --- Keep LoRA/Embedding Params in signature --- vvv
    loras: Optional[List[Dict[str, Any]]] = None,
    embeddings: Optional[List[str]] = None,
    # --- Keep LoRA/Embedding Params in signature --- ^^^
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors: bool = True,
    variant: Optional[str] = "fp16" if torch.cuda.is_available() and dtype == torch.float16 else None,
    force_reload: bool = False
) -> Optional[DiffusionPipeline]:
    """
    Loads a diffusion pipeline, optionally applying LoRAs and Textual Inversions
    by calling dedicated helper functions.
    """
    # Key Generation (based on base model + VAE) - unchanged
    vae_key_part = f":vae={vae_id}" if vae_id else ""
    try: dtype_str = str(dtype).split('.')[-1]
    except Exception: dtype_str = 'unknown_dtype'
    pipeline_key = f"{pipeline_type}:{model_path_or_id}:{variant}:{dtype_str}:{device}{vae_key_part}"
    pipeline_cache_hit = False

    # Cache Check
    if not force_reload and pipeline_key in loaded_pipelines:
        logger.info(f"Returning cached pipeline: {pipeline_key}. Applying LoRAs/Embeddings if specified.")
        resource_manager.update_usage(pipeline_key)
        pipeline = loaded_pipelines[pipeline_key]
        pipeline_cache_hit = True
        # --- Unload existing LoRAs from cached object using helper ---
        unload_lora_weights(pipeline) # Call helper
    else:
        # --- Load Base Pipeline (If not in cache) ---
        # ... (Path resolution logic using find_model_path - unchanged) ...
        model_load_path_str = model_path_or_id # Placeholder
        load_from_single_file = False
        resolved_diffusers_path = find_model_path(model_path_or_id, "diffusers")
        resolved_ckpt_path = find_model_path(model_path_or_id, "checkpoints")
        if resolved_diffusers_path and resolved_diffusers_path.is_dir(): model_load_path_str = str(resolved_diffusers_path)
        elif resolved_ckpt_path and resolved_ckpt_path.is_file(): model_load_path_str = str(resolved_ckpt_path); load_from_single_file = True
        else: logger.info(f"Assuming Hub ID or incorrect local path: {model_load_path_str}")

        # VAE Loading - unchanged
        vae = None
        if vae_id:
            # ... (logic using find_model_path as before) ...
            resolved_vae_path = find_model_path(vae_id, "vae")
            vae_load_path_str = str(resolved_vae_path) if resolved_vae_path else vae_id
            try:
                vae = AutoencoderKL.from_pretrained(vae_load_path_str, torch_dtype=dtype).to(device)
            except Exception as e:
                logger.error(f"Failed to load specified VAE '{vae_id}': {e}", exc_info=True); return None


        # VRAM Allocation - unchanged
        estimated_vram_mb = 4 * 1024; # TODO: Improve estimate
        component_type = f"pipeline_{pipeline_type}"
        try: await resource_manager.allocate(pipeline_key, estimated_vram_mb, component_type)
        except MemoryError as e: return None

        # Load Main Pipeline Logic - unchanged
        pipeline = None
        try:
            load_args = {"torch_dtype": dtype, **({"vae": vae} if vae else {})}
            if variant: load_args["variant"] = variant
            diffusers_module = __import__('diffusers', fromlist=[pipeline_type])
            pipeline_class = getattr(diffusers_module, pipeline_type)
            if load_from_single_file and hasattr(pipeline_class, 'from_single_file'):
                 pipeline = pipeline_class.from_single_file(model_load_path_str, **load_args)
            else:
                 load_args["use_safetensors"] = use_safetensors
                 pipeline = pipeline_class.from_pretrained(model_load_path_str, **load_args)
            pipeline.to(device)
            if hasattr(pipeline, "safety_checker"): pipeline.safety_checker = None
        except Exception as e:
            logger.error(f"Error loading base pipeline {pipeline_key}: {e}", exc_info=True)
            await resource_manager.release(pipeline_key); return None

        logger.info(f"Successfully loaded base pipeline: {pipeline_key}")
        loaded_pipelines[pipeline_key] = pipeline # Add base pipeline to cache

    # --- Apply LoRAs using helper --- vvv
    if pipeline and loras:
        await apply_loras_to_pipeline(pipeline, loras, device, dtype)
    # --- Apply LoRAs using helper --- ^^^

    # --- Load Embeddings using helper --- vvv
    if pipeline and embeddings:
        await load_textual_inversions(pipeline, embeddings)
    # --- Load Embeddings using helper --- ^^^

    # --- Final Check & Return ---
    if pipeline:
        if pipeline_cache_hit: # Update usage timestamp if from cache
            resource_manager.update_usage(pipeline_key)
        return pipeline
    else:
        return None # Should have returned earlier if loading failed

# --- Modified unload_pipeline ---
async def unload_pipeline(pipeline_key: str):
    """Unloads pipeline, ensures LoRAs are detached using helper."""
    if pipeline_key in loaded_pipelines:
        logger.info(f"Unloading pipeline: {pipeline_key}")
        await resource_manager.release(pipeline_key)
        pipeline = loaded_pipelines.pop(pipeline_key)
        # --- Unload LoRAs via helper ---
        unload_lora_weights(pipeline)
        # --- Delete components ---
        # ... (del pipeline.vae, pipeline.unet, etc.) ...
        if hasattr(pipeline, 'vae'): del pipeline.vae
        if hasattr(pipeline, 'unet'): del pipeline.unet
        if hasattr(pipeline, 'text_encoder'): del pipeline.text_encoder
        if hasattr(pipeline, 'text_encoder_2'): del pipeline.text_encoder_2
        del pipeline
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"Pipeline {pipeline_key} unloaded.")
    else:
        logger.warning(f"Attempted to unload non-cached pipeline: {pipeline_key}")