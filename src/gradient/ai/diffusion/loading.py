# src/gradient/ai/diffusion/loading.py

import logging
import torch
# Use torch_dtype alias for torch.dtype to avoid conflict
from torch import dtype as torch_dtype_alias
from diffusers import DiffusionPipeline, AutoencoderKL
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from gradient.core.resource_manager import resource_manager
from gradient.core.config import get_setting, PROJECT_ROOT
from .loading_loras import apply_loras_to_pipeline, unload_lora_weights
from .loading_embeddings import load_textual_inversions

logger = logging.getLogger(__name__)

# Try to import optional dependencies
_xformers_available = False
try:
    import xformers # Pylance will complain if not installed, but code handles it
    _xformers_available = True
    logger.info("xformers successfully imported.")
except ImportError:
    logger.info("xformers not installed or importable. xformers attention optimization will not be available.")

_bitsandbytes_available = False
try:
    import bitsandbytes
    _bitsandbytes_available = True
    logger.info("bitsandbytes successfully imported.")
except ImportError:
    logger.info("bitsandbytes not installed or importable. Quantization features (4-bit/8-bit) will not be available.")


# Cache
loaded_pipelines: Dict[str, Any] = {}
pipeline_original_devices: Dict[str, str] = {}


def get_model_search_dirs() -> List[Path]:
    search_dirs_config = get_setting("paths.models.search_dirs", ["./data/models"])
    resolved_dirs = []
    for dir_str in search_dirs_config:
        path = Path(dir_str)
        if not path.is_absolute(): path = (PROJECT_ROOT / path).resolve()
        resolved_dirs.append(path)
    return resolved_dirs

def find_model_path(model_id_or_partial_path: str, model_type: str) -> Optional[Path]:
    input_path = Path(model_id_or_partial_path)
    if input_path.is_absolute():
        return input_path.resolve() if input_path.exists() else None

    search_dirs = get_model_search_dirs()
    subdir_name = get_setting(f"paths.models.{model_type}", model_type)

    for base_dir in search_dirs:
        potential_path = (base_dir / subdir_name / model_id_or_partial_path).resolve()
        if potential_path.exists():
            return potential_path

        if model_type in ["checkpoints", "vae", "lora", "embeddings", "llm"]:
            common_suffixes = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']
            model_id_str = str(model_id_or_partial_path)
            if not any(model_id_str.lower().endswith(s) for s in common_suffixes):
                for suffix in common_suffixes:
                    potential_file_path = (base_dir / subdir_name / (model_id_str + suffix)).resolve()
                    if potential_file_path.is_file():
                        return potential_file_path
    return None


async def load_diffusion_pipeline(
    model_path_or_id: str,
    pipeline_type_str: str = "StableDiffusionPipeline",
    vae_id: Optional[str] = None,
    loras: Optional[List[Dict[str, Any]]] = None,
    embeddings: Optional[List[str]] = None,

    enable_cpu_offload: bool = False,
    attention_optimization: Optional[str] = "auto",
    vae_optimization: Optional[str] = "sliced",
    quantization_bits: Optional[int] = None,

    target_device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_torch_dtype: torch_dtype_alias = torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors: bool = True,
    # variant will be determined inside the function using model_torch_dtype
    variant_in: Optional[str] = None, # Pass as argument if needed, otherwise determine dynamically
    force_reload: bool = False
) -> Optional[DiffusionPipeline]:

    # Determine variant based on model_torch_dtype if not explicitly passed
    variant = variant_in
    if variant is None: # If no specific variant is passed in, determine it
        variant = "fp16" if torch.cuda.is_available() and model_torch_dtype == torch.float16 else None

    pipeline_key_parts = [
        pipeline_type_str, model_path_or_id,
        str(variant), # Now variant is defined before use
        str(model_torch_dtype),
        target_device_str,
        f"vae={vae_id}" if vae_id else "",
        f"cpu_offload={enable_cpu_offload}",
        f"attn_opt={attention_optimization}",
        f"vae_opt={vae_optimization}",
        f"quant={'{}b'.format(quantization_bits) if quantization_bits else 'full'}"
    ]
    pipeline_key = ":".join(filter(None, pipeline_key_parts))
    logger.debug(f"Generated pipeline key: {pipeline_key}")

    # Cache Check
    if not force_reload and pipeline_key in loaded_pipelines:
        logger.info(f"Returning cached and configured pipeline: {pipeline_key}.")
        pipeline = loaded_pipelines[pipeline_key]

        is_currently_offloaded = pipeline_original_devices.get(pipeline_key) == "cpu_offloaded"

        if enable_cpu_offload:
            if not is_currently_offloaded:
                logger.info(f"Cached pipeline {pipeline_key} was not offloaded but current request enables it. Enabling.")
                pipeline.enable_sequential_cpu_offload(gpu_id=0 if target_device_str == "cuda" else -1)
                pipeline_original_devices[pipeline_key] = "cpu_offloaded"
        else:
            if is_currently_offloaded:
                logger.info(f"Cached pipeline {pipeline_key} was offloaded but current request disables it. Disabling and moving to {target_device_str}.")
                if hasattr(pipeline, "disable_sequential_cpu_offload"):
                    pipeline.disable_sequential_cpu_offload()
                pipeline.to(target_device_str)
                pipeline_original_devices[pipeline_key] = target_device_str
            elif str(pipeline.device) != target_device_str:
                logger.info(f"Moving cached pipeline {pipeline_key} from {pipeline.device} to {target_device_str}.")
                pipeline.to(target_device_str)

        resource_manager.update_usage(pipeline_key)
        if hasattr(pipeline, '_loras_applied_this_request'):
            del pipeline._loras_applied_this_request
        unload_lora_weights(pipeline)

    else:
        logger.info(f"Loading pipeline: {pipeline_key}")

        model_load_path_str = model_path_or_id
        load_from_single_file = False
        resolved_diffusers_path = find_model_path(model_path_or_id, "diffusers")
        resolved_ckpt_path = find_model_path(model_path_or_id, "checkpoints")

        if resolved_diffusers_path and resolved_diffusers_path.is_dir():
            model_load_path_str = str(resolved_diffusers_path)
            logger.info(f"Using local diffusers model directory: {model_load_path_str}")
        elif resolved_ckpt_path and resolved_ckpt_path.is_file():
            model_load_path_str = str(resolved_ckpt_path)
            load_from_single_file = True
            logger.info(f"Using local single file model (checkpoint): {model_load_path_str}")
        else:
            logger.info(f"Model '{model_path_or_id}' not found locally. Assuming Hub ID or direct path.")

        vae_component = None
        if vae_id:
            resolved_vae_path_str = vae_id
            found_vae_path = find_model_path(vae_id, "vae")
            if found_vae_path: resolved_vae_path_str = str(found_vae_path)
            try:
                vae_dtype_load = torch.float32 if target_device_str == "cpu" else model_torch_dtype
                vae_component = AutoencoderKL.from_pretrained(resolved_vae_path_str, torch_dtype=vae_dtype_load)
                logger.info(f"Loaded separate VAE '{vae_id}' with dtype {vae_dtype_load}.")
            except Exception as e: logger.error(f"Failed to load VAE '{vae_id}': {e}.", exc_info=True)

        quant_args = {}
        if quantization_bits and _bitsandbytes_available:
            logger.info(f"Attempting to apply {quantization_bits}-bit quantization.")
            if quantization_bits == 8:
                quant_args["load_in_8bit"] = True
            elif quantization_bits == 4:
                quant_args["load_in_4bit"] = True
                quant_args["bnb_4bit_quant_type"] = "nf4"
                quant_args["bnb_4bit_compute_dtype"] = model_torch_dtype if model_torch_dtype in [torch.float16, torch.bfloat16] else torch.float16
                quant_args["bnb_4bit_use_double_quant"] = True
        elif quantization_bits and not _bitsandbytes_available:
            logger.warning(f"bitsandbytes not available. Cannot apply {quantization_bits}-bit q.")

        estimated_vram_mb = 4096
        if "xl" in model_path_or_id.lower(): estimated_vram_mb = 8192
        if enable_cpu_offload: estimated_vram_mb = max(512, estimated_vram_mb // 4)
        elif quantization_bits == 8: estimated_vram_mb = max(1024, int(estimated_vram_mb * 0.6))
        elif quantization_bits == 4: estimated_vram_mb = max(1024, int(estimated_vram_mb * 0.4))

        component_type = f"pipeline_{pipeline_type_str}"
        try:
            await resource_manager.allocate(pipeline_key, estimated_vram_mb, component_type)
        except MemoryError as e:
            logger.error(f"VRAM allocation failed for {pipeline_key}: {e}")
            return None

        pipeline: Optional[DiffusionPipeline] = None
        try:
            pipeline_class_module = __import__('diffusers', fromlist=[pipeline_type_str])
            ConcretePipelineClass = getattr(pipeline_class_module, pipeline_type_str)

            load_args = {"torch_dtype": model_torch_dtype, **quant_args}
            if vae_component: load_args["vae"] = vae_component

            if not load_from_single_file:
                if variant: load_args["variant"] = variant # variant is now defined
                if not use_safetensors: load_args["use_safetensors"] = False
            
            if not enable_cpu_offload and not quant_args.get("load_in_4bit") and not quant_args.get("load_in_8bit"):
                if not load_from_single_file:
                    load_args["low_cpu_mem_usage"] = get_setting("performance.low_cpu_mem_usage_on_load", True)
                    if load_args.get("low_cpu_mem_usage"):
                        logger.info("Using low_cpu_mem_usage=True for pipeline loading.")

            if load_from_single_file:
                pipeline = ConcretePipelineClass.from_single_file(model_load_path_str, **load_args)
            else:
                pipeline = ConcretePipelineClass.from_pretrained(model_load_path_str, **load_args)

            if hasattr(pipeline, "safety_checker"): pipeline.safety_checker = None

            if enable_cpu_offload:
                logger.info(f"Enabling sequential CPU offload for {pipeline_key}.")
                pipeline.enable_sequential_cpu_offload(gpu_id=0 if target_device_str == "cuda" else -1)
                pipeline_original_devices[pipeline_key] = "cpu_offloaded"
            else:
                logger.info(f"Moving pipeline {pipeline_key} to device: {target_device_str}")
                pipeline.to(target_device_str)
                pipeline_original_devices[pipeline_key] = target_device_str

            if vae_optimization == "sliced" and hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing(); logger.info("VAE slicing enabled.")
            elif vae_optimization == "tiled" and hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling(); logger.info("VAE tiling enabled.")

            if attention_optimization != "none":
                applied_attn_opt = False
                if attention_optimization == "auto" or attention_optimization == "xformers":
                    if _xformers_available and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                        try:
                            pipeline.enable_xformers_memory_efficient_attention()
                            logger.info("xformers memory efficient attention enabled.")
                            applied_attn_opt = True
                        except Exception as e_xformers:
                            logger.warning(f"Failed to enable xformers: {e_xformers}.")
                            if attention_optimization != "auto": applied_attn_opt = False
                    elif attention_optimization == "xformers":
                             logger.warning("xformers specifically requested but not available/functional.")

                if not applied_attn_opt and (attention_optimization == "auto" or attention_optimization == "sliced"):
                    if hasattr(pipeline, "enable_attention_slicing"):
                        pipeline.enable_attention_slicing(); logger.info("Attention slicing enabled.")
                    else: logger.warning("Requested attention slicing, but method not available.")
            
            logger.info(f"Successfully loaded and configured base pipeline: {pipeline_key} (Device: {pipeline.device})")
            loaded_pipelines[pipeline_key] = pipeline

        except Exception as e:
            logger.error(f"Error loading new pipeline {pipeline_key}: {e}", exc_info=True)
            await resource_manager.release(pipeline_key)
            return None

    if pipeline:
        if not getattr(pipeline, '_loras_applied_this_request', False):
            if loras:
                await apply_loras_to_pipeline(pipeline, loras, target_device_str, model_torch_dtype)
            if embeddings:
                await load_textual_inversions(pipeline, embeddings)
            pipeline._loras_applied_this_request = True
        
        resource_manager.update_usage(pipeline_key)
        return pipeline
    
    return None

async def unload_pipeline(pipeline_key: str):
    if pipeline_key in loaded_pipelines:
        logger.info(f"Unloading pipeline: {pipeline_key}")
        pipeline = loaded_pipelines.pop(pipeline_key)
        pipeline_original_devices.pop(pipeline_key, None)

        unload_lora_weights(pipeline)

        if hasattr(pipeline, "_hf_hook") and pipeline._hf_hook is not None : # Check if offload was active
            if hasattr(pipeline, "disable_sequential_cpu_offload"):
                logger.debug("Disabling sequential CPU offload before full unload.")
                try: pipeline.disable_sequential_cpu_offload()
                except Exception as e: logger.warning(f"Error disabling sequential CPU offload: {e}")
        
        if hasattr(pipeline, "disable_vae_slicing"): pipeline.disable_vae_slicing()
        if hasattr(pipeline, "disable_vae_tiling"): pipeline.disable_vae_tiling()
        if hasattr(pipeline, "disable_xformers_memory_efficient_attention"): pipeline.disable_xformers_memory_efficient_attention()
        if hasattr(pipeline, "disable_attention_slicing"): pipeline.disable_attention_slicing()
        
        logger.debug(f"Attempting to move pipeline components of {pipeline_key} to CPU before deletion.")
        if hasattr(pipeline, "components") and isinstance(pipeline.components, dict):
            for name, component in pipeline.components.items():
                if component is not None and hasattr(component, "to") and hasattr(component, "device") and component.device.type != 'cpu':
                    try:
                        logger.debug(f"Moving component '{name}' from {component.device} to CPU.")
                        component.to("cpu")
                    except Exception as e: logger.warning(f"Could not move component '{name}' to CPU: {e}")
        elif hasattr(pipeline, "to") and hasattr(pipeline, "device") and pipeline.device.type != 'cpu':
             try:
                 logger.debug(f"Moving main pipeline object from {pipeline.device} to CPU.")
                 pipeline.to("cpu")
             except Exception as e: logger.warning(f"Could not move main pipeline object to CPU: {e}")

        del pipeline
        if torch.cuda.is_available():
            logger.debug("Calling torch.cuda.empty_cache().")
            torch.cuda.empty_cache()
        
        await resource_manager.release(pipeline_key)
        logger.info(f"Pipeline {pipeline_key} unloaded and VRAM release signaled.")
    else:
        logger.warning(f"Attempted to unload non-cached pipeline: {pipeline_key}")