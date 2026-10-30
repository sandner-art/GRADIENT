# gradient-project-root/src/gradient/ai/llm/loading.py

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

# Import llama_cpp only if available, handle potential import error gracefully?
# Using TYPE_CHECKING to satisfy linters/type checkers about the Llama type
# while still allowing the code to run if llama_cpp is not installed.
if TYPE_CHECKING:
    from llama_cpp import Llama
else:
    try:
        from llama_cpp import Llama
    except ImportError:
        Llama = None  # Allows basic code structure even if install failed
        logging.getLogger(__name__).info(
            "llama-cpp-python is not installed; LLM chat is disabled. Install with: pip install 'gradient-workbench[llm]'"
        )


# Assuming resource_manager is accessible
from gradient.core.config import get_primary_model_dir
from gradient.core.resource_manager import resource_manager
from gradient.model_manager.inventory import find_model_path

from .models import LLMConfig

logger = logging.getLogger(__name__)

# Cache for loaded LLMs
# key = unique config identifier, value = Llama object
# Use forward reference string for Dict value type hint
loaded_llms: dict[str, "Llama"] = {}


# --- Model Path ---
def get_llm_base_path() -> Path:
    """Primary folder for GGUF models (paths.models.llm in the first search dir)."""
    return get_primary_model_dir("llm")


def resolve_llm_path(model_id: str) -> Path | None:
    path = find_model_path(model_id, "llm")
    return path if path and path.is_file() else None


# --- Model Loading ---


async def load_llm(config: LLMConfig, force_reload: bool = False) -> Optional["Llama"]:
    """
    Loads a GGUF model using llama-cpp-python.
    Handles caching and basic VRAM allocation.
    Returns the Llama instance or None if loading fails.
    """
    # Check if Llama class was successfully imported at the module level
    if Llama is None:
        # Error was already logged when module was imported
        logger.error("Cannot load LLM: llama-cpp-python is not available.")
        return None

    # Create a unique key based on relevant config parameters
    llm_key = f"{config.model_id}:{config.n_ctx}:{config.n_gpu_layers}"

    # Check cache first
    if not force_reload and llm_key in loaded_llms:
        cached_llm = loaded_llms[llm_key]
        # Runtime check to ensure cached item is actually a Llama instance
        # (Handles edge cases where Llama might change or cache gets corrupted)
        if isinstance(cached_llm, Llama):
            logger.info(f"Returning cached LLM: {llm_key}")
            resource_manager.update_usage(llm_key)
            return cached_llm
        else:
            logger.warning(f"Cached object for {llm_key} is not a valid Llama instance. Reloading.")
            loaded_llms.pop(llm_key, None)  # Remove invalid entry

    # --- Determine Model Path ---
    model_load_path = resolve_llm_path(config.model_id)
    if model_load_path is None:
        logger.error(f"GGUF model not found: {config.model_id!r} (looked in the llm model folders)")
        return None

    # --- VRAM Estimation & Allocation ---
    estimated_vram_mb = 0
    if config.n_gpu_layers != 0:
        try:
            # Ensure model_load_path is not None before accessing stat()
            if model_load_path:
                model_size_bytes = model_load_path.stat().st_size
                model_size_gb = model_size_bytes / (1024**3)
                # Simple estimation: fraction of layers * size * efficiency factor
                # TODO: Refine this based on quantization type and actual layer sizes if possible
                offload_fraction = (
                    1.0 if config.n_gpu_layers == -1 else min(1.0, abs(config.n_gpu_layers) / 50.0)
                )  # Guess: 50 layers total
                estimated_vram_mb = int(
                    model_size_gb * offload_fraction * 0.9 * 1024
                )  # Assume 90% efficiency?
                logger.info(
                    f"Estimated VRAM for GGUF layers: {estimated_vram_mb} MB (Size: {model_size_gb:.2f} GB, n_gpu_layers: {config.n_gpu_layers})"
                )
            else:
                # Should not happen if path checks above work, but defensively handle
                raise ValueError("Model load path is None, cannot estimate size.")
        except Exception as e:
            logger.error(
                f"Failed to estimate VRAM based on file size for {model_load_path}: {e}",
                exc_info=True,
            )
            estimated_vram_mb = 4096  # Fallback VRAM estimate

    component_type = "llm"

    # Keep at most one LLM resident; evict older ones first.
    for other in [k for k in loaded_llms if k != llm_key]:
        await unload_llm(other)

    # --- Load the LLM ---
    llm_instance: Llama | None = None  # Use string ref for local type hint
    load_start_time = time.time()
    try:
        logger.info(f"Initializing llama_cpp.Llama for model: {model_load_path.name}")
        # Convert model_load_path to string for llama_cpp
        model_path_str = str(model_load_path)

        llm_instance = await asyncio.to_thread(
            Llama,
            model_path=model_path_str,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            # Add other relevant parameters from LLMConfig if needed
            # seed=config.seed if config.seed is not None else -1, # Example
            verbose=False,
        )
        load_end_time = time.time()
        logger.info(f"Successfully loaded LLM: {llm_key} in {load_end_time - load_start_time:.2f}s")

        # Add the successfully loaded instance to the cache
        loaded_llms[llm_key] = llm_instance

        resource_manager.register(
            llm_key, component_type, estimated_vram_mb, unload=lambda: unload_llm(llm_key)
        )

        return llm_instance  # Return the loaded Llama object

    except Exception as e:
        logger.error(f"Error during Llama initialization for {llm_key}: {e}", exc_info=True)
        return None  # Return None on failure


async def unload_llm(llm_key: str):
    """Unloads a specific LLM from memory and cache."""
    if llm_key in loaded_llms:
        logger.info(f"Unloading LLM: {llm_key}")
        resource_manager.release(llm_key)

        # Remove from cache and delete object
        llm_instance = loaded_llms.pop(llm_key)
        del llm_instance  # Helps Python's garbage collector

        logger.info(f"LLM {llm_key} unloaded.")
    else:
        logger.warning(f"Attempted to unload non-cached LLM: {llm_key}")


# --- Coordination with ResourceManager ---
# TODO: Implement proper unload callback registration with ResourceManager for LLMs
# Example: resource_manager.register_unload_handler("llm", unload_llm)
