# src/gradient/ai/diffusion/loading_loras.py (NEW FILE)

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from diffusers import DiffusionPipeline
# Import PEFT if using it for more advanced LoRA handling
# from peft import PeftModel
from huggingface_hub import hf_hub_download

from gradient.core.config import get_setting # For finding LoRA directory

logger = logging.getLogger(__name__)

# --- Path Helper (Copied or Imported from loading.py/config.py) ---
# It's often better to have a central path finding utility.
# For now, let's reuse find_model_path concept focusing on loras.
# Assuming find_model_path exists elsewhere (e.g., loading.py or a new paths.py)
# from .loading import find_model_path # Or from gradient.core.paths import find_model_path
# Placeholder implementation if not imported:
def find_lora_path(lora_name: str) -> Optional[Path]:
     # Simplified version: searches only the 'lora' subdir in configured paths
     from .loading import find_model_path # Temporary import if not refactored
     return find_model_path(lora_name, "lora")


# --- Core LoRA Loading Logic ---

async def apply_loras_to_pipeline(
    pipeline: DiffusionPipeline,
    loras: Optional[List[Dict[str, Any]]] = None,
    device: str = 'cpu', # Match pipeline device
    dtype: torch.dtype = torch.float32 # Match pipeline dtype
):
    """
    Applies a list of LoRAs with weights to a loaded DiffusionPipeline.
    Uses diffusers' built-in methods primarily.

    Args:
        pipeline: The loaded DiffusionPipeline object.
        loras: A list of dictionaries, e.g., [{'name': 'lora_name.safetensors', 'weight': 0.8}, ...]
        device: Target device for LoRA weights if loaded separately.
        dtype: Target dtype for LoRA weights if loaded separately.
    """
    if not loras:
        return # Nothing to apply

    if not hasattr(pipeline, 'load_lora_weights') or not callable(pipeline.load_lora_weights):
        logger.warning(f"Pipeline type {type(pipeline).__name__} does not support `load_lora_weights`. Skipping LoRA application.")
        return

    # --- 1. Unload any existing LoRAs first ---
    # This is crucial if the same base pipeline object is reused.
    unload_lora_weights(pipeline) # Use helper function below

    # --- 2. Load and Apply New LoRAs ---
    loaded_adapters = []
    adapter_weights = []

    logger.info(f"Applying {len(loras)} LoRA(s) to pipeline...")
    for i, lora_info in enumerate(loras):
        lora_name = lora_info.get('name')
        # Diffusers' load_lora_weights often handles scaling internally or via kwargs,
        # but PEFT integration uses adapter_weights. Let's store the requested weight.
        lora_weight = float(lora_info.get('weight', 1.0))
        adapter_name = f"lora_{i}" # Unique name for this instance

        if not lora_name:
            logger.warning(f"Skipping LoRA at index {i}: 'name' is missing.")
            continue

        logger.info(f"Preparing LoRA '{lora_name}' with target weight {lora_weight} (adapter: {adapter_name})")
        lora_path_or_id = lora_name
        resolved_lora_path = find_lora_path(lora_name) # Use helper

        if resolved_lora_path and resolved_lora_path.is_file():
            lora_path_or_id = str(resolved_lora_path)
            logger.debug(f"Found local LoRA path: {lora_path_or_id}")
            weight_name = resolved_lora_path.name # Pass filename for local files
        else:
            logger.warning(f"LoRA file '{lora_name}' not found locally. Assuming Hugging Face Hub ID.")
            # For Hub IDs, weight_name is usually None or specific standard names like 'pytorch_lora_weights.safetensors'
            weight_name = None # Let diffusers handle Hub ID resolution
            # Optional: Pre-download using hf_hub_download if specific file needed


        # Load weights directly into the pipeline object using unique adapter name
        try:
            # Note: The `weight` parameter within `load_lora_weights` itself is often DEPRECATED or limited.
            # The standard way involves PEFT or cross_attention_kwargs.
            pipeline.load_lora_weights(
                lora_path_or_id,
                weight_name=weight_name,
                adapter_name=adapter_name
                # Optional kwargs like device_map could go here
            )
            logger.info(f"Loaded LoRA weights for '{lora_name}' into adapter '{adapter_name}'.")
            loaded_adapters.append(adapter_name)
            adapter_weights.append(lora_weight) # Store weight for potential PEFT application

        except Exception as e:
            logger.error(f"Failed to load LoRA '{lora_name}' (Path/ID: '{lora_path_or_id}'): {e}", exc_info=True)
            # Optionally continue to try loading other LoRAs? Or fail hard? Let's continue.


    # --- 3. Activate and Set Weights (Requires PEFT integration usually) ---
    # This part depends heavily on how diffusers integrates with PEFT.
    # If the pipeline object becomes a PeftModel or similar after loading...
    if loaded_adapters:
        if hasattr(pipeline, "set_adapters") and callable(pipeline.set_adapters):
             try:
                 logger.info(f"Activating adapters: {loaded_adapters} with weights {adapter_weights}")
                 pipeline.set_adapters(loaded_adapters, adapter_weights=adapter_weights)
                 # If fuse/unfuse needed for efficiency (check docs/performance)
                 # if hasattr(pipeline, "fuse_lora"):
                 #    pipeline.fuse_lora(adapter_names=loaded_adapters) # Fuse requires specifying adapters
                 logger.info("Successfully set LoRA adapters and weights via set_adapters.")
             except Exception as e:
                 logger.error(f"Failed to set LoRA adapters/weights via PEFT: {e}. LoRAs might not be correctly weighted.", exc_info=True)
                 logger.warning("Falling back to potentially using cross_attention_kwargs (if applicable in pipeline call).")
        else:
             logger.warning("Pipeline does not have 'set_adapters'. Cannot explicitly set adapter weights. Relying on cross_attention_kwargs or internal scaling.")

    logger.info("LoRA application process finished.")


def unload_lora_weights(pipeline: DiffusionPipeline):
    """Safely unloads/detaches all LoRA weights from a pipeline."""
    if hasattr(pipeline, 'unload_lora_weights') and callable(pipeline.unload_lora_weights):
         logger.debug(f"Unloading all LoRA weights from pipeline using unload_lora_weights().")
         try:
              # Unload all LoRAs loaded with adapter names
              # Need to know which adapters were loaded? Or does it unload all? Check diffusers docs.
              # Assuming it unloads all non-base adapters if called without args?
              pipeline.unload_lora_weights()
         except Exception as e:
              logger.warning(f"Error unloading LoRA weights via unload_lora_weights(): {e}")
              # Fallback to resetting processors just in case
              if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'set_default_attn_processor'):
                   try: pipeline.unet.set_default_attn_processor()
                   except Exception as e_unet: logger.warning(f"Error resetting UNet processors as fallback: {e_unet}")

    elif hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'set_default_attn_processor'):
         logger.debug(f"Resetting UNet attention processors for pipeline.")
         try: pipeline.unet.set_default_attn_processor()
         except Exception as e: logger.warning(f"Error resetting UNet processors: {e}")
    # TODO: Add similar logic for text_encoder(s) if they have LoRAs applied separately