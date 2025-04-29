# src/gradient/ai/diffusion/loading_loras.py

import logging
from pathlib import Path
# --- Import LoRAParameter --- vvv
from typing import List, Dict, Any, Optional
from .models import LoRAParameter # Import the Pydantic model
# --- Import LoRAParameter --- ^^^

import torch
from diffusers import DiffusionPipeline
# from peft import PeftModel # Keep commented for now
from huggingface_hub import hf_hub_download

# Assuming find_lora_path helper exists or is defined here/imported
# from .loading import find_model_path
def find_lora_path(lora_name: str) -> Optional[Path]:
     from .loading import find_model_path
     return find_model_path(lora_name, "lora")


logger = logging.getLogger(__name__)


async def apply_loras_to_pipeline(
    pipeline: DiffusionPipeline,
    # --- Correct Type Hint --- vvv
    loras: Optional[List[LoRAParameter]] = None, # Use the Pydantic model type
    # --- Correct Type Hint --- ^^^
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32
):
    """ Applies LoRAs defined by LoRAParameter objects to a pipeline. """
    if not loras: return

    if not hasattr(pipeline, 'load_lora_weights') or not callable(pipeline.load_lora_weights):
        logger.warning(f"Pipeline {type(pipeline).__name__} lacks load_lora_weights. Skipping LoRAs.")
        return

    unload_lora_weights(pipeline) # Unload previous

    loaded_adapters = []
    adapter_weights = []
    logger.info(f"Applying {len(loras)} LoRA(s) to pipeline...")

    for i, lora_info in enumerate(loras): # lora_info is type LoRAParameter
        # --- Access attributes directly --- vvv
        lora_name = lora_info.name
        lora_weight = lora_info.weight # Already a float
        # --- Access attributes directly --- ^^^
        adapter_name = f"lora_{i}"

        if not lora_name: logger.warning(f"Skipping LoRA {i}: name is empty."); continue

        logger.info(f"Preparing LoRA '{lora_name}' weight={lora_weight} adapter='{adapter_name}'")
        lora_path_or_id = lora_name
        resolved_lora_path = find_lora_path(lora_name)
        weight_name = None # Default for Hub IDs

        if resolved_lora_path and resolved_lora_path.is_file():
            lora_path_or_id = str(resolved_lora_path)
            weight_name = resolved_lora_path.name # Use filename for local safetensors
            logger.debug(f"Found local LoRA path: {lora_path_or_id}")
        else: logger.warning(f"LoRA file '{lora_name}' not found locally. Assuming Hub ID.")

        try:
            # Load weights into unique adapter
            pipeline.load_lora_weights(
                lora_path_or_id,
                weight_name=weight_name,
                adapter_name=adapter_name
            )
            logger.info(f"Loaded LoRA weights '{lora_name}' -> adapter '{adapter_name}'.")
            loaded_adapters.append(adapter_name)
            adapter_weights.append(lora_weight) # Store requested weight

        except Exception as e: logger.error(f"Failed load LoRA '{lora_name}': {e}", exc_info=True)

    # --- Activate and Set Weights (Attempt using set_adapters if available) ---
    if loaded_adapters:
        if hasattr(pipeline, "set_adapters") and callable(pipeline.set_adapters):
             try:
                 logger.info(f"Activating adapters: {loaded_adapters} with weights {adapter_weights}")
                 pipeline.set_adapters(loaded_adapters, adapter_weights=adapter_weights)
                 logger.info("Successfully set LoRA adapters and weights.")
             except Exception as e: logger.error(f"Failed PEFT set_adapters: {e}. LoRAs might use default scaling.", exc_info=True)
        else: logger.warning("Pipeline lacks 'set_adapters'. Cannot set weights explicitly. Relying on internal scaling/kwargs.")

    logger.info("LoRA application process finished.")


def unload_lora_weights(pipeline: DiffusionPipeline):
    """Safely unloads/detaches all LoRA weights from a pipeline."""
    # --- (Implementation unchanged, keep PEFT warning for now) ---
    if hasattr(pipeline, 'unload_lora_weights') and callable(pipeline.unload_lora_weights):
         logger.debug(f"Unloading all LoRA weights from pipeline.")
         try: pipeline.unload_lora_weights()
         except Exception as e: logger.warning(f"Error unloading LoRA weights via unload_lora_weights(): {e}") # Keep PEFT warning
         finally: # Ensure fallback runs even if unload errors
              if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'set_default_attn_processor'):
                   try: pipeline.unet.set_default_attn_processor()
                   except Exception as e_unet: logger.warning(f"Error resetting UNet processors as fallback: {e_unet}")
    elif hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'set_default_attn_processor'):
         logger.debug(f"Resetting UNet attention processors.")
         try: pipeline.unet.set_default_attn_processor()
         except Exception as e: logger.warning(f"Error resetting UNet processors: {e}")