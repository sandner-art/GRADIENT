# src/gradient/ai/diffusion/loading_loras.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .models import LoRAParameter 

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
import json 

def find_lora_path(lora_name: str) -> Optional[Path]:
     from .loading import find_model_path 
     return find_model_path(lora_name, "lora")

logger = logging.getLogger(__name__)

async def apply_loras_to_pipeline(
    pipeline: DiffusionPipeline,
    loras: Optional[List[LoRAParameter]] = None, 
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32
):
    if not loras: return

    if not hasattr(pipeline, 'load_lora_weights') or not callable(pipeline.load_lora_weights):
        logger.warning(f"Pipeline {type(pipeline).__name__} lacks load_lora_weights. Skipping LoRAs.")
        return

    logger.debug("Calling unload_lora_weights before applying new LoRAs.")
    unload_lora_weights(pipeline) 

    # These lists will store info for adapters that are successfully loaded AND recognized
    adapters_to_activate = []
    weights_for_activation = []

    logger.info(f"Applying {len(loras)} LoRA(s) to pipeline...")

    for i, lora_info in enumerate(loras): 
        lora_name = lora_info.name
        lora_weight = lora_info.weight
        # This is the name `load_lora_weights` will use to register the adapter internally with PEFT if it's a PEFT LoRA,
        # or as a general identifier if it's a standard LoRA that set_adapters will use.
        adapter_key_for_pipeline = f"g_adapter_{i}" 

        if not lora_name:
            logger.warning(f"Skipping LoRA {i}: name is empty.")
            continue

        logger.info(f"Preparing LoRA '{lora_name}' (Weight: {lora_weight}, Pipeline Adapter Key: '{adapter_key_for_pipeline}')")
        
        lora_path_or_id = lora_name 
        resolved_lora_path = find_lora_path(lora_name)
        
        diffusers_weight_name_arg = None 
        load_source = lora_path_or_id
        is_peft_directory_style = False

        if resolved_lora_path and resolved_lora_path.exists():
            load_source = str(resolved_lora_path)
            if resolved_lora_path.is_file():
                diffusers_weight_name_arg = resolved_lora_path.name 
                logger.debug(f"LoRA '{lora_name}' is local file: {load_source}. `weight_name` for diffusers: '{diffusers_weight_name_arg}'.")
            elif resolved_lora_path.is_dir() and (resolved_lora_path / "adapter_config.json").is_file():
                is_peft_directory_style = True
                logger.debug(f"LoRA '{lora_name}' is local PEFT-style directory: {load_source}. `weight_name` for diffusers is None.")
                # diffusers_weight_name_arg remains None
            else: # Path exists but not a file or recognized PEFT dir
                logger.warning(f"Resolved LoRA path '{load_source}' for '{lora_name}' is not a recognized file or PEFT directory. Treating as Hub ID.")
        else:
            logger.warning(f"LoRA '{lora_name}' not found locally. Treating as Hub ID. `weight_name` for diffusers is None.")

        try:
            pipeline.load_lora_weights(
                load_source, 
                weight_name=diffusers_weight_name_arg, 
                adapter_name=adapter_key_for_pipeline 
            )
            logger.info(f"Call to pipeline.load_lora_weights for '{lora_name}' (adapter_key='{adapter_key_for_pipeline}') completed.")
            
            # After load_lora_weights, we assume it's ready to be used by set_adapters with adapter_key_for_pipeline.
            # The "No LoRA keys associated..." warning from diffusers is the main indicator if weights didn't actually apply.
            adapters_to_activate.append(adapter_key_for_pipeline)
            weights_for_activation.append(lora_weight)

        except ValueError as ve: 
            logger.error(f"ValueError loading LoRA '{lora_name}': {ve}", exc_info=True) # Still log PEFT target module errors
        except Exception as e: 
            logger.error(f"Generic error loading LoRA '{lora_name}' (adapter_key='{adapter_key_for_pipeline}'): {e}", exc_info=True)
            
    # After attempting to load all requested LoRAs
    if adapters_to_activate:
        if hasattr(pipeline, "set_adapters") and callable(pipeline.set_adapters):
            try:
                logger.info(f"Attempting to activate adapters: {adapters_to_activate} with weights: {weights_for_activation}")
                pipeline.set_adapters(adapters_to_activate, adapter_weights=weights_for_activation)
                logger.info(f"Successfully called set_adapters for: {adapters_to_activate}.")
            except ValueError as ve_set: 
                 logger.error(f"ValueError during pipeline.set_adapters: {ve_set}.", exc_info=True)
                 # This error (Adapter name X not in list of present adapters Y) means that
                 # `load_lora_weights` with `adapter_name=X` did not actually make X available
                 # to `set_adapters`. This usually happens if no LoRA weights from the file/dir
                 # were actually applied to any component (UNet, TE) under that adapter name.
                 # The "No LoRA keys associated..." warning is the clue.
                 logger.error(
                     f"This typically means that although load_lora_weights was called for adapter(s) [{', '.join(adapters_to_activate)}], "
                     f"no actual LoRA layers were successfully matched and applied to the pipeline's components for these adapter names. "
                     f"Check for 'No LoRA keys associated...' warnings during the load_lora_weights call for these adapters."
                 )
            except Exception as e_set:
                 logger.error(f"Generic error during pipeline.set_adapters: {e_set}.", exc_info=True)
        else: 
            logger.warning("Pipeline lacks 'set_adapters'. Cannot explicitly set weights/activate adapters.")
    else:
        logger.info("No LoRAs were successfully prepared for activation in this session, skipping set_adapters.")

    logger.info("LoRA application process finished.")

# unload_lora_weights function remains the same as the previous good version.
def unload_lora_weights(pipeline: DiffusionPipeline):
    logger.debug("Starting LoRA unload process...")
    
    if hasattr(pipeline, 'unload_lora_weights') and callable(pipeline.unload_lora_weights):
        logger.debug("Attempting unload via pipeline.unload_lora_weights().")
        try:
            pipeline.unload_lora_weights()
            logger.info("Successfully called pipeline.unload_lora_weights().")
        except Exception as e:
            logger.warning(f"Error calling pipeline.unload_lora_weights(): {e}. Will attempt manual resets.", exc_info=True)

    components_to_reset_attn_procs = []
    if hasattr(pipeline, 'unet'): components_to_reset_attn_procs.append(pipeline.unet)
    if hasattr(pipeline, 'text_encoder'): components_to_reset_attn_procs.append(pipeline.text_encoder)
    if hasattr(pipeline, 'text_encoder_2'): components_to_reset_attn_procs.append(pipeline.text_encoder_2)

    for component in components_to_reset_attn_procs:
        if component and hasattr(component, 'set_default_attn_processor'):
            component_name = component.__class__.__name__
            logger.debug(f"Resetting {component_name} attention processors.")
            try: component.set_default_attn_processor()
            except Exception as e_comp_reset: logger.error(f"Error resetting {component_name} attention processors: {e_comp_reset}", exc_info=True)
    
    for component_name in ["unet", "text_encoder", "text_encoder_2"]:
        if hasattr(pipeline, component_name):
            component = getattr(pipeline, component_name)
            if component and hasattr(component, "peft_config"):
                if isinstance(component.peft_config, dict) and len(component.peft_config) > 0:
                    logger.debug(f"Clearing PEFT config from {component_name}.")
                    component.peft_config = {} 
            if component and hasattr(component, "active_adapters"): 
                if component.active_adapters:
                    logger.debug(f"Clearing active_adapters from {component_name}.")
                    component.active_adapters = []
            if component and hasattr(component, "lora_layers") and isinstance(component.lora_layers, torch.nn.ModuleDict):
                if len(component.lora_layers) > 0:
                    logger.debug(f"Clearing {len(component.lora_layers)} entries from lora_layers in {component_name}.")
                    component.lora_layers.clear()

    logger.info("LoRA unload process completed.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()