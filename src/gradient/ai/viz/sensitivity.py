# gradient-project-root/src/gradient/ai/viz/sensitivity.py

import logging
import time
import torch
import numpy as np
from typing import Optional, Dict, Any

# Import diffusion components needed
from gradient.ai.diffusion.loading import load_diffusion_pipeline
from gradient.ai.diffusion.models import Text2ImageRequest
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

# Import our models
from .models import SensitivityAnalysisRequest, SensitivityAnalysisResponse, SensitivityMapData

logger = logging.getLogger(__name__)

async def analyze_parameter_sensitivity(request: SensitivityAnalysisRequest) -> Optional[SensitivityAnalysisResponse]:
    start_time = time.time()
    base_req = request.base_request
    target_param_name = request.target_parameter.name
    logger.info(f"Starting sensitivity analysis: Target='{target_param_name}', Model='{base_req.model_id}'")

    # --- Basic Validation ---
    if request.analysis_target != "latent":
        logger.error(f"Unsupported analysis target: {request.analysis_target}. Only 'latent' is supported currently.")
        return None
    if target_param_name != "guidance_scale":
        logger.error(f"Unsupported target parameter: {target_param_name}. Only 'guidance_scale' is supported currently.")
        return None

    # --- Determine Device and Dtype ---
    # Determine device string first
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # Convert to torch.device object
    device = torch.device(device_str) # <--- Create torch.device object
    dtype = torch.float16 if device.type == "cuda" else torch.float32 # Check device.type now

    # --- Load Pipeline ---
    pipeline = await load_diffusion_pipeline(
        model_path_or_id=base_req.model_id,
        pipeline_type="StableDiffusionPipeline",
        device=device_str, # Pass string to loader function (it likely handles it)
        dtype=dtype
    )

    if not pipeline or not isinstance(pipeline, StableDiffusionPipeline):
        logger.error(f"Failed to load StableDiffusionPipeline for sensitivity analysis: {base_req.model_id}")
        return None

    # Ensure pipeline is on the correct device object (redundant if loader worked, but safe)
    pipeline.to(device)

    # --- Prepare Inputs ---
    height = base_req.height
    width = base_req.width
    num_inference_steps = base_req.num_inference_steps
    guidance_scale = base_req.guidance_scale
    negative_prompt = base_req.negative_prompt
    prompt = base_req.prompt
    seed = base_req.seed
    batch_size = 1

    scheduler: SchedulerMixin = pipeline.scheduler
    # Pass torch.device object to set_timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    num_channels_latents = pipeline.unet.config.in_channels
    shape = (batch_size, num_channels_latents, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor)

    generator = None
    if seed is not None:
        # Pass torch.device object to Generator
        generator = torch.Generator(device=device).manual_seed(seed)

    # Pass torch.device object to prepare_latents
    latents = pipeline.prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator)

    # Pass torch.device object to _encode_prompt (it likely handles string too, but device object is safer)
    text_embeddings = pipeline._encode_prompt(
        prompt, device, 1, True, negative_prompt
    )

    # --- Modified Diffusion Loop ---
    # Pass torch.device object for tensor creation
    guidance_scale_tensor = torch.tensor(float(guidance_scale), dtype=dtype, device=device, requires_grad=True)
    latents_grad = latents.clone().detach().requires_grad_(True)

    logger.info(f"Running modified diffusion loop for gradient calculation w.r.t. guidance_scale={guidance_scale_tensor.item()}")
    for i, t in enumerate(pipeline.progress_bar(timesteps)):
        latent_model_input = torch.cat([latents_grad] * 2)
        # Pass torch.device object if scale_model_input needs it (check specific scheduler)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            # Ensure encoder_hidden_states are on the correct device if not already
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings.to(device)).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_guided = noise_pred_uncond + guidance_scale_tensor * (noise_pred_text - noise_pred_uncond)

        # Pass torch.device object if scheduler.step needs it
        step_output = scheduler.step(noise_pred_guided, t, latents_grad)
        latents_prev = step_output.prev_sample

        # Detach/reattach (simplification) - device is implicitly handled here by latents_prev
        latents_grad = latents_prev.detach().requires_grad_(True)


    # --- Final Latent & Gradient Calculation ---
    final_latent = latents_grad
    loss = final_latent.abs().mean()

    logger.info("Performing backward pass to calculate gradients...")
    try:
        loss.backward()
    except Exception as e:
        logger.error(f"Error during backward pass: {e}", exc_info=True)
        # Cleanup before returning None
        del pipeline, latents_grad, final_latent, loss, guidance_scale_tensor, text_embeddings
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    # --- Extract and Process Gradient ---
    map_data = None # Initialize
    if final_latent.grad is not None:
         sensitivity_map_raw = final_latent.grad.abs().detach().cpu().numpy()
         if sensitivity_map_raw.shape[0] == 1:
             sensitivity_map_raw = sensitivity_map_raw.squeeze(0)
         logger.info(f"Generated raw sensitivity map (gradient w.r.t final latent loss) with shape: {sensitivity_map_raw.shape}")

         mean_sens = float(np.mean(sensitivity_map_raw))
         max_sens = float(np.max(sensitivity_map_raw))
         min_sens = float(np.min(sensitivity_map_raw))
         std_dev_sens = float(np.std(sensitivity_map_raw))
         heatmap_b64 = None # TODO

         map_data = SensitivityMapData(
             shape=list(sensitivity_map_raw.shape),
             mean_sensitivity=mean_sens,
             max_sensitivity=max_sens,
             min_sensitivity=min_sens,
             std_dev_sensitivity=std_dev_sens,
             heatmap_base64=heatmap_b64
         )
    else:
         logger.error("Gradient w.r.t final latent is None after backward pass.")
         # map_data remains None


    # --- Cleanup ---
    # Delete large objects explicitly to help release memory sooner
    del pipeline, latents_grad, final_latent, loss, guidance_scale_tensor, text_embeddings
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not map_data:
         logger.error("Failed to generate sensitivity map data.")
         return None

    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Sensitivity analysis completed in {performance_ms:.2f} ms.")

    return SensitivityAnalysisResponse(
        request_params=request,
        sensitivity_map=map_data,
        performance_ms=performance_ms
    )