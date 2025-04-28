# gradient-project-root/src/gradient/ai/diffusion/pipelines.py

import logging
import torch
import time
import base64
from io import BytesIO
from PIL import Image
from typing import List, Optional # Ensure List, Optional are imported

from .loading import load_diffusion_pipeline
from .models import Text2ImageRequest, Text2ImageResponse, ImageResult
from diffusers.schedulers.scheduling_utils import SchedulerMixin # Import base scheduler

logger = logging.getLogger(__name__)

async def run_text_to_image(request: Text2ImageRequest) -> Optional[Text2ImageResponse]:
    """
    Runs the text-to-image pipeline based on the request.
    Supports custom noise schedules via `custom_sigmas`.
    """
    start_time = time.time()
    logger.info(f"Received text-to-image request: {request.prompt[:50]}...")

    # 1. Load Pipeline
    # TODO: Determine pipeline_type based on model_id if possible
    pipeline_type = "StableDiffusionPipeline" # Assume SD for now
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # TODO: Pass vae_id from request if added to Text2ImageRequest model later
    pipeline = await load_diffusion_pipeline(
        model_path_or_id=request.model_id,
        pipeline_type=pipeline_type,
        device=device_str, # Loader might expect string
        dtype=dtype,
        # vae_id=request.vae_id # Add when VAE is part of request
    )

    if not pipeline:
        logger.error(f"Failed to load pipeline for model: {request.model_id}")
        return None

    pipeline.to(device) # Ensure pipeline is on correct device object

    # 2. Prepare Generator
    generator = None
    effective_seed = request.seed
    if effective_seed is None:
        effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        logger.info(f"Generated random seed: {effective_seed}")
    else:
         logger.info(f"Using provided seed: {effective_seed}")
    generator = torch.Generator(device=device).manual_seed(effective_seed)

    # --- 3. Prepare Timesteps/Sigmas --- vvv
    timesteps = None
    num_inference_steps = request.num_inference_steps

    scheduler: SchedulerMixin = pipeline.scheduler
    scheduler_config = scheduler.config

    if request.custom_sigmas:
        # Use user-provided sigmas
        if len(request.custom_sigmas) < 2:
             logger.error("Custom sigmas must contain at least two values.")
             return None # Or raise validation error

        try:
            # Convert to tensor on the correct device
            sigmas_tensor = torch.tensor(request.custom_sigmas, dtype=torch.float32, device=device) # Sigmas usually float32
            # Some schedulers might expect descending order, others ascending. Diffusers usually handles this.
            # We need to get corresponding timesteps from these sigmas
            # This is scheduler-dependent and less direct than setting timesteps.
            # A common approach is to find timesteps closest to the sigma values.
            # This requires interpolation and knowledge of the scheduler's sigma<->timestep mapping.

            # --- SIMPLER APPROACH for schedulers supporting set_timesteps with tensor: ---
            # If the scheduler can derive timesteps from sigmas directly, or if we can calculate them.
            # For now, let's *try* setting timesteps based on the *number* of sigmas provided,
            # assuming the user wants that many effective steps, and then *later* find a way
            # to directly use the sigma values if the scheduler supports it easily.
            # This isn't true schedule sculpting yet, but allows varying step count via list length.

            # Option A (Simpler, uses length but not values): Set steps based on list length
            num_inference_steps = len(request.custom_sigmas) - 1 # Number of steps is usually len(sigmas/timesteps) - 1
            scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = scheduler.timesteps
            logger.info(f"Using custom sigma list length to set num_inference_steps={num_inference_steps}. Actual sigma values NOT YET USED.")
            # TODO: Implement actual sigma injection into the scheduler/loop if possible.

            # Option B (More complex, scheduler-dependent - Needs research per scheduler type):
            # Directly manipulate scheduler.sigmas if possible (might break internal state)
            # scheduler.sigmas = sigmas_tensor
            # timesteps = scheduler.timesteps # Recalculate based on new sigmas? Risky.
            # Or: Find timesteps corresponding to sigmas (requires inverse mapping/interpolation)
            # timesteps = find_timesteps_for_sigmas(scheduler, sigmas_tensor) # Needs custom function

        except Exception as e:
            logger.error(f"Failed to process custom_sigmas: {e}", exc_info=True)
            return None
    else:
        # Use default num_inference_steps
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        logger.info(f"Using standard scheduler timesteps for {num_inference_steps} steps.")


    # --- 4. Prepare Latents and Embeddings ---
    # ... (Latent prep using pipeline.prepare_latents as before) ...
    height = request.height
    width = request.width
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
         1, # batch_size=1
         num_channels_latents,
         height,
         width,
         dtype,
         device,
         generator
    )

    # Encode prompts (using updated method)
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=request.prompt,
        device=device,
        num_images_per_prompt=request.num_outputs, # Use num_outputs from request here
        do_classifier_free_guidance=request.guidance_scale > 1.0, # Standard check
        negative_prompt=request.negative_prompt
        # TODO: Handle prompt embeds for SDXL later (pooled_prompt_embeds etc.)
    )

    if request.guidance_scale > 1.0:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # --- 5. Run Inference Loop ---
    images_pil: List[Image.Image] = []
    try:
        logger.info(f"Running inference loop with {len(timesteps)} timesteps...")
        pipeline.to(device) # Ensure device

        # Iterate through timesteps (obtained from scheduler.set_timesteps)
        for i, t in enumerate(pipeline.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if request.guidance_scale > 1.0 else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds
                ).sample

            # perform guidance
            if request.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + request.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 6. Post-process latents (decode via VAE)
        # Scale latents based on VAE config
        if hasattr(pipeline.vae.config, 'scaling_factor'):
            latents = latents / pipeline.vae.config.scaling_factor
        else:
            latents = 1 / pipeline.vae.config.get("scaling_factor", 0.18215) * latents # Use default or get from config

        with torch.no_grad():
            image = pipeline.vae.decode(latents.to(pipeline.vae.dtype)).sample # Ensure dtype matches VAE

        # Process image (already handled in decode_latents_batch - use similar logic)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy() # Should be batch 1 now

        # Convert to PIL
        images_pil = pipeline.numpy_to_pil(image)

        logger.info(f"Inference completed. Generated {len(images_pil)} image(s).")

    except Exception as e:
        logger.error(f"Error during diffusion inference loop: {e}", exc_info=True)
        return None

    # 7. Encode Results
    # ... (Encoding logic remains the same as before) ...
    results: List[ImageResult] = []
    final_params = request.model_dump(exclude_unset=True)
    final_params['seed'] = effective_seed # Store the seed used

    for i, img_pil in enumerate(images_pil):
        try:
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            # Use the same effective seed for all images in the batch run
            results.append(ImageResult(image_base64=img_base64, seed=effective_seed, parameters=final_params))
        except Exception as e:
            logger.error(f"Error encoding image {i}: {e}", exc_info=True)

    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Text-to-image request processed in {performance_ms:.2f} ms.")

    return Text2ImageResponse(
        images=results,
        request_params=request, # Echo back original request
        performance_ms=performance_ms
    )