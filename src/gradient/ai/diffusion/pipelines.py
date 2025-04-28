# gradient-project-root/src/gradient/ai/diffusion/pipelines.py

import logging
import torch
import time
import base64
import uuid # Import uuid
import json # Import json
import datetime # Import datetime
from io import BytesIO
from pathlib import Path # Import Path
from PIL import Image
from typing import List, Optional

# DB Imports
from sqlalchemy.ext.asyncio import AsyncSession # Type hint for session
from gradient.database.session import get_db_session, update_run_status # Import DB session and updater
from gradient.database.models import ExperimentRun, GeneratedOutput # Import DB models

# AI Imports
from .loading import load_diffusion_pipeline # Import the loading function
from .models import Text2ImageRequest, Text2ImageResponse, ImageResult # Import pydantic models
from diffusers.schedulers.scheduling_utils import SchedulerMixin # Import base scheduler class for type hint
from diffusers import StableDiffusionPipeline # Example specific pipeline for type checking, adjust if needed

logger = logging.getLogger(__name__)

# --- TODO: Configure output path ---
def get_output_base_path() -> Path:
     """Gets the base path for saving experiment outputs."""
     # Use project root from config? Needs config module import
     # from gradient.core.config import PROJECT_ROOT
     # output_path = PROJECT_ROOT / "experiments"
     # For now, assume relative to where script is run or project root
     output_path = Path("./experiments/")
     output_path.mkdir(parents=True, exist_ok=True) # Ensure it exists
     return output_path

async def run_text_to_image(
    request: Text2ImageRequest,
    # Optional: Pass run_uuid if called from another process like xyz_grid
    parent_run_uuid: Optional[str] = None
) -> Optional[Text2ImageResponse]:
    """
    Runs the text-to-image pipeline based on the request.
    Logs experiment details to the database and saves generated images.
    Supports custom noise schedules via `custom_sigmas` (length determines steps).
    """
    start_time = time.time()
    run_uuid = parent_run_uuid or str(uuid.uuid4()) # Use parent or generate new UUID
    logger.info(f"Starting text-to-image run {run_uuid}: {request.prompt[:50]}...")

    db_run: Optional[ExperimentRun] = None
    db_run_id: Optional[int] = None # Store DB ID for potential status updates on failure

    # --- DB Logging Start ---
    # Only create a new ExperimentRun record if this is not part of a parent run
    if not parent_run_uuid:
        try:
            async with get_db_session() as db:
                db_run = ExperimentRun(
                    run_uuid=run_uuid,
                    run_type="text_to_image",
                    start_time=datetime.datetime.now(tz=datetime.timezone.utc), # Use timezone aware UTC
                    status="running",
                    base_parameters=request.model_dump(mode='json') # Serialize request to JSON
                )
                db.add(db_run)
                await db.commit()
                await db.refresh(db_run) # Get the automatically assigned ID
                db_run_id = db_run.id
                logger.info(f"Created ExperimentRun record ID: {db_run_id} for run {run_uuid}")
        except Exception as e:
            logger.error(f"Failed to create initial DB record for run {run_uuid}: {e}", exc_info=True)
            db_run = None # Ensure db_run is None if creation failed
            db_run_id = None
            # Decide if we should fail the whole run here - maybe return None
            # return None # Fail early if DB logging fails? Or proceed without DB logging?
    # --- End DB Logging Start ---

    # --- Core Generation Logic ---
    try:
        # --- Setup Device & Dtype ---
        pipeline_type = "StableDiffusionPipeline" # TODO: Make dynamic based on model?
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        logger.debug(f"Run {run_uuid}: Using device='{device.type}' dtype='{dtype}'")

        # --- Load Pipeline ---
        # TODO: Pass vae_id from request if added to Text2ImageRequest model later
        pipeline = await load_diffusion_pipeline(
            model_path_or_id=request.model_id,
            pipeline_type=pipeline_type,
            device=device_str, # Loader might expect string
            dtype=dtype,
            # vae_id=request.vae_id # Add when VAE is part of request
        )

        if not pipeline:
            error_msg = f"Failed to load pipeline for model: {request.model_id}"
            logger.error(f"Run {run_uuid}: {error_msg}")
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None

        pipeline.to(device) # Ensure pipeline components are on the correct device object

        # --- Prepare Generator ---
        generator = None
        effective_seed = request.seed
        if effective_seed is None:
            effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            logger.info(f"Run {run_uuid}: Generated random seed: {effective_seed}")
        else:
            # Ensure seed is int if passed as float/string etc.
            try:
                effective_seed = int(effective_seed)
                logger.info(f"Run {run_uuid}: Using provided seed: {effective_seed}")
            except (ValueError, TypeError):
                logger.warning(f"Run {run_uuid}: Invalid seed value '{request.seed}'. Generating random seed.")
                effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device=device).manual_seed(effective_seed)

        # --- Prepare Timesteps/Sigmas ---
        timesteps = None
        num_inference_steps = request.num_inference_steps # Start with request value
        scheduler: SchedulerMixin = pipeline.scheduler

        if request.custom_sigmas is not None:
            if len(request.custom_sigmas) < 2:
                error_msg = f"Custom sigmas list must contain at least 2 values, but got {len(request.custom_sigmas)}."
                logger.error(f"Run {run_uuid}: {error_msg}")
                if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
                return None

            calculated_steps = len(request.custom_sigmas) - 1
            if calculated_steps <= 0:
                error_msg = f"Custom sigmas list length resulted in non-positive steps ({calculated_steps}). Provide more sigma values."
                logger.error(f"Run {run_uuid}: {error_msg}")
                if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
                return None

            num_inference_steps = calculated_steps # Override request value
            logger.info(f"Run {run_uuid}: Using custom sigma list length to set num_inference_steps={num_inference_steps}. Actual sigma values NOT YET USED.")
            try:
                scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = scheduler.timesteps
            except Exception as e:
                error_msg = f"Failed to set scheduler timesteps based on custom_sigmas length: {e}"
                logger.error(f"Run {run_uuid}: {error_msg}", exc_info=True)
                if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
                return None
        else:
            # Use num_inference_steps from request if custom_sigmas is None
            if num_inference_steps <= 0:
                error_msg = f"num_inference_steps must be positive, but got {num_inference_steps}."
                logger.error(f"Run {run_uuid}: {error_msg}")
                if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
                return None
            logger.info(f"Run {run_uuid}: Using standard scheduler timesteps for {num_inference_steps} steps.")
            try:
                scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = scheduler.timesteps
            except Exception as e:
                error_msg = f"Failed to set scheduler timesteps: {e}"
                logger.error(f"Run {run_uuid}: {error_msg}", exc_info=True)
                if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
                return None

        if timesteps is None: # Defensive check
            error_msg = "Scheduler timesteps were not set."
            logger.error(f"Run {run_uuid}: {error_msg}")
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None

        # --- Latent & Embedding Prep ---
        height = request.height
        width = request.width
        if height <= 0 or width <= 0 or height % 8 != 0 or width % 8 != 0 : # Check positive and multiple of 8
             error_msg = f"Image height and width must be positive multiples of 8, got W={width}, H={height}."
             logger.error(f"Run {run_uuid}: {error_msg}")
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
             return None

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            request.num_outputs, # Use num_outputs for batch size
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator
        )

        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=request.prompt, device=device, num_images_per_prompt=request.num_outputs, # Pass correct num_outputs
            do_classifier_free_guidance=request.guidance_scale > 1.0,
            negative_prompt=request.negative_prompt
        )

        # Guidance handling (no repeats needed here as encode_prompt handles num_images_per_prompt)
        if request.guidance_scale > 1.0:
            prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            prompt_embeds_cfg = prompt_embeds

        # --- Inference Loop ---
        images_pil: List[Image.Image] = []
        num_actual_steps = len(timesteps) # Use actual length of timesteps for loop
        logger.info(f"Run {run_uuid}: Running inference loop with {num_actual_steps} timesteps for {request.num_outputs} image(s)...")

        for i, t in enumerate(pipeline.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if request.guidance_scale > 1.0 else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = pipeline.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds_cfg.to(device)
                ).sample

            if request.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + request.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample
        # --- End Loop ---

        # --- VAE Decode ---
        # Use VAE scaling factor from pipeline config
        latents_scaled = 1 / pipeline.vae.config.get("scaling_factor", 0.18215) * latents
        with torch.no_grad():
            # Ensure latents are on the same device and dtype as VAE expects
            image = pipeline.vae.decode(latents_scaled.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1) # Rescale to [0, 1]
        image = image.cpu().permute(0, 2, 3, 1).float().numpy() # Move to CPU for PIL conversion
        images_pil = pipeline.numpy_to_pil(image) # Convert to PIL list
        logger.info(f"Run {run_uuid}: Inference completed. Generated {len(images_pil)} image(s).")

    except Exception as e: # Catch any exception during the core logic
        error_during_run = f"{type(e).__name__}: {e}"
        logger.error(f"Run {run_uuid}: Error during diffusion core logic: {e}", exc_info=True)
        if db_run_id: await update_run_status(db_run_id, "failed", error=error_during_run)
        return None # Return None if loop failed

    # --- Process, Save, and Log Outputs ---
    results: List[ImageResult] = []
    output_base_path = get_output_base_path() / run_uuid # Save in folder named after run UUID
    output_base_path.mkdir(parents=True, exist_ok=True)
    # Prepare final parameters dict once, updating seed per image
    final_params = request.model_dump(exclude_unset=True)
    final_params['num_inference_steps'] = num_inference_steps # Store actual steps used
    if 'custom_sigmas' in final_params and not final_params['custom_sigmas']:
         del final_params['custom_sigmas'] # Don't store empty list if parsed back to None

    output_records_to_add = [] # Batch DB additions

    for i, img_pil in enumerate(images_pil):
        current_seed = effective_seed + i # Simple seed increment for batch
        filename = f"output_{i:03d}_seed{current_seed}.png"
        file_path = output_base_path / filename
        file_path_str = str(file_path.resolve())

        logger.debug(f"Processing image {i}: Saving to {file_path_str}, Seed: {current_seed}")
        output_params = final_params.copy()
        output_params['seed'] = current_seed

        # Save the image with metadata attempt
        pnginfo = None
        try:
            # --- Metadata Injection ---
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            # Simple JSON dump of parameters for metadata
            # Consider standard formats like A1111 later for compatibility
            pnginfo.add_text("grad_parameters", json.dumps(output_params))
        except Exception as meta_e:
             logger.warning(f"Could not prepare PNG metadata: {meta_e}")
             pnginfo = None # Ensure it's None if error occurs

        try:
            img_pil.save(file_path_str, "PNG", pnginfo=pnginfo) # Pass pnginfo object
        except Exception as e:
            logger.error(f"Failed to save image {file_path_str}: {e}", exc_info=True)
            continue # Skip this image if saving failed

        # Encode for API response
        img_base64 = None
        try:
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG") # Re-save for base64, could optimize later
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {i} to base64: {e}", exc_info=True)
            # Continue, but image won't be in API response

        # Add result for API response if encoded
        if img_base64:
            results.append(ImageResult(
                image_base64=img_base64,
                seed=current_seed,
                parameters=output_params
            ))

        # Prepare output record for DB logging (only if initial run record was created)
        if db_run_id:
            output_record = GeneratedOutput(
                run_id=db_run_id,
                output_type="image",
                file_path=file_path_str, # Store resolved path
                parameters=output_params,
                creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
                # output_metadata field could store VAE used, timings, etc.
            )
            output_records_to_add.append(output_record)

    # Batch add output records to DB
    if output_records_to_add:
        try:
             async with get_db_session() as db:
                 db.add_all(output_records_to_add)
                 await db.commit()
                 logger.info(f"Logged {len(output_records_to_add)} output records to DB for run {run_uuid}")
        except Exception as e:
             logger.error(f"Failed to log output records to DB for run {run_uuid}: {e}", exc_info=True)


    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Run {run_uuid}: Text-to-image request processed in {performance_ms:.2f} ms.")

    # Update final status in DB
    if db_run_id: await update_run_status(db_run_id, "completed", end_time=end_time)

    # Return API response
    return Text2ImageResponse(
        images=results,
        request_params=request, # Echo original request
        performance_ms=performance_ms
    )