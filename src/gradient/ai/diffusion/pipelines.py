# gradient-project-root/src/gradient/ai/diffusion/pipelines.py

import logging
import torch
import time
import base64
import uuid
import json
import datetime
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple # <-- Added Tuple

# DB Imports
from sqlalchemy.ext.asyncio import AsyncSession
from gradient.database.session import get_db_session, update_run_status
from gradient.database.models import ExperimentRun, GeneratedOutput

# AI Imports
from .loading import load_diffusion_pipeline
# Import specific request/response types used
from .models import Text2ImageRequest, Text2ImageResponse, ImageResult, ImageToImageRequest, ImageToImageResponse
from diffusers.schedulers.scheduling_utils import SchedulerMixin
# Import specific pipeline types if needed for type hints or checks
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DiffusionPipeline

logger = logging.getLogger(__name__)

# --- Output Path (Unchanged) ---
def get_output_base_path() -> Path:
     output_path = Path("./experiments/")
     output_path.mkdir(parents=True, exist_ok=True)
     return output_path

# --- Helper Function to Prepare Image (for Img2Img) ---
def prepare_init_image(image_base64: str, width: int, height: int) -> Optional[Image.Image]:
    """Decodes, validates, and resizes the initial image."""
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # TODO: Add more robust resizing/cropping options later?
        # For now, just resize to target dimensions.
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Failed to prepare initial image: {e}", exc_info=True)
        return None

# --- Helper Function for Scheduler Loading/Setting ---
def setup_scheduler_and_timesteps(
    pipeline: DiffusionPipeline,
    request_scheduler_id: Optional[str],
    request_custom_sigmas: Optional[List[float]],
    request_num_steps: int,
    device: torch.device,
    run_uuid: str # For logging
) -> Optional[Tuple[SchedulerMixin, torch.Tensor, int]]:
    """Loads scheduler if requested, sets timesteps, returns scheduler, timesteps, and effective steps."""
    scheduler = pipeline.scheduler # Start with the default
    effective_num_steps = request_num_steps # Start with request value

    # --- 1. Override Scheduler if requested ---
    if request_scheduler_id:
        logger.info(f"Run {run_uuid}: Attempting to load requested scheduler: {request_scheduler_id}")
        original_scheduler_name = scheduler.__class__.__name__
        try:
            # Dynamically import the scheduler class
            # Ensure 'diffusers' is installed and scheduler_id is a valid class name within it
            scheduler_module = __import__('diffusers', fromlist=[request_scheduler_id])
            scheduler_class = getattr(scheduler_module, request_scheduler_id)

            if issubclass(scheduler_class, SchedulerMixin):
                scheduler = scheduler_class.from_config(pipeline.scheduler.config)
                pipeline.scheduler = scheduler # IMPORTANT: Update the pipeline's scheduler instance
                logger.info(f"Run {run_uuid}: Successfully switched scheduler to {request_scheduler_id}")
            else:
                logger.warning(f"Run {run_uuid}: '{request_scheduler_id}' is not a valid Scheduler class. Using default '{original_scheduler_name}'.")
                scheduler = pipeline.scheduler # Revert to default just in case
        except (ImportError, AttributeError) as e:
            logger.warning(f"Run {run_uuid}: Could not find or import scheduler '{request_scheduler_id}'. Using default '{original_scheduler_name}'. Error: {e}")
            scheduler = pipeline.scheduler # Revert to default
        except Exception as e:
            logger.error(f"Run {run_uuid}: Unexpected error loading scheduler '{request_scheduler_id}'. Using default '{original_scheduler_name}'. Error: {e}", exc_info=True)
            scheduler = pipeline.scheduler # Revert to default
    else:
        logger.info(f"Run {run_uuid}: Using default scheduler: {scheduler.__class__.__name__}")

    # --- 2. Set Timesteps based on custom sigmas or num_steps ---
    timesteps = None
    if request_custom_sigmas is not None:
        # --- Custom Sigmas Logic (Simplified - assumes length determines steps) ---
        if len(request_custom_sigmas) < 2:
            logger.error(f"Run {run_uuid}: Custom sigmas list must contain at least 2 values.")
            return None # Signal error
        calculated_steps = len(request_custom_sigmas) - 1
        if calculated_steps <= 0:
             logger.error(f"Run {run_uuid}: Custom sigmas list length resulted in non-positive steps.")
             return None # Signal error
        effective_num_steps = calculated_steps # Override request value
        logger.info(f"Run {run_uuid}: Using custom sigma list length to set num_inference_steps={effective_num_steps}. Actual sigma values NOT YET USED by standard schedulers.")
        # Fallthrough to set timesteps based on calculated steps
    else:
        # Use num_inference_steps from request if custom_sigmas is None
        if effective_num_steps <= 0:
            logger.error(f"Run {run_uuid}: num_inference_steps must be positive, but got {effective_num_steps}.")
            return None # Signal error

    # --- 3. Set Timesteps on the *final* scheduler ---
    try:
        scheduler.set_timesteps(effective_num_steps, device=device)
        timesteps = scheduler.timesteps
        logger.info(f"Run {run_uuid}: Timesteps set on scheduler '{scheduler.__class__.__name__}' for {len(timesteps)} steps.")
        if timesteps is None or len(timesteps) == 0: # Defensive check
             raise ValueError("Scheduler returned empty or None timesteps.")
        return scheduler, timesteps, effective_num_steps # Return the final scheduler, timesteps, and steps count
    except Exception as e:
        logger.error(f"Run {run_uuid}: Failed to set timesteps ({effective_num_steps} steps) on scheduler '{scheduler.__class__.__name__}': {e}", exc_info=True)
        return None # Signal error


# --- Modified run_text_to_image ---
async def run_text_to_image(
    request: Text2ImageRequest,
    parent_run_uuid: Optional[str] = None
) -> Optional[Text2ImageResponse]:
    """Runs the text-to-image pipeline, logs to DB, saves images with metadata."""
    start_time = time.time()
    run_uuid = parent_run_uuid or str(uuid.uuid4())
    logger.info(f"Starting text-to-image run {run_uuid}: {request.prompt[:50]}...")
    db_run_id: Optional[int] = None

    # --- DB Logging Start ---
    if not parent_run_uuid:
        try:
            async with get_db_session() as db:
                # Store effective parameters including potentially defaulted scheduler
                params_to_log = request.model_dump(mode='json')
                # Note: default scheduler name isn't known until pipeline load
                db_run = ExperimentRun(
                    run_uuid=run_uuid, run_type="text_to_image",
                    start_time=datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc),
                    status="running", base_parameters=params_to_log
                )
                db.add(db_run)
                await db.commit()
                await db.refresh(db_run)
                db_run_id = db_run.id
                logger.info(f"Created ExperimentRun record ID: {db_run_id} for run {run_uuid}")
        except Exception as e:
            logger.error(f"Failed to create initial DB record for run {run_uuid}: {e}", exc_info=True)
            # Proceed without DB logging? For now, yes, but log error. db_run_id remains None.

    # --- Core Generation Logic ---
    images_pil: List[Image.Image] = []
    final_params_dict = request.model_dump(exclude_unset=True) # Base params for saving later
    actual_scheduler_name = "Unknown" # Placeholder

    try:
        # --- Setup Device & Dtype ---
        # TODO: Make pipeline_type dynamic? Needs mapping from model_id potentially.
        pipeline_type = "StableDiffusionPipeline"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        logger.debug(f"Run {run_uuid}: Using device='{device.type}' dtype='{dtype}'")

        # --- Load Pipeline ---
        pipeline: Optional[DiffusionPipeline] = await load_diffusion_pipeline(
            model_path_or_id=request.model_id, pipeline_type=pipeline_type,
            device=device_str, dtype=dtype # vae_id=request.vae_id
        )
        if not pipeline:
            error_msg = f"Failed to load pipeline for model: {request.model_id}"
            logger.error(f"Run {run_uuid}: {error_msg}")
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None
        pipeline.to(device) # Ensure on correct device

        # --- Setup Scheduler and Timesteps ---
        setup_result = setup_scheduler_and_timesteps(
             pipeline, request.scheduler_id, request.custom_sigmas,
             request.num_inference_steps, device, run_uuid
        )
        if setup_result is None:
             error_msg = "Failed to setup scheduler or timesteps."
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
             return None # Error already logged by helper

        scheduler, timesteps, effective_num_steps = setup_result
        actual_scheduler_name = scheduler.__class__.__name__
        final_params_dict['num_inference_steps'] = effective_num_steps # Update final params
        final_params_dict['scheduler_id'] = actual_scheduler_name # Log the actual scheduler used

        # --- Prepare Generator ---
        generator = None
        effective_seed = request.seed
        if effective_seed is None: effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            try: effective_seed = int(effective_seed)
            except (ValueError, TypeError): effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(effective_seed)
        final_params_dict['seed'] = effective_seed # Use the resolved seed

        # --- Latent & Embedding Prep ---
        height, width = request.height, request.width
        if height <= 0 or width <= 0 or height % 8 != 0 or width % 8 != 0 :
             error_msg = f"Invalid dimensions W={width}, H={height}."
             logger.error(f"Run {run_uuid}: {error_msg}")
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
             return None

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            request.num_outputs, num_channels_latents, height, width, dtype, device, generator
        )
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=request.prompt, device=device, num_images_per_prompt=request.num_outputs,
            do_classifier_free_guidance=request.guidance_scale > 1.0,
            negative_prompt=request.negative_prompt
        )
        if request.guidance_scale > 1.0:
            prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            prompt_embeds_cfg = prompt_embeds

        # --- Inference Loop ---
        logger.info(f"Run {run_uuid}: Running inference loop with {len(timesteps)} steps using {actual_scheduler_name}...")
        # Use pipeline.progress_bar for default tqdm
        for i, t in enumerate(pipeline.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if request.guidance_scale > 1.0 else latents
            # Use the correct scheduler instance
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = pipeline.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds_cfg.to(device)
                ).sample

            if request.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + request.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Use the correct scheduler instance
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # --- VAE Decode ---
        latents_scaled = 1 / pipeline.vae.config.get("scaling_factor", 0.18215) * latents
        with torch.no_grad():
             image = pipeline.vae.decode(latents_scaled.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        images_pil = pipeline.numpy_to_pil(image)
        logger.info(f"Run {run_uuid}: Inference completed. Generated {len(images_pil)} image(s).")

    except Exception as e:
        error_during_run = f"{type(e).__name__}: {e}"
        logger.error(f"Run {run_uuid}: Error during diffusion core logic: {e}", exc_info=True)
        if db_run_id: await update_run_status(db_run_id, "failed", error=error_during_run)
        return None

    # --- Process, Save, and Log Outputs ---
    results: List[ImageResult] = []
    output_base_path = get_output_base_path() / run_uuid
    output_base_path.mkdir(parents=True, exist_ok=True)
    output_records_to_add = []

    # Clean up final_params_dict for logging (remove None custom_sigmas if not used)
    if 'custom_sigmas' in final_params_dict and not request.custom_sigmas:
        del final_params_dict['custom_sigmas']

    for i, img_pil in enumerate(images_pil):
        current_seed = effective_seed + i
        filename = f"output_{i:03d}_seed{current_seed}.png"
        file_path = output_base_path / filename
        file_path_str = str(file_path.resolve())

        output_params = final_params_dict.copy()
        output_params['seed'] = current_seed # Set specific seed for this output

        # --- Metadata Injection ---
        pnginfo = None
        try:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            # Store the final, resolved parameters used for *this* image
            pnginfo.add_text("grad_parameters", json.dumps(output_params))
        except Exception as meta_e:
             logger.warning(f"Could not prepare PNG metadata: {meta_e}")
             pnginfo = None

        # --- Save Image ---
        try:
            img_pil.save(file_path_str, "PNG", pnginfo=pnginfo)
            logger.debug(f"Saved image {i} to {file_path_str} with metadata.")
        except Exception as e:
            logger.error(f"Failed to save image {file_path_str}: {e}", exc_info=True)
            continue # Skip logging/returning this image

        # --- Base64 Encode for API ---
        img_base64 = None
        try:
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {i} to base64: {e}")

        # --- Append API Result ---
        if img_base64:
            results.append(ImageResult(
                image_base64=img_base64,
                seed=current_seed,
                parameters=output_params # Store resolved params
            ))

        # --- Prepare DB Output Record ---
        if db_run_id:
            output_record = GeneratedOutput(
                run_id=db_run_id,
                output_type="image",
                file_path=file_path_str,
                parameters=output_params, # Log resolved params
                creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
                # metadata= could store extra info like timing breakdown?
            )
            output_records_to_add.append(output_record)

    # --- Batch Log Outputs to DB ---
    if output_records_to_add:
        try:
             async with get_db_session() as db:
                 db.add_all(output_records_to_add)
                 await db.commit()
                 logger.info(f"Logged {len(output_records_to_add)} output records to DB for run {run_uuid}")
        except Exception as e:
             logger.error(f"Failed to log output records to DB for run {run_uuid}: {e}", exc_info=True)

    # --- Finalize Run ---
    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Run {run_uuid}: Text-to-image request processed in {performance_ms:.2f} ms.")
    if db_run_id: await update_run_status(db_run_id, "completed", end_time=end_time)

    return Text2ImageResponse(
        images=results,
        request_params=request, # Echo original request
        performance_ms=performance_ms
    )


# --- NEW: run_image_to_image ---
async def run_image_to_image(
    request: ImageToImageRequest,
    parent_run_uuid: Optional[str] = None
) -> Optional[ImageToImageResponse]:
    """Runs the image-to-image pipeline, logs to DB, saves images with metadata."""
    start_time = time.time()
    run_uuid = parent_run_uuid or str(uuid.uuid4())
    logger.info(f"Starting image-to-image run {run_uuid}: {request.prompt[:50]}...")
    db_run_id: Optional[int] = None

    # --- DB Logging Start ---
    if not parent_run_uuid:
        try:
            async with get_db_session() as db:
                params_to_log = request.model_dump(mode='json', exclude={'image_base64'}) # Exclude large image data
                params_to_log['image_provided'] = True # Indicate image was input
                db_run = ExperimentRun(
                    run_uuid=run_uuid, run_type="image_to_image",
                    start_time=datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc),
                    status="running", base_parameters=params_to_log
                )
                db.add(db_run)
                await db.commit()
                await db.refresh(db_run)
                db_run_id = db_run.id
                logger.info(f"Created ExperimentRun record ID: {db_run_id} for run {run_uuid}")
        except Exception as e:
            logger.error(f"Failed to create initial DB record for run {run_uuid}: {e}", exc_info=True)

    # --- Core Generation Logic ---
    images_pil: List[Image.Image] = []
    final_params_dict = request.model_dump(exclude_unset=True, exclude={'image_base64'}) # Base params for saving
    actual_scheduler_name = "Unknown"

    try:
        # --- Setup Device & Dtype ---
        # Check if a specific Img2Img pipeline is needed? Assume standard works for now.
        pipeline_type = "StableDiffusionPipeline" # Or StableDiffusionImg2ImgPipeline if needed
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        logger.debug(f"Run {run_uuid}: Using device='{device.type}' dtype='{dtype}'")

        # --- Load Pipeline ---
        pipeline: Optional[DiffusionPipeline] = await load_diffusion_pipeline(
            model_path_or_id=request.model_id, pipeline_type=pipeline_type,
            device=device_str, dtype=dtype # vae_id=request.vae_id
        )
        if not pipeline:
            error_msg = f"Failed to load pipeline for model: {request.model_id}"
            logger.error(f"Run {run_uuid}: {error_msg}")
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None
        pipeline.to(device)

        # --- Prepare Input Image ---
        init_image_pil = prepare_init_image(request.image_base64, request.width, request.height)
        if init_image_pil is None:
            error_msg = "Failed to load or process initial image."
            logger.error(f"Run {run_uuid}: {error_msg}")
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None

        # --- Setup Scheduler and Timesteps ---
        setup_result = setup_scheduler_and_timesteps(
             pipeline, request.scheduler_id, request.custom_sigmas,
             request.num_inference_steps, device, run_uuid
        )
        if setup_result is None:
             error_msg = "Failed to setup scheduler or timesteps."
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
             return None # Error already logged by helper

        scheduler, timesteps, effective_num_steps = setup_result
        actual_scheduler_name = scheduler.__class__.__name__
        final_params_dict['num_inference_steps'] = effective_num_steps # Update final params
        final_params_dict['scheduler_id'] = actual_scheduler_name # Log actual scheduler

        # --- Prepare Generator ---
        generator = None
        effective_seed = request.seed
        if effective_seed is None: effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
             try: effective_seed = int(effective_seed)
             except (ValueError, TypeError): effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(effective_seed)
        final_params_dict['seed'] = effective_seed # Use the resolved seed

        # --- Run Inference ---
        # The standard pipeline call often handles Img2Img directly.
        # Check diffusers docs for the specific pipeline class version being used.
        # Common arguments: prompt, image, strength, guidance_scale, num_inference_steps, generator, etc.
        # `timesteps` might not be passed directly if using the high-level call.
        logger.info(f"Run {run_uuid}: Running Img2Img inference with {effective_num_steps} steps using {actual_scheduler_name}...")

        # Ensure the pipeline object actually has the __call__ method configured for img2img
        # This might depend on the pipeline_type loaded ('StableDiffusionImg2ImgPipeline' is explicit)
        # If using generic 'StableDiffusionPipeline', it might work, but verify arguments.
        if not hasattr(pipeline, "__call__"):
            raise TypeError(f"Loaded pipeline of type {type(pipeline)} is not callable.")

        # Prepare args for the pipeline call
        pipeline_args = {
            "prompt": request.prompt,
            "image": init_image_pil,
            "strength": request.strength,
            "num_inference_steps": effective_num_steps, # Pass effective steps
            "guidance_scale": request.guidance_scale,
            "negative_prompt": request.negative_prompt,
            "num_images_per_prompt": request.num_outputs,
            "generator": generator,
            # 'output_type': 'pil' # Default is usually PIL
        }

        # Call the pipeline
        with torch.no_grad():
            pipeline_output = pipeline(**pipeline_args)
            images_pil = pipeline_output.images # Access the 'images' attribute of the output

        logger.info(f"Run {run_uuid}: Inference completed. Generated {len(images_pil)} image(s).")

    except Exception as e:
        error_during_run = f"{type(e).__name__}: {e}"
        logger.error(f"Run {run_uuid}: Error during diffusion core logic: {e}", exc_info=True)
        if db_run_id: await update_run_status(db_run_id, "failed", error=error_during_run)
        return None

    # --- Process, Save, and Log Outputs (Mostly same as T2I) ---
    results: List[ImageResult] = []
    output_base_path = get_output_base_path() / run_uuid
    output_base_path.mkdir(parents=True, exist_ok=True)
    output_records_to_add = []

    # Clean up final_params_dict for logging
    if 'custom_sigmas' in final_params_dict and not request.custom_sigmas:
        del final_params_dict['custom_sigmas']
    # Add strength to logged params
    final_params_dict['strength'] = request.strength

    for i, img_pil in enumerate(images_pil):
        current_seed = effective_seed + i # Simple seed increment for batch
        filename = f"output_{i:03d}_seed{current_seed}.png"
        file_path = output_base_path / filename
        file_path_str = str(file_path.resolve())

        output_params = final_params_dict.copy()
        output_params['seed'] = current_seed

        # Metadata Injection
        pnginfo = None
        try:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            pnginfo.add_text("grad_parameters", json.dumps(output_params))
        except Exception as meta_e:
             logger.warning(f"Could not prepare PNG metadata: {meta_e}")
             pnginfo = None

        # Save Image
        try:
            img_pil.save(file_path_str, "PNG", pnginfo=pnginfo)
            logger.debug(f"Saved image {i} to {file_path_str} with metadata.")
        except Exception as e:
            logger.error(f"Failed to save image {file_path_str}: {e}", exc_info=True)
            continue

        # Base64 Encode
        img_base64 = None
        try:
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {i} to base64: {e}")

        # Append API Result
        if img_base64:
            results.append(ImageResult(
                image_base64=img_base64,
                seed=current_seed,
                parameters=output_params
            ))

        # Prepare DB Output Record
        if db_run_id:
            output_record = GeneratedOutput(
                run_id=db_run_id, output_type="image", file_path=file_path_str,
                parameters=output_params,
                creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
            )
            output_records_to_add.append(output_record)

    # Batch Log Outputs to DB
    if output_records_to_add:
        try:
             async with get_db_session() as db:
                 db.add_all(output_records_to_add)
                 await db.commit()
                 logger.info(f"Logged {len(output_records_to_add)} output records to DB for run {run_uuid}")
        except Exception as e:
             logger.error(f"Failed to log output records to DB for run {run_uuid}: {e}", exc_info=True)

    # Finalize Run
    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Run {run_uuid}: Image-to-image request processed in {performance_ms:.2f} ms.")
    if db_run_id: await update_run_status(db_run_id, "completed", end_time=end_time)

    return ImageToImageResponse(
        images=results,
        request_params=request, # Echo original request
        performance_ms=performance_ms
    )