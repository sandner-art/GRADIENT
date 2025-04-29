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
from typing import List, Optional, Tuple, Union # Added Union

# DB Imports
from sqlalchemy.ext.asyncio import AsyncSession
from gradient.database.session import get_db_session, update_run_status
from gradient.database.models import ExperimentRun, GeneratedOutput

# AI Imports
from .loading import load_diffusion_pipeline
# Import specific request/response types used
from .models import (
    Text2ImageRequest, Text2ImageResponse,
    ImageToImageRequest, ImageToImageResponse,
    ImageResult
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
# Import specific pipeline types needed
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline, # Explicitly import
    DiffusionPipeline,
    AutoencoderKL
)
# Import specific scheduler types for type hinting or potential checks (optional)
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler

logger = logging.getLogger(__name__)

# --- Output Path ---
def get_output_base_path() -> Path:
     # Consider moving this to config/paths later
     output_path = Path("./experiments/")
     output_path.mkdir(parents=True, exist_ok=True)
     return output_path

# --- Helper Function to Prepare Image (for Img2Img) ---
def prepare_init_image(image_base64: str, width: int, height: int) -> Optional[Image.Image]:
    """Decodes, validates, and resizes the initial image."""
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        logger.info(f"Input image loaded: mode={image.mode}, size={image.size}")
        # Resize to target dimensions. LANCZOS is high quality.
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        logger.info(f"Input image resized to: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to prepare initial image: {e}", exc_info=True) # Log errors
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
            scheduler_module = __import__('diffusers', fromlist=[request_scheduler_id])
            scheduler_class = getattr(scheduler_module, request_scheduler_id)

            if issubclass(scheduler_class, SchedulerMixin):
                # Create new scheduler from the pipeline's current scheduler config
                scheduler = scheduler_class.from_config(pipeline.scheduler.config)
                pipeline.scheduler = scheduler # IMPORTANT: Update the pipeline instance
                logger.info(f"Run {run_uuid}: Successfully switched scheduler to {request_scheduler_id}")
            else:
                logger.warning(f"Run {run_uuid}: '{request_scheduler_id}' is not a valid Scheduler class. Using default '{original_scheduler_name}'.")
                scheduler = pipeline.scheduler # Ensure we use the original default
        except (ImportError, AttributeError) as e:
            logger.warning(f"Run {run_uuid}: Could not find or import scheduler '{request_scheduler_id}'. Using default '{original_scheduler_name}'. Error: {e}")
            scheduler = pipeline.scheduler
        except Exception as e:
            logger.error(f"Run {run_uuid}: Unexpected error loading scheduler '{request_scheduler_id}'. Using default '{original_scheduler_name}'. Error: {e}", exc_info=True)
            scheduler = pipeline.scheduler
    else:
        logger.info(f"Run {run_uuid}: Using default scheduler: {scheduler.__class__.__name__}")

    # --- 2. Set Timesteps based on custom sigmas or num_steps ---
    timesteps = None
    if request_custom_sigmas is not None:
        if len(request_custom_sigmas) < 2:
            logger.error(f"Run {run_uuid}: Custom sigmas list must contain at least 2 values.")
            return None
        calculated_steps = len(request_custom_sigmas) - 1
        if calculated_steps <= 0:
             logger.error(f"Run {run_uuid}: Custom sigmas list length resulted in non-positive steps.")
             return None
        effective_num_steps = calculated_steps
        logger.info(f"Run {run_uuid}: Using custom sigma list length to set num_inference_steps={effective_num_steps}. Actual sigma values NOT YET USED by standard schedulers.")
    else:
        if effective_num_steps <= 0:
            logger.error(f"Run {run_uuid}: num_inference_steps must be positive, but got {effective_num_steps}.")
            return None

    # --- 3. Set Timesteps on the *final* scheduler ---
    try:
        # For Img2Img, strength modifies *which* timesteps are used internally by the pipeline call.
        # We still set the total number of steps here for the scheduler instance itself.
        scheduler.set_timesteps(effective_num_steps, device=device)
        timesteps = scheduler.timesteps # Get the full array of potential timesteps
        logger.info(f"Run {run_uuid}: Timesteps generated by scheduler '{scheduler.__class__.__name__}' for max {effective_num_steps} steps. Count: {len(timesteps)}.")
        if timesteps is None or len(timesteps) == 0:
             raise ValueError("Scheduler returned empty or None timesteps.")
        return scheduler, timesteps, effective_num_steps
    except Exception as e:
        logger.error(f"Run {run_uuid}: Failed to set timesteps ({effective_num_steps} steps) on scheduler '{scheduler.__class__.__name__}': {e}", exc_info=True)
        return None

# --- Helper Function for Scheduled CFG (Placeholder/Future) ---
def get_scheduled_guidance_scale(
    step_index: int,
    total_steps: int,
    schedule_definition: Union[float, str, List[Tuple[int, float]]]
) -> float:
    """Calculates the guidance scale for the current step based on a schedule."""
    # TODO: Implement actual scheduling logic here based on definition type
    if isinstance(schedule_definition, (float, int)):
        return float(schedule_definition)
    # Add logic for string patterns or lists later
    logger.warning(f"Guidance scheduling not fully implemented for type {type(schedule_definition)}. Using fixed value 7.5 as fallback.")
    return 7.5 # Fallback


# --- run_text_to_image ---
async def run_text_to_image(
    request: Text2ImageRequest,
    parent_run_uuid: Optional[str] = None
) -> Optional[Text2ImageResponse]:
    """Runs the text-to-image pipeline, logs to DB, saves images with metadata."""
    start_time = time.time()
    run_uuid = parent_run_uuid or str(uuid.uuid4())
    logger.info(f"Starting text-to-image run {run_uuid} Prompt: '{request.prompt[:50]}...'")
    db_run_id: Optional[int] = None

    # --- DB Logging Start ---
    if not parent_run_uuid:
        try:
            async with get_db_session() as db:
                params_to_log = request.model_dump(mode='json') # Log initial request
                db_run = ExperimentRun(
                    run_uuid=run_uuid, run_type="text_to_image",
                    start_time=datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc),
                    status="running", base_parameters=params_to_log
                )
                db.add(db_run)
                await db.commit()
                await db.refresh(db_run)
                db_run_id = db_run.id
                logger.info(f"Run {run_uuid}: Created ExperimentRun record ID: {db_run_id}")
        except Exception as e:
            logger.error(f"Run {run_uuid}: Failed to create initial DB record: {e}", exc_info=True)

    # --- Core Generation Logic ---
    images_pil: List[Image.Image] = []
    final_params_dict = request.model_dump(exclude_unset=True) # Base params for saving later
    actual_scheduler_name = "Unknown"

    try:
        # --- Setup Device & Dtype ---
        pipeline_type = "StableDiffusionPipeline" # Standard T2I pipeline
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        logger.debug(f"Run {run_uuid}: Using device='{device.type}' dtype='{dtype}' Pipeline Type: {pipeline_type}")

        # --- Load Pipeline ---
        pipeline: Optional[DiffusionPipeline] = await load_diffusion_pipeline(
            model_path_or_id=request.model_id, pipeline_type=pipeline_type,
            device=device_str, dtype=dtype # vae_id=request.vae_id
        )
        if not pipeline:
            error_msg = f"Failed to load pipeline for model: {request.model_id}"
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None
        pipeline.to(device)
        logger.info(f"Run {run_uuid}: Successfully loaded pipeline object of type: {type(pipeline).__name__}")


        # --- Setup Scheduler and Timesteps ---
        setup_result = setup_scheduler_and_timesteps(
             pipeline, request.scheduler_id, request.custom_sigmas,
             request.num_inference_steps, device, run_uuid
        )
        if setup_result is None:
             error_msg = "Failed to setup scheduler or timesteps."
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
             return None
        scheduler, timesteps, effective_num_steps = setup_result
        actual_scheduler_name = scheduler.__class__.__name__
        final_params_dict['num_inference_steps'] = effective_num_steps
        final_params_dict['scheduler_id'] = actual_scheduler_name

        # --- Prepare Generator ---
        generator = None
        effective_seed = request.seed
        if effective_seed is None: effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            try: effective_seed = int(effective_seed)
            except (ValueError, TypeError): effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(effective_seed)
        final_params_dict['seed'] = effective_seed

        # --- Latent & Embedding Prep ---
        height, width = request.height, request.width
        if height <= 0 or width <= 0 or height % 8 != 0 or width % 8 != 0 :
             error_msg = f"Invalid dimensions W={width}, H={height}."
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
             return None

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            request.num_outputs, num_channels_latents, height, width, dtype, device, generator
        )
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=request.prompt, device=device, num_images_per_prompt=request.num_outputs,
            do_classifier_free_guidance=True, # Always prepare for CFG, decide use in loop
            negative_prompt=request.negative_prompt
        )
        if negative_prompt_embeds is None and isinstance(request.guidance_scale, (float, int)) and request.guidance_scale > 1.0:
            logger.warning(f"Run {run_uuid}: CFG scale > 1 requested but negative prompt embeddings are None. Disabling CFG.")

        # --- Inference Loop ---
        logger.info(f"Run {run_uuid}: Running T2I inference loop with {len(timesteps)} steps using {actual_scheduler_name}...")
        # Use pipeline.progress_bar for default tqdm
        for i, t in enumerate(pipeline.progress_bar(timesteps)):
            # Prepare input for UNet
            latent_model_input = torch.cat([latents] * 2) if isinstance(request.guidance_scale, (float, int)) and request.guidance_scale > 1.0 and negative_prompt_embeds is not None else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Concatenate embeddings for CFG
            if isinstance(request.guidance_scale, (float, int)) and request.guidance_scale > 1.0 and negative_prompt_embeds is not None:
                 current_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                 current_prompt_embeds = prompt_embeds

            # Predict noise
            with torch.no_grad():
                noise_pred = pipeline.unet(
                    latent_model_input, t, encoder_hidden_states=current_prompt_embeds.to(device, dtype=dtype) # Ensure dtype match
                ).sample

            # Perform guidance
            if isinstance(request.guidance_scale, (float, int)) and request.guidance_scale > 1.0 and negative_prompt_embeds is not None:
                # Calculate or get scheduled guidance scale (using placeholder helper for now)
                current_guidance_scale = get_scheduled_guidance_scale(
                    step_index=i, total_steps=len(timesteps), schedule_definition=request.guidance_scale
                )
                if current_guidance_scale > 1.0: # Only apply if > 1
                     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                     noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)
                # else: use original noise_pred (effectively no guidance for this step)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # --- VAE Decode ---
        # Use getattr for safety, provide default SD 1.5 factor
        vae_scaling_factor = getattr(pipeline.vae.config, 'scaling_factor', 0.18215)
        latents_scaled = 1 / vae_scaling_factor * latents
        with torch.no_grad():
             image = pipeline.vae.decode(latents_scaled.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        images_pil = pipeline.numpy_to_pil(image)
        if not images_pil: raise ValueError("VAE decoding completed but produced no images.")
        logger.info(f"Run {run_uuid}: T2I Inference completed. Generated {len(images_pil)} image(s).")

    except Exception as e:
        error_during_run = f"{type(e).__name__}: {e}"
        logger.error(f"Run {run_uuid}: Error during T2I core logic: {e}", exc_info=True)
        if db_run_id: await update_run_status(db_run_id, "failed", error=error_during_run)
        return None

    # --- Process, Save, Log Outputs ---
    results: List[ImageResult] = []
    output_base_path = get_output_base_path() / run_uuid
    output_base_path.mkdir(parents=True, exist_ok=True)
    output_records_to_add = []

    # Clean up final params for logging
    if 'custom_sigmas' in final_params_dict and not request.custom_sigmas:
        del final_params_dict['custom_sigmas']
    # Ensure guidance_scale logged is the original request (string/list or float)
    final_params_dict['guidance_scale'] = request.guidance_scale

    for i, img_pil in enumerate(images_pil):
        current_seed = effective_seed + i
        filename = f"t2i_output_{i:03d}_seed{current_seed}.png" # Slightly more specific filename
        file_path = output_base_path / filename
        file_path_str = str(file_path.resolve())

        output_params = final_params_dict.copy()
        output_params['seed'] = current_seed # Set specific seed

        # Metadata Injection
        pnginfo = None
        try:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            pnginfo.add_text("grad_parameters", json.dumps(output_params))
        except Exception as meta_e:
             logger.warning(f"Run {run_uuid}: Could not prepare PNG metadata: {meta_e}")
             pnginfo = None

        # Save Image
        try:
            img_pil.save(file_path_str, "PNG", pnginfo=pnginfo)
            logger.debug(f"Run {run_uuid}: Saved image {i} to {file_path_str}")
        except Exception as e:
            logger.error(f"Run {run_uuid}: Failed to save image {file_path_str}: {e}", exc_info=True)
            continue # Skip logging/returning this image

        # Base64 Encode
        img_base64 = None
        try:
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Run {run_uuid}: Error encoding image {i} to base64: {e}")

        # Append API Result
        if img_base64:
            results.append(ImageResult(image_base64=img_base64, seed=current_seed, parameters=output_params))

        # Prepare DB Output Record
        if db_run_id:
            output_record = GeneratedOutput(
                run_id=db_run_id, output_type="image", file_path=file_path_str,
                parameters=output_params, creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
            )
            output_records_to_add.append(output_record)

    # Batch Log Outputs to DB
    if output_records_to_add:
        try:
             async with get_db_session() as db:
                 db.add_all(output_records_to_add)
                 await db.commit()
                 logger.info(f"Run {run_uuid}: Logged {len(output_records_to_add)} output records to DB.")
        except Exception as e:
             logger.error(f"Run {run_uuid}: Failed to log output records to DB: {e}", exc_info=True)

    # Finalize Run
    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Run {run_uuid}: Text-to-image request processed in {performance_ms:.2f} ms.")
    if db_run_id: await update_run_status(db_run_id, "completed", end_time=end_time)

    return Text2ImageResponse(images=results, request_params=request, performance_ms=performance_ms)


# --- run_image_to_image (with Debugging Logs) ---
async def run_image_to_image(
    request: ImageToImageRequest,
    parent_run_uuid: Optional[str] = None
) -> Optional[ImageToImageResponse]:
    """Runs the image-to-image pipeline, logs to DB, saves images with metadata."""
    start_time = time.time()
    run_uuid = parent_run_uuid or str(uuid.uuid4())
    logger.info(f"Starting image-to-image run {run_uuid} Prompt: '{request.prompt[:50]}...' Strength: {request.strength}")
    db_run_id: Optional[int] = None

    # --- DB Logging Start ---
    if not parent_run_uuid:
        try:
            async with get_db_session() as db:
                # Exclude large base64 image from DB log, indicate image was provided
                params_to_log = request.model_dump(mode='json', exclude={'image_base64'})
                params_to_log['image_provided'] = True
                db_run = ExperimentRun(
                    run_uuid=run_uuid, run_type="image_to_image",
                    start_time=datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc),
                    status="running", base_parameters=params_to_log
                )
                db.add(db_run)
                await db.commit()
                await db.refresh(db_run)
                db_run_id = db_run.id
                logger.info(f"Run {run_uuid}: Created ExperimentRun record ID: {db_run_id}")
        except Exception as e:
            logger.error(f"Run {run_uuid}: Failed to create initial DB record: {e}", exc_info=True)

    # --- Core Generation Logic ---
    images_pil: List[Image.Image] = []
    final_params_dict = request.model_dump(exclude_unset=True, exclude={'image_base64'})
    actual_scheduler_name = "Unknown"

    try:
        # --- Setup Device & Dtype ---
        pipeline_type = "StableDiffusionImg2ImgPipeline" # Use specific I2I pipeline
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        logger.debug(f"Run {run_uuid}: Using device='{device.type}' dtype='{dtype}' Pipeline Type: {pipeline_type}")

        # --- Load Pipeline ---
        pipeline: Optional[DiffusionPipeline] = await load_diffusion_pipeline(
            model_path_or_id=request.model_id, pipeline_type=pipeline_type,
            device=device_str, dtype=dtype # vae_id=request.vae_id
        )
        if not pipeline:
            error_msg = f"Failed to load pipeline for model: {request.model_id}"
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None
        pipeline.to(device)
        logger.info(f"Run {run_uuid}: Successfully loaded pipeline object of type: {type(pipeline).__name__}")

        # --- Prepare Input Image ---
        logger.info(f"Run {run_uuid}: Preparing initial image...")
        init_image_pil = prepare_init_image(request.image_base64, request.width, request.height)
        if init_image_pil is None:
            error_msg = "Failed to load or process initial image."
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None

        # --- Setup Scheduler and Timesteps ---
        # Note: The pipeline() call for Img2Img often determines the *actual* number of steps
        # and which timesteps to use based on 'strength'. This setup prepares the scheduler instance.
        setup_result = setup_scheduler_and_timesteps(
             pipeline, request.scheduler_id, None, # Ignore custom_sigmas for standard I2I for now
             request.num_inference_steps, device, run_uuid
        )
        if setup_result is None:
             error_msg = "Failed to setup scheduler or timesteps."
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
             return None
        scheduler, timesteps, effective_num_steps = setup_result # effective_num_steps is the *max* requested
        actual_scheduler_name = scheduler.__class__.__name__
        # Log requested steps, actual steps might differ due to strength
        final_params_dict['num_inference_steps'] = effective_num_steps
        final_params_dict['scheduler_id'] = actual_scheduler_name

        # --- Prepare Generator ---
        generator = None
        effective_seed = request.seed
        if effective_seed is None: effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            try: effective_seed = int(effective_seed)
            except (ValueError, TypeError): effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(effective_seed)
        final_params_dict['seed'] = effective_seed

        # --- Run Inference ---
        # The number of steps actually run is often int(num_inference_steps * strength)
        logger.info(f"Run {run_uuid}: Running Img2Img inference with strength={request.strength}, max_steps={effective_num_steps}, scheduler={actual_scheduler_name}...")

        if not isinstance(pipeline, StableDiffusionImg2ImgPipeline):
             logger.warning(f"Run {run_uuid}: Loaded pipeline is type {type(pipeline).__name__}, not StableDiffusionImg2ImgPipeline. Strength behavior might differ.")

        # Prepare args for the pipeline call
        pipeline_args = {
            "prompt": request.prompt,
            "image": init_image_pil,
            "strength": request.strength,
            "num_inference_steps": effective_num_steps, # Pipeline uses this *with* strength
            "guidance_scale": request.guidance_scale, # TODO: Allow scheduled guidance here too?
            "negative_prompt": request.negative_prompt,
            "num_images_per_prompt": request.num_outputs,
            "generator": generator,
        }
        # --- DETAILED LOGGING ---
        logger.debug(f"Run {run_uuid}: Calling {type(pipeline).__name__}.__call__ with args: "
                     f"strength={pipeline_args['strength']}, "
                     f"num_inference_steps={pipeline_args['num_inference_steps']}, "
                     f"guidance_scale={pipeline_args['guidance_scale']}, "
                     f"num_outputs={pipeline_args['num_images_per_prompt']}, "
                     f"prompt='{str(pipeline_args['prompt'])[:50]}...', "
                     f"image=<PIL Image {init_image_pil.size}>, "
                     f"negative_prompt='{str(pipeline_args['negative_prompt'])[:50]}...', "
                     f"generator=<torch Generator>")

        # Call the pipeline
        with torch.no_grad():
            pipeline_output = pipeline(**pipeline_args)
            images_pil = pipeline_output.images

        if not images_pil: raise ValueError("Pipeline call completed but returned no images.")
        logger.info(f"Run {run_uuid}: Img2Img Inference completed. Generated {len(images_pil)} image(s).")

    except Exception as e:
        error_during_run = f"{type(e).__name__}: {e}"
        logger.error(f"Run {run_uuid}: Error during Img2Img core logic: {e}", exc_info=True)
        if db_run_id: await update_run_status(db_run_id, "failed", error=error_during_run)
        return None

    # --- Process, Save, Log Outputs ---
    results: List[ImageResult] = []
    output_base_path = get_output_base_path() / run_uuid
    output_base_path.mkdir(parents=True, exist_ok=True)
    output_records_to_add = []

    # Clean up final params for logging
    if 'custom_sigmas' in final_params_dict: del final_params_dict['custom_sigmas'] # Not used here
    final_params_dict['strength'] = request.strength # Ensure strength is logged

    for i, img_pil in enumerate(images_pil):
        current_seed = effective_seed + i
        filename = f"i2i_output_{i:03d}_seed{current_seed}.png" # More specific filename
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
             logger.warning(f"Run {run_uuid}: Could not prepare PNG metadata: {meta_e}")
             pnginfo = None

        # Save Image
        try:
            img_pil.save(file_path_str, "PNG", pnginfo=pnginfo)
            logger.debug(f"Run {run_uuid}: Saved image {i} to {file_path_str}")
        except Exception as e:
            logger.error(f"Run {run_uuid}: Failed to save image {file_path_str}: {e}", exc_info=True)
            continue

        # Base64 Encode
        img_base64 = None
        try:
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Run {run_uuid}: Error encoding image {i} to base64: {e}")

        # Append API Result
        if img_base64:
            results.append(ImageResult(image_base64=img_base64, seed=current_seed, parameters=output_params))

        # Prepare DB Output Record
        if db_run_id:
            output_record = GeneratedOutput(
                run_id=db_run_id, output_type="image", file_path=file_path_str,
                parameters=output_params, creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
            )
            output_records_to_add.append(output_record)

    # Batch Log Outputs to DB
    if output_records_to_add:
        try:
             async with get_db_session() as db:
                 db.add_all(output_records_to_add)
                 await db.commit()
                 logger.info(f"Run {run_uuid}: Logged {len(output_records_to_add)} output records to DB.")
        except Exception as e:
             logger.error(f"Run {run_uuid}: Failed to log output records to DB: {e}", exc_info=True)

    # Finalize Run
    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Run {run_uuid}: Image-to-image request processed in {performance_ms:.2f} ms.")
    if db_run_id: await update_run_status(db_run_id, "completed", end_time=end_time)

    return ImageToImageResponse(images=results, request_params=request, performance_ms=performance_ms)