# src/gradient/ai/diffusion/pipelines.py

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
from typing import List, Optional, Tuple, Union

# DB Imports
from sqlalchemy.ext.asyncio import AsyncSession
from gradient.database.session import get_db_session, update_run_status
from gradient.database.models import ExperimentRun, GeneratedOutput

# AI Imports
from .loading import load_diffusion_pipeline # Assuming loading helpers are now separate
# Import specific request/response types used
from .models import (
    Text2ImageRequest, Text2ImageResponse,
    ImageToImageRequest, ImageToImageResponse,
    ImageResult,
    LoRAParameter # Import LoRAParameter Pydantic model
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
# Import specific pipeline types needed
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline, # Explicitly import
    DiffusionPipeline,
    AutoencoderKL
)
# Import specific scheduler types (optional, for type hinting etc.)
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler

logger = logging.getLogger(__name__)

# --- Output Path ---
def get_output_base_path() -> Path:
     # TODO: Read from config
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
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        logger.info(f"Input image resized to: {image.size}")
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
    run_uuid: str
) -> Optional[Tuple[SchedulerMixin, torch.Tensor, int]]:
    """Loads scheduler if requested, sets timesteps, returns scheduler, timesteps, and effective steps."""
    scheduler = pipeline.scheduler
    effective_num_steps = request_num_steps

    # --- 1. Override Scheduler ---
    if request_scheduler_id:
        logger.info(f"Run {run_uuid}: Attempting to load requested scheduler: {request_scheduler_id}")
        original_scheduler_name = scheduler.__class__.__name__
        try:
            scheduler_module = __import__('diffusers', fromlist=[request_scheduler_id])
            scheduler_class = getattr(scheduler_module, request_scheduler_id)
            if issubclass(scheduler_class, SchedulerMixin):
                scheduler = scheduler_class.from_config(pipeline.scheduler.config)
                pipeline.scheduler = scheduler
                logger.info(f"Run {run_uuid}: Successfully switched scheduler to {request_scheduler_id}")
            else:
                logger.warning(f"Run {run_uuid}: '{request_scheduler_id}' not valid Scheduler. Using default '{original_scheduler_name}'.")
                scheduler = pipeline.scheduler
        except (ImportError, AttributeError) as e:
            logger.warning(f"Run {run_uuid}: Could not import scheduler '{request_scheduler_id}'. Using default '{original_scheduler_name}'. Error: {e}")
            scheduler = pipeline.scheduler
        except Exception as e:
            logger.error(f"Run {run_uuid}: Error loading scheduler '{request_scheduler_id}'. Using default '{original_scheduler_name}'. Error: {e}", exc_info=True)
            scheduler = pipeline.scheduler
    else:
        logger.info(f"Run {run_uuid}: Using default scheduler: {scheduler.__class__.__name__}")

    # --- 2. Determine Effective Steps ---
    timesteps = None
    if request_custom_sigmas is not None:
        if len(request_custom_sigmas) < 2: logger.error(f"Run {run_uuid}: Custom sigmas list needs >= 2 values."); return None
        calculated_steps = len(request_custom_sigmas) - 1
        if calculated_steps <= 0: logger.error(f"Run {run_uuid}: Custom sigmas list length resulted in <= 0 steps."); return None
        effective_num_steps = calculated_steps
        logger.info(f"Run {run_uuid}: Using custom sigma list length for num_inference_steps={effective_num_steps}.")
    else:
        if effective_num_steps <= 0: logger.error(f"Run {run_uuid}: num_inference_steps must be positive ({effective_num_steps})."); return None

    # --- 3. Set Timesteps on Scheduler ---
    try:
        scheduler.set_timesteps(effective_num_steps, device=device)
        timesteps = scheduler.timesteps
        logger.info(f"Run {run_uuid}: Timesteps set for scheduler '{scheduler.__class__.__name__}'. Count: {len(timesteps)} (Max requested: {effective_num_steps}).")
        if timesteps is None or len(timesteps) == 0: raise ValueError("Scheduler returned empty timesteps.")
        return scheduler, timesteps, effective_num_steps
    except Exception as e:
        logger.error(f"Run {run_uuid}: Failed to set timesteps ({effective_num_steps} steps) on scheduler '{scheduler.__class__.__name__}': {e}", exc_info=True)
        return None

# --- Helper Function for Scheduled CFG (Placeholder) ---
def get_scheduled_guidance_scale(
    step_index: int,
    total_steps: int,
    schedule_definition: Union[float, str, List[Tuple[int, float]]]
) -> float:
    """Calculates the guidance scale for the current step based on a schedule."""
    if isinstance(schedule_definition, (float, int)):
        return float(schedule_definition)
    # TODO: Implement parsing for string patterns and list of tuples
    logger.warning(f"Guidance scheduling not implemented for type {type(schedule_definition)}. Using fixed 7.5.")
    return 7.5


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
                # Log initial request params (LoRAs/Embeddings included if sent)
                params_to_log = request.model_dump(mode='json')
                db_run = ExperimentRun(
                    run_uuid=run_uuid, run_type="text_to_image",
                    start_time=datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc),
                    status="running", base_parameters=params_to_log
                )
                db.add(db_run)
                await db.commit(); await db.refresh(db_run)
                db_run_id = db_run.id
                logger.info(f"Run {run_uuid}: Created ExperimentRun record ID: {db_run_id}")
        except Exception as e:
            logger.error(f"Run {run_uuid}: Failed to create initial DB record: {e}", exc_info=True)

    # --- Core Generation Logic ---
    images_pil: List[Image.Image] = []
    # Start with request params, exclude unset ones
    final_params_dict = request.model_dump(exclude_unset=True)
    actual_scheduler_name = "Unknown"
    effective_num_steps = request.num_inference_steps # Default

    try:
        # --- Setup Device & Dtype ---
        pipeline_type = "StableDiffusionPipeline"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        logger.debug(f"Run {run_uuid}: Device='{device.type}' DType='{dtype}' Pipeline='{pipeline_type}'")

        # --- Load Pipeline (now passes loras/embeddings) ---
        pipeline: Optional[DiffusionPipeline] = await load_diffusion_pipeline(
            model_path_or_id=request.model_id, pipeline_type=pipeline_type, device=device_str, dtype=dtype,
            loras=getattr(request, 'loras', None), # Pass LoRA list
            embeddings=getattr(request, 'embeddings', None) # Pass embedding list
            # vae_id=request.vae_id
        )
        if not pipeline:
            error_msg = f"Failed to load pipeline for model: {request.model_id}"
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None
        pipeline.to(device)
        logger.info(f"Run {run_uuid}: Loaded pipeline object type: {type(pipeline).__name__}")

        # --- Setup Scheduler and Timesteps ---
        setup_result = setup_scheduler_and_timesteps(
             pipeline, request.scheduler_id, request.custom_sigmas, request.num_inference_steps, device, run_uuid
        )
        if setup_result is None:
             error_msg = "Failed to setup scheduler or timesteps."
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg); return None
        scheduler, timesteps, effective_num_steps = setup_result
        actual_scheduler_name = scheduler.__class__.__name__
        # Update final params with actual values used
        final_params_dict['num_inference_steps'] = effective_num_steps
        final_params_dict['scheduler_id'] = actual_scheduler_name

        # --- Prepare Generator ---
        generator = None; effective_seed = request.seed
        if effective_seed is None: effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            try: effective_seed = int(effective_seed)
            except (ValueError, TypeError): effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(effective_seed)
        final_params_dict['seed'] = effective_seed # Log resolved seed

        # --- Latent & Embedding Prep ---
        height, width = request.height, request.width
        if height <= 0 or width <= 0 or height % 8 != 0 or width % 8 != 0 :
             error_msg = f"Invalid dimensions W={width}, H={height}."
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg); return None

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(request.num_outputs, num_channels_latents, height, width, dtype, device, generator)
        guidance_float = request.guidance_scale if isinstance(request.guidance_scale, (float, int)) else 7.5
        do_cfg = guidance_float > 1.0

        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=request.prompt, device=device, num_images_per_prompt=request.num_outputs,
            do_classifier_free_guidance=do_cfg, negative_prompt=request.negative_prompt
        )
        if do_cfg and negative_prompt_embeds is None:
            logger.warning(f"Run {run_uuid}: CFG > 1 requested but no negative embeddings. Disabling CFG.")
            do_cfg = False

        # --- Inference Loop ---
        logger.info(f"Run {run_uuid}: Running T2I inference loop with {len(timesteps)} steps using {actual_scheduler_name}...")
        for i, t in enumerate(pipeline.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            current_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]) if do_cfg else prompt_embeds

            with torch.no_grad():
                noise_pred = pipeline.unet(
                    latent_model_input, t,
                    encoder_hidden_states=current_prompt_embeds.to(device, dtype=dtype)
                    # cross_attention_kwargs removed
                ).sample

            if do_cfg:
                current_guidance_scale = get_scheduled_guidance_scale(i, len(timesteps), request.guidance_scale)
                if current_guidance_scale > 1.0:
                     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                     noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # --- VAE Decode ---
        vae_scaling_factor = getattr(pipeline.vae.config, 'scaling_factor', 0.18215)
        latents_scaled = 1 / vae_scaling_factor * latents
        with torch.no_grad(): image = pipeline.vae.decode(latents_scaled.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1); image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        images_pil = pipeline.numpy_to_pil(image)
        if not images_pil: raise ValueError("VAE decoding produced no images.")
        logger.info(f"Run {run_uuid}: T2I Inference completed. Generated {len(images_pil)} image(s).")

    except Exception as e:
        error_during_run = f"{type(e).__name__}: {e}"
        logger.error(f"Run {run_uuid}: Error during T2I core logic: {e}", exc_info=True)
        if db_run_id: await update_run_status(db_run_id, "failed", error=error_during_run)
        return None

    # --- Process, Save, Log Outputs ---
    results: List[ImageResult] = []; output_base_path = get_output_base_path() / run_uuid
    output_base_path.mkdir(parents=True, exist_ok=True); output_records_to_add = []

    # Prepare final params dict for serialization
    if 'custom_sigmas' in final_params_dict and not request.custom_sigmas: del final_params_dict['custom_sigmas']
    final_params_dict['guidance_scale'] = request.guidance_scale # Log original request value

    # Convert LoRA objects to dicts for serialization before assigning to final_params_dict
    serializable_loras = None
    if getattr(request, 'loras', None):
        try:
             # Use model_dump() for Pydantic v2
             serializable_loras = [lora.model_dump() for lora in request.loras]
        except AttributeError: # Fallback for potential raw dicts? Should not happen with Pydantic validation
             logger.warning("Could not dump LoRA parameters to dict, attempting direct use.")
             serializable_loras = request.loras

    final_params_dict['loras'] = serializable_loras # Assign serializable list (or None)
    final_params_dict['embeddings'] = getattr(request, 'embeddings', None) # Strings are fine

    for i, img_pil in enumerate(images_pil):
        current_seed = effective_seed + i
        filename = f"t2i_output_{i:03d}_seed{current_seed}.png"
        file_path = output_base_path / filename; file_path_str = str(file_path.resolve())

        # output_params now contains serializable LoRA list
        output_params = final_params_dict.copy()
        output_params['seed'] = current_seed

        # Metadata Injection
        pnginfo = None
        try:
            from PIL.PngImagePlugin import PngInfo; pnginfo = PngInfo()
            # default=str handles types json doesn't know natively (like datetime if it was in params)
            pnginfo.add_text("grad_parameters", json.dumps(output_params, default=str))
        except Exception as meta_e: logger.warning(f"Run {run_uuid}: Could not prepare PNG metadata: {meta_e}")

        # Save Image
        try: img_pil.save(file_path_str, "PNG", pnginfo=pnginfo)
        except Exception as e: logger.error(f"Run {run_uuid}: Failed save image {file_path_str}: {e}"); continue

        # Base64 Encode
        img_base64 = None
        try: buffered = BytesIO(); img_pil.save(buffered, format="PNG"); img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e: logger.error(f"Run {run_uuid}: Error encoding image {i} base64: {e}")

        # Append API Result
        if img_base64: results.append(ImageResult(image_base64=img_base64, seed=current_seed, parameters=output_params))

        # Prepare DB Output Record
        if db_run_id:
            output_record = GeneratedOutput(
                run_id=db_run_id, output_type="image", file_path=file_path_str,
                parameters=output_params, # output_params is now JSON serializable
                creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
            )
            output_records_to_add.append(output_record)

    # Batch Log Outputs to DB (Should succeed now)
    if output_records_to_add:
        try:
             async with get_db_session() as db: db.add_all(output_records_to_add); await db.commit()
             logger.info(f"Run {run_uuid}: Logged {len(output_records_to_add)} output records to DB.")
        except Exception as e: logger.error(f"Run {run_uuid}: Failed to log output records to DB: {e}", exc_info=True)

    # Finalize Run
    end_time = time.time(); performance_ms = (end_time - start_time) * 1000
    logger.info(f"Run {run_uuid}: Text-to-image request processed in {performance_ms:.2f} ms.")
    if db_run_id: await update_run_status(db_run_id, "completed", end_time=end_time)

    return Text2ImageResponse(images=results, request_params=request, performance_ms=performance_ms)


# --- run_image_to_image ---
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
                params_to_log = request.model_dump(mode='json', exclude={'image_base64'})
                params_to_log['image_provided'] = True
                db_run = ExperimentRun(
                    run_uuid=run_uuid, run_type="image_to_image",
                    start_time=datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc),
                    status="running", base_parameters=params_to_log
                )
                db.add(db_run); await db.commit(); await db.refresh(db_run)
                db_run_id = db_run.id
                logger.info(f"Run {run_uuid}: Created ExperimentRun record ID: {db_run_id}")
        except Exception as e: logger.error(f"Run {run_uuid}: Failed to create initial DB record: {e}", exc_info=True)

    # --- Core Generation Logic ---
    images_pil: List[Image.Image] = []
    final_params_dict = request.model_dump(exclude_unset=True, exclude={'image_base64'})
    actual_scheduler_name = "Unknown"
    effective_num_steps = request.num_inference_steps

    try:
        # --- Setup Device & Dtype ---
        pipeline_type = "StableDiffusionImg2ImgPipeline"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str); dtype = torch.float16 if device.type == "cuda" else torch.float32
        logger.debug(f"Run {run_uuid}: Device='{device.type}' DType='{dtype}' Pipeline='{pipeline_type}'")

        # --- Load Pipeline (now passes loras/embeddings) ---
        pipeline: Optional[DiffusionPipeline] = await load_diffusion_pipeline(
            model_path_or_id=request.model_id, pipeline_type=pipeline_type, device=device_str, dtype=dtype,
            loras=getattr(request, 'loras', None), embeddings=getattr(request, 'embeddings', None)
            # vae_id=request.vae_id
        )
        if not pipeline:
            error_msg = f"Failed load pipeline {request.model_id}"; logger.error(f"Run {run_uuid}: {error_msg}")
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg); return None
        pipeline.to(device)
        logger.info(f"Run {run_uuid}: Loaded pipeline object type: {type(pipeline).__name__}")

        # --- Prepare Input Image ---
        logger.info(f"Run {run_uuid}: Preparing initial image...")
        init_image_pil = prepare_init_image(request.image_base64, request.width, request.height)
        if init_image_pil is None:
            error_msg = "Failed process init image."; logger.error(f"Run {run_uuid}: {error_msg}")
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg); return None

        # --- Setup Scheduler and Timesteps ---
        setup_result = setup_scheduler_and_timesteps(
             pipeline, request.scheduler_id, None, request.num_inference_steps, device, run_uuid
        )
        if setup_result is None:
             error_msg = "Failed setup scheduler/timesteps."; logger.error(f"Run {run_uuid}: {error_msg}")
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg); return None
        scheduler, _, effective_num_steps = setup_result
        actual_scheduler_name = scheduler.__class__.__name__
        final_params_dict['num_inference_steps'] = effective_num_steps
        final_params_dict['scheduler_id'] = actual_scheduler_name

        # --- Prepare Generator ---
        generator = None; effective_seed = request.seed
        if effective_seed is None: effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            try: effective_seed = int(effective_seed)
            except (ValueError, TypeError): effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(effective_seed)
        final_params_dict['seed'] = effective_seed

        # --- Run Inference ---
        logger.info(f"Run {run_uuid}: Running Img2Img inference strength={request.strength}, max_steps={effective_num_steps}...")
        if not isinstance(pipeline, StableDiffusionImg2ImgPipeline): logger.warning(f"Run {run_uuid}: Pipeline is {type(pipeline).__name__}, not StableDiffusionImg2ImgPipeline.")

        pipeline_args = {
            "prompt": request.prompt, "image": init_image_pil, "strength": request.strength,
            "num_inference_steps": effective_num_steps, "guidance_scale": request.guidance_scale,
            "negative_prompt": request.negative_prompt, "num_images_per_prompt": request.num_outputs,
            "generator": generator,
            # cross_attention_kwargs removed
        }
        logger.debug(f"Run {run_uuid}: Calling {type(pipeline).__name__}.__call__ with args: strength={pipeline_args['strength']}, num_inference_steps={pipeline_args['num_inference_steps']}, guidance_scale={pipeline_args['guidance_scale']}, ...")

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
    results: List[ImageResult] = []; output_base_path = get_output_base_path() / run_uuid
    output_base_path.mkdir(parents=True, exist_ok=True); output_records_to_add = []

    # Prepare final params dict for serialization
    if 'custom_sigmas' in final_params_dict: del final_params_dict['custom_sigmas']
    final_params_dict['strength'] = request.strength
    final_params_dict['guidance_scale'] = request.guidance_scale
    # Convert LoRA objects to dicts for serialization
    serializable_loras = None
    if getattr(request, 'loras', None):
        try: serializable_loras = [lora.model_dump() for lora in request.loras]
        except AttributeError: logger.warning("Could not dump LoRA params to dict"); serializable_loras = request.loras
    final_params_dict['loras'] = serializable_loras
    final_params_dict['embeddings'] = getattr(request, 'embeddings', None)

    for i, img_pil in enumerate(images_pil):
        current_seed = effective_seed + i
        filename = f"i2i_output_{i:03d}_seed{current_seed}.png"
        file_path = output_base_path / filename; file_path_str = str(file_path.resolve())
        output_params = final_params_dict.copy(); output_params['seed'] = current_seed

        # Metadata Injection
        pnginfo = None
        try: from PIL.PngImagePlugin import PngInfo; pnginfo = PngInfo(); pnginfo.add_text("grad_parameters", json.dumps(output_params, default=str))
        except Exception as meta_e: logger.warning(f"Run {run_uuid}: Could not prep PNG metadata: {meta_e}")

        # Save Image
        try: img_pil.save(file_path_str, "PNG", pnginfo=pnginfo)
        except Exception as e: logger.error(f"Run {run_uuid}: Failed save image {file_path_str}: {e}"); continue

        # Base64 Encode
        img_base64 = None
        try: buffered = BytesIO(); img_pil.save(buffered, format="PNG"); img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e: logger.error(f"Run {run_uuid}: Error encoding image {i} base64: {e}")

        # Append API Result
        if img_base64: results.append(ImageResult(image_base64=img_base64, seed=current_seed, parameters=output_params))

        # Prepare DB Output Record
        if db_run_id:
            output_record = GeneratedOutput(
                run_id=db_run_id, output_type="image", file_path=file_path_str,
                parameters=output_params, # output_params is now JSON serializable
                creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
            )
            output_records_to_add.append(output_record)

    # Batch Log Outputs to DB
    if output_records_to_add:
        try:
             async with get_db_session() as db: db.add_all(output_records_to_add); await db.commit()
             logger.info(f"Run {run_uuid}: Logged {len(output_records_to_add)} output records to DB.")
        except Exception as e: logger.error(f"Run {run_uuid}: Failed log output records DB: {e}", exc_info=True)

    # Finalize Run
    end_time = time.time(); performance_ms = (end_time - start_time) * 1000
    logger.info(f"Run {run_uuid}: Image-to-image request processed in {performance_ms:.2f} ms.")
    if db_run_id: await update_run_status(db_run_id, "completed", end_time=end_time)

    return ImageToImageResponse(images=results, request_params=request, performance_ms=performance_ms)