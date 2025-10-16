# src/gradient/ai/diffusion/pipelines.py (Corrected Sections)

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

from sqlalchemy.ext.asyncio import AsyncSession
from gradient.database.session import get_db_session, update_run_status
from gradient.database.models import ExperimentRun, GeneratedOutput

# Use the new parameter names from loading.py
from .loading import load_diffusion_pipeline
from .models import (
    Text2ImageRequest, Text2ImageResponse,
    ImageToImageRequest, ImageToImageResponse,
    ImageResult,
    LoRAParameter 
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline, 
    DiffusionPipeline, # Keep this for general type hint
    AutoencoderKL
)
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler

logger = logging.getLogger(__name__)

# --- Output Path (Unchanged) ---
def get_output_base_path() -> Path:
     output_path = Path("./experiments/") # TODO: Read from config
     output_path.mkdir(parents=True, exist_ok=True)
     return output_path

# --- Helper Function to Prepare Image (Unchanged) ---
def prepare_init_image(image_base64: str, width: int, height: int) -> Optional[Image.Image]:
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Failed to prepare initial image: {e}", exc_info=True)
        return None

# --- Helper Function for Scheduler Loading/Setting (Unchanged) ---
def setup_scheduler_and_timesteps(
    pipeline: DiffusionPipeline, # Use the general DiffusionPipeline type hint
    request_scheduler_id: Optional[str],
    request_custom_sigmas: Optional[List[float]],
    request_num_steps: int,
    device: torch.device, # Keep as torch.device for internal use
    run_uuid: str
) -> Optional[Tuple[SchedulerMixin, torch.Tensor, int]]:
    # ... (implementation unchanged)
    scheduler = pipeline.scheduler
    effective_num_steps = request_num_steps
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
        except Exception as e:
            logger.error(f"Run {run_uuid}: Error loading scheduler '{request_scheduler_id}'. Defaulting. Error: {e}", exc_info=True)
    else:
        logger.info(f"Run {run_uuid}: Using default scheduler: {scheduler.__class__.__name__}")

    timesteps = None
    if request_custom_sigmas is not None and len(request_custom_sigmas) >= 2:
        effective_num_steps = len(request_custom_sigmas) - 1
        logger.info(f"Run {run_uuid}: Using custom sigma list length for num_inference_steps={effective_num_steps}.")
    
    if effective_num_steps <= 0:
        logger.error(f"Run {run_uuid}: effective_num_steps must be positive ({effective_num_steps}).")
        return None
    try:
        scheduler.set_timesteps(effective_num_steps, device=device)
        timesteps = scheduler.timesteps
        if timesteps is None or len(timesteps) == 0: raise ValueError("Scheduler returned empty timesteps.")
        logger.info(f"Run {run_uuid}: Timesteps set for '{scheduler.__class__.__name__}'. Count: {len(timesteps)}.")
        return scheduler, timesteps, effective_num_steps
    except Exception as e:
        logger.error(f"Run {run_uuid}: Failed to set timesteps on scheduler '{scheduler.__class__.__name__}': {e}", exc_info=True)
        return None

# --- Helper Function for Scheduled CFG (Placeholder - Unchanged) ---
def get_scheduled_guidance_scale(
    step_index: int,
    total_steps: int,
    schedule_definition: Union[float, str, List[Tuple[int, float]]]
) -> float:
    if isinstance(schedule_definition, (float, int)): return float(schedule_definition)
    logger.warning(f"Guidance scheduling not implemented for {type(schedule_definition)}. Using fixed 7.5.")
    return 7.5


# --- run_text_to_image ---
async def run_text_to_image(
    request: Text2ImageRequest,
    parent_run_uuid: Optional[str] = None
) -> Optional[Text2ImageResponse]:
    start_time = time.time()
    run_uuid = parent_run_uuid or str(uuid.uuid4())
    logger.info(f"Starting T2I run {run_uuid} Prompt: '{request.prompt[:50]}...'")
    db_run_id: Optional[int] = None

    if not parent_run_uuid: # Log initial request if not part of a larger run (like XYZ grid)
        try:
            async with get_db_session() as db:
                params_to_log = request.model_dump(mode='json')
                db_run = ExperimentRun(
                    run_uuid=run_uuid, run_type="text_to_image",
                    start_time=datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc),
                    status="running", base_parameters=params_to_log
                )
                db.add(db_run); await db.commit(); await db.refresh(db_run)
                db_run_id = db_run.id
        except Exception as e: logger.error(f"Run {run_uuid}: DB log fail: {e}", exc_info=True)

    images_pil: List[Image.Image] = []
    final_params_dict = request.model_dump(exclude_unset=True) # For image metadata
    actual_scheduler_name = "Unknown"
    effective_num_steps = request.num_inference_steps

    try:
        # --- Setup Device & Dtype for this pipeline execution context ---
        # These are distinct from model_torch_dtype which is for loading
        exec_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        exec_device = torch.device(exec_device_str)
        # Dtype for computation, usually float16 on CUDA, float32 on CPU
        # This might differ from model_torch_dtype if model was loaded in, e.g., 8-bit
        exec_dtype = torch.float16 if exec_device.type == "cuda" else torch.float32
        logger.debug(f"Run {run_uuid}: Exec Device='{exec_device.type}' Exec DType='{exec_dtype}'")

        # --- Load Pipeline ---
        # CORRECTIONS: Use new param names and pass VRAM opt flags
        pipeline: Optional[DiffusionPipeline] = await load_diffusion_pipeline(
            model_path_or_id=request.model_id,
            pipeline_type_str="StableDiffusionPipeline", # Use new param name
            target_device_str=exec_device_str,          # Use new param name
            model_torch_dtype=exec_dtype,               # Use new param name (or request.model_dtype if that's a field)
            loras=getattr(request, 'loras', None),
            embeddings=getattr(request, 'embeddings', None),
            # vae_id=request.vae_id, # Add if VAE selection is separate
            
            # Pass VRAM optimization flags from the request
            enable_cpu_offload=request.enable_cpu_offload,
            attention_optimization=request.attention_optimization,
            vae_optimization=request.vae_optimization,
            quantization_bits=request.quantization_bits
        )
        if not pipeline:
            error_msg = f"Failed to load pipeline for model: {request.model_id}"
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg)
            return None
        # `pipeline.to(exec_device)` is handled by load_diffusion_pipeline or by offload manager

        # --- Setup Scheduler and Timesteps ---
        # Note: setup_scheduler_and_timesteps uses `pipeline.scheduler.config` and `pipeline.device`
        # `pipeline.device` should be correctly set by load_diffusion_pipeline or offloading logic.
        setup_result = setup_scheduler_and_timesteps(
             pipeline, request.scheduler_id, request.custom_sigmas, request.num_inference_steps, exec_device, run_uuid
        )
        if setup_result is None:
             error_msg = "Failed to setup scheduler or timesteps."
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg); return None
        scheduler, timesteps, effective_num_steps = setup_result
        actual_scheduler_name = scheduler.__class__.__name__
        final_params_dict['num_inference_steps'] = effective_num_steps
        final_params_dict['scheduler_id'] = actual_scheduler_name

        # --- Prepare Generator ---
        generator = None; effective_seed = request.seed
        if effective_seed is None: effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            try: effective_seed = int(effective_seed)
            except (ValueError, TypeError): effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=pipeline.device).manual_seed(effective_seed) # Use pipeline's device for generator
        final_params_dict['seed'] = effective_seed 

        # --- Latent & Embedding Prep ---
        # (Standard T2I loop, should adapt to pipeline.device and exec_dtype)
        # ... (T2I inference loop as before, ensuring tensors created are on pipeline.device with exec_dtype) ...
        height, width = request.height, request.width
        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            request.num_outputs, num_channels_latents, height, width, 
            pipeline.unet.dtype, # Use UNet's dtype for latents
            pipeline.device, generator
        )
        guidance_float = request.guidance_scale if isinstance(request.guidance_scale, (float, int)) else 7.5
        do_cfg = guidance_float > 1.0

        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=request.prompt, device=pipeline.device, num_images_per_prompt=request.num_outputs,
            do_classifier_free_guidance=do_cfg, negative_prompt=request.negative_prompt,
            prompt_embeds=None, negative_prompt_embeds=None # Allow pre-computed later
        )
        # Ensure prompt_embeds are on correct device and dtype
        prompt_embeds = prompt_embeds.to(device=pipeline.device, dtype=pipeline.text_encoder.dtype if hasattr(pipeline, 'text_encoder') else exec_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=pipeline.device, dtype=pipeline.text_encoder.dtype if hasattr(pipeline, 'text_encoder') else exec_dtype)


        logger.info(f"Run {run_uuid}: T2I loop with {len(timesteps)} steps using {actual_scheduler_name} on {pipeline.device}...")
        for i, t in enumerate(pipeline.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            current_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]) if do_cfg and negative_prompt_embeds is not None else prompt_embeds

            with torch.no_grad():
                noise_pred = pipeline.unet(
                    latent_model_input, t,
                    encoder_hidden_states=current_prompt_embeds
                ).sample

            if do_cfg:
                current_guidance = get_scheduled_guidance_scale(i, len(timesteps), request.guidance_scale)
                if negative_prompt_embeds is not None: # Only do CFG if negative prompt was provided and embeds exist
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + current_guidance * (noise_pred_text - noise_pred_uncond)
                # else: just use noise_pred_text (i.e., noise_pred as is)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # VAE Decode
        # Ensure latents are on VAE device and correct dtype for VAE
        latents_for_vae = latents.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
        vae_scaling_factor = getattr(pipeline.vae.config, 'scaling_factor', 0.18215)
        latents_for_vae = 1 / vae_scaling_factor * latents_for_vae
        with torch.no_grad(): 
            image = pipeline.vae.decode(latents_for_vae).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy() # Move to CPU for PIL conversion
        images_pil = pipeline.numpy_to_pil(image)


        if not images_pil: raise ValueError("VAE decoding produced no images.")
        logger.info(f"Run {run_uuid}: T2I Inference completed. Generated {len(images_pil)} image(s).")

    except Exception as e:
        error_during_run = f"{type(e).__name__}: {e}"
        logger.error(f"Run {run_uuid}: Error during T2I core logic: {e}", exc_info=True)
        if db_run_id: await update_run_status(db_run_id, "failed", error=error_during_run)
        return None

    # --- Process, Save, Log Outputs (ensure serializable_loras logic uses .model_dump()) ---
    results: List[ImageResult] = []; output_base_path = get_output_base_path() / run_uuid
    output_base_path.mkdir(parents=True, exist_ok=True); output_records_to_add = []

    if 'custom_sigmas' in final_params_dict and not request.custom_sigmas: del final_params_dict['custom_sigmas']
    final_params_dict['guidance_scale'] = request.guidance_scale 

    serializable_loras = None
    if getattr(request, 'loras', None):
        try: serializable_loras = [lora.model_dump() for lora in request.loras] # Use model_dump
        except AttributeError: logger.warning("Could not dump LoRA params"); serializable_loras = request.loras
    final_params_dict['loras'] = serializable_loras
    final_params_dict['embeddings'] = getattr(request, 'embeddings', None)
    # Add VRAM optimization params to metadata
    final_params_dict['enable_cpu_offload'] = request.enable_cpu_offload
    final_params_dict['attention_optimization'] = request.attention_optimization
    final_params_dict['vae_optimization'] = request.vae_optimization
    final_params_dict['quantization_bits'] = request.quantization_bits


    for i, img_pil in enumerate(images_pil):
        current_seed = effective_seed + i # Assuming multiple images in batch increment seed
        filename = f"t2i_output_{i:03d}_seed{current_seed}.png"
        file_path_str = str((output_base_path / filename).resolve())
        output_params = final_params_dict.copy(); output_params['seed'] = current_seed

        pnginfo = None
        try:
            from PIL.PngImagePlugin import PngInfo; pnginfo = PngInfo()
            pnginfo.add_text("grad_parameters", json.dumps(output_params, default=str))
        except Exception as meta_e: logger.warning(f"Run {run_uuid}: PNG meta error: {meta_e}")

        try: img_pil.save(file_path_str, "PNG", pnginfo=pnginfo)
        except Exception as e: logger.error(f"Run {run_uuid}: Failed save {file_path_str}: {e}"); continue
        
        img_base64 = None
        try: 
            buffered = BytesIO(); img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e: logger.error(f"Run {run_uuid}: Base64 encode error: {e}")

        if img_base64: results.append(ImageResult(image_base64=img_base64, seed=current_seed, parameters=output_params))
        if db_run_id:
            output_records_to_add.append(GeneratedOutput(
                run_id=db_run_id, output_type="image", file_path=file_path_str,
                parameters=output_params, creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
            ))

    if output_records_to_add:
        try:
             async with get_db_session() as db: db.add_all(output_records_to_add); await db.commit()
        except Exception as e: logger.error(f"Run {run_uuid}: DB log outputs fail: {e}", exc_info=True)

    end_time = time.time(); performance_ms = (end_time - start_time) * 1000
    logger.info(f"Run {run_uuid}: T2I request processed in {performance_ms:.2f} ms.")
    if db_run_id: await update_run_status(db_run_id, "completed", end_time=end_time)

    return Text2ImageResponse(images=results, request_params=request, performance_ms=performance_ms)


# --- run_image_to_image ---
async def run_image_to_image(
    request: ImageToImageRequest,
    parent_run_uuid: Optional[str] = None
) -> Optional[ImageToImageResponse]:
    start_time = time.time()
    run_uuid = parent_run_uuid or str(uuid.uuid4())
    logger.info(f"Starting I2I run {run_uuid} Prompt: '{request.prompt[:50]}...' Strength: {request.strength}")
    db_run_id: Optional[int] = None

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
        except Exception as e: logger.error(f"Run {run_uuid}: DB log fail: {e}", exc_info=True)

    images_pil: List[Image.Image] = []
    final_params_dict = request.model_dump(exclude_unset=True, exclude={'image_base64'})
    actual_scheduler_name = "Unknown"
    effective_num_steps = request.num_inference_steps

    try:
        exec_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        exec_device = torch.device(exec_device_str)
        exec_dtype = torch.float16 if exec_device.type == "cuda" else torch.float32
        logger.debug(f"Run {run_uuid}: Exec Device='{exec_device.type}' Exec DType='{exec_dtype}'")

        # --- Load Pipeline ---
        # CORRECTIONS: Use new param names and pass VRAM opt flags
        pipeline: Optional[DiffusionPipeline] = await load_diffusion_pipeline(
            model_path_or_id=request.model_id,
            pipeline_type_str="StableDiffusionImg2ImgPipeline", # Use new param name
            target_device_str=exec_device_str,                  # Use new param name
            model_torch_dtype=exec_dtype,                       # Use new param name
            loras=getattr(request, 'loras', None),
            embeddings=getattr(request, 'embeddings', None),
            # vae_id=request.vae_id,

            enable_cpu_offload=request.enable_cpu_offload,
            attention_optimization=request.attention_optimization,
            vae_optimization=request.vae_optimization,
            quantization_bits=request.quantization_bits
        )
        if not pipeline:
            error_msg = f"Failed load pipeline {request.model_id}"
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg); return None
        
        init_image_pil = prepare_init_image(request.image_base64, request.width, request.height)
        if init_image_pil is None:
            error_msg = "Failed process init image."
            if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg); return None

        setup_result = setup_scheduler_and_timesteps(
             pipeline, request.scheduler_id, None, request.num_inference_steps, exec_device, run_uuid # No custom_sigmas for i2i for now
        )
        if setup_result is None:
             error_msg = "Failed setup scheduler/timesteps."
             if db_run_id: await update_run_status(db_run_id, "failed", error=error_msg); return None
        scheduler, _, effective_num_steps = setup_result
        actual_scheduler_name = scheduler.__class__.__name__
        final_params_dict['num_inference_steps'] = effective_num_steps
        final_params_dict['scheduler_id'] = actual_scheduler_name

        generator = None; effective_seed = request.seed
        if effective_seed is None: effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            try: effective_seed = int(effective_seed)
            except (ValueError, TypeError): effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=pipeline.device).manual_seed(effective_seed)
        final_params_dict['seed'] = effective_seed

        # Ensure prompt embeddings are handled correctly for Img2Img pipeline
        # Some Img2Img pipelines might re-encode prompts or expect them differently.
        # The standard StableDiffusionImg2ImgPipeline handles prompt, negative_prompt directly.

        pipeline_args = {
            "prompt": request.prompt, 
            "image": init_image_pil, 
            "strength": request.strength,
            "num_inference_steps": effective_num_steps, 
            "guidance_scale": request.guidance_scale if isinstance(request.guidance_scale, (float,int)) else 7.5, # Ensure float
            "negative_prompt": request.negative_prompt, 
            "num_images_per_prompt": request.num_outputs,
            "generator": generator,
            # No cross_attention_kwargs passed by default
        }
        
        with torch.no_grad():
            # Note: The pipeline call will use its own device and dtypes for internal ops.
            # Our exec_dtype was for loading primarily.
            pipeline_output = pipeline(**pipeline_args)
            images_pil = pipeline_output.images

        if not images_pil: raise ValueError("Pipeline call completed but returned no images.")
        logger.info(f"Run {run_uuid}: Img2Img Inference completed. Generated {len(images_pil)} image(s).")

    except Exception as e:
        error_during_run = f"{type(e).__name__}: {e}"
        logger.error(f"Run {run_uuid}: Error during I2I core logic: {e}", exc_info=True)
        if db_run_id: await update_run_status(db_run_id, "failed", error=error_during_run)
        return None

    # --- Process, Save, Log Outputs (ensure serializable_loras logic uses .model_dump()) ---
    results: List[ImageResult] = []; output_base_path = get_output_base_path() / run_uuid
    output_base_path.mkdir(parents=True, exist_ok=True); output_records_to_add = []

    # Add VRAM optimization params to metadata
    final_params_dict['enable_cpu_offload'] = request.enable_cpu_offload
    final_params_dict['attention_optimization'] = request.attention_optimization
    final_params_dict['vae_optimization'] = request.vae_optimization
    final_params_dict['quantization_bits'] = request.quantization_bits

    serializable_loras = None # For metadata
    if getattr(request, 'loras', None):
        try: serializable_loras = [lora.model_dump() for lora in request.loras] # Use model_dump
        except AttributeError: logger.warning("Could not dump LoRA params"); serializable_loras = request.loras
    final_params_dict['loras'] = serializable_loras
    final_params_dict['embeddings'] = getattr(request, 'embeddings', None)


    for i, img_pil in enumerate(images_pil):
        current_seed = effective_seed + i
        filename = f"i2i_output_{i:03d}_seed{current_seed}.png"
        file_path_str = str((output_base_path / filename).resolve())
        output_params = final_params_dict.copy(); output_params['seed'] = current_seed

        pnginfo = None
        try: 
            from PIL.PngImagePlugin import PngInfo; pnginfo = PngInfo()
            pnginfo.add_text("grad_parameters", json.dumps(output_params, default=str))
        except Exception as meta_e: logger.warning(f"Run {run_uuid}: PNG meta error: {meta_e}")

        try: img_pil.save(file_path_str, "PNG", pnginfo=pnginfo)
        except Exception as e: logger.error(f"Run {run_uuid}: Failed save {file_path_str}: {e}"); continue
        
        img_base64 = None
        try:
            buffered = BytesIO(); img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e: logger.error(f"Run {run_uuid}: Base64 encode error: {e}")

        if img_base64: results.append(ImageResult(image_base64=img_base64, seed=current_seed, parameters=output_params))
        if db_run_id:
            output_records_to_add.append(GeneratedOutput(
                run_id=db_run_id, output_type="image", file_path=file_path_str,
                parameters=output_params, creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
            ))

    if output_records_to_add:
        try:
             async with get_db_session() as db: db.add_all(output_records_to_add); await db.commit()
        except Exception as e: logger.error(f"Run {run_uuid}: DB log outputs fail: {e}", exc_info=True)

    end_time = time.time(); performance_ms = (end_time - start_time) * 1000
    logger.info(f"Run {run_uuid}: I2I request processed in {performance_ms:.2f} ms.")
    if db_run_id: await update_run_status(db_run_id, "completed", end_time=end_time)

    return ImageToImageResponse(images=results, request_params=request, performance_ms=performance_ms)