# gradient-project-root/src/gradient/ai/diffusion/pipelines.py

import logging
import torch
import time
import base64
from io import BytesIO
from PIL import Image
from typing import List, Optional # <--- ADD THIS IMPORT

from .loading import load_diffusion_pipeline # Import the loading function
from .models import Text2ImageRequest, Text2ImageResponse, ImageResult # Import pydantic models

logger = logging.getLogger(__name__)

async def run_text_to_image(request: Text2ImageRequest) -> Optional[Text2ImageResponse]:
    """
    Runs the text-to-image pipeline based on the request.
    """
    start_time = time.time()
    logger.info(f"Received text-to-image request: {request.prompt[:50]}...")

    # 1. Load Pipeline (using the loading function)
    pipeline_type = "StableDiffusionPipeline" # Assume SD for now
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Access dtype correctly from torch
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = await load_diffusion_pipeline(
        model_path_or_id=request.model_id,
        pipeline_type=pipeline_type,
        device=device,
        dtype=dtype
    )

    if not pipeline:
        logger.error(f"Failed to load pipeline for model: {request.model_id}")
        return None

    # 2. Prepare Generator (for seeding)
    generator = None
    effective_seed = request.seed
    if effective_seed is None:
        effective_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        logger.info(f"Generated random seed: {effective_seed}")
    else:
         logger.info(f"Using provided seed: {effective_seed}")

    generator = torch.Generator(device=device).manual_seed(effective_seed)


    # 3. Run Inference
    images_pil: List[Image.Image] = []
    try:
        logger.info(f"Running {pipeline_type} inference...")
        pipeline.to(device) # Ensure pipeline is on the correct device

        pipeline_output = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_outputs,
            generator=generator,
        )
        # Check if output exists and has images
        if hasattr(pipeline_output, 'images'):
            images_pil = pipeline_output.images
            logger.info(f"Inference completed. Generated {len(images_pil)} image(s).")
        else:
            logger.error("Pipeline output did not contain 'images' attribute.")
            return None # Indicate failure

    except Exception as e:
        logger.error(f"Error during diffusion inference: {e}", exc_info=True)
        return None

    # 4. Process and Encode Results
    results: List[ImageResult] = []
    for i, img_pil in enumerate(images_pil):
        try:
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG") # Save as PNG
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            # Generate distinct seeds for batch if needed, or use the main seed
            current_seed = effective_seed + i if request.num_outputs > 1 else effective_seed

            results.append(ImageResult(image_base64=img_base64, seed=current_seed))
        except Exception as e:
            logger.error(f"Error encoding image {i}: {e}", exc_info=True)

    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Text-to-image request processed in {performance_ms:.2f} ms.")

    # Echo back the parameters used, including the effective seed
    request_params_echo = request.copy()
    request_params_echo.seed = effective_seed

    return Text2ImageResponse(
        images=results,
        request_params=request_params_echo,
        performance_ms=performance_ms
    )

# --- TODO: Add other pipeline runners (img2img, inpainting, etc.) ---