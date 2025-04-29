# gradient-project-root/src/gradient/api/endpoints/generate.py

from fastapi import APIRouter, HTTPException, Body
import logging

# Import Pydantic models and the pipeline runner/grid functions
from gradient.ai.diffusion.models import (
    Text2ImageRequest, Text2ImageResponse,
    XYZGridRequest, XYZGridResponse,
    ImageToImageRequest, ImageToImageResponse # <-- Import Img2Img models
)
from gradient.ai.diffusion.pipelines import run_text_to_image, run_image_to_image # <-- Import Img2Img runner
from gradient.ai.diffusion.xyz_grid import run_xyz_grid

# Import task scheduler if running as background task (later phase)
# from gradient.core.scheduling import task_scheduler

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Existing Text2Image Endpoint (Unchanged) ---
@router.post(
    "/text-to-image",
    response_model=Text2ImageResponse,
    summary="Generate images from text prompts",
    tags=["Generation"]
)
async def handle_text_to_image(
    request: Text2ImageRequest = Body(...)
):
    """Accepts text prompt and parameters, generates images using a diffusion model."""
    logger.info(f"Received API request for text-to-image: {request.model_id}")
    try:
        result = await run_text_to_image(request)
        if result is None:
            raise HTTPException(status_code=500, detail="Image generation failed internally.")
        if not result.images:
             raise HTTPException(status_code=500, detail="Generation completed but produced no images.")
        return result
    except MemoryError as e:
        logger.error(f"Out of memory error during text-to-image: {e}", exc_info=True)
        raise HTTPException(status_code=507, detail=f"Insufficient memory for generation: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during text-to-image API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- NEW ImageToImage Endpoint --- vvv
@router.post(
    "/image-to-image",
    response_model=ImageToImageResponse,
    summary="Generate images based on an initial image and text prompt",
    tags=["Generation"]
)
async def handle_image_to_image(
    request: ImageToImageRequest = Body(...)
):
    """Accepts an initial image, text prompt, strength, and other parameters to generate variations."""
    logger.info(f"Received API request for image-to-image: {request.model_id}")
    try:
        result = await run_image_to_image(request)
        if result is None:
            raise HTTPException(status_code=500, detail="Image generation failed internally.")
        if not result.images:
             raise HTTPException(status_code=500, detail="Generation completed but produced no images.")
        return result
    except MemoryError as e:
        logger.error(f"Out of memory error during image-to-image: {e}", exc_info=True)
        raise HTTPException(status_code=507, detail=f"Insufficient memory for generation: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during image-to-image API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
# --- NEW ImageToImage Endpoint --- ^^^


# --- Existing XYZ Grid Endpoint (Unchanged) ---
@router.post(
    "/xyz-grid",
    response_model=XYZGridResponse,
    summary="Generate a grid of images varying parameters across X/Y/Z axes",
    tags=["Generation"]
)
async def handle_xyz_grid(
    request: XYZGridRequest = Body(...)
):
    """Generates a grid of images based on parameter variations."""
    logger.info(f"Received API request for XYZ grid generation: Model={request.base_parameters.model_id}")
    try:
        grid_results = await run_xyz_grid(request)
        return XYZGridResponse(
            grid_results=grid_results,
            request_params=request
        )
    except MemoryError as e:
        logger.error(f"Out of memory error during XYZ grid processing: {e}", exc_info=True)
        raise HTTPException(status_code=507, detail=f"Insufficient memory during grid generation: {e}")
    except ValueError as e:
         logger.error(f"Value error during XYZ grid processing: {e}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid grid configuration: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during XYZ grid API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during grid generation: {e}")