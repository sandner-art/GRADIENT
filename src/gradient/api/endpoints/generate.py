# gradient-project-root/src/gradient/api/endpoints/generate.py

from fastapi import APIRouter, HTTPException, Body
import logging

# Import Pydantic models and the pipeline runner/grid functions
from gradient.ai.diffusion.models import (
    Text2ImageRequest, Text2ImageResponse,
    XYZGridRequest, XYZGridResponse # <-- Import Grid models
)
from gradient.ai.diffusion.pipelines import run_text_to_image
from gradient.ai.diffusion.xyz_grid import run_xyz_grid # <-- Import Grid function

# Import task scheduler if running as background task (later phase)
# from gradient.core.scheduling import task_scheduler

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Existing Text2Image Endpoint ---
@router.post(
    "/text-to-image", # Keep original path for direct generation
    response_model=Text2ImageResponse,
    summary="Generate images from text prompts",
    tags=["Generation"]
)
async def handle_text_to_image(
    request: Text2ImageRequest = Body(...)
):
    """
    Accepts text prompt and parameters, generates images using a diffusion model.
    """
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


# --- New XYZ Grid Endpoint ---
@router.post(
    "/xyz-grid", # New path for grid generation
    response_model=XYZGridResponse,
    summary="Generate a grid of images varying parameters across X/Y/Z axes",
    tags=["Generation"]
)
async def handle_xyz_grid(
    request: XYZGridRequest = Body(...)
):
    """
    Accepts base parameters and axis definitions, generates a grid of images.
    Runs synchronously for now. Each cell generation is awaited sequentially.
    """
    logger.info(f"Received API request for XYZ grid generation: Model={request.base_parameters.model_id}")
    # TODO: Add background task support later for long grids
    try:
        grid_results = await run_xyz_grid(request)
        # We could potentially combine results into a single grid image here,
        # but for now, just return the list of individual cell results.
        return XYZGridResponse(
            grid_results=grid_results,
            request_params=request
        )
    except MemoryError as e:
        # This might occur if many runs consume accumulating resources,
        # though individual runs also check VRAM.
        logger.error(f"Out of memory error during XYZ grid processing: {e}", exc_info=True)
        raise HTTPException(status_code=507, detail=f"Insufficient memory during grid generation: {e}")
    except ValueError as e: # Catch potential validation errors from grid logic
         logger.error(f"Value error during XYZ grid processing: {e}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid grid configuration: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during XYZ grid API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during grid generation: {e}")