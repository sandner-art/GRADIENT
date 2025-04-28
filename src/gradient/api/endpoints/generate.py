# gradient-project-root/src/gradient/api/endpoints/generate.py (NEW FILE)

from fastapi import APIRouter, HTTPException, Body
import logging

# Import Pydantic models and the pipeline runner function
from gradient.ai.diffusion.models import Text2ImageRequest, Text2ImageResponse
from gradient.ai.diffusion.pipelines import run_text_to_image

# Import task scheduler if running as background task (later phase)
# from gradient.core.scheduling import task_scheduler

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/generate/text-to-image",
    response_model=Text2ImageResponse,
    summary="Generate images from text prompts",
    tags=["Generation"] # Add a new tag for generation endpoints
)
async def handle_text_to_image(
    request: Text2ImageRequest = Body(...) # Get request data from body
):
    """
    Accepts text prompt and parameters, generates images using a diffusion model.
    This currently runs synchronously. Background task implementation TBD.
    """
    logger.info(f"Received API request for text-to-image: {request.model_id}")

    # --- Option 1: Run synchronously (for Phase 1 simplicity) ---
    try:
        result = await run_text_to_image(request)
        if result is None:
            # Logged error in run_text_to_image
            raise HTTPException(status_code=500, detail="Image generation failed internally.")
        if not result.images:
             # Handle case where generation technically succeeded but produced no images
             logger.warning("Text-to-image completed but returned no images.")
             # Depending on expected behavior, could be error or empty success
             raise HTTPException(status_code=500, detail="Generation completed but produced no images.")

        return result

    except MemoryError as e:
        logger.error(f"Out of memory error during text-to-image: {e}", exc_info=True)
        raise HTTPException(status_code=507, detail=f"Insufficient memory for generation: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during text-to-image API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    # --- Option 2: Run as background task (Later Phase) ---
    # task_id = await task_scheduler.submit_task(run_text_to_image, request)
    # return {"message": "Text-to-image task submitted", "task_id": task_id}
    # (Would require separate endpoints for task status/results)