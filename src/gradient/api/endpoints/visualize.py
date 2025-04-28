# gradient-project-root/src/gradient/api/endpoints/visualize.py (NEW FILE)

import logging
from fastapi import APIRouter, HTTPException, Body

# Import models and generator function
from gradient.ai.viz.models import LatentSpaceVizRequest, LatentSpaceVizResponse
from gradient.ai.viz.latent_space import generate_latent_space_visualization

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/visualize/latent-space",
    response_model=LatentSpaceVizResponse,
    summary="Generate data for visualizing the latent space",
    tags=["Visualization"] # New tag
)
async def handle_latent_space_visualization(
    request: LatentSpaceVizRequest = Body(...)
):
    """
    Generates random latent samples, optionally decodes them using a VAE,
    and projects them into 2D/3D using UMAP for visualization.
    """
    logger.info(f"Received API request for latent space visualization: Model={request.model_id}, Samples={request.num_samples}")
    # TODO: Add background task support later for large sample counts
    try:
        result = await generate_latent_space_visualization(request)
        if result is None:
             # Error logged internally (e.g., umap not found, VAE load failed)
             raise HTTPException(status_code=500, detail="Latent space visualization generation failed internally.")
        return result
    except MemoryError as e:
        logger.error(f"Out of memory error during latent space visualization: {e}", exc_info=True)
        raise HTTPException(status_code=507, detail=f"Insufficient memory during visualization: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during latent space visualization API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during visualization: {e}")