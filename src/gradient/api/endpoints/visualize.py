# gradient-project-root/src/gradient/api/endpoints/visualize.py

import logging
from fastapi import APIRouter, HTTPException, Body

# Import models and generator/analyzer functions
from gradient.ai.viz.models import (
    LatentSpaceVizRequest, LatentSpaceVizResponse,
    SensitivityAnalysisRequest, SensitivityAnalysisResponse # <-- Import Sensitivity models
)
from gradient.ai.viz.latent_space import generate_latent_space_visualization
from gradient.ai.viz.sensitivity import analyze_parameter_sensitivity # <-- Import Sensitivity function

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Existing Latent Space Endpoint ---
@router.post(
    "/latent-space", # Keep original path
    response_model=LatentSpaceVizResponse,
    summary="Generate data for visualizing the latent space",
    tags=["Visualization"]
)
async def handle_latent_space_visualization(
    request: LatentSpaceVizRequest = Body(...)
):
    """
    Generates random latent samples, optionally decodes them using a VAE,
    and projects them into 2D/3D using UMAP for visualization.
    """
    logger.info(f"Received API request for latent space visualization: Model={request.model_id}, Samples={request.num_samples}")
    try:
        result = await generate_latent_space_visualization(request)
        if result is None:
             raise HTTPException(status_code=500, detail="Latent space visualization generation failed internally.")
        return result
    except MemoryError as e:
        logger.error(f"Out of memory error during latent space visualization: {e}", exc_info=True)
        raise HTTPException(status_code=507, detail=f"Insufficient memory during visualization: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during latent space visualization API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during visualization: {e}")


# --- NEW Sensitivity Analysis Endpoint ---
@router.post(
    "/parameter-sensitivity", # New path
    response_model=SensitivityAnalysisResponse,
    summary="Analyze sensitivity of generation output to a specific parameter",
    tags=["Visualization"]
)
async def handle_parameter_sensitivity(
    request: SensitivityAnalysisRequest = Body(...)
):
    """
    Performs a base generation and calculates the gradient of the output
    (currently final latent's loss) with respect to a target parameter
    (currently guidance_scale) to estimate sensitivity.
    """
    logger.info(f"Received API request for parameter sensitivity analysis: Param={request.target_parameter.name}")
    # TODO: Add background task support later for potentially slow analysis
    try:
        result = await analyze_parameter_sensitivity(request)
        if result is None:
             # Error logged internally
             raise HTTPException(status_code=500, detail="Parameter sensitivity analysis failed internally.")
        return result
    except MemoryError as e:
        logger.error(f"Out of memory error during sensitivity analysis: {e}", exc_info=True)
        raise HTTPException(status_code=507, detail=f"Insufficient memory during analysis: {e}")
    except ValueError as e: # Catch validation errors from analysis logic
         logger.error(f"Value error during sensitivity analysis: {e}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid request for sensitivity analysis: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during sensitivity analysis API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during sensitivity analysis: {e}")