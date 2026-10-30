# gradient-project-root/src/gradient/api/endpoints/visualize.py

import logging

from fastapi import APIRouter, Body

from gradient.ai.viz.latent_space import generate_latent_space_visualization
from gradient.ai.viz.models import (
    LatentSpaceVizRequest,
    LatentSpaceVizResponse,
    SensitivityAnalysisRequest,
    SensitivityAnalysisResponse,
)
from gradient.ai.viz.sensitivity import analyze_parameter_sensitivity

from .generate import to_http_error

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/latent-space",
    response_model=LatentSpaceVizResponse,
    summary="(Prototype) Project random latents with UMAP",
)
async def handle_latent_space_visualization(request: LatentSpaceVizRequest = Body(...)):
    try:
        return await generate_latent_space_visualization(request)
    except Exception as e:
        raise to_http_error(e, "latent space visualization") from e


@router.post(
    "/parameter-sensitivity",
    response_model=SensitivityAnalysisResponse,
    summary="Where in the image does a parameter matter? (central finite differences)",
)
async def handle_parameter_sensitivity(request: SensitivityAnalysisRequest = Body(...)):
    """Renders the same seed at p-d and p+d and returns a per-pixel sensitivity heatmap."""
    try:
        return await analyze_parameter_sensitivity(request)
    except Exception as e:
        raise to_http_error(e, "sensitivity analysis") from e
