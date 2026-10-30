# gradient-project-root/src/gradient/api/endpoints/generate.py

import logging

import torch
from fastapi import APIRouter, Body, HTTPException

from gradient.ai.diffusion.models import (
    ImageToImageRequest,
    ImageToImageResponse,
    Text2ImageRequest,
    Text2ImageResponse,
    XYZGridRequest,
    XYZGridResponse,
)
from gradient.ai.diffusion.pipelines import run_image_to_image, run_text_to_image
from gradient.ai.diffusion.xyz_grid import run_xyz_grid

router = APIRouter()
logger = logging.getLogger(__name__)


def to_http_error(e: Exception, what: str) -> HTTPException:
    """Map generation errors to useful HTTP responses."""
    if isinstance(e, HTTPException):
        return e
    if isinstance(e, (ValueError, FileNotFoundError)):
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, (MemoryError, torch.OutOfMemoryError)):
        return HTTPException(
            status_code=507,
            detail=f"Out of memory during {what}. Try a smaller size, fewer images, or performance.memory_mode: low. ({e})",
        )
    logger.exception("Unexpected error during %s", what)
    return HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post(
    "/text-to-image", response_model=Text2ImageResponse, summary="Generate images from text prompts"
)
async def handle_text_to_image(request: Text2ImageRequest = Body(...)):
    logger.info("text-to-image request: %s", request.model_id)
    try:
        return await run_text_to_image(request)
    except Exception as e:
        raise to_http_error(e, "text-to-image") from e


@router.post(
    "/image-to-image",
    response_model=ImageToImageResponse,
    summary="Generate images based on an initial image and text prompt",
)
async def handle_image_to_image(request: ImageToImageRequest = Body(...)):
    logger.info("image-to-image request: %s", request.model_id)
    try:
        return await run_image_to_image(request)
    except Exception as e:
        raise to_http_error(e, "image-to-image") from e


@router.post(
    "/xyz-grid",
    response_model=XYZGridResponse,
    summary="Generate a grid of images varying parameters across X/Y/Z axes",
)
async def handle_xyz_grid(request: XYZGridRequest = Body(...)):
    logger.info("XYZ grid request: %s", request.base_parameters.model_id)
    try:
        grid_results = await run_xyz_grid(request)
        return XYZGridResponse(grid_results=grid_results, request_params=request)
    except Exception as e:
        raise to_http_error(e, "XYZ grid generation") from e
