# src/gradient/api/endpoints/lab.py
"""Research tools: Taylor preview, attention maps, token ablation, layer listing.

Structural modulation lanes are part of every generation request (`modulation` field),
so they work from /generate/* as well as from the Lab UI.
"""

import logging

from fastapi import APIRouter, Body, Query

from gradient.ai.diffusion.loading import get_loaded_entry
from gradient.ai.lab.ablation import token_ablation
from gradient.ai.lab.attention import attention_maps
from gradient.ai.lab.models import (
    AblationRequest,
    AblationResponse,
    AttentionMapRequest,
    AttentionMapResponse,
    TaylorRequest,
    TaylorResponse,
)
from gradient.ai.lab.probes import attention_layers
from gradient.ai.lab.taylor import taylor_preview
from gradient.core.scheduling import run_gpu_task

from .generate import to_http_error

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/lab/taylor-preview",
    response_model=TaylorResponse,
    summary="Image + d(image)/d(guidance) for instant previews",
)
async def handle_taylor(request: TaylorRequest = Body(...)):
    try:
        return await run_gpu_task(taylor_preview, request)
    except Exception as e:
        raise to_http_error(e, "Taylor preview") from e


@router.post(
    "/lab/attention-maps",
    response_model=AttentionMapResponse,
    summary="Per-word cross-attention maps (SD 1.x/SDXL)",
)
async def handle_attention_maps(request: AttentionMapRequest = Body(...)):
    try:
        return await run_gpu_task(attention_maps, request)
    except Exception as e:
        raise to_http_error(e, "attention maps") from e


@router.post(
    "/lab/token-ablation",
    response_model=AblationResponse,
    summary="Leave-one-word-out influence, measured with DINO features",
)
async def handle_token_ablation(request: AblationRequest = Body(...)):
    try:
        return await run_gpu_task(token_ablation, request)
    except Exception as e:
        raise to_http_error(e, "token ablation") from e


@router.get(
    "/lab/layers",
    summary="Attention layer names of a model (for modulation targets and attention maps)",
)
async def handle_layers(model_id: str = Query(...)):
    try:
        entry = await run_gpu_task(get_loaded_entry, model_id)
        layers = attention_layers(entry.pipeline)
    except Exception as e:
        raise to_http_error(e, "layer listing") from e
    prefixes = sorted({name.split(".")[0] for name in (layer["name"] for layer in layers)})
    suggestions = [f"{p}.*" for p in prefixes]
    if any(layer["kind"] == "cross" for layer in layers):
        suggestions = ["*attn2", *[f"{p}.*attn2" for p in prefixes], *suggestions]
    return {
        "model_id": model_id,
        "pipeline": entry.pipeline_class,
        "layers": layers,
        "suggestions": suggestions,
    }
