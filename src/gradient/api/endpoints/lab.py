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
from gradient.ai.lab.concept_attention import concept_attention_maps, concept_support
from gradient.ai.lab.explorer import explore_branches, explore_map, explore_start, explore_tokens
from gradient.ai.lab.models import (
    AblationRequest,
    AblationResponse,
    AttentionMapRequest,
    AttentionMapResponse,
    ConceptAttentionRequest,
    ConceptAttentionResponse,
    ExploreBranchesRequest,
    ExploreBranchesResponse,
    ExploreMapRequest,
    ExploreMapResponse,
    ExploreStartRequest,
    ExploreStartResponse,
    ExploreTokensRequest,
    ExploreTokensResponse,
    QuantReportRequest,
    QuantReportResponse,
    TaylorRequest,
    TaylorResponse,
)
from gradient.ai.lab.probes import attention_layers
from gradient.ai.lab.quant_report import quantization_report
from gradient.ai.lab.taylor import taylor_preview
from gradient.api.job_kinds import label_for, register_lab_kind
from gradient.core.jobs import PRIORITY_INTERACTIVE, job_queue
from gradient.core.scheduling import run_gpu_task

from .generate import to_http_error

logger = logging.getLogger(__name__)
router = APIRouter()

# Lab tools run as interactive queue jobs (ahead of batch renders) and are saved as runs
TAYLOR = register_lab_kind("taylor", taylor_preview, TaylorRequest)
ATTENTION = register_lab_kind("attention", attention_maps, AttentionMapRequest)
CONCEPTS = register_lab_kind("concept_attention", concept_attention_maps, ConceptAttentionRequest)
ABLATION = register_lab_kind("token_ablation", token_ablation, AblationRequest)
QUANTIZATION = register_lab_kind("quantization", quantization_report, QuantReportRequest)
EXPLORE_START = register_lab_kind("explore_start", explore_start, ExploreStartRequest, save=False)
EXPLORE_MAP = register_lab_kind("explore_map", explore_map, ExploreMapRequest)
EXPLORE_TOKENS = register_lab_kind(
    "explore_tokens", explore_tokens, ExploreTokensRequest, save=False
)
EXPLORE_BRANCHES = register_lab_kind("explore_branches", explore_branches, ExploreBranchesRequest)


async def _run_lab(kind: str, request, priority: int = PRIORITY_INTERACTIVE):
    payload = request.model_dump(mode="json")
    return await job_queue.run(kind, payload, label_for(kind, payload), priority)


@router.post(
    "/lab/quantization-report",
    response_model=QuantReportResponse,
    summary="Render with full precision and quantized variants; compare images and analyses",
)
async def handle_quantization_report(request: QuantReportRequest = Body(...)):
    try:
        # long: waits its turn like a render instead of jumping the queue
        return await _run_lab(QUANTIZATION, request, priority=0)
    except Exception as e:
        raise to_http_error(e, "quantization report") from e


@router.post(
    "/lab/explore/start",
    response_model=ExploreStartResponse,
    summary="Explorer: render (or replay a path) and show how the image formed",
)
async def handle_explore_start(request: ExploreStartRequest = Body(...)):
    try:
        return await _run_lab(EXPLORE_START, request)
    except Exception as e:
        raise to_http_error(e, "explorer") from e


@router.post(
    "/lab/explore/map",
    response_model=ExploreMapResponse,
    summary="Explorer: when does a region settle? (branches at several steps, measured)",
)
async def handle_explore_map(request: ExploreMapRequest = Body(...)):
    try:
        return await _run_lab(EXPLORE_MAP, request)
    except Exception as e:
        raise to_http_error(e, "explorer map") from e


@router.post(
    "/lab/explore/tokens",
    response_model=ExploreTokensResponse,
    summary="Explorer: which words act on a region, with alternatives",
)
async def handle_explore_tokens(request: ExploreTokensRequest = Body(...)):
    try:
        return await _run_lab(EXPLORE_TOKENS, request)
    except Exception as e:
        raise to_http_error(e, "explorer words") from e


@router.post(
    "/lab/explore/branches",
    response_model=ExploreBranchesResponse,
    summary="Explorer: render branches of a region (seeds, words, mixes, LoRA weights)",
)
async def handle_explore_branches(request: ExploreBranchesRequest = Body(...)):
    try:
        return await _run_lab(EXPLORE_BRANCHES, request)
    except Exception as e:
        raise to_http_error(e, "explorer branches") from e


@router.post(
    "/lab/taylor-preview",
    response_model=TaylorResponse,
    summary="Image + d(image)/d(guidance) for instant previews",
)
async def handle_taylor(request: TaylorRequest = Body(...)):
    try:
        return await _run_lab(TAYLOR, request)
    except Exception as e:
        raise to_http_error(e, "Taylor preview") from e


@router.post(
    "/lab/attention-maps",
    response_model=AttentionMapResponse,
    summary="Per-word cross-attention maps (SD 1.x/SDXL)",
)
async def handle_attention_maps(request: AttentionMapRequest = Body(...)):
    try:
        return await _run_lab(ATTENTION, request)
    except Exception as e:
        raise to_http_error(e, "attention maps") from e


@router.post(
    "/lab/concept-attention",
    response_model=ConceptAttentionResponse,
    summary="ConceptAttention saliency maps and segmentation (FLUX.1, FLUX.2, SD3/3.5)",
)
async def handle_concept_attention(request: ConceptAttentionRequest = Body(...)):
    try:
        return await _run_lab(CONCEPTS, request)
    except Exception as e:
        raise to_http_error(e, "concept attention") from e


@router.post(
    "/lab/token-ablation",
    response_model=AblationResponse,
    summary="Leave-one-word-out influence, measured with DINO features",
)
async def handle_token_ablation(request: AblationRequest = Body(...)):
    try:
        return await _run_lab(ABLATION, request)
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
    support = concept_support(entry.pipeline)
    return {
        "model_id": model_id,
        "pipeline": entry.pipeline_class,
        "layers": layers,
        "suggestions": suggestions,
        # ConceptAttention: number of double-stream blocks (0 = use cross-attention maps instead)
        "concept_blocks": support[1] if support else 0,
    }
