# src/gradient/ai/lab/models.py
"""Request/response models for the Lab endpoints (/api/v1/lab/*)."""

from typing import Literal

from pydantic import BaseModel, Field

from gradient.ai.diffusion.models import Text2ImageRequest


class TaylorRequest(BaseModel):
    base_request: Text2ImageRequest
    method: Literal["auto", "jvp", "finite_difference"] = Field(
        default="auto",
        description="jvp: exact forward-mode derivative (SD 1.x/SDXL); finite_difference: any model (3 renders)",
    )
    delta: float | None = Field(
        default=None, gt=0, description="Finite-difference step (default 0.5)"
    )


class TaylorResponse(BaseModel):
    method: str
    guidance_scale: float
    seed: int
    base_image_base64: str
    tangent_png_base64: str = Field(
        ..., description="RGB tangent encoded as 8-bit: value = 128 + 127 * tangent / tangent_scale"
    )
    tangent_scale: float
    magnitude_heatmap_base64: str
    mean_abs_tangent: float
    performance_ms: float
    note: str | None = Field(
        default=None, description="E.g. why 'auto' fell back from JVP to finite differences"
    )


class AttentionMapRequest(BaseModel):
    base_request: Text2ImageRequest
    words: list[str] | None = Field(
        default=None, description="Words to map (default: every word in the prompt)"
    )
    layers: str = Field(
        default="*attn2", description="Glob over attention layer names (comma-separated allowed)"
    )
    start: float = Field(default=0.0, ge=0, le=1, description="First step to record (fraction)")
    end: float = Field(
        default=1.0, ge=0, le=1, description="Stop recording at this fraction of the steps"
    )


class WordMap(BaseModel):
    word: str
    token_positions: list[int]
    attention_share: float = Field(..., description="Mean attention probability the word receives")
    heatmap_base64: str
    overlay_base64: str


class AttentionMapResponse(BaseModel):
    image_base64: str
    seed: int
    maps: list[WordMap]
    layers_recorded: list[str]
    performance_ms: float


class AblationRequest(BaseModel):
    base_request: Text2ImageRequest
    mode: Literal["remove_word", "mask_embedding"] = Field(
        default="remove_word",
        description="remove_word: drop the word from the prompt (any model); mask_embedding: replace its token "
        "embeddings with the empty prompt's (SD 1.x/SDXL)",
    )
    words: list[str] | None = Field(
        default=None, description="Words to ablate (default: content words of the prompt)"
    )
    max_words: int = Field(default=12, ge=1, le=40)
    feature_model: str | None = Field(
        default=None,
        description="Vision model for distances (default: DINOv3 ViT-S/16, falls back to DINOv2)",
    )


class AblationResult(BaseModel):
    word: str
    ablated_prompt: str | None = None
    global_distance: float = Field(
        ..., description="1 - cosine similarity of the global (CLS) features"
    )
    patch_distance: float = Field(..., description="Mean per-patch feature distance")
    heatmap_base64: str
    overlay_base64: str
    ablated_image_base64: str


class AblationResponse(BaseModel):
    base_image_base64: str
    seed: int
    mode: str
    feature_model: str
    feature_note: str | None = None
    results: list[AblationResult]
    performance_ms: float


class ConceptAttentionRequest(BaseModel):
    base_request: Text2ImageRequest
    concepts: list[str] | None = Field(
        default=None,
        description="Concept words to locate (they need not occur in the prompt; single words work best). "
        "Default: the prompt's content words plus 'background'",
    )
    blocks: str | None = Field(
        default=None,
        description="Double-stream blocks to average, e.g. '9-18', '12,15,18' or 'all' "
        "(default: the deeper half)",
    )
    start: float = Field(default=0.0, ge=0, le=1, description="First step to record (fraction)")
    end: float = Field(
        default=1.0, ge=0, le=1, description="Stop recording at this fraction of the steps"
    )


class ConceptMap(BaseModel):
    concept: str
    color: str = Field(..., description="Colour of this concept in the segmentation image")
    area: float = Field(..., description="Fraction of image patches where this concept wins")
    mean_probability: float = Field(..., description="Mean concept probability over the image")
    heatmap_base64: str
    overlay_base64: str


class ConceptAttentionResponse(BaseModel):
    image_base64: str
    seed: int
    segmentation_base64: str = Field(
        ..., description="Image tinted by the winning concept per patch"
    )
    concepts: list[ConceptMap]
    blocks_recorded: list[int]
    num_blocks: int = Field(..., description="Double-stream blocks in this model")
    steps_recorded: int
    performance_ms: float


# ----------------------------------------------------------------- quantization fidelity
class QuantReportRequest(BaseModel):
    base_request: Text2ImageRequest = Field(
        ..., description="Settings to render (its quantization is replaced per variant)"
    )
    variants: list[str] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="Quantizations to test, e.g. ['torchao-int8', 'bnb-4bit', 'gguf:<file>']",
    )
    reference: str = Field(default="none", description="What the variants are compared with")
    num_seeds: int = Field(
        default=3, ge=2, le=8, description="Seeds per variant (seed, seed+1, ...)"
    )
    taylor: bool = Field(
        default=True, description="Also compare ∂image/∂guidance (do Lab analyses still agree?)"
    )
    feature_model: str | None = Field(
        default=None,
        description="Vision model for the feature distances (default: DINOv3 ViT-S/16, DINOv2 small)",
    )


class QuantVariantReport(BaseModel):
    quantization: str
    label: str
    error: str | None = None
    weights_mb: dict[str, float] = Field(default_factory=dict)
    denoiser_mb: float | None = None
    load_seconds: float | None = None
    seconds_per_image: float | None = None
    peak_memory_mb: float | None = Field(default=None, description="CUDA peak while rendering")
    psnr: float | None = Field(
        default=None, description="dB against the reference (higher: closer)"
    )
    ssim: float | None = None
    mean_abs_diff: float | None = Field(default=None, description="0..255 per channel")
    feature_distance: float | None = Field(
        default=None, description="Mean DINO patch cosine distance to the reference"
    )
    relative_change: float | None = Field(
        default=None,
        description="feature_distance divided by the distance between two reference seeds: "
        "0 = identical, 1 = as different as a new seed",
    )
    taylor_cosine: float | None = Field(
        default=None, description="Cosine similarity of ∂image/∂w with the reference's"
    )
    taylor_method: str | None = None
    verdict: str = ""
    strip_base64: str | None = Field(
        default=None, description="Per seed: reference | variant | difference"
    )


class QuantReportResponse(BaseModel):
    model_id: str
    seeds: list[int]
    reference: str
    seed_spread: float | None = Field(
        default=None, description="Mean DINO distance between reference images of different seeds"
    )
    feature_model: str | None = None
    feature_note: str | None = None
    variants: list[QuantVariantReport] = Field(..., description="The reference first")
    image_base64: str = Field(
        ..., description="Overview: one row per quantization, one column per seed"
    )
    summary: str
    performance_ms: float
