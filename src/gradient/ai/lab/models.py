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
