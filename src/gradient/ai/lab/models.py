# src/gradient/ai/lab/models.py
"""Request/response models for the Lab endpoints (/api/v1/lab/*)."""

from typing import Annotated, Literal

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


# ----------------------------------------------------------------- explorer
class ExploreOp(BaseModel):
    """One step on an exploration path: the region is opened again at `step` and denoised
    again (new noise, maybe another prompt); everything outside it is kept."""

    step: int = Field(..., ge=0, description="Step (of this render's steps) to branch at")
    mask_base64: str | None = Field(
        default=None, description="PNG, white (or opaque) = the region; null = the whole image"
    )
    kind: Literal["seed", "prompt", "mix", "lora"] = "seed"
    seed: int = Field(..., ge=0, description="Seed of the fresh noise in the region")
    prompt: str | None = Field(default=None, description="Prompt while branching (null: base)")
    mix_prompt: str | None = Field(
        default=None, description="Blend the prompt's embeddings with this prompt's"
    )
    mix_weight: float = Field(default=0.5, ge=0, le=1)
    lora_weight: float | None = Field(
        default=None, description="Weight of the (first) LoRA while branching (null: as set)"
    )
    label: str | None = None


class ExploreBase(BaseModel):
    base_request: Text2ImageRequest = Field(..., description="The render to explore")
    init_image_base64: str | None = Field(
        default=None, description="Explore an image-to-image render: its input image"
    )
    strength: float | None = Field(default=None, gt=0, le=1, description="Its strength")
    path: list[ExploreOp] = Field(
        default_factory=list, description="Branches taken so far, in order"
    )


class ExploreStartRequest(ExploreBase):
    frames: int = Field(default=8, ge=2, le=16, description="Denoised estimates to show")


class ExploreFrame(BaseModel):
    step: int
    sigma: float
    image_base64: str = Field(..., description="What the model predicted at this step")
    still_changing_base64: str = Field(
        ..., description="Where that prediction still changes before the end (free estimate)"
    )
    still_changing: float = Field(..., description="Mean of that map, 1 at the first frame")


class ExploreStartResponse(BaseModel):
    image_base64: str
    seed: int
    steps: int = Field(..., description="Steps of this render (branch steps are 0..steps-1)")
    scheduler: str
    prompt: str = Field(..., description="The prompt in effect after the path")
    frames: list[ExploreFrame]
    note: str | None = None
    performance_ms: float


class ExploreMapRequest(ExploreBase):
    mask_base64: str | None = Field(default=None, description="The region (null: whole image)")
    steps: list[int] | None = Field(default=None, description="Steps to test (default: spread)")
    num_steps: int = Field(default=6, ge=2, le=12)
    branches: int = Field(default=3, ge=1, le=8, description="New seeds per step")
    seed: int | None = Field(default=None, description="First branch seed (default: seed + 1000)")
    feature_model: str | None = None


class ExploreMapPoint(BaseModel):
    step: int
    sigma: float
    change: float = Field(
        ..., description="How much the region changed, relative to a completely new region (step 0)"
    )
    distance: float = Field(..., description="Mean distance of the branches from the image")
    spread: float = Field(..., description="Standard deviation of that distance over branches")
    estimate: float | None = Field(
        default=None, description="The free estimate from the render itself (same scale)"
    )
    images_base64: list[str] = Field(..., description="Thumbnails of the branches")
    heatmap_base64: str = Field(..., description="Where the branches differed")


class ExploreMapResponse(BaseModel):
    image_base64: str = Field(..., description="Overview: one row per step")
    reference_image_base64: str
    seed: int
    points: list[ExploreMapPoint]
    window: list[float] | None = Field(
        default=None, description="Steps where the change falls from 0.8 to 0.2 (interpolated)"
    )
    suggested: dict[str, int] = Field(
        default_factory=dict, description="Steps for 'big' changes, 'variations' and 'details'"
    )
    estimate_error: float | None = Field(
        default=None, description="Mean |estimate - change| over the steps"
    )
    estimate_correlation: float | None = Field(
        default=None, description="Spatial correlation of the free map with the measured one"
    )
    metric: str = Field(..., description="'dino' (feature patches) or 'pixels'")
    feature_model: str | None = None
    feature_note: str | None = None
    summary: str
    performance_ms: float


class ExploreTokensRequest(ExploreBase):
    mask_base64: str | None = None
    method: Literal["auto", "attention", "ablation"] = Field(
        default="auto",
        description="attention: one render (cross-attention or ConceptAttention); "
        "ablation: one render per word (any model)",
    )
    max_words: int = Field(default=8, ge=1, le=20)
    alternatives: int = Field(default=6, ge=0, le=20)
    feature_model: str | None = None


class TokenSuggestion(BaseModel):
    word: str
    score: float = Field(..., description="0..1, how much the word acts in the region")
    strength: float = Field(..., description="Its attention (or ablation change) in the region")
    lift: float = Field(..., description="How concentrated it is in the region (1 = evenly spread)")
    alternatives: list[str] = Field(default_factory=list)
    overlay_base64: str | None = None


class ExploreTokensResponse(BaseModel):
    image_base64: str
    prompt: str
    method: str
    words: list[TokenSuggestion]
    note: str | None = None
    performance_ms: float


class ExploreBranchSpec(BaseModel):
    kind: Literal["seed", "prompt", "mix", "lora"] = "seed"
    seed: int | None = Field(default=None, description="Default: a new seed per branch")
    prompt: str | None = None
    mix_prompt: str | None = None
    mix_weight: float = Field(default=0.5, ge=0, le=1)
    lora_weight: float | None = None
    label: str | None = None


class ExploreBranchesRequest(ExploreBase):
    mask_base64: str | None = None
    step: int = Field(..., ge=0)
    branches: list[ExploreBranchSpec] = Field(..., min_length=1, max_length=12)
    feature_model: str | None = None


class ExploreBranch(BaseModel):
    label: str
    op: ExploreOp = Field(..., description="Append it to the path to continue from this branch")
    image_base64: str
    parameters: dict = Field(..., description="Saved with the image (gallery, PNG metadata)")
    change: float | None = Field(default=None, description="Distance from the image in the region")
    outside_change: float | None = Field(default=None, description="Distance outside the region")


class ExploreBranchesResponse(BaseModel):
    reference_image_base64: str
    seed: int
    step: int
    sigma: float
    branches: list[ExploreBranch]
    metric: str
    feature_note: str | None = None
    performance_ms: float


class EmbedRequest(BaseModel):
    """Embeddings for other apps' metrics (POST /lab/embed)."""

    images: list[str] = Field(
        default_factory=list,
        max_length=64,
        description="Base64 PNG, JPEG or WEBP images (data: URLs work)",
    )
    output_ids: list[int] = Field(
        default_factory=list, max_length=256, description="Gallery images, after `images`"
    )
    texts: list[Annotated[str, Field(max_length=2000)]] = Field(
        default_factory=list,
        max_length=256,
        description="Texts to embed in the same space (needs a CLIP or SigLIP model)",
    )
    feature_model: str | None = Field(
        default=None,
        description="A vision model (folder name or Hub id): DINO for images only (default: "
        "the Lab's DINOv3/DINOv2), CLIP or SigLIP for images and texts",
    )


class EmbedResponse(BaseModel):
    model: str
    kind: Literal["dino", "clip"]
    dim: int
    image_embeddings: list[list[float]] = Field(
        ..., description="L2-normalized, in request order (images, then output_ids)"
    )
    text_embeddings: list[list[float]] | None = None
    note: str | None = None


# ----------------------------------------------------------------- style check
StylePosition = Literal["template", "before", "after", "chunk", "blocks"]


class StyleCheckRequest(BaseModel):
    base_request: Text2ImageRequest = Field(
        ..., description="Settings to render (its styles are replaced per variant)"
    )
    styles: list[str] = Field(..., min_length=1, max_length=16, description="Style names")
    positions: list[StylePosition] = Field(
        default_factory=lambda: ["template"], min_length=1, max_length=4
    )
    style_starts: list[Annotated[float, Field(ge=0.0, lt=1.0)]] = Field(
        default_factory=lambda: [0.0],
        min_length=1,
        max_length=4,
        description="Fractions of the steps from which the styles apply",
    )
    num_seeds: int = Field(default=2, ge=1, le=4, description="Seeds (seed, seed+1, ...)")
    feature_model: str | None = Field(
        default=None, description="DINO model for style strength (default: DINOv3/DINOv2)"
    )
    clip_model: str | None = Field(
        default=None,
        description=(
            "CLIP or SigLIP model for 'content kept' (default: the first one installed in the "
            "vision folder; 'none' skips it)"
        ),
    )


class StyleCheckCell(BaseModel):
    style: str | None  # None: the unstyled image
    position: str | None = None
    start: float | None = None
    seed: int
    image_base64: str
    parameters: dict


class StyleCheckRow(BaseModel):
    style: str
    position: str
    start: float
    strength: float = Field(..., description="Mean DINO distance to the unstyled image")
    strength_rel: float | None = Field(
        default=None, description="strength / distance between unstyled seeds (1 = a new seed)"
    )
    content: float | None = Field(
        default=None, description="Mean CLIP similarity of the images with the unstyled prompt"
    )
    content_rel: float | None = Field(
        default=None, description="content / the unstyled images' (1 = kept as well)"
    )
    consistency: float | None = Field(
        default=None, description="Mean DINO similarity between the style's images across seeds"
    )
    tokens: int | None = None
    chunks: int | None = None
    notes: list[str] = Field(default_factory=list)
    cells: list[int] = Field(default_factory=list, description="Indexes into cells, by seed")


class StyleCheckResponse(BaseModel):
    model_id: str
    prompt: str
    seeds: list[int]
    feature_model: str
    clip_model: str | None = None
    seed_distance: float | None = Field(
        default=None, description="Mean DINO distance between the unstyled images of two seeds"
    )
    baseline_content: float | None = None
    baseline_consistency: float | None = None
    baseline: list[int] = Field(default_factory=list, description="Unstyled cells, by seed")
    rows: list[StyleCheckRow]
    cells: list[StyleCheckCell]
    image_base64: str = Field(..., description="Contact sheet: unstyled row, then each variant")
    notes: list[str] = Field(default_factory=list)
    elapsed_s: float
