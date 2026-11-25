# src/gradient/ai/diffusion/models.py

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# --- Helper Models ---


class LoRAParameter(BaseModel):
    """Defines parameters for a single LoRA."""

    name: str = Field(..., description="Name/filename/path/Hub ID of the LoRA")
    weight: float = Field(
        default=1.0,
        description="Weight/scale to apply to the LoRA (effectiveness depends on backend loading method)",
    )

    # Add Pydantic v2 config if needed, e.g., for ORM mode if saved to DB directly
    # model_config = ConfigDict(from_attributes=True)


class ModulationLane(BaseModel):
    """One lane on the denoising timeline, active for steps in [start, end) (fractions of the run)."""

    type: Literal["guidance", "prompt", "attention_scale", "lora_weight"] = Field(
        ..., description="What the lane changes while active"
    )
    start: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Start of the lane as a fraction of the steps"
    )
    end: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="End of the lane (exclusive) as a fraction of the steps",
    )
    value: float | None = Field(
        default=None,
        description="guidance: CFG value; attention_scale: output scale (0 skips the layers); lora_weight: weight; prompt: blend factor (blend mode)",
    )
    prompt: str | None = Field(
        default=None, description="prompt lanes: the prompt to switch/blend to"
    )
    mode: Literal["replace", "blend"] = Field(
        default="replace", description="prompt lanes: replace or blend the embeddings"
    )
    targets: str | None = Field(
        default=None,
        description="attention_scale lanes: glob over attention layer names, e.g. 'up_blocks.*attn2' or 'transformer_blocks.1?.attn'",
    )
    lora_index: int | None = Field(
        default=None, ge=0, description="lora_weight lanes: index into the request's loras"
    )

    @model_validator(mode="after")
    def _check(self) -> "ModulationLane":
        if self.end <= self.start:
            raise ValueError("lane end must be greater than start")
        if self.type in ("guidance", "attention_scale", "lora_weight") and self.value is None:
            raise ValueError(f"{self.type} lanes need a value")
        if self.type == "prompt" and not self.prompt:
            raise ValueError("prompt lanes need a prompt")
        if self.type == "attention_scale" and not self.targets:
            raise ValueError("attention_scale lanes need targets (a layer-name pattern)")
        if self.type == "lora_weight" and self.lora_index is None:
            raise ValueError("lora_weight lanes need lora_index")
        return self


# --- Base Diffusion Task Model ---


class DiffusionTaskBase(BaseModel):
    """Base model for common diffusion parameters."""

    model_id: str = Field(..., description="Identifier for the base model (path or HF ID)")
    prompt: str
    negative_prompt: str | None = ""  # Allow empty string
    prompt_syntax: Literal["a1111", "comfy", "plain"] = Field(
        default="a1111",
        description=(
            "How SD 1.x/2.x and SDXL read the prompts: 'a1111' (weights like (word:1.2), "
            "prompts longer than 75 tokens, BREAK, [from:to:when] editing and [a|b] "
            "alternation; weights applied as A1111 does), 'comfy' (the same, weights applied "
            "as ComfyUI does) or 'plain' (the text as it is, cut at 77 tokens). Other model "
            "families always get the text as it is."
        ),
    )
    styles: list[str] | None = Field(
        default=None,
        description=(
            "Prompt styles by name (GET /styles), applied in order: A1111 styles.csv, where "
            "{prompt} in a style marks the place of the prompt"
        ),
    )
    style_position: Literal["template", "before", "after", "chunk"] = Field(
        default="template",
        description=(
            "Where the styles' text goes: 'template' (A1111: into {prompt}, else after the "
            "prompt), 'before' or 'after' the prompt, or 'chunk' (after it, in its own "
            "75-token chunk via BREAK on SD 1.x/2.x and SDXL)"
        ),
    )
    style_start: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Add the styles only from this fraction of the steps (prompt editing; SD 1.x/2.x "
            "and SDXL), so the composition comes from the prompt alone"
        ),
    )
    height: int = Field(default=512, gt=0)
    width: int = Field(default=512, gt=0)
    num_inference_steps: int = Field(default=25, gt=0)
    guidance_scale: float | str | list[tuple[int, float]] = Field(
        default=7.5,
        description=(
            "Guidance scale. A number, or a schedule (applied per step on classic-CFG models such as SD1.5/SDXL/SD3): "
            "'linear(9,3)', 'cosine(9,3)', keyframes '9@0,5@15,3@30' (value@step, linearly interpolated), "
            "or a list of [step, value] pairs."
        ),
    )
    seed: int | None = None
    vae_id: str | None = Field(
        default=None, description="Optional VAE override (name in the vae folder or Hub id)"
    )
    quantization: str | None = Field(
        default=None,
        description=(
            "Load the model quantized: none, bnb-8bit, bnb-4bit, torchao-int8, "
            "torchao-int8-dynamic, torchao-fp8 or gguf:<file>; add '+text' to quantize the text "
            "encoders too. Empty: the model's default (usually none). See GET /system/quantization."
        ),
    )

    custom_sigmas: list[float] | None = Field(
        default=None,
        description=(
            "Custom noise schedule passed to the scheduler's set_timesteps(sigmas=...). "
            "Overrides num_inference_steps. Requires a scheduler that accepts sigmas "
            "(e.g. EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler)."
        ),
    )
    scheduler_id: str | None = Field(
        default=None,
        description="Optional identifier for the scheduler (e.g., 'DDIMScheduler'). If None, uses pipeline default.",
    )

    # --- LoRA/Embedding Fields --- vvv
    loras: list[LoRAParameter] | None = Field(
        default=None, description="List of LoRAs to apply with their weights."
    )
    # Embeddings are often handled via prompt, but explicit list can help loading/validation
    embeddings: list[str] | None = Field(
        default=None,
        description="List of Textual Inversion embedding triggers/filenames to ensure are loaded (if backend supports explicit loading).",
    )
    # --- LoRA/Embedding Fields --- ^^^

    modulation: list[ModulationLane] | None = Field(
        default=None,
        description="Structural modulation lanes applied during denoising (see docs/user_guide.md)",
    )

    @field_validator("styles", mode="before")
    @classmethod
    def _styles_list(cls, v):
        """A single name (e.g. an XYZ axis value) is a list of one; 'none' or '' is no style."""
        if isinstance(v, str):
            names = [n.strip() for n in v.split(" + ")]
            return [n for n in names if n and n.lower() != "none"]
        return v

    # Validation for height/width multiples of 8
    @field_validator("height", "width")
    @classmethod
    def check_multiple_of_8(cls, value):
        if value is not None and value % 8 != 0:
            raise ValueError(f"Value must be a multiple of 8, got {value}")
        return value

    @field_validator("guidance_scale")
    @classmethod
    def check_guidance(cls, value):
        if isinstance(value, (int, float)) and value < 0:
            raise ValueError("guidance_scale must be >= 0")
        if isinstance(value, str):
            from .guidance import parse_guidance_schedule

            parse_guidance_schedule(value, 10)  # raises ValueError on bad syntax
        return value

    # Add Pydantic v2 config if needed
    # model_config = ConfigDict(extra='allow') # Example if extra fields needed


# --- Image Result Model ---


class ImageResult(BaseModel):
    """Represents a single generated image result."""

    image_base64: str
    seed: int
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters used for this specific image"
    )
    output_id: int | None = Field(default=None, description="Gallery id of the saved image")
    run_uuid: str | None = Field(default=None, description="Run this image belongs to")


# --- Text-to-Image Models ---


class Text2ImageRequest(DiffusionTaskBase):
    """Request model for Text-to-Image generation."""

    num_outputs: int = Field(default=1, gt=0, le=4)  # Limit batch size for now


class Text2ImageResponse(BaseModel):
    """Response model for Text-to-Image generation."""

    images: list[ImageResult]
    request_params: Text2ImageRequest  # Echo original request
    performance_ms: float | None = None
    run_uuid: str | None = None


# --- Image-to-Image Models ---


class ImageToImageRequest(DiffusionTaskBase):
    """Request model for Image-to-Image generation."""

    image_base64: str = Field(..., description="Base64 encoded initial image")
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Strength of the diffusion process (0.0 preserves original, 1.0 ignores original structure).",
    )
    num_outputs: int = Field(default=1, gt=0, le=4)


class ImageToImageResponse(BaseModel):
    """Response model for Image-to-Image generation."""

    images: list[ImageResult]
    request_params: ImageToImageRequest  # Echo original request
    performance_ms: float | None = None
    run_uuid: str | None = None


# --- XYZ Grid Specific Models ---


class AxisParameter(BaseModel):
    """Defines which parameter varies along an axis and its values."""

    parameter: str = Field(
        ...,
        description="Name of the parameter to vary (e.g., 'seed', 'guidance_scale', 'prompt', 'loras')",
    )  # Added 'loras' example
    values: list[Any] = Field(
        ..., min_length=1, description="List of values for the parameter along this axis"
    )

    # Basic validation - more complex type checks might be needed in XYZ grid logic
    # @field_validator('parameter')
    # def validate_parameter_name(cls, v):
    #     allowed_base = DiffusionTaskBase.model_fields.keys()
    #     # Add specific fields from request types if needed
    #     if v not in allowed_base: # Simplified check
    #          # raise ValueError(f"Invalid parameter name for XYZ grid axis: {v}")
    #          pass # Allow custom keys for now? Or validate against known T2I/I2I keys?
    #     return v


class XYZGridRequest(BaseModel):
    """Request model for generating an XYZ grid of images."""

    base_parameters: DiffusionTaskBase  # Base now includes loras/embeddings
    x_axis: AxisParameter | None = None
    y_axis: AxisParameter | None = None
    z_axis: AxisParameter | None = None

    @field_validator("x_axis", "y_axis", "z_axis")
    def check_axis_values_non_empty(cls, v):
        if v is not None and not v.values:
            raise ValueError("Axis 'values' list cannot be empty")
        return v

    # TODO: Add model_validator to ensure parameter names are unique across x, y, z axes.


class XYZGridResult(BaseModel):
    """Represents a single cell in the generated grid."""

    x_value: Any | None = None
    y_value: Any | None = None
    z_value: Any | None = None
    parameters: dict[str, Any]  # Full parameters used for this cell (resolved from base + axes)
    image_result: ImageResult | None = None  # The generated image
    error: str | None = None  # Error message if generation failed for this cell


class XYZGridResponse(BaseModel):
    """Response containing the grid results."""

    grid_results: list[XYZGridResult]
    request_params: XYZGridRequest  # Echo back request
    run_uuid: str | None = None
