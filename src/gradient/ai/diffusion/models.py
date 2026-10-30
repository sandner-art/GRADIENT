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
    # Add other metadata later (safety status, timings, etc.)


# --- Text-to-Image Models ---


class Text2ImageRequest(DiffusionTaskBase):
    """Request model for Text-to-Image generation."""

    num_outputs: int = Field(default=1, gt=0, le=4)  # Limit batch size for now


class Text2ImageResponse(BaseModel):
    """Response model for Text-to-Image generation."""

    images: list[ImageResult]
    request_params: Text2ImageRequest  # Echo original request
    performance_ms: float | None = None


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
