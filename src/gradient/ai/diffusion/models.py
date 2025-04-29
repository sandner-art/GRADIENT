# src/gradient/ai/diffusion/models.py

from pydantic import BaseModel, Field, field_validator, ConfigDict # Import ConfigDict for Pydantic v2
from typing import Optional, List, Union, Any, Dict, Tuple

# --- Helper Models ---

class LoRAParameter(BaseModel):
    """Defines parameters for a single LoRA."""
    name: str = Field(..., description="Name/filename/path/Hub ID of the LoRA")
    weight: float = Field(default=1.0, description="Weight/scale to apply to the LoRA (effectiveness depends on backend loading method)")

    # Add Pydantic v2 config if needed, e.g., for ORM mode if saved to DB directly
    # model_config = ConfigDict(from_attributes=True)


# --- Base Diffusion Task Model ---

class DiffusionTaskBase(BaseModel):
    """Base model for common diffusion parameters."""
    model_id: str = Field(..., description="Identifier for the base model (path or HF ID)")
    prompt: str
    negative_prompt: Optional[str] = "" # Allow empty string
    height: int = Field(default=512, gt=0)
    width: int = Field(default=512, gt=0)
    num_inference_steps: int = Field(default=25, gt=0)
    guidance_scale: Union[float, str, List[Tuple[int, float]]] = Field( # Allow scheduled CFG later
        default=7.5,
        description="Guidance scale. Can be float or definition for scheduled CFG (future).",
        ge=0.0 # Ensure float values are non-negative if applicable
    )
    seed: Optional[int] = None
    # vae_id: Optional[str] = None # Add later when VAE selection is robust

    custom_sigmas: Optional[List[float]] = Field(
        default=None,
        description="Optional list of sigma values for a custom noise schedule. If provided, num_inference_steps might be ignored or validated against list length."
    )
    scheduler_id: Optional[str] = Field(
        default=None,
        description="Optional identifier for the scheduler (e.g., 'DDIMScheduler'). If None, uses pipeline default."
    )

    # --- LoRA/Embedding Fields --- vvv
    loras: Optional[List[LoRAParameter]] = Field(
        default=None,
        description="List of LoRAs to apply with their weights."
    )
    # Embeddings are often handled via prompt, but explicit list can help loading/validation
    embeddings: Optional[List[str]] = Field(
        default=None,
        description="List of Textual Inversion embedding triggers/filenames to ensure are loaded (if backend supports explicit loading)."
    )
    # --- LoRA/Embedding Fields --- ^^^

    # Validation for height/width multiples of 8
    @field_validator('height', 'width')
    def check_multiple_of_8(cls, value):
        if value is not None and value % 8 != 0:
            raise ValueError(f'Value must be a multiple of 8, got {value}')
        return value

    # Add Pydantic v2 config if needed
    # model_config = ConfigDict(extra='allow') # Example if extra fields needed


# --- Image Result Model ---

class ImageResult(BaseModel):
    """Represents a single generated image result."""
    image_base64: str
    seed: int
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used for this specific image")
    # Add other metadata later (safety status, timings, etc.)


# --- Text-to-Image Models ---

class Text2ImageRequest(DiffusionTaskBase):
    """Request model for Text-to-Image generation."""
    num_outputs: int = Field(default=1, gt=0, le=4) # Limit batch size for now


class Text2ImageResponse(BaseModel):
    """Response model for Text-to-Image generation."""
    images: List[ImageResult]
    request_params: Text2ImageRequest # Echo original request
    performance_ms: Optional[float] = None


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
    images: List[ImageResult]
    request_params: ImageToImageRequest # Echo original request
    performance_ms: Optional[float] = None


# --- XYZ Grid Specific Models ---

class AxisParameter(BaseModel):
    """Defines which parameter varies along an axis and its values."""
    parameter: str = Field(..., description="Name of the parameter to vary (e.g., 'seed', 'guidance_scale', 'prompt', 'loras')") # Added 'loras' example
    values: List[Any] = Field(..., min_length=1, description="List of values for the parameter along this axis")

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
    base_parameters: DiffusionTaskBase # Base now includes loras/embeddings
    x_axis: Optional[AxisParameter] = None
    y_axis: Optional[AxisParameter] = None
    z_axis: Optional[AxisParameter] = None

    @field_validator('x_axis', 'y_axis', 'z_axis')
    def check_axis_values_non_empty(cls, v):
        if v is not None and not v.values:
             raise ValueError("Axis 'values' list cannot be empty")
        return v

    # TODO: Add model_validator to ensure parameter names are unique across x, y, z axes.


class XYZGridResult(BaseModel):
    """Represents a single cell in the generated grid."""
    x_value: Optional[Any] = None
    y_value: Optional[Any] = None
    z_value: Optional[Any] = None
    parameters: Dict[str, Any] # Full parameters used for this cell (resolved from base + axes)
    image_result: Optional[ImageResult] = None # The generated image
    error: Optional[str] = None # Error message if generation failed for this cell


class XYZGridResponse(BaseModel):
    """Response containing the grid results."""
    grid_results: List[XYZGridResult]
    request_params: XYZGridRequest # Echo back request