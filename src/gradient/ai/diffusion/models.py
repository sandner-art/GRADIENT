# gradient-project-root/src/gradient/ai/diffusion/models.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union, Any, Dict

# --- Base Model Update ---
class DiffusionTaskBase(BaseModel):
    """Base model for common diffusion parameters."""
    model_id: str = Field(..., description="Identifier for the base model (path or HF ID)")
    prompt: str
    negative_prompt: Optional[str] = ""
    height: int = Field(default=512, gt=0)
    width: int = Field(default=512, gt=0)
    num_inference_steps: int = Field(default=25, gt=0)
    guidance_scale: float = Field(default=7.5, ge=0)
    seed: Optional[int] = None
    # vae_id: Optional[str] = None # Keep commented until fully implemented

    custom_sigmas: Optional[List[float]] = Field(
        default=None,
        description="Optional list of sigma values for a custom noise schedule. If provided, num_inference_steps might be ignored or validated against list length."
    )

    # --- NEW: Scheduler Selection --- vvv
    scheduler_id: Optional[str] = Field(
        default=None,
        description="Optional identifier for the scheduler (e.g., 'DDIMScheduler', 'DPMSolverMultistepScheduler'). If None, uses pipeline default."
    )
    # --- NEW: Scheduler Selection --- ^^^

# --- Image Result (Unchanged) ---
class ImageResult(BaseModel):
    """Represents a single generated image result."""
    image_base64: str
    seed: int
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used for this specific image")
    # Add other metadata later (safety status, timings, etc.)

# --- Text2Image Models (Inherit scheduler_id) ---
class Text2ImageRequest(DiffusionTaskBase):
    """Request model for Text-to-Image generation."""
    num_outputs: int = Field(default=1, gt=0, le=4) # Limit batch size for now

class Text2ImageResponse(BaseModel):
    """Response model for Text-to-Image generation."""
    images: List[ImageResult]
    request_params: Text2ImageRequest # Keep original request type here
    performance_ms: Optional[float] = None

# --- Img2Img Models --- vvv
class ImageToImageRequest(DiffusionTaskBase):
    """Request model for Image-to-Image generation."""
    image_base64: str = Field(..., description="Base64 encoded initial image")
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Strength of the diffusion process. 1.0 means full diffusion (ignore initial image structure), 0.0 means no diffusion (return initial image).",
    )
    num_outputs: int = Field(default=1, gt=0, le=4) # Match T2I for consistency

class ImageToImageResponse(BaseModel):
    """Response model for Image-to-Image generation."""
    images: List[ImageResult]
    request_params: ImageToImageRequest
    performance_ms: Optional[float] = None
# --- Img2Img Models --- ^^^


# --- XYZ Grid Models (Unchanged for now, but base_parameters now includes scheduler_id) ---
class AxisParameter(BaseModel):
    """Defines which parameter varies along an axis and its values."""
    parameter: str = Field(..., description="Name of the parameter to vary (e.g., 'seed', 'guidance_scale', 'prompt', 'num_inference_steps', 'scheduler_id')") # Added scheduler_id example
    values: List[Any] = Field(..., min_length=1, description="List of values for the parameter along this axis")

    @field_validator('parameter')
    def validate_parameter_name(cls, v):
        # TODO: Add validation later to ensure parameter name is valid for DiffusionTaskBase or loaded model
        allowed_params = DiffusionTaskBase.model_fields.keys() # Get valid fields dynamically
        # Also allow img2img specific fields if context allows? Complex.
        # For now, just check base fields. Grid might need separate validation.
        # if v not in allowed_params:
        #     raise ValueError(f"Invalid parameter name for XYZ grid axis (check DiffusionTaskBase): {v}")
        return v

class XYZGridRequest(BaseModel):
    """Request model for generating an XYZ grid of images."""
    base_parameters: DiffusionTaskBase # This now includes scheduler_id by inheritance
    x_axis: Optional[AxisParameter] = None
    y_axis: Optional[AxisParameter] = None
    z_axis: Optional[AxisParameter] = None

    @field_validator('x_axis', 'y_axis', 'z_axis')
    def check_axis_parameter_uniqueness(cls, v, info):
         if v is None: return v
         if not v.values: raise ValueError("Axis values list cannot be empty")
         return v

class XYZGridResult(BaseModel):
    """Represents a single cell in the generated grid."""
    x_value: Optional[Any] = None
    y_value: Optional[Any] = None
    z_value: Optional[Any] = None
    parameters: Dict[str, Any] # Full parameters for this cell
    image_result: Optional[ImageResult] = None # The generated image
    error: Optional[str] = None # Error message if generation failed for this cell

class XYZGridResponse(BaseModel):
    """Response containing the grid results."""
    grid_results: List[XYZGridResult]
    request_params: XYZGridRequest # Echo back request