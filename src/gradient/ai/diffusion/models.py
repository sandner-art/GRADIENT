# gradient-project-root/src/gradient/ai/diffusion/models.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union, Any, Dict

# --- Existing Models ---
class DiffusionTaskBase(BaseModel):
    """Base model for common diffusion parameters."""
    model_id: str = Field(..., description="Identifier for the base model (path or HF ID)")
    prompt: str
    negative_prompt: Optional[str] = ""
    height: int = Field(default=512, gt=0)
    width: int = Field(default=512, gt=0)
    num_inference_steps: int = Field(default=25, gt=0)
    guidance_scale: float = Field(default=7.5, ge=0)
    seed: Optional[int] = None # For grids, this might be a starting seed or ignored if axis uses seed
    # Add VAE later
    # vae_id: Optional[str] = None

    # --- NEW: Custom Noise Schedule --- vvv
    custom_sigmas: Optional[List[float]] = Field(
        default=None,
        description="Optional list of sigma values for a custom noise schedule. If provided, num_inference_steps might be ignored or validated against list length."
    )
    # --- NEW: Custom Noise Schedule --- ^^^

class ImageResult(BaseModel):
    """Represents a single generated image result."""
    image_base64: str
    seed: int
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used for this specific image")
    # Add other metadata later (safety status, timings, etc.)

# --- XYZ Grid Specific Models ---

class AxisParameter(BaseModel):
    """Defines which parameter varies along an axis and its values."""
    parameter: str = Field(..., description="Name of the parameter to vary (e.g., 'seed', 'guidance_scale', 'prompt', 'num_inference_steps')")
    values: List[Any] = Field(..., min_length=1, description="List of values for the parameter along this axis")

    @field_validator('parameter')
    def validate_parameter_name(cls, v):
        # TODO: Add validation later to ensure parameter name is valid for DiffusionTaskBase or loaded model
        # allowed_params = {'seed', 'guidance_scale', 'prompt', 'negative_prompt', 'num_inference_steps', ...}
        # if v not in allowed_params:
        #     raise ValueError(f"Invalid parameter name for XYZ grid axis: {v}")
        return v

class XYZGridRequest(BaseModel):
    """Request model for generating an XYZ grid of images."""
    # Base parameters that are fixed unless overridden by an axis
    base_parameters: DiffusionTaskBase

    # Define up to 3 axes
    x_axis: Optional[AxisParameter] = None
    y_axis: Optional[AxisParameter] = None
    z_axis: Optional[AxisParameter] = None # Optional third dimension

    @field_validator('x_axis', 'y_axis', 'z_axis')
    def check_axis_parameter_uniqueness(cls, v, info):
        """Ensure the same parameter isn't used for multiple axes."""
        # This basic check runs *before* all fields are validated, so it's complex to check across axes here.
        # A model validator is better suited for cross-field checks.
        # For now, we rely on later logic to handle potential overlaps gracefully.
        # A more robust implementation would use model_validator.
        if v is None:
            return v
        if not v.values:
             raise ValueError("Axis values list cannot be empty")
        # Basic type check - refine later based on parameter
        # param = v.parameter
        # expected_type = getattr(DiffusionTaskBase.__annotations__, param, None) # Needs refinement
        # for val in v.values: check type if possible
        return v

    # TODO: Add model_validator to ensure parameter names are unique across x, y, z axes.

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
    # Maybe add combined grid image later?
    # grid_image_base64: Optional[str] = None
    request_params: XYZGridRequest # Echo back request

# --- Update existing models slightly ---
class Text2ImageRequest(DiffusionTaskBase):
    """Request model for Text-to-Image generation."""
    num_outputs: int = Field(default=1, gt=0, le=4) # Limit batch size for now

class Text2ImageResponse(BaseModel):
    """Response model for Text-to-Image generation."""
    images: List[ImageResult]
    request_params: Text2ImageRequest # Keep original request type here
    performance_ms: Optional[float] = None