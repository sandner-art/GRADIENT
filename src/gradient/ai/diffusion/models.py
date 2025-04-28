# gradient-project-root/src/gradient/ai/diffusion/models.py

from pydantic import BaseModel, Field
from typing import Optional, List, Union

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
    # Add LoRA/ControlNet configs later
    # loras: Optional[List[LoraConfig]] = None
    # controlnets: Optional[List[ControlNetConfig]] = None
    # Add VAE later
    # vae_id: Optional[str] = None

class Text2ImageRequest(DiffusionTaskBase):
    """Request model for Text-to-Image generation."""
    num_outputs: int = Field(default=1, gt=0, le=4) # Limit batch size for now

class ImageResult(BaseModel):
    """Represents a single generated image result."""
    image_base64: str # Base64 encoded image data
    seed: int
    # Add other metadata later (safety status, timings, etc.)

class Text2ImageResponse(BaseModel):
    """Response model for Text-to-Image generation."""
    images: List[ImageResult]
    request_params: Text2ImageRequest # Echo back parameters used
    performance_ms: Optional[float] = None # Time taken