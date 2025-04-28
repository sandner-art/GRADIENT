# gradient-project-root/src/gradient/ai/viz/models.py (NEW FILE)

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class LatentPoint(BaseModel):
    """Represents a single point in the projected latent space."""
    id: int # Simple identifier for the point
    x: float # Projected X coordinate
    y: float # Projected Y coordinate
    z: Optional[float] = None # Optional projected Z coordinate (for 3D)
    image_preview_base64: Optional[str] = None # Small Base64 encoded preview image

class LatentSpaceVizRequest(BaseModel):
    """Request to generate data for latent space visualization."""
    model_id: str = Field(..., description="Base model ID (used to infer VAE if not specified)")
    vae_id: Optional[str] = None # Allow specifying the VAE to use for decoding
    num_samples: int = Field(default=500, gt=50, le=5000, description="Number of points to sample")
    seed: Optional[int] = None # Seed for reproducibility
    # Add parameters later for UMAP (n_neighbors, min_dist) or other methods
    projection_method: str = Field(default="umap", description="Dimensionality reduction method (currently only 'umap')")
    generate_previews: bool = Field(default=True, description="Whether to decode samples and generate image previews")
    preview_resolution: int = Field(default=64, description="Resolution for generated preview images") # Small previews

class LatentSpaceVizResponse(BaseModel):
    """Response containing the projected points and potentially previews."""
    request_params: LatentSpaceVizRequest
    points: List[LatentPoint]
    projection_method: str
    performance_ms: Optional[float] = None