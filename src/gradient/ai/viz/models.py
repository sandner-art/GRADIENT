# gradient-project-root/src/gradient/ai/viz/models.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sys # <--- Import sys

# Import the modules containing types referenced in string hints
# This ensures they are loaded into sys.modules
from gradient.ai.diffusion import models as diffusion_models
# If other models reference types from other modules via strings, import them here too

# --- Define all your models ---

class LatentPoint(BaseModel):
    # ... (definition)
    id: int
    x: float
    y: float
    z: Optional[float] = None
    image_preview_base64: Optional[str] = None

class LatentSpaceVizRequest(BaseModel):
    # ... (definition)
    model_id: str = Field(...)
    vae_id: Optional[str] = None
    num_samples: int = Field(default=500, gt=50, le=5000)
    seed: Optional[int] = None
    projection_method: str = Field(default="umap")
    generate_previews: bool = Field(default=True)
    preview_resolution: int = Field(default=64)

class LatentSpaceVizResponse(BaseModel):
    # ... (definition)
    request_params: LatentSpaceVizRequest
    points: List[LatentPoint]
    projection_method: str
    performance_ms: Optional[float] = None

class SensitivityTargetParameter(BaseModel):
    # ... (definition)
    name: str = Field(...)

class SensitivityAnalysisRequest(BaseModel):
    # --- Use ABSOLUTE path string for Forward Reference --- vvv
    base_request: "gradient.ai.diffusion.models.Text2ImageRequest"
    # --- Use ABSOLUTE path string for Forward Reference --- ^^^
    target_parameter: SensitivityTargetParameter
    analysis_target: str = Field(default="latent")

class SensitivityMapData(BaseModel):
    # ... (definition)
    shape: List[int]
    mean_sensitivity: float
    max_sensitivity: float
    min_sensitivity: float
    std_dev_sensitivity: float
    heatmap_base64: Optional[str] = None

class SensitivityAnalysisResponse(BaseModel):
    # ... (definition)
    request_params: SensitivityAnalysisRequest
    sensitivity_map: SensitivityMapData
    performance_ms: Optional[float] = None


# --- Call model_rebuild() at the end WITH updated globals --- vvv
# Create a dictionary containing the necessary modules/names for resolution
update_ns = {
    'gradient': sys.modules.get('gradient'), # Get the top-level gradient module
    # Add other top-level names needed for resolving type hints if any
}
# Filter out None values in case a module wasn't imported correctly
update_ns = {k: v for k, v in update_ns.items() if v is not None}

# Pass the current globals() updated with necessary modules
current_globals = globals()
current_globals.update(update_ns)

SensitivityAnalysisRequest.model_rebuild(
    _types_namespace=current_globals # Pass the updated namespace
)
SensitivityAnalysisResponse.model_rebuild(
     _types_namespace=current_globals
)
LatentSpaceVizResponse.model_rebuild(
     _types_namespace=current_globals
)
# Add other models needing rebuild here...