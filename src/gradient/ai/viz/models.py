# gradient-project-root/src/gradient/ai/viz/models.py

from typing import Literal

from pydantic import BaseModel, Field

from gradient.ai.diffusion.models import Text2ImageRequest


class LatentPoint(BaseModel):
    id: int
    x: float
    y: float
    z: float | None = None
    image_preview_base64: str | None = None


class LatentSpaceVizRequest(BaseModel):
    model_id: str = Field(...)
    vae_id: str | None = None
    num_samples: int = Field(default=500, gt=50, le=5000)
    seed: int | None = None
    projection_method: str = Field(default="umap")
    generate_previews: bool = Field(default=True)
    preview_resolution: int = Field(default=64)


class LatentSpaceVizResponse(BaseModel):
    request_params: LatentSpaceVizRequest
    points: list[LatentPoint]
    projection_method: str
    performance_ms: float | None = None
    note: str | None = None


class SensitivityTargetParameter(BaseModel):
    name: str = Field(..., description="'guidance_scale' or 'lora_weight:<index>'")
    delta: float | None = Field(
        default=None,
        gt=0,
        description="Finite-difference step (default: 5% of the value, min 0.05)",
    )


class SensitivityAnalysisRequest(BaseModel):
    base_request: Text2ImageRequest
    target_parameter: SensitivityTargetParameter
    analysis_target: Literal["pixel"] = Field(
        default="pixel", description="Space in which the change is measured"
    )


class SensitivityMapData(BaseModel):
    shape: list[int]
    mean_sensitivity: float
    max_sensitivity: float
    min_sensitivity: float
    std_dev_sensitivity: float
    heatmap_base64: str | None = None
    base_image_base64: str | None = None
    method: str = "central_difference"
    parameter_value: float | None = None
    delta: float | None = None


class SensitivityAnalysisResponse(BaseModel):
    request_params: SensitivityAnalysisRequest
    sensitivity_map: SensitivityMapData
    performance_ms: float | None = None
