# gradient-project-root/src/gradient/ai/viz/sensitivity.py
"""Parameter sensitivity maps by central finite differences.

For a parameter p (guidance scale or a LoRA weight) we render the same seed at
p - d and p + d and measure, per pixel, |I(p+d) - I(p-d)| / (2d). The result shows
*where* in the image the parameter matters, for any model family (no gradients
through the sampler are needed, so it also works with offloading/quantization).

The previous prototype differentiated a detached latent with respect to itself,
which always produced a constant map; see docs/research/research_review_2026-10.md.
"""

import base64
import logging
import time
from io import BytesIO

import numpy as np
from PIL import Image

from gradient.ai.diffusion.models import Text2ImageRequest
from gradient.ai.diffusion.pipelines import generate_images, resolve_seed
from gradient.core.scheduling import run_gpu_task

from .models import SensitivityAnalysisRequest, SensitivityAnalysisResponse, SensitivityMapData

logger = logging.getLogger(__name__)

# 6-stop "magma-like" colour ramp (black -> purple -> orange -> pale yellow), no matplotlib needed
_RAMP = np.array(
    [[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]],
    dtype=np.float32,
)


def colorize(values: np.ndarray) -> Image.Image:
    v = values.astype(np.float32)
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99.5))
    v = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1) * (len(_RAMP) - 1)
    idx = np.floor(v).astype(int).clip(0, len(_RAMP) - 2)
    frac = (v - idx)[..., None]
    rgb = _RAMP[idx] * (1 - frac) + _RAMP[idx + 1] * frac
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def _b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _with_param(base: Text2ImageRequest, name: str, value: float) -> Text2ImageRequest:
    req = base.model_copy(deep=True)
    if name == "guidance_scale":
        req.guidance_scale = value
    elif name.startswith("lora_weight:"):
        idx = int(name.split(":", 1)[1])
        req.loras[idx].weight = value  # type: ignore[index]
    return req


def _current_value(base: Text2ImageRequest, name: str) -> float:
    if name == "guidance_scale":
        if not isinstance(base.guidance_scale, (int, float)):
            raise ValueError("Sensitivity analysis needs a constant guidance_scale")
        return float(base.guidance_scale)
    if name.startswith("lora_weight:"):
        try:
            idx = int(name.split(":", 1)[1])
            return float(base.loras[idx].weight)  # type: ignore[index]
        except (ValueError, IndexError, TypeError):
            raise ValueError(f"'{name}' does not refer to a LoRA in base_request.loras") from None
    raise ValueError("target_parameter.name must be 'guidance_scale' or 'lora_weight:<index>'")


def _render(req: Text2ImageRequest) -> np.ndarray:
    result = generate_images(req, "text2image", num_outputs=1)
    return np.asarray(result.images[0].convert("RGB"), dtype=np.float32) / 255.0


async def analyze_parameter_sensitivity(
    request: SensitivityAnalysisRequest,
) -> SensitivityAnalysisResponse:
    start = time.time()
    name = request.target_parameter.name
    base = request.base_request.model_copy(deep=True)
    base.seed = resolve_seed(base.seed)
    value = _current_value(base, name)
    delta = request.target_parameter.delta or max(0.05, abs(value) * 0.05)

    low = await run_gpu_task(_render, _with_param(base, name, value - delta))
    high = await run_gpu_task(_render, _with_param(base, name, value + delta))
    center = await run_gpu_task(
        generate_images, _with_param(base, name, value), "text2image", None, None, 1
    )

    sens = np.abs(high - low).mean(axis=-1) / (2 * delta)  # H x W, per unit of the parameter
    data = SensitivityMapData(
        shape=list(sens.shape),
        mean_sensitivity=float(sens.mean()),
        max_sensitivity=float(sens.max()),
        min_sensitivity=float(sens.min()),
        std_dev_sensitivity=float(sens.std()),
        heatmap_base64=_b64(colorize(sens)),
        base_image_base64=_b64(center.images[0]),
        method="central_difference",
        parameter_value=value,
        delta=delta,
    )
    logger.info(
        "Sensitivity of %s at %.3f (d=%.3f): mean %.4f", name, value, delta, data.mean_sensitivity
    )
    return SensitivityAnalysisResponse(
        request_params=request, sensitivity_map=data, performance_ms=(time.time() - start) * 1000
    )
