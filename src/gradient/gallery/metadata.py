# src/gradient/gallery/metadata.py
"""Metadata written into GRADIENT's PNGs.

Every image carries two text chunks:

* `grad_parameters`: the full JSON of the generation (exact reproduction in GRADIENT);
* `parameters`: the A1111-style summary that most image tools read (Kurator, ArtAgents,
  image viewers, Civitai...), e.g.

      a red lighthouse on a rocky coast
      Negative prompt: blurry
      Steps: 25, Sampler: DPMSolverMultistep, CFG scale: 7.0, Seed: 42, Size: 512x512,
      Model: sd15-base, Lora: detail-tweaker:0.8, Version: GRADIENT 0.5.0
"""

from __future__ import annotations

import json
from typing import Any

from PIL.PngImagePlugin import PngInfo


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("gradient-workbench")
    except Exception:
        return "dev"


def _clean(value: Any) -> str:
    # A1111 separates fields with ", " and keys with ": "; keep values on one line
    return str(value).replace("\n", " ").strip()


def a1111_parameters(params: dict[str, Any]) -> str:
    """A1111-compatible `parameters` text for a generation's parameter dict."""
    lines = [_clean(params.get("prompt") or "")]
    negative = params.get("negative_prompt")
    if negative:
        lines.append(f"Negative prompt: {_clean(negative)}")
    fields: list[tuple[str, Any]] = [
        ("Steps", params.get("num_inference_steps")),
        ("Sampler", params.get("scheduler_id") or "default"),
        ("CFG scale", params.get("guidance_scale")),
        ("Seed", params.get("seed")),
    ]
    if params.get("width") and params.get("height"):
        fields.append(("Size", f"{params['width']}x{params['height']}"))
    fields.append(("Model", params.get("model_id")))
    if params.get("vae_id"):
        fields.append(("VAE", params["vae_id"]))
    loras = params.get("loras") or []
    if loras:
        fields.append(
            (
                "Lora",
                "; ".join(
                    f"{lo.get('name')}:{float(lo.get('weight', 1.0)):g}"
                    for lo in loras
                    if isinstance(lo, dict)
                ),
            )
        )
    if params.get("strength") is not None:
        fields.append(("Denoising strength", params["strength"]))
    if params.get("family"):
        fields.append(("Model family", params["family"]))
    if params.get("quantization") not in (None, "", "none"):
        fields.append(("Quantization", params["quantization"]))
    fields.append(("Version", f"GRADIENT {_version()}"))
    lines.append(", ".join(f"{k}: {_clean(v)}" for k, v in fields if v is not None and v != ""))
    return "\n".join(lines)


def png_info(params: dict[str, Any]) -> PngInfo:
    info = PngInfo()
    info.add_text("grad_parameters", json.dumps(params, default=str))
    info.add_text("parameters", a1111_parameters(params))
    return info


def output_fields(params: dict[str, Any]) -> dict[str, Any]:
    """Columns of GeneratedOutput that are copied out of the parameters for filtering."""

    def as_int(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "model_id": params.get("model_id"),
        "prompt": params.get("prompt"),
        "seed": as_int(params.get("seed")),
        "width": as_int(params.get("width")),
        "height": as_int(params.get("height")),
    }
