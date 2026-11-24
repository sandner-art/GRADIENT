# src/gradient/ai/diffusion/quantization.py
"""Quantized model loading.

A quantization is named by a short spec, chosen per render (Text2ImageRequest.quantization),
so the same model can be rendered at full precision and quantized side by side:

    none                  full precision (the default)
    bnb-8bit, bnb-4bit    bitsandbytes LLM.int8 / NF4 (CUDA only; diffusers-format models)
    torchao-int8          int8 weights (torchao), any device
    torchao-int8-dynamic  int8 weights and activations (torchao)
    torchao-fp8           float8 weights (torchao)
    gguf:<file>           a GGUF file for the denoiser (city96-style FLUX/SD3.5/... files, or
                          any GGUF holding the denoiser's weights), other parts from the model

Append "+text" to also quantize the text encoders (useful for T5, Mistral or Qwen encoders),
e.g. "bnb-4bit+text". By default only the denoiser (UNet / transformer) is quantized: it is
the largest part and the one whose precision matters least for memory/quality trade-offs.

Diffusers-format models are quantized while loading (weights never exist at full precision on
the GPU). Single-file checkpoints are loaded first, then quantized in place (torchao only).
"""

from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gradient.core.config import get_setting

logger = logging.getLogger(__name__)

BUILTIN: dict[str, dict[str, Any]] = {
    "none": {"label": "Full precision", "backend": None, "factor": 1.0},
    "bnb-8bit": {
        "label": "8-bit (bitsandbytes)",
        "backend": "bitsandbytes",
        "factor": 0.55,
        "cuda": True,
        "about": "LLM.int8 weights; about half the memory, close to full quality.",
    },
    "bnb-4bit": {
        "label": "4-bit NF4 (bitsandbytes)",
        "backend": "bitsandbytes",
        "factor": 0.3,
        "cuda": True,
        "about": "NF4 weights; about a third of the memory, small quality loss.",
    },
    "torchao-int8": {
        "label": "int8 weights (torchao)",
        "backend": "torchao",
        "factor": 0.55,
        "about": "int8 weight-only; about half the memory, very close to full quality.",
    },
    "torchao-int8-dynamic": {
        "label": "int8 weights + activations (torchao)",
        "backend": "torchao",
        "factor": 0.55,
        "about": "int8 matrix multiplies; faster on GPUs with int8 kernels, slightly lossier.",
    },
    "torchao-fp8": {
        "label": "fp8 weights (torchao)",
        "backend": "torchao",
        "factor": 0.55,
        "about": "float8 weight-only; best on GPUs with fp8 support (Ada/Hopper and newer).",
    },
}
PACKAGES = {"bitsandbytes": "bitsandbytes", "torchao": "torchao", "gguf": "gguf"}


@dataclass(frozen=True)
class QuantSpec:
    name: str  # a BUILTIN key or "gguf"
    text: bool = False  # also quantize the text encoders
    gguf_file: str | None = None

    @property
    def key(self) -> str:
        base = f"gguf:{self.gguf_file}" if self.name == "gguf" else self.name
        return base + ("+text" if self.text else "")

    @property
    def backend(self) -> str | None:
        return "gguf" if self.name == "gguf" else BUILTIN[self.name]["backend"]

    @property
    def factor(self) -> float:
        """Rough share of full-precision memory (for choosing a memory mode)."""
        return 0.35 if self.name == "gguf" else BUILTIN[self.name]["factor"]


def parse(spec: str | None) -> QuantSpec:
    raw = (spec or "").strip() or str(get_setting("performance.default_quantization", "none"))
    text = raw.endswith("+text")
    if text:
        raw = raw[: -len("+text")]
    if raw.startswith("gguf:"):
        name = raw[len("gguf:") :].strip()
        if not name:
            raise ValueError("gguf: needs a file name, e.g. gguf:flux1-dev-Q4_K_S.gguf")
        return QuantSpec("gguf", text, name)
    if raw not in BUILTIN:
        raise ValueError(
            f"Unknown quantization {raw!r}; use one of {', '.join(BUILTIN)} or gguf:<file>"
        )
    if raw == "none" and text:
        raise ValueError("'+text' needs a quantization, e.g. torchao-int8+text")
    return QuantSpec(raw, text)


def _installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def unavailable_reason(spec: QuantSpec) -> str | None:
    """Why this quantization can't be used here (None if it can)."""
    backend = spec.backend
    if backend is None:
        return None
    if not _installed(PACKAGES[backend]):
        return (
            f"needs the '{PACKAGES[backend]}' package: install GRADIENT's quant extras "
            "(pip install -e .[quant])"
        )
    if spec.name != "gguf" and BUILTIN[spec.name].get("cuda") and not torch.cuda.is_available():
        return "bitsandbytes needs an NVIDIA GPU"
    return None


def gguf_files() -> list[str]:
    """GGUF denoiser files in the checkpoints folders (LLM GGUFs live in models/llm)."""
    from gradient.model_manager.inventory import list_model_names

    return sorted(n for n in list_model_names("checkpoints") if n.lower().endswith(".gguf"))


def options() -> list[dict[str, Any]]:
    """Every quantization with availability, for the UI."""
    out = []
    for name, meta in BUILTIN.items():
        spec = QuantSpec(name)
        out.append(
            {
                "value": name,
                "label": meta["label"],
                "about": meta.get("about", ""),
                "backend": meta["backend"],
                "available": unavailable_reason(spec) is None,
                "reason": unavailable_reason(spec),
            }
        )
    gguf_reason = unavailable_reason(QuantSpec("gguf", gguf_file="x"))
    for f in gguf_files():
        out.append(
            {
                "value": f"gguf:{f}",
                "label": f"GGUF: {f}",
                "about": "Pre-quantized denoiser weights; the rest comes from the selected model.",
                "backend": "gguf",
                "available": gguf_reason is None,
                "reason": gguf_reason,
            }
        )
    return out


# ------------------------------------------------------------------------------ configs
def _torchao_config(name: str):
    import torchao.quantization as tq

    return {
        "torchao-int8": tq.Int8WeightOnlyConfig,
        "torchao-int8-dynamic": tq.Int8DynamicActivationInt8WeightConfig,
        "torchao-fp8": tq.Float8WeightOnlyConfig,
    }[name]()


def _component_config(spec: QuantSpec, dtype: torch.dtype, library: str):
    """A quantization config for one component, from diffusers or transformers."""
    if library == "transformers":
        import transformers as lib
    else:
        import diffusers as lib
    if spec.name == "bnb-8bit":
        return lib.BitsAndBytesConfig(load_in_8bit=True)
    if spec.name == "bnb-4bit":
        return lib.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    return lib.TorchAoConfig(_torchao_config(spec.name))


def read_components(load_path: str) -> dict[str, tuple[str, str]]:
    """{component: (library, class)} from model_index.json (local folder or Hub repo)."""
    index = Path(load_path) / "model_index.json"
    try:
        if index.is_file():
            data = json.loads(index.read_text(encoding="utf-8"))
        else:
            from diffusers import DiffusionPipeline

            data = DiffusionPipeline.load_config(load_path)
    except Exception as e:  # unknown layout: quantize what we can find later
        logger.info("Could not read the components of %s: %s", load_path, e)
        return {}
    return {
        k: (v[0], v[1])
        for k, v in data.items()
        if not k.startswith("_") and isinstance(v, list) and len(v) == 2 and v[0] and v[1]
    }


def denoiser_name(components: dict[str, tuple[str, str]]) -> str:
    for name in ("transformer", "unet", "prior", "decoder"):
        if name in components:
            return name
    return "transformer"


def text_encoder_names(components: dict[str, tuple[str, str]]) -> list[str]:
    return [k for k, (lib, _) in components.items() if k.startswith("text_encoder")]


def load_args(spec: QuantSpec, load_path: str, dtype: torch.dtype) -> dict[str, Any]:
    """Extra from_pretrained arguments for a diffusers-format model."""
    if spec.name == "none":
        return {}
    reason = unavailable_reason(spec)
    if reason:
        raise ValueError(f"Can't use {spec.key}: {reason}")
    components = read_components(load_path)
    denoiser = denoiser_name(components)
    args: dict[str, Any] = {}
    mapping: dict[str, Any] = {}
    if spec.name == "gguf":
        args[denoiser] = load_gguf_denoiser(spec, load_path, components, dtype)
    else:
        mapping[denoiser] = _component_config(spec, dtype, "diffusers")
    if spec.text:
        text_spec = spec if spec.name != "gguf" else QuantSpec("torchao-int8")
        if unavailable_reason(text_spec):
            raise ValueError(f"Can't quantize the text encoders: {unavailable_reason(text_spec)}")
        for name in text_encoder_names(components):
            library = components[name][0]
            mapping[name] = _component_config(text_spec, dtype, library)
    if mapping:
        from diffusers import PipelineQuantizationConfig

        args["quantization_config"] = PipelineQuantizationConfig(quant_mapping=mapping)
    return args


def load_gguf_denoiser(spec: QuantSpec, load_path: str, components, dtype: torch.dtype):
    import diffusers
    from diffusers import GGUFQuantizationConfig

    from gradient.model_manager.inventory import find_model_path

    path = find_model_path(spec.gguf_file or "", "checkpoints")
    if path is None or not path.is_file():
        raise ValueError(
            f"GGUF file {spec.gguf_file!r} not found; put it in the diffusion checkpoints folder"
        )
    name = denoiser_name(components)
    cls_name = components.get(name, ("diffusers", "FluxTransformer2DModel"))[1]
    cls = getattr(diffusers, cls_name)
    logger.info("Loading %s from %s (GGUF)", cls_name, path.name)
    try:
        return cls.from_single_file(
            str(path),
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            config=load_path,
            subfolder=name,
            dtype=dtype,
        )
    except Exception as e:
        raise ValueError(
            f"{path.name} doesn't fit this model's {cls_name} ({e}). Pick the GGUF made for "
            "this model (same family and size)."
        ) from e


def quantize_in_place(pipe, spec: QuantSpec) -> None:
    """For single-file checkpoints: quantize the loaded denoiser (and text encoders) with torchao."""
    if spec.name == "none":
        return
    reason = unavailable_reason(spec)
    if reason:
        raise ValueError(f"Can't use {spec.key}: {reason}")
    if spec.backend != "torchao":
        raise ValueError(
            f"{spec.key} needs a diffusers-format model; for single-file checkpoints use "
            "torchao-int8, torchao-int8-dynamic or torchao-fp8"
        )
    from torchao.quantization import quantize_

    names = [n for n in ("transformer", "unet") if getattr(pipe, n, None) is not None][:1]
    if spec.text:
        names += [
            n
            for n in ("text_encoder", "text_encoder_2", "text_encoder_3")
            if getattr(pipe, n, None)
        ]
    for n in names:
        quantize_(getattr(pipe, n), _torchao_config(spec.name))
        logger.info("Quantized %s in place (%s)", n, spec.key)


# ------------------------------------------------------------------------------- report
def weight_bytes(module: torch.nn.Module | None) -> int:
    """Bytes the module's weights occupy (quantized tensors counted by their storage)."""
    if module is None:
        return 0
    total = 0
    for p in list(module.parameters()) + list(module.buffers()):
        total += _tensor_bytes(p)
    return total


def _tensor_bytes(t: torch.Tensor) -> int:
    flatten = getattr(type(t), "__tensor_flatten__", None)
    if flatten is not None and type(t).__name__ not in ("Parameter", "Tensor"):
        try:
            names, _ = t.__tensor_flatten__()
            return sum(_tensor_bytes(getattr(t, n)) for n in names)
        except Exception:
            pass
    quant_state = getattr(t, "quant_state", None)  # bitsandbytes 4-bit
    extra = 0
    if quant_state is not None:
        absmax = getattr(quant_state, "absmax", None)
        extra = absmax.numel() * absmax.element_size() if torch.is_tensor(absmax) else 0
    return t.numel() * t.element_size() + extra


def describe(pipe) -> dict[str, Any]:
    """Weight sizes of the main parts, for reports."""
    out = {}
    for n in ("transformer", "unet", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
        m = getattr(pipe, n, None)
        if isinstance(m, torch.nn.Module):
            out[n] = round(weight_bytes(m) / 1024**2, 3)
    return out
