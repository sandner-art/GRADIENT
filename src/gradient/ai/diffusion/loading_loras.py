# src/gradient/ai/diffusion/loading_loras.py
"""LoRA handling via diffusers' PEFT integration.

Supports single-file LoRAs (.safetensors, kohya/A1111/diffusers key formats) and
PEFT directories (adapter_config.json + adapter_model.safetensors), local or on
the Hugging Face Hub. Multiple LoRAs are combined with per-adapter weights.
"""

import logging

from diffusers import DiffusionPipeline

from gradient.model_manager.inventory import find_model_path

logger = logging.getLogger(__name__)


def apply_loras_to_pipeline(pipeline: DiffusionPipeline, loras: list | None) -> list[str]:
    """Load the given LoRAs (objects with .name and .weight) and activate them. Returns adapter names."""
    if not loras:
        return []
    if not callable(getattr(pipeline, "load_lora_weights", None)):
        raise ValueError(f"{type(pipeline).__name__} does not support LoRAs")

    names: list[str] = []
    weights: list[float] = []
    for i, lora in enumerate(loras):
        if isinstance(lora, dict):
            lora_name, weight = lora.get("name"), float(lora.get("weight", 1.0))
        else:
            lora_name, weight = lora.name, float(lora.weight)
        if not lora_name:
            continue
        adapter = f"g_adapter_{i}"
        path = find_model_path(lora_name, "lora")
        kwargs = {"adapter_name": adapter}
        if path is not None and path.is_file():
            source = str(path.parent)
            kwargs["weight_name"] = path.name
        elif path is not None and path.is_dir():
            source = str(path)
            if not (path / "adapter_config.json").is_file():
                files = sorted(path.glob("*.safetensors"))
                if len(files) == 1:
                    kwargs["weight_name"] = files[0].name
        else:
            source = lora_name  # Hub repo id
        logger.info("Loading LoRA %s (weight %.2f) as %s", lora_name, weight, adapter)
        try:
            pipeline.load_lora_weights(source, **kwargs)
        except Exception as e:
            raise ValueError(f"Could not load LoRA '{lora_name}': {e}") from e
        names.append(adapter)
        weights.append(weight)

    if names:
        loaded = set()
        if callable(getattr(pipeline, "get_list_adapters", None)):
            for adapters in pipeline.get_list_adapters().values():
                loaded.update(adapters)
        missing = [n for n in names if loaded and n not in loaded]
        if missing:
            logger.warning(
                "LoRA adapters %s did not match any layers of %s (wrong base model family?)",
                missing,
                type(pipeline).__name__,
            )
            keep = [(n, w) for n, w in zip(names, weights, strict=True) if n not in missing]
            names, weights = [n for n, _ in keep], [w for _, w in keep]
        if names:
            pipeline.set_adapters(names, adapter_weights=weights)
    return names


def unload_lora_weights(pipeline: DiffusionPipeline) -> None:
    if callable(getattr(pipeline, "unload_lora_weights", None)):
        try:
            pipeline.unload_lora_weights()
            logger.debug("Unloaded LoRA weights")
        except Exception as e:
            logger.warning("unload_lora_weights failed: %s", e)
