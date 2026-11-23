# src/gradient/ai/diffusion/loading_loras.py
"""LoRA handling via diffusers' PEFT integration, with adapters kept loaded.

Supports single-file LoRAs (.safetensors, kohya/A1111/diffusers key formats) and
PEFT directories (adapter_config.json + adapter_model.safetensors), local or on
the Hugging Face Hub.

Each LoRA is loaded once per pipeline as a named adapter and stays in memory
(up to `performance.max_loaded_loras`, least recently used evicted). Changing
weights or switching between recently used LoRAs is then just `set_adapters`,
with no reload from disk. Without LoRAs the adapters are disabled, not removed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from diffusers import DiffusionPipeline

from gradient.core.config import get_setting
from gradient.model_manager.inventory import find_model_path

logger = logging.getLogger(__name__)


def adapter_name(lora_name: str) -> str:
    """Stable, PEFT-safe adapter name for a LoRA (same LoRA -> same adapter)."""
    slug = re.sub(r"[^A-Za-z0-9_]", "_", lora_name)[-40:].strip("_") or "lora"
    return f"g_{slug}_{hashlib.sha1(lora_name.encode()).hexdigest()[:6]}"


def _spec(lora: Any) -> tuple[str | None, float]:
    if isinstance(lora, dict):
        return lora.get("name"), float(lora.get("weight", 1.0))
    return getattr(lora, "name", None), float(getattr(lora, "weight", 1.0))


def _all_adapters(pipeline: DiffusionPipeline) -> set[str]:
    loaded: set[str] = set()
    if callable(getattr(pipeline, "get_list_adapters", None)):
        for adapters in pipeline.get_list_adapters().values():
            loaded.update(adapters)
    return loaded


def load_lora_adapter(pipeline: DiffusionPipeline, lora_name: str, adapter: str) -> bool:
    """Load one LoRA as `adapter`; False if it matched no layer (wrong model family)."""
    path = find_model_path(lora_name, "lora")
    kwargs: dict[str, Any] = {"adapter_name": adapter}
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
    logger.info("Loading LoRA %s as adapter %s", lora_name, adapter)
    try:
        pipeline.load_lora_weights(source, **kwargs)
    except Exception as e:
        raise ValueError(f"Could not load LoRA '{lora_name}': {e}") from e
    if adapter not in _all_adapters(pipeline):
        logger.warning(
            "LoRA %s did not match any layers of %s (wrong base model family?)",
            lora_name,
            type(pipeline).__name__,
        )
        return False
    return True


def activate_loras(pipeline: DiffusionPipeline, entry, loras: list | None) -> list[str]:
    """Make exactly `loras` (with their weights) active on `pipeline`.

    `entry` is the cached LoadedPipeline: it remembers which adapters are loaded
    (`entry.adapters`, name -> adapter, LRU order), which are active, and whether LoRA
    layers are currently disabled. Returns the active adapter names, in request order
    (LoRAs that matched no layer are skipped).
    """
    wanted = [(n, w) for n, w in (_spec(lo) for lo in (loras or [])) if n]
    if wanted and not callable(getattr(pipeline, "load_lora_weights", None)):
        raise ValueError(f"{type(pipeline).__name__} does not support LoRAs")

    if not wanted:
        if entry.active_adapters and callable(getattr(pipeline, "disable_lora", None)):
            pipeline.disable_lora()
            entry.lora_disabled = True
        entry.active_adapters = []
        return []

    names: list[str] = []
    weights: list[float] = []
    for lora_name, weight in wanted:
        adapter = entry.adapters.get(lora_name)
        if adapter is None:
            adapter = adapter_name(lora_name)
            if adapter in _all_adapters(pipeline):  # loaded before under this name
                entry.adapters[lora_name] = adapter
            elif load_lora_adapter(pipeline, lora_name, adapter):
                entry.adapters[lora_name] = adapter
            else:
                entry.unmatched.add(lora_name)
                continue
        entry.adapters.move_to_end(lora_name)
        if adapter not in names:
            names.append(adapter)
            weights.append(weight)

    # evict least recently used adapters beyond the limit (never the ones in use)
    limit = max(1, int(get_setting("performance.max_loaded_loras", 6)))
    for lora_name in list(entry.adapters):
        if len(entry.adapters) <= max(limit, len(names)):
            break
        adapter = entry.adapters[lora_name]
        if adapter in names:
            continue
        try:
            pipeline.delete_adapters(adapter)
            logger.info("Unloaded LoRA %s (keeping %d loaded)", lora_name, limit)
        except Exception as e:
            logger.warning("Could not unload LoRA %s: %s", lora_name, e)
        del entry.adapters[lora_name]

    if not names:
        if callable(getattr(pipeline, "disable_lora", None)):
            pipeline.disable_lora()
            entry.lora_disabled = True
        entry.active_adapters = []
        return []
    if entry.lora_disabled and callable(getattr(pipeline, "enable_lora", None)):
        pipeline.enable_lora()
        entry.lora_disabled = False
    pipeline.set_adapters(names, adapter_weights=weights)
    entry.active_adapters = names
    return names


def unload_lora_weights(pipeline: DiffusionPipeline) -> None:
    if callable(getattr(pipeline, "unload_lora_weights", None)):
        try:
            pipeline.unload_lora_weights()
            logger.debug("Unloaded LoRA weights")
        except Exception as e:
            logger.warning("unload_lora_weights failed: %s", e)
