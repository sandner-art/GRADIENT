# src/gradient/model_manager/inventory.py
"""Discover, describe and delete locally installed models."""

import json
import logging
import shutil
from pathlib import Path

from gradient.core.config import get_model_search_dirs, get_model_subdir_name
from gradient.core.families import family_from_class

from . import manifest
from .schemas import GenerationDefaults, LocalModel, ModelType

logger = logging.getLogger(__name__)

WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft"}


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        # skip huggingface_hub's local metadata cache
        if ".cache" in p.parts:
            continue
        if p.is_file():
            total += p.stat().st_size
    return total


def _classify(item: Path, model_type: str) -> str | None:
    """Return the format of a candidate entry, or None if it is not a model."""
    if _is_hidden(item):
        return None
    if item.is_file():
        if item.suffix.lower() not in WEIGHT_SUFFIXES:
            return None
        if model_type == "diffusers":
            return None  # diffusers folder only holds directories
        if item.suffix.lower() == ".gguf":
            return "gguf"
        return "single_file"
    if item.is_dir():
        if model_type == "diffusers" and (item / "model_index.json").is_file():
            return "diffusers"
        if model_type == "lora" and (item / "adapter_config.json").is_file():
            return "peft"
        if model_type == "lora" and any(item.glob("*.safetensors")):
            return "directory"
        if model_type in ("vae", "controlnet") and (item / "config.json").is_file():
            return "diffusers"
        if model_type == "vision" and (item / "config.json").is_file():
            return "directory"
    return None


def iter_type_dirs(model_type: str) -> list[Path]:
    sub = get_model_subdir_name(model_type)
    return [base / sub for base in get_model_search_dirs()]


def manifest_key(model_type: str, name: str) -> str:
    return f"{get_model_subdir_name(model_type)}/{name}"


def _pipeline_info(folder: Path) -> dict[str, str]:
    """Pipeline class and family from a diffusers folder's model_index.json (models added by hand)."""
    try:
        cls = json.loads((folder / "model_index.json").read_text(encoding="utf-8")).get(
            "_class_name"
        )
    except (OSError, ValueError):
        return {}
    if not isinstance(cls, str):
        return {}
    info = {"pipeline": cls}
    family = family_from_class(cls)
    if family:
        info["family"] = family
    return info


def list_local_models(model_type: ModelType, with_sizes: bool = True) -> list[LocalModel]:
    entries = manifest.all_entries()
    found: dict[str, LocalModel] = {}
    for type_dir in iter_type_dirs(model_type):
        if not type_dir.is_dir():
            continue
        try:
            children = sorted(type_dir.iterdir())
        except OSError as e:
            logger.error("Cannot scan %s: %s", type_dir, e)
            continue
        for item in children:
            fmt = _classify(item, model_type)
            if fmt is None or item.name in found:
                continue  # first search dir wins on name clashes
            meta = entries.get(manifest_key(model_type, item.name)) or {}
            if fmt == "diffusers" and not meta.get("pipeline"):
                meta = {**meta, **_pipeline_info(item)}
            defaults = meta.get("defaults")
            found[item.name] = LocalModel(
                name=item.name,
                type=model_type,
                format=fmt,  # type: ignore[arg-type]
                path=str(item),
                size_bytes=dir_size(item) if with_sizes else 0,
                family=meta.get("family"),
                pipeline=meta.get("pipeline"),
                source={
                    k: meta[k]
                    for k in ("repo_id", "revision", "url", "catalog_id", "license")
                    if meta.get(k)
                }
                or None,
                defaults=GenerationDefaults.model_validate(defaults) if defaults else None,
            )
    return list(found.values())


def list_model_names(model_type: ModelType) -> list[str]:
    return [m.name for m in list_local_models(model_type, with_sizes=False)]


def find_model_path(model_id_or_partial_path: str, model_type: str) -> Path | None:
    """Resolve a model name to a path: absolute path, or name inside a configured type folder."""
    candidate = Path(model_id_or_partial_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve() if candidate.exists() else None
    if ".." in candidate.parts:
        return None
    for type_dir in iter_type_dirs(model_type):
        path = (type_dir / candidate).resolve()
        if path.exists() and path.is_relative_to(type_dir.resolve()):
            return path
    return None


def get_manifest_meta(model_type: str, name: str) -> dict:
    return manifest.get_entry(manifest_key(model_type, name)) or {}


def delete_local_model(model_type: ModelType, name: str) -> Path:
    """Delete a model folder/file inside a configured models dir. Returns the deleted path."""
    if not name or "/" in name or "\\" in name or name in (".", "..") or _is_hidden(Path(name)):
        raise ValueError(f"Invalid model name: {name!r}")
    for type_dir in iter_type_dirs(model_type):
        path = type_dir / name
        if not path.exists():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(type_dir.resolve()):
            raise ValueError("Refusing to delete outside the models directory")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        manifest.remove_entry(manifest_key(model_type, name))
        logger.info("Deleted local model %s", resolved)
        return resolved
    raise FileNotFoundError(f"{model_type} model {name!r} not found")
