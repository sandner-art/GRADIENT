# src/gradient/model_manager/manifest.py
"""A small JSON manifest that records where downloaded models came from.

Stored at <first models search dir>/manifest.json and keyed by
"<type subdir>/<name>" (e.g. "diffusers/stable-diffusion-xl-base-1.0").
It lets the UI show source/licence/family info and recommended defaults.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

from gradient.core.config import get_model_search_dirs

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def manifest_path() -> Path:
    base = get_model_search_dirs()[0]
    base.mkdir(parents=True, exist_ok=True)
    return base / "manifest.json"


def _read() -> dict[str, Any]:
    path = manifest_path()
    if not path.is_file():
        return {"version": 1, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("entries", {})
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read model manifest %s: %s", path, e)
        return {"version": 1, "entries": {}}


def _write(data: dict[str, Any]) -> None:
    path = manifest_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def all_entries() -> dict[str, dict[str, Any]]:
    with _lock:
        return dict(_read()["entries"])


def get_entry(key: str) -> dict[str, Any] | None:
    return all_entries().get(key)


def set_entry(key: str, meta: dict[str, Any]) -> None:
    with _lock:
        data = _read()
        data["entries"][key] = meta
        _write(data)


def remove_entry(key: str) -> None:
    with _lock:
        data = _read()
        if data["entries"].pop(key, None) is not None:
            _write(data)


def find_by_catalog_id(catalog_id: str) -> tuple[str, dict[str, Any]] | None:
    for key, meta in all_entries().items():
        if meta.get("catalog_id") == catalog_id:
            return key, meta
    return None
