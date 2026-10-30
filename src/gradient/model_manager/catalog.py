# src/gradient/model_manager/catalog.py
"""Loads the curated model catalog (config/model_catalog.yaml)."""

import functools
import logging
from pathlib import Path

import yaml

from gradient.core.config import get_setting, resolve_project_path

from .schemas import CatalogEntry

logger = logging.getLogger(__name__)


def catalog_path() -> Path:
    return resolve_project_path(
        get_setting("downloads.catalog_file", "./config/model_catalog.yaml")
    )


@functools.cache
def _load(path_str: str, mtime: float) -> tuple[CatalogEntry, ...]:
    path = Path(path_str)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for item in raw.get("models", []):
        try:
            entry = CatalogEntry.model_validate(item)
        except Exception as e:
            logger.warning("Skipping invalid catalog entry %r: %s", item.get("id"), e)
            continue
        if entry.id in seen:
            logger.warning("Duplicate catalog id %r ignored", entry.id)
            continue
        seen.add(entry.id)
        entries.append(entry)
    logger.info("Loaded %d catalog entries from %s", len(entries), path)
    return tuple(entries)


def load_catalog() -> list[CatalogEntry]:
    path = catalog_path()
    if not path.is_file():
        logger.warning("Model catalog not found at %s", path)
        return []
    return list(_load(str(path), path.stat().st_mtime))


def get_catalog_entry(catalog_id: str) -> CatalogEntry | None:
    return next((e for e in load_catalog() if e.id == catalog_id), None)
