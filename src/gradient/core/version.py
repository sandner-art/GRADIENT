# src/gradient/core/version.py
"""GRADIENT's version: from pyproject.toml in a source checkout (so `git am` / `git pull`
show the new version right away), else from the installed package's metadata."""

import tomllib
from functools import cache
from importlib.metadata import PackageNotFoundError, version

from .config import PROJECT_ROOT

PACKAGE = "gradient-workbench"


@cache
def installed_version() -> str | None:
    try:
        return version(PACKAGE)
    except PackageNotFoundError:
        return None


@cache
def app_version() -> str:
    try:
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            return str(tomllib.load(f)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return installed_version() or "unknown"


def install_is_stale() -> bool:
    """The checkout is newer than what was installed (dependencies may be missing)."""
    installed = installed_version()
    return installed is not None and installed != app_version()
