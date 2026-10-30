# gradient-project-root/src/gradient/core/config.py

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root is 3 levels up from src/gradient/core/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "default_settings.yaml"
USER_SETTINGS_PATH = CONFIG_DIR / "config.yaml"  # User override file
LOGGING_CONFIG_PATH = CONFIG_DIR / "logging_config.yaml"

# Load .env from the project root (not the current working directory).
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

settings: dict[str, Any] = {}
logging_config: dict[str, Any] = {}

# Model type -> settings key under paths.models
MODEL_TYPES = (
    "diffusers",
    "checkpoints",
    "vae",
    "llm",
    "lora",
    "controlnet",
    "embeddings",
    "vision",
)


def load_yaml_config(file_path: Path) -> dict[str, Any]:
    """Loads configuration from a YAML file (empty dict if missing/invalid)."""
    if not file_path.is_file():
        logger.debug(f"Configuration file not found: {file_path}")
        return {}
    try:
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        logger.error(f"Error reading config file {file_path}: {e}")
        return {}


def deep_update(source: dict, overrides: dict) -> dict:
    """Recursively update a dictionary."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(source.get(key), dict):
            source[key] = deep_update(source[key], value)
        else:
            source[key] = value
    return source


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_settings() -> None:
    """Loads default settings, merges user overrides, then environment overrides."""
    global settings
    merged = deep_update(
        load_yaml_config(DEFAULT_SETTINGS_PATH), load_yaml_config(USER_SETTINGS_PATH)
    )

    app = merged.setdefault("app", {})
    if host := os.getenv("GRADIENT_APP_HOST"):
        app["host"] = host
    if port := os.getenv("GRADIENT_APP_PORT"):
        try:
            app["port"] = int(port)
        except ValueError:
            logger.warning(f"Invalid GRADIENT_APP_PORT: {port!r} (must be an integer)")
    if (reload_flag := _env_bool("GRADIENT_APP_RELOAD")) is not None:
        app["reload"] = reload_flag

    perf = merged.setdefault("performance", {})
    if mode := os.getenv("GRADIENT_MEMORY_MODE"):
        perf["memory_mode"] = mode
    if device := os.getenv("GRADIENT_DEVICE"):
        perf["device"] = device

    paths_cfg = merged.setdefault("paths", {})
    if data_dir := os.getenv("GRADIENT_DATA_DIR"):
        paths_cfg["data"] = data_dir
    if outputs_dir := os.getenv("GRADIENT_OUTPUTS_DIR"):
        paths_cfg["outputs"] = outputs_dir

    models_cfg = paths_cfg.setdefault("models", {})
    if model_dir := os.getenv("GRADIENT_MODEL_DIR"):
        dirs = list(models_cfg.get("search_dirs") or [])
        if model_dir not in dirs:
            models_cfg["search_dirs"] = [model_dir, *dirs]

    settings = merged


def load_logging_config() -> None:
    """Loads the logging configuration and ensures the log directory exists."""
    global logging_config
    logging_config = load_yaml_config(LOGGING_CONFIG_PATH)
    # Log files are relative to the project, not to the folder the server was started from.
    for handler in (logging_config.get("handlers") or {}).values():
        filename = handler.get("filename") if isinstance(handler, dict) else None
        if filename:
            path = Path(filename)
            path = path if path.is_absolute() else PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            handler["filename"] = str(path)


load_settings()
load_logging_config()


def get_setting(key_path: str, default: Any = None) -> Any:
    """Get a setting using dot notation, e.g. 'app.host'."""
    value: Any = settings
    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return default if value is None else value


def resolve_project_path(path_like: str | os.PathLike) -> Path:
    """Resolve a config path: absolute paths stay as-is, relative ones are anchored at PROJECT_ROOT."""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_model_search_dirs() -> list[Path]:
    """Configured model base directories, in priority order."""
    return [
        resolve_project_path(d) for d in get_setting("paths.models.search_dirs", ["./data/models"])
    ]


def get_model_subdir_name(model_type: str) -> str:
    return get_setting(f"paths.models.{model_type}", model_type)


def get_primary_model_dir(model_type: str, create: bool = True) -> Path:
    """Directory where newly downloaded models of `model_type` are stored."""
    path = get_model_search_dirs()[0] / get_model_subdir_name(model_type)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_outputs_dir(create: bool = True) -> Path:
    path = resolve_project_path(get_setting("paths.outputs", "./experiments"))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir(create: bool = True) -> Path:
    path = resolve_project_path(get_setting("paths.data", "./data"))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    print(f"App Host: {get_setting('app.host', 'default_host')}")
    print(f"App Port: {get_setting('app.port', 'default_port')}")
    print(f"Model dirs: {get_model_search_dirs()}")
