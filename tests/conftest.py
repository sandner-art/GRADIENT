# tests/conftest.py
"""Shared fixtures. All tests run offline, on CPU, in a temporary workspace."""

import os
import sys
import tempfile
from pathlib import Path

# Isolate every path BEFORE gradient is imported (config is read at import time).
_WORKSPACE = Path(tempfile.mkdtemp(prefix="gradient-tests-"))
os.environ["GRADIENT_MODEL_DIR"] = str(_WORKSPACE / "models")
os.environ["GRADIENT_DATA_DIR"] = str(_WORKSPACE / "data")
os.environ["GRADIENT_OUTPUTS_DIR"] = str(_WORKSPACE / "experiments")
os.environ["GRADIENT_DEVICE"] = "cpu"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(var, None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from helpers import build_tiny_sd  # noqa: E402


@pytest.fixture(scope="session")
def workspace() -> Path:
    return _WORKSPACE


@pytest.fixture(scope="session")
def tiny_sd_source(workspace: Path) -> Path:
    """A tiny random SD pipeline built once per session (outside the models dir)."""
    return build_tiny_sd(workspace / "sources" / "tiny-sd")


@pytest.fixture(scope="session")
def tiny_sd(workspace: Path, tiny_sd_source: Path) -> str:
    """The tiny pipeline installed into the models dir; returns its model_id."""
    import shutil

    target = workspace / "models" / "diffusers" / "tiny-sd"
    if not target.exists():
        shutil.copytree(tiny_sd_source, target)
    return "tiny-sd"


@pytest.fixture(scope="session")
def fake_hub():
    from fake_hub import FakeHub

    hub = FakeHub().start()
    old = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = hub.url
    yield hub
    hub.stop()
    if old is None:
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ["HF_ENDPOINT"] = old


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from gradient.api.main import app

    with TestClient(app) as c:
        yield c
