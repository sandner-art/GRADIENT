# tests/test_style_check.py
"""Lab style check: styles × positions × seeds with DINO and CLIP scores."""

import pytest

from gradient.ai.diffusion import styles

CSV = """name,prompt,negative_prompt
Ink: Wash,"ink wash painting of {prompt}, sepia",photo
Bold: Neon,"neon glow, vivid magenta and cyan",
"""
BASE = {
    "prompt": "a red dog on a hill",
    "negative_prompt": "",
    "width": 64,
    "height": 64,
    "num_inference_steps": 3,
    "guidance_scale": 5.0,
    "seed": 21,
}


@pytest.fixture
def style_file():
    path = styles.styles_dir() / "check.csv"
    path.write_text(CSV, encoding="utf-8")
    yield
    path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def tiny_clip(workspace):
    from helpers import build_tiny_clip

    target = workspace / "models" / "vision" / "clip-tiny"
    if not target.exists():
        build_tiny_clip(target)
    return "clip-tiny"


def test_style_check(client, tiny_sd, tiny_vision, tiny_clip, style_file):
    body = {
        "base_request": {"model_id": tiny_sd, **BASE},
        "styles": ["Ink: Wash", "Bold: Neon", "Gone: Missing"],
        "positions": ["template", "chunk"],
        "num_seeds": 2,
        "feature_model": "dinov2-tiny",
        "clip_model": tiny_clip,
    }
    r = client.post("/api/v1/lab/style-check", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["seeds"] == [21, 22] and data["clip_model"] == tiny_clip
    assert len(data["rows"]) == 6 and len(data["cells"]) == 2 + 6 * 2
    assert data["seed_distance"] > 0 and data["baseline_content"] is not None
    rows = {(row["style"], row["position"]): row for row in data["rows"]}
    ink = rows[("Ink: Wash", "template")]
    assert ink["strength"] > 0 and ink["strength_rel"] > 0
    assert ink["content_rel"] is not None and ink["consistency"] is not None
    assert ink["tokens"] == 37 and ink["chunks"] == 1  # a token per letter here
    assert [data["cells"][i]["seed"] for i in ink["cells"]] == [21, 22]
    assert data["cells"][ink["cells"][0]]["parameters"]["styled_prompt"].startswith("ink wash")
    chunk = rows[("Bold: Neon", "chunk")]
    assert data["cells"][chunk["cells"][0]]["parameters"]["prompt_chunks"] == 2
    missing = rows[("Gone: Missing", "template")]
    assert "style not found" in missing["notes"] and missing["strength"] < 1e-6
    assert data["image_base64"]


def test_style_check_limits(client, tiny_sd):
    body = {
        "base_request": {"model_id": tiny_sd, **BASE},
        "styles": [f"s{i}" for i in range(16)],
        "positions": ["template", "before", "after"],
    }
    r = client.post("/api/v1/lab/style-check", json=body)
    assert r.status_code >= 400 and "at most 32" in r.text
