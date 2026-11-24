# tests/test_embed.py
"""POST /lab/embed: DINO image embeddings and CLIP image + text embeddings for other apps."""

import base64
import io

import numpy as np
import pytest
from PIL import Image


def b64(color, size=(48, 40), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def tiny_clip(workspace):
    from helpers import build_tiny_clip

    target = workspace / "models" / "vision" / "clip-tiny"
    if not target.exists():
        build_tiny_clip(target)
    return "clip-tiny"


def test_dino_image_embeddings(client, tiny_vision):
    body = {
        "images": [b64((255, 0, 0)), "data:image/png;base64," + b64((0, 0, 255)), b64((255, 0, 0))],
        "feature_model": "dinov2-tiny",
    }
    r = client.post("/api/v1/lab/embed", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["kind"] == "dino" and data["model"] == "dinov2-tiny" and data["dim"] == 32
    vectors = np.array(data["image_embeddings"])
    assert vectors.shape == (3, 32) and data["text_embeddings"] is None
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert np.allclose(vectors[0], vectors[2], atol=1e-5)  # same image, same embedding


def test_texts_need_a_clip_model(client, tiny_vision):
    r = client.post("/api/v1/lab/embed", json={"texts": ["a cat"], "feature_model": "dinov2-tiny"})
    assert r.status_code == 400 and "CLIP or SigLIP" in r.json()["detail"]


def test_clip_images_and_texts(client, tiny_clip):
    body = {
        "images": [b64((10, 200, 30), fmt="JPEG"), b64((10, 200, 30), size=(64, 64), fmt="WEBP")],
        "texts": ["txcl painting of a fox", "a car"],
        "feature_model": tiny_clip,
    }
    r = client.post("/api/v1/lab/embed", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["kind"] == "clip" and data["dim"] == 16
    images, texts = np.array(data["image_embeddings"]), np.array(data["text_embeddings"])
    assert images.shape == (2, 16) and texts.shape == (2, 16)
    assert np.allclose(np.linalg.norm(texts, axis=1), 1, atol=1e-5)
    only_texts = client.post(
        "/api/v1/lab/embed", json={"texts": ["a car"], "feature_model": tiny_clip}
    ).json()
    assert only_texts["image_embeddings"] == [] and np.allclose(
        only_texts["text_embeddings"][0], texts[1], atol=1e-5
    )


def test_gallery_images_by_id(client, tiny_sd, tiny_vision):
    r = client.post(
        "/api/v1/generate/text-to-image",
        json={
            "model_id": tiny_sd,
            "prompt": "a lighthouse",
            "width": 64,
            "height": 64,
            "num_inference_steps": 2,
            "seed": 3,
        },
    )
    assert r.status_code == 200, r.text
    output_id = r.json()["images"][0]["output_id"]
    data = client.post(
        "/api/v1/lab/embed", json={"output_ids": [output_id], "feature_model": "dinov2-tiny"}
    ).json()
    assert len(data["image_embeddings"]) == 1
    missing = client.post(
        "/api/v1/lab/embed", json={"output_ids": [987654], "feature_model": "dinov2-tiny"}
    )
    assert missing.status_code == 404


@pytest.mark.parametrize(
    "body, status",
    [
        ({}, 422),
        ({"images": ["not base64!"]}, 400),
        ({"images": [base64.b64encode(b"<svg/>").decode()]}, 400),
        ({"images": [b64((1, 2, 3), fmt="GIF")]}, 400),
        ({"images": [b64((1, 2, 3))] * 65}, 422),
    ],
)
def test_bad_requests(client, tiny_vision, body, status):
    body = {**body, "feature_model": "dinov2-tiny"} if body else body
    assert client.post("/api/v1/lab/embed", json=body).status_code == status


def test_vision_models_list_their_family(client, tiny_vision, tiny_clip):
    models = {m["name"]: m for m in client.get("/api/v1/models/local?type=vision").json()}
    assert (
        models["clip-tiny"]["family"] == "clip" and models["clip-tiny"]["pipeline"] == "CLIPModel"
    )
    assert (
        models["dinov2-tiny"]["family"] == "dinov2" and models["dinov3-tiny"]["family"] == "dinov3"
    )
