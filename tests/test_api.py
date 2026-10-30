# tests/test_api.py
"""API tests with a tiny random SD pipeline on CPU (no downloads)."""

import base64
import io

import numpy as np
import pytest
from PIL import Image


def _decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.int16)


BASE = {
    "prompt": "a lighthouse",
    "negative_prompt": "blurry",
    "width": 64,
    "height": 64,
    "num_inference_steps": 3,
    "seed": 7,
}


def test_status_and_info(client):
    assert client.get("/api/v1/status").json()["status"] == "OK"
    info = client.get("/api/v1/system/info").json()
    assert info["device"]["device"] == "cpu"
    assert info["versions"]["diffusers"]
    assert client.get("/api/v1/system/status").json()["queue_depth"] == 0


def test_model_lists(client, tiny_sd):
    assert tiny_sd in client.get("/api/v1/system/models/diffusion").json()
    local = client.get("/api/v1/models/local", params={"type": "diffusers"}).json()
    entry = next(m for m in local if m["name"] == tiny_sd)
    assert entry["format"] == "diffusers" and entry["size_bytes"] > 0
    catalog = client.get("/api/v1/models/catalog").json()
    assert catalog and all("installed" in c for c in catalog)
    assert not any("test" in c["tags"] for c in catalog)
    assert any(
        "test" in c["tags"]
        for c in client.get("/api/v1/models/catalog", params={"include_tests": True}).json()
    )
    storage = client.get("/api/v1/models/storage").json()
    assert storage["download_dir"].endswith("models")


def test_text_to_image_and_reproducibility(client, tiny_sd):
    r = client.post(
        "/api/v1/generate/text-to-image",
        json={**BASE, "model_id": tiny_sd, "num_outputs": 2, "guidance_scale": "linear(8,4)"},
    )
    assert r.status_code == 200, r.text
    images = r.json()["images"]
    assert [i["seed"] for i in images] == [7, 8]
    params = images[0]["parameters"]
    assert params["guidance_schedule_applied"] is True
    assert params["pipeline"] == "StableDiffusionPipeline"
    # image #2 regenerates alone from its logged seed
    r2 = client.post(
        "/api/v1/generate/text-to-image",
        json={**BASE, "model_id": tiny_sd, "seed": 8, "guidance_scale": "linear(8,4)"},
    )
    assert (
        np.abs(
            _decode(images[1]["image_base64"]) - _decode(r2.json()["images"][0]["image_base64"])
        ).max()
        <= 1
    )


def test_image_to_image(client, tiny_sd):
    init = Image.new("RGB", (64, 64), (200, 30, 30))
    buf = io.BytesIO()
    init.save(buf, "PNG")
    r = client.post(
        "/api/v1/generate/image-to-image",
        json={
            **BASE,
            "model_id": tiny_sd,
            "image_base64": base64.b64encode(buf.getvalue()).decode(),
            "strength": 0.5,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["images"][0]["parameters"]["pipeline"] == "StableDiffusionImg2ImgPipeline"


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"model_id": "does-not-exist"}, 400),
        ({"scheduler_id": "FlowMatchEulerDiscreteScheduler"}, 400),
        ({"scheduler_id": "DDIMScheduler", "custom_sigmas": [3.0, 1.0, 0.0]}, 400),
        ({"guidance_scale": "zigzag"}, 422),
        ({"width": 63}, 422),
    ],
)
def test_generation_errors(client, tiny_sd, payload, status):
    body = {**BASE, "model_id": tiny_sd, **payload}
    assert client.post("/api/v1/generate/text-to-image", json=body).status_code == status


def test_xyz_grid(client, tiny_sd):
    r = client.post(
        "/api/v1/generate/xyz-grid",
        json={
            "base_parameters": {**BASE, "model_id": tiny_sd},
            "x_axis": {"parameter": "guidance_scale", "values": [2.0, 6.0]},
            "y_axis": {"parameter": "num_inference_steps", "values": [2, 3]},
        },
    )
    assert r.status_code == 200, r.text
    cells = r.json()["grid_results"]
    assert len(cells) == 4 and all(c["image_result"] for c in cells)
    assert len({c["image_result"]["seed"] for c in cells}) == 1  # one shared seed across the grid


def test_parameter_sensitivity(client, tiny_sd):
    r = client.post(
        "/api/v1/visualize/parameter-sensitivity",
        json={
            "base_request": {**BASE, "model_id": tiny_sd, "guidance_scale": 5.0},
            "target_parameter": {"name": "guidance_scale"},
        },
    )
    assert r.status_code == 200, r.text
    m = r.json()["sensitivity_map"]
    assert m["shape"] == [64, 64]
    assert m["max_sensitivity"] > m["min_sensitivity"]  # a real, non-constant map
    assert m["heatmap_base64"] and m["base_image_base64"]


def test_presets_roundtrip(client):
    body = {
        "model_identifier": "tiny-sd",
        "preset_name": "fast",
        "settings": {"num_inference_steps": 4},
    }
    created = client.post("/api/v1/presets", json=body)
    assert created.status_code == 201, created.text
    assert client.post("/api/v1/presets", json=body).status_code == 409
    listed = client.get("/api/v1/presets", params={"model_identifier": "tiny-sd"}).json()
    assert [p["preset_name"] for p in listed] == ["fast"]
    assert client.delete(f"/api/v1/presets/{created.json()['id']}").status_code == 204


def test_delete_local_model_via_api(client, workspace, tiny_sd_source):
    import shutil

    shutil.copytree(tiny_sd_source, workspace / "models" / "diffusers" / "to-delete")
    assert client.delete("/api/v1/models/local/diffusers/to-delete").status_code == 200
    assert not (workspace / "models" / "diffusers" / "to-delete").exists()
    assert client.delete("/api/v1/models/local/diffusers/to-delete").status_code == 404


def test_download_endpoint_validation(client):
    assert client.post("/api/v1/models/downloads", json={"catalog_id": "nope"}).status_code == 400
    assert (
        client.post(
            "/api/v1/models/downloads",
            json={"url": "http://insecure/x.safetensors", "model_type": "lora"},
        ).status_code
        == 400
    )
    assert client.get("/api/v1/models/downloads").status_code == 200
