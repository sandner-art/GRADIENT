# tests/test_quantization.py
"""Quantized loading (torchao, GGUF; bitsandbytes needs CUDA) and the fidelity report."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gradient.ai.diffusion import quantization as quant

BASE = {
    "prompt": "a red lighthouse on a rocky coast",
    "width": 64,
    "height": 64,
    "num_inference_steps": 4,
    "guidance_scale": 5.0,
    "seed": 21,
}


def _img(b64: str) -> np.ndarray:
    import base64

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), np.float32)


def render(client, model_id, **extra):
    r = client.post("/api/v1/generate/text-to-image", json={**BASE, "model_id": model_id, **extra})
    return r


def write_gguf_unet(sd_path: Path, target: Path) -> Path:
    """The tiny UNet as a GGUF file (Q8_0 where the shapes allow, diffusers key names)."""
    gguf = pytest.importorskip("gguf")
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
    target.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(target), arch="sd1")
    for k, v in unet.state_dict().items():
        a = v.float().numpy()
        if a.ndim >= 2 and a.shape[-1] % 32 == 0:
            q = gguf.GGMLQuantizationType.Q8_0
            w.add_tensor(k, gguf.quants.quantize(a, q), raw_dtype=q)
        else:
            w.add_tensor(k, a.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return target


def test_parse_specs():
    assert quant.parse(None).key == "none"
    assert quant.parse("torchao-int8+text") == quant.QuantSpec("torchao-int8", True)
    g = quant.parse("gguf:flux1-dev-Q4_K_S.gguf")
    assert g.name == "gguf" and g.gguf_file == "flux1-dev-Q4_K_S.gguf" and g.backend == "gguf"
    for bad in ("int3", "gguf:", "none+text"):
        with pytest.raises(ValueError):
            quant.parse(bad)


def test_options_and_errors(client, tiny_sd):
    opts = {o["value"]: o for o in client.get("/api/v1/system/quantization").json()["options"]}
    assert opts["none"]["available"]
    if not torch.cuda.is_available():
        assert not opts["bnb-4bit"]["available"] and "GPU" in opts["bnb-4bit"]["reason"]
        r = render(client, tiny_sd, quantization="bnb-4bit")
        assert r.status_code in (400, 422, 500) and "GPU" in r.text
    r = render(client, tiny_sd, quantization="int3")
    assert r.status_code in (400, 422) and "Unknown quantization" in r.text


def test_torchao_render_lora_and_jvp(client, tiny_sd, tiny_sd_source, workspace):
    pytest.importorskip("torchao")
    from helpers import build_tiny_lora

    lora = build_tiny_lora(
        tiny_sd_source, workspace / "models" / "lora" / "quant-lora.safetensors", seed=3
    )
    full = render(client, tiny_sd)
    assert full.status_code == 200, full.text
    q = render(client, tiny_sd, quantization="torchao-int8")
    assert q.status_code == 200, q.text
    a = _img(full.json()["images"][0]["image_base64"])
    b = _img(q.json()["images"][0]["image_base64"])
    assert 0 < np.abs(a - b).mean() < 4  # 8-bit levels: close, but not the same weights
    params = q.json()["images"][0]["parameters"]
    assert params["quantization"] == "torchao-int8"
    loaded = client.get("/api/v1/system/info").json()["loaded_pipelines"]
    assert any(p["quantization"] == "torchao-int8" for p in loaded)

    # A1111 text records it
    out = client.get(f"/api/v1/history/outputs/{q.json()['images'][0]['output_id']}").json()
    png = Image.open(out["file_path"])
    assert "Quantization: torchao-int8" in png.info["parameters"]

    # LoRAs load on top of quantized weights
    r = render(
        client,
        tiny_sd,
        quantization="torchao-int8",
        loras=[{"name": lora.name, "weight": 0.8}],
    )
    assert r.status_code == 200, r.text

    # the exact JVP runs through quantized weights
    t = client.post(
        "/api/v1/lab/taylor-preview",
        json={"base_request": {**BASE, "model_id": tiny_sd, "quantization": "torchao-int8"}},
    )
    assert t.status_code == 200, t.text
    assert t.json()["method"] == "jvp" and t.json()["note"] is None


def test_gguf_denoiser(client, tiny_sd, tiny_sd_source, workspace):
    pytest.importorskip("gguf")
    target = workspace / "models" / "checkpoints" / "tiny-unet-Q8_0.gguf"
    write_gguf_unet(tiny_sd_source, target)
    models = client.get("/api/v1/system/models/diffusion").json()
    assert "tiny-unet-Q8_0.gguf" not in models  # a denoiser file, not a model
    opts = {o["value"] for o in client.get("/api/v1/system/quantization").json()["options"]}
    assert "gguf:tiny-unet-Q8_0.gguf" in opts

    full = render(client, tiny_sd)
    q = render(client, tiny_sd, quantization="gguf:tiny-unet-Q8_0.gguf")
    assert q.status_code == 200, q.text
    a = _img(full.json()["images"][0]["image_base64"])
    b = _img(q.json()["images"][0]["image_base64"])
    assert np.abs(a - b).mean() < 2  # Q8_0 is nearly lossless

    missing = render(client, tiny_sd, quantization="gguf:nope.gguf")
    assert missing.status_code in (400, 500) and "not found" in missing.text


def test_weight_bytes_counts_quantized_storage(tiny_sd_source):
    pytest.importorskip("torchao")
    from diffusers import PipelineQuantizationConfig, StableDiffusionPipeline, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    full = StableDiffusionPipeline.from_pretrained(str(tiny_sd_source), safety_checker=None)
    q = StableDiffusionPipeline.from_pretrained(
        str(tiny_sd_source),
        safety_checker=None,
        dtype=torch.float32,
        quantization_config=PipelineQuantizationConfig(
            quant_mapping={"unet": TorchAoConfig(Int8WeightOnlyConfig())}
        ),
    )
    assert quant.weight_bytes(q.unet) < quant.weight_bytes(full.unet)
    assert set(quant.describe(q)) >= {"unet", "text_encoder", "vae"}


def test_quantization_report(client, tiny_sd, tiny_sd_source, workspace, tiny_vision):
    pytest.importorskip("torchao")
    pytest.importorskip("gguf")
    target = workspace / "models" / "checkpoints" / "tiny-unet-Q8_0.gguf"
    if not target.exists():
        write_gguf_unet(tiny_sd_source, target)
    r = client.post(
        "/api/v1/lab/quantization-report",
        json={
            "base_request": {**BASE, "model_id": tiny_sd},
            "variants": ["torchao-int8", "gguf:tiny-unet-Q8_0.gguf", "bnb-4bit"],
            "num_seeds": 2,
            "feature_model": tiny_vision[0],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["seeds"] == [21, 22] and data["reference"] == "none"
    assert data["feature_model"] == tiny_vision[0] and data["seed_spread"] > 0
    rows = {v["quantization"]: v for v in data["variants"]}
    assert list(rows)[0] == "none" and rows["none"]["verdict"] == "Reference."
    for key in ("torchao-int8", "gguf:tiny-unet-Q8_0.gguf"):
        v = rows[key]
        assert v["error"] is None, v
        assert v["psnr"] > 20 and 0 < v["ssim"] <= 1
        assert v["relative_change"] is not None and v["relative_change"] < 0.5
        assert v["taylor_cosine"] > 0.9 and v["taylor_method"] == "jvp"
        assert v["denoiser_mb"] < rows["none"]["denoiser_mb"]
        assert v["strip_base64"] and v["verdict"].endswith(".")
    if not torch.cuda.is_available():
        assert "GPU" in rows["bnb-4bit"]["error"]
    assert "torchao-int8" in data["summary"]
    overview = _img(data["image_base64"])
    assert overview.shape[0] > 192 * 2  # a row per successful quantization

    # saved as a Lab run with its strips and report
    runs = client.get("/api/v1/history/runs", params={"run_type": "lab_quantization"}).json()
    assert runs["items"] and runs["items"][0]["count"] == 1

    bad = client.post(
        "/api/v1/lab/quantization-report",
        json={"base_request": {**BASE, "model_id": tiny_sd}, "variants": ["none"]},
    )
    assert bad.status_code in (400, 422) and "differs from the reference" in bad.text
