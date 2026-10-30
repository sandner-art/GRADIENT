# tests/test_lab.py
"""Lab tools on the tiny random SD pipeline (CPU, offline)."""

import base64
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

BASE = {
    "prompt": "a red lighthouse on a rocky coast",
    "negative_prompt": "blurry",
    "width": 64,
    "height": 64,
    "num_inference_steps": 6,
    "guidance_scale": 5.0,
    "seed": 5,
}


def _img(b64: str) -> np.ndarray:
    return np.asarray(
        Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.float32
    )


@pytest.fixture(scope="session")
def tiny_vision(workspace):
    """Tiny random DINOv2 and DINOv3 encoders in the vision folder."""
    from transformers import Dinov2Config, Dinov2Model, DINOv3ViTConfig, DINOv3ViTModel

    root = workspace / "models" / "vision"
    torch.manual_seed(0)
    Dinov2Model(
        Dinov2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            patch_size=14,
            image_size=56,
        )
    ).save_pretrained(root / "dinov2-tiny")
    DINOv3ViTModel(
        DINOv3ViTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            patch_size=16,
            num_register_tokens=4,
        )
    ).save_pretrained(root / "dinov3-tiny")
    return ["dinov2-tiny", "dinov3-tiny"]


# --------------------------------------------------------------------------- modulation
def test_modulation_lanes_change_output(client, tiny_sd):
    def gen(**extra):
        r = client.post(
            "/api/v1/generate/text-to-image", json={**BASE, "model_id": tiny_sd, **extra}
        )
        assert r.status_code == 200, r.text
        return r.json()["images"][0]

    base = _img(gen()["image_base64"])
    neutral = gen(modulation=[{"type": "attention_scale", "targets": "*attn2", "value": 1.0}])
    assert np.abs(_img(neutral["image_base64"]) - base).max() == 0
    for lanes in (
        [
            {
                "type": "attention_scale",
                "targets": "up_blocks.*attn2",
                "value": 0.0,
                "start": 0.0,
                "end": 0.5,
            }
        ],
        [{"type": "prompt", "prompt": "a blue dog in a field", "start": 0.0, "end": 0.5}],
        [{"type": "prompt", "prompt": "a blue dog in a field", "mode": "blend", "value": 0.5}],
    ):
        out = gen(modulation=lanes)
        assert np.abs(_img(out["image_base64"]) - base).mean() > 0
        assert out["parameters"]["modulation_applied"]
    # hooks/processors are removed again afterwards
    assert np.abs(_img(gen()["image_base64"]) - base).max() == 0


@pytest.mark.parametrize(
    "lanes",
    [
        [{"type": "attention_scale", "targets": "nothing.matches*", "value": 0.0}],
        [{"type": "lora_weight", "lora_index": 0, "value": 0.5}],
    ],
)
def test_modulation_errors(client, tiny_sd, lanes):
    r = client.post(
        "/api/v1/generate/text-to-image", json={**BASE, "model_id": tiny_sd, "modulation": lanes}
    )
    assert r.status_code == 400, r.text


def test_lane_validation(client, tiny_sd):
    bad = [{"type": "prompt", "start": 0.6, "end": 0.4, "prompt": "x"}]
    r = client.post(
        "/api/v1/generate/text-to-image", json={**BASE, "model_id": tiny_sd, "modulation": bad}
    )
    assert r.status_code == 422


def test_layers_endpoint(client, tiny_sd):
    r = client.get("/api/v1/lab/layers", params={"model_id": tiny_sd})
    assert r.status_code == 200, r.text
    data = r.json()
    assert any(layer["kind"] == "cross" for layer in data["layers"])
    assert "*attn2" in data["suggestions"]


# --------------------------------------------------------------------------- Taylor
def test_jvp_matches_finite_differences(tiny_sd):
    from gradient.ai.diffusion.loading import get_pipeline
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import _set_scheduler
    from gradient.ai.lab import unet_sampler

    req = Text2ImageRequest(model_id=tiny_sd, **{k: v for k, v in BASE.items() if k != "seed"})
    pipe, entry = get_pipeline(tiny_sd)
    _set_scheduler(pipe, entry, None)
    _, tangent = unet_sampler.guidance_jvp(pipe, req, 5)

    def render(w):
        r = req.model_copy(update={"guidance_scale": w})
        with torch.no_grad():
            return (
                unet_sampler.decode(pipe, unet_sampler.sample(pipe, r, 5))[0]
                .permute(1, 2, 0)
                .numpy()
            )

    fd = (render(5.01) - render(4.99)) / 0.02
    t = tangent.numpy()
    cosine = float((t * fd).sum() / (np.linalg.norm(t) * np.linalg.norm(fd)))
    assert cosine > 0.999


def test_own_sampler_reproduces_pipeline(tiny_sd):
    from gradient.ai.diffusion.loading import get_pipeline
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import _set_scheduler, generate_images
    from gradient.ai.lab import unet_sampler

    req = Text2ImageRequest(model_id=tiny_sd, **BASE)
    ref = np.asarray(generate_images(req).images[0], dtype=np.float32) / 255
    pipe, entry = get_pipeline(tiny_sd)
    _set_scheduler(pipe, entry, None)
    with torch.no_grad():
        own = (
            unet_sampler.decode(pipe, unet_sampler.sample(pipe, req, BASE["seed"]))[0]
            .permute(1, 2, 0)
            .numpy()
        )
    assert np.abs(own - ref).max() < 2 / 255


@pytest.mark.parametrize("method", ["auto", "finite_difference"])
def test_taylor_endpoint(client, tiny_sd, method):
    r = client.post(
        "/api/v1/lab/taylor-preview",
        json={"base_request": {**BASE, "model_id": tiny_sd}, "method": method},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["method"] == ("jvp" if method == "auto" else "finite_difference")
    assert data["tangent_scale"] > 0
    tangent = (_img(data["tangent_png_base64"]) - 128) / 127 * data["tangent_scale"]
    assert np.abs(tangent).mean() > 0
    assert _img(data["base_image_base64"]).shape == (64, 64, 3)


# --------------------------------------------------------------------------- attention maps
def test_attention_maps(client, tiny_sd):
    r = client.post(
        "/api/v1/lab/attention-maps",
        json={
            "base_request": {**BASE, "model_id": tiny_sd},
            "layers": "*attn2",
            "start": 0.2,
            "end": 0.8,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    words = [m["word"] for m in data["maps"]]
    assert "lighthouse" in words and "coast" in words
    assert data["layers_recorded"]
    assert all(0 <= m["attention_share"] <= 1 for m in data["maps"])
    # recording must not change the image
    plain = client.post("/api/v1/generate/text-to-image", json={**BASE, "model_id": tiny_sd}).json()
    assert np.abs(_img(data["image_base64"]) - _img(plain["images"][0]["image_base64"])).max() <= 1


def test_attention_maps_errors(client, tiny_sd):
    req = {"base_request": {**BASE, "model_id": tiny_sd}, "words": ["zebra"]}
    assert client.post("/api/v1/lab/attention-maps", json=req).status_code == 400


def test_word_token_positions(tiny_sd_source):
    from transformers import CLIPTokenizer

    from gradient.ai.lab.probes import word_token_positions

    tok = CLIPTokenizer.from_pretrained(Path(tiny_sd_source) / "tokenizer")
    words = word_token_positions(tok, "a cat, sitting")
    assert [w for w, _ in words] == ["a", "cat", "sitting"]
    flat = [p for _, pos in words for p in pos]
    assert flat == sorted(flat) and flat[0] == 1
    # positions agree with tokenizing the full prompt
    full = tok("a cat, sitting").input_ids
    first_word_ids = tok("cat", add_special_tokens=False).input_ids
    cat_pos = dict(words)["cat"]
    assert [full[p] for p in cat_pos] == first_word_ids


# --------------------------------------------------------------------------- token ablation
def test_ablation_helpers():
    from gradient.ai.lab.ablation import content_words, remove_word

    assert content_words("a photo of a red cat on the moon", 10) == ["photo", "red", "cat", "moon"]
    assert remove_word("a red cat, on the moon", "cat") == "a red, on the moon"
    assert remove_word("Cat and catalog", "cat") == "and catalog"


@pytest.mark.parametrize("mode", ["remove_word", "mask_embedding"])
def test_token_ablation(client, tiny_sd, tiny_vision, mode):
    r = client.post(
        "/api/v1/lab/token-ablation",
        json={
            "base_request": {**BASE, "model_id": tiny_sd},
            "mode": mode,
            "feature_model": tiny_vision[0],
            "max_words": 3,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["feature_model"] == tiny_vision[0]
    results = data["results"]
    assert len(results) == 3
    assert [r["patch_distance"] for r in results] == sorted(
        (r["patch_distance"] for r in results), reverse=True
    )
    assert all(r["global_distance"] >= -1e-6 and r["heatmap_base64"] for r in results)


def test_dinov3_features_and_fallback(tiny_vision):
    from gradient.ai.lab.features import distances, load_feature_model

    fm, note = load_feature_model(tiny_vision[1])
    assert fm.patch == 16 and note is None
    a = Image.new("RGB", (64, 64), (200, 20, 20))
    b = a.copy()
    b.paste((20, 20, 200), (0, 0, 32, 32))
    g, patch_map = distances(fm, a, b)
    assert patch_map.shape == (28, 28)
    assert g >= 0 and patch_map[:14, :14].mean() > patch_map[14:, 14:].mean()
    # an unavailable model reports a clear error
    with pytest.raises(ValueError):
        load_feature_model("does/not-exist")
