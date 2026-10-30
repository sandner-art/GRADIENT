# tests/test_concept_attention.py
"""ConceptAttention on tiny random FLUX.1, SD3.5-style and FLUX.2 [klein] pipelines (CPU, offline)."""

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

BASE = {
    "prompt": "a red lighthouse on a rocky coast",
    "negative_prompt": "blurry",
    "width": 64,
    "height": 64,
    "num_inference_steps": 4,
    "guidance_scale": 4.0,
    "seed": 3,
}
MODELS = ["tiny_flux", "tiny_sd3", "tiny_flux2"]


def _img(b64: str) -> np.ndarray:
    return np.asarray(
        Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.float32
    )


@pytest.fixture(params=MODELS)
def dit(request):
    return request.getfixturevalue(request.param)


def test_parse_blocks():
    from gradient.ai.lab.concept_attention import parse_blocks

    assert parse_blocks(None, 19) == list(range(9, 19))
    assert parse_blocks("all", 4) == [0, 1, 2, 3]
    assert parse_blocks("1-2, 0", 4) == [0, 1, 2]
    with pytest.raises(ValueError, match="0-3"):
        parse_blocks("2-7", 4)
    with pytest.raises(ValueError, match="Invalid"):
        parse_blocks("last", 4)


def test_concept_token_positions(tiny_flux, tiny_sd3, tiny_flux2):
    from gradient.ai.diffusion.loading import get_pipeline
    from gradient.ai.lab.concept_attention import concept_support, encode_concepts

    expected = {
        tiny_flux: [0, 0],  # T5: first token of the word
        tiny_sd3: [10, 3],  # character-level test CLIP: BOS + 10 letters -> last letter
        tiny_flux2: [2, 2],  # "<|im_start|> user <word>" in the chat template
    }
    for model_id, positions in expected.items():
        pipe, _ = get_pipeline(model_id)
        embeds, pos = encode_concepts(pipe, concept_support(pipe)[0], ["lighthouse", "sky"])
        assert pos == positions, model_id
        assert embeds.shape[:2] == (1, 2)
        assert not torch.allclose(embeds[0, 0], embeds[0, 1])


def test_concept_stream_reproduces_text_stream(dit):
    """With the prompt's own tokens as concepts, concept attention sees exactly the keys of the
    real text stream, so its outputs must equal the model's text-stream attention outputs in
    every block. This checks the hooks, norms, RoPE, modulation and CFG slicing end to end."""
    from gradient.ai.diffusion.loading import get_pipeline
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import generate_images
    from gradient.ai.lab.concept_attention import ConceptAttentionProbe, concept_support

    req = Text2ImageRequest(model_id=dit, **{**BASE, "num_inference_steps": 2})
    pipe, _ = get_pipeline(dit)
    kind, n = concept_support(pipe)
    transformer = pipe.transformer
    # DiTs load in bf16; compare in fp32 so kernel-order rounding can't hide a real mismatch
    # (bf16 -> fp32 -> bf16 is lossless, so later tests see the same weights)
    model_dtype = transformer.dtype
    pipe.to(dtype=torch.float32)

    captured = {}

    def grab_text(_m, _a, kwargs):
        captured.setdefault("ehs", kwargs["encoder_hidden_states"][-1:].clone())

    text_out: dict[int, torch.Tensor] = {}
    handles = [transformer.register_forward_pre_hook(grab_text, with_kwargs=True)]
    for i, block in enumerate(transformer.transformer_blocks):
        if getattr(block.attn, "to_add_out", None) is not None:

            def grab_out(_m, args, i=i):
                text_out.setdefault(i, args[0][-1:].clone())

            handles.append(block.attn.to_add_out.register_forward_pre_hook(grab_out))
    try:
        generate_images(req)
        text_embeds = captured["ehs"]
        n_text = text_embeds.shape[1]
        probe = ConceptAttentionProbe(
            kind, text_embeds, list(range(n_text)), list(range(n)), zero_pooled=False
        )
        probe.trace = {}
        text_out.clear()
        generate_images(req, probes=[probe])
    finally:
        for h in handles:
            h.remove()
        pipe.to(dtype=model_dtype)

    assert text_out, "the model never called to_add_out"
    for i, expected in text_out.items():
        torch.testing.assert_close(probe.trace[i], expected, atol=2e-5, rtol=1e-4)


def test_endpoint_maps_and_unchanged_image(client, dit):
    r = client.post(
        "/api/v1/lab/concept-attention",
        json={"base_request": {**BASE, "model_id": dit}, "concepts": ["lighthouse", "sea", "sky"]},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    concepts = data["concepts"]
    assert [c["concept"] for c in concepts] == ["lighthouse", "sea", "sky"]
    # softmax over concepts: probabilities and winning areas add up to 1
    assert sum(c["mean_probability"] for c in concepts) == pytest.approx(1, abs=1e-4)
    assert sum(c["area"] for c in concepts) == pytest.approx(1, abs=1e-6)
    assert len({c["color"] for c in concepts}) == 3
    # default blocks: the deeper half; every step recorded
    n = data["num_blocks"]
    assert data["blocks_recorded"] == list(range(n // 2, n))
    assert data["steps_recorded"] == BASE["num_inference_steps"]
    assert _img(data["segmentation_base64"]).shape == (64, 64, 3)
    # hooks only read activations: the image is bit-identical to a normal render
    plain = client.post("/api/v1/generate/text-to-image", json={**BASE, "model_id": dit}).json()
    assert np.array_equal(_img(data["image_base64"]), _img(plain["images"][0]["image_base64"]))


def test_defaults_and_step_window(client, tiny_flux):
    r = client.post(
        "/api/v1/lab/concept-attention",
        json={
            "base_request": {**BASE, "model_id": tiny_flux},
            "blocks": "all",
            "start": 0.5,
            "end": 1.0,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    names = [c["concept"] for c in data["concepts"]]
    assert names == ["red", "lighthouse", "rocky", "coast", "background"]
    assert data["blocks_recorded"] == [0, 1, 2, 3]
    assert data["steps_recorded"] == 2


def test_errors(client, tiny_sd, tiny_flux):
    # UNet models: ConceptAttention points to cross-attention maps, and vice versa
    r = client.post(
        "/api/v1/lab/concept-attention", json={"base_request": {**BASE, "model_id": tiny_sd}}
    )
    assert r.status_code == 400 and "cross-attention" in r.text
    r = client.post(
        "/api/v1/lab/attention-maps", json={"base_request": {**BASE, "model_id": tiny_flux}}
    )
    assert r.status_code == 400 and "concept-attention" in r.text
    r = client.post(
        "/api/v1/lab/concept-attention",
        json={"base_request": {**BASE, "model_id": tiny_flux}, "blocks": "9"},
    )
    assert r.status_code == 400 and "0-3" in r.text


def test_layers_report_concept_blocks(client, tiny_sd, tiny_flux, tiny_sd3):
    def blocks(model_id):
        r = client.get("/api/v1/lab/layers", params={"model_id": model_id})
        assert r.status_code == 200, r.text
        return r.json()["concept_blocks"]

    assert (blocks(tiny_sd), blocks(tiny_flux), blocks(tiny_sd3)) == (0, 4, 3)


def test_local_models_report_family(client, tiny_sd, tiny_flux, tiny_sd3):
    models = {m["name"]: m for m in client.get("/api/v1/models/local?type=diffusers").json()}
    assert models[tiny_flux]["family"] == "flux1"
    assert models[tiny_flux]["pipeline"] == "FluxPipeline"
    assert models[tiny_sd3]["family"] == "sd3"
    assert models[tiny_sd]["family"] == "sd15"
