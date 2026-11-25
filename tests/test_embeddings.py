# tests/test_embeddings.py
"""Textual inversion: file formats, tokens in prompts, SD/SDXL loading, misfit reports."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gradient.ai.diffusion import loading_embeddings as emb

BASE = {"width": 64, "height": 64, "num_inference_steps": 2, "guidance_scale": 5.0, "seed": 3}


@pytest.fixture
def emb_dir(workspace: Path):
    folder = workspace / "models" / "embeddings"
    folder.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    def put(name: str, state) -> Path:
        path = folder / name
        if path.suffix == ".safetensors":
            save_file(state, str(path))
        else:
            torch.save(state, str(path))
        made.append(path)
        return path

    yield put
    for path in made:
        path.unlink(missing_ok=True)


def test_file_formats_and_families(emb_dir):
    torch.manual_seed(0)
    a1111 = torch.randn(2, 768)
    emb_dir("a1111.pt", {"string_to_param": {"*": a1111}, "name": "x", "step": 100})
    emb_dir("trained.safetensors", {"emb_params": torch.randn(3, 1024)})
    emb_dir("plain.bin", {"<cat>": torch.randn(768)})
    emb_dir("xl.safetensors", {"clip_l": torch.randn(2, 768), "clip_g": torch.randn(2, 1280)})

    info = emb.embedding_info("a1111.pt")
    assert (info.token, info.family, info.vectors, info.error) == ("a1111", "sd1", 2, None)
    assert torch.equal(emb.read_tensors(emb.find_model_path("a1111.pt", "embeddings"))[""], a1111)
    assert emb.embedding_info("trained.safetensors").family == "sd2"
    assert emb.embedding_info("plain.bin").vectors == 1
    assert emb.embedding_info("xl.safetensors").family == "sdxl"
    assert emb.embedding_info("missing.pt").error == "not found"
    assert emb.fits("sdxl", "sdxl") and emb.fits("sd1", "sd15") and not emb.fits("sd1", "sdxl")
    assert emb.fits("sd1", None) is None


def test_tokens_in_prompts():
    installed = ["EasyNegative.safetensors", "bad-hands-5.pt", "v1.3neg.pt", "other.pt"]
    found = emb.embeddings_in_prompts(
        ["photo of a cat, (bad-hands-5:1.2)", "EasyNegative, blurry"], installed
    )
    assert found == ["EasyNegative.safetensors", "bad-hands-5.pt"]
    assert emb.embeddings_in_prompts(["bad-hands-50, xEasyNegative", None], installed) == []
    assert emb.embeddings_in_prompts(["v1.3neg."], installed) == ["v1.3neg.pt"]
    assert emb.normalize_prompt("embedding:EasyNegative.pt, embedding:bad-hands-5") == (
        "EasyNegative, bad-hands-5"
    )
    assert emb.normalize_prompt("no embeddings here") == "no embeddings here"


def _row(encoder, tokenizer, token):
    return encoder.get_input_embeddings().weight[tokenizer.convert_tokens_to_ids(token)]


def test_render_loads_embeddings_named_in_the_prompt(tiny_sd, emb_dir):
    from gradient.ai.diffusion.loading import get_pipeline
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import generate_images

    torch.manual_seed(1)
    vec = torch.randn(1, 16) * 3
    emb_dir("tinyemb.safetensors", {"emb_params": vec})
    emb_dir("tinyneg.pt", {"string_to_param": {"*": torch.randn(2, 16)}, "name": "n"})

    plain = generate_images(Text2ImageRequest(model_id=tiny_sd, prompt="a red dog", **BASE))
    assert not plain.params.get("embeddings")

    req = Text2ImageRequest(
        model_id=tiny_sd,
        prompt="a red dog, tinyemb",
        negative_prompt="embedding:tinyneg.pt",
        **BASE,
    )
    result = generate_images(req)
    assert result.params["embeddings"] == ["tinyemb.safetensors", "tinyneg.pt"]
    assert "embeddings_skipped" not in result.params
    pipe, entry = get_pipeline(tiny_sd)
    assert {"tinyemb", "tinyneg", "tinyneg_1"} <= set(pipe.tokenizer.added_tokens_encoder)
    assert torch.allclose(_row(pipe.text_encoder, pipe.tokenizer, "tinyemb"), vec[0])

    # the token changes the image; the same request again gives the same image
    import numpy as np

    first = np.asarray(result.images[0])
    assert not np.array_equal(first, np.asarray(plain.images[0]))
    assert np.array_equal(first, np.asarray(generate_images(req).images[0]))


def test_misfits_are_reported_not_loaded(tiny_sd, emb_dir):
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import generate_images

    emb_dir("xlonly.safetensors", {"clip_l": torch.randn(1, 16), "clip_g": torch.randn(1, 16)})
    emb_dir("wide.pt", torch.randn(1, 768))
    emb_dir("x.pt", torch.randn(1, 16))  # 'x' is already a word of the tokenizer
    req = Text2ImageRequest(
        model_id=tiny_sd, prompt="xlonly wide", embeddings=["x.pt", "gone.pt"], **BASE
    )
    params = generate_images(req).params
    reasons = params["embeddings_skipped_reasons"]
    assert set(params["embeddings_skipped"]) == {"xlonly.safetensors", "wide.pt", "x.pt", "gone.pt"}
    assert "SDXL embedding" in reasons["xlonly.safetensors"]
    assert "width 768" in reasons["wide.pt"]
    assert "ordinary word" in reasons["x.pt"]
    assert reasons["gone.pt"].startswith("not found")
    assert params["embeddings"] == []


def test_sdxl_embedding_goes_into_both_encoders(tiny_sdxl, emb_dir):
    from gradient.ai.diffusion.loading import get_pipeline
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import generate_images

    torch.manual_seed(2)
    clip_l, clip_g = torch.randn(2, 16), torch.randn(2, 16)
    emb_dir("xlstyle.safetensors", {"clip_l": clip_l, "clip_g": clip_g})
    emb_dir("sd1only.pt", torch.randn(1, 16))
    req = Text2ImageRequest(model_id=tiny_sdxl, prompt="a dog, xlstyle, sd1only", **BASE)
    params = generate_images(req).params
    assert params["embeddings"] == ["xlstyle.safetensors"]
    assert "needs SDXL embeddings" in params["embeddings_skipped_reasons"]["sd1only.pt"]
    pipe, _ = get_pipeline(tiny_sdxl)
    assert torch.allclose(_row(pipe.text_encoder, pipe.tokenizer, "xlstyle_1"), clip_l[1])
    assert torch.allclose(_row(pipe.text_encoder_2, pipe.tokenizer_2, "xlstyle"), clip_g[0])


def test_details_endpoint(client, tiny_sd, emb_dir):
    emb_dir("sd1style.pt", torch.randn(1, 768))
    emb_dir("xlstyle2.safetensors", {"clip_l": torch.randn(1, 768), "clip_g": torch.randn(1, 1280)})
    body = client.get(f"/api/v1/system/models/embeddings/details?model_id={tiny_sd}").json()
    assert body["model_family"] == "sd15"
    rows = {r["name"]: r for r in body["embeddings"]}
    assert rows["sd1style.pt"]["token"] == "sd1style"
    assert rows["sd1style.pt"]["fits"] is True
    assert rows["xlstyle2.safetensors"]["family_label"] == "SDXL"
    assert rows["xlstyle2.safetensors"]["fits"] is False
    no_model = client.get("/api/v1/system/models/embeddings/details").json()
    assert {r["fits"] for r in no_model["embeddings"]} == {None}


def test_job_summary_lists_skipped_embeddings():
    from types import SimpleNamespace

    from gradient.api.job_kinds import _image_summary

    image = SimpleNamespace(
        output_id=1, seed=2, parameters={"embeddings_skipped": ["b.pt", "a.pt"]}
    )
    response = SimpleNamespace(run_uuid="r", images=[image], performance_ms=1.0)
    assert _image_summary(response)["embeddings_skipped"] == ["a.pt", "b.pt"]
