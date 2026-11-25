# tests/test_memory.py
"""Freeing memory: image models, the vision models used for metrics, caches; the report."""

import pytest


@pytest.fixture(scope="module")
def tiny_clip(workspace):
    from helpers import build_tiny_clip

    target = workspace / "models" / "vision" / "clip-tiny"
    if not target.exists():
        build_tiny_clip(target)
    return "clip-tiny"


def _tracked(prefix):
    from gradient.core.resource_manager import resource_manager

    return sorted(r["id"] for r in resource_manager.snapshot() if r["id"].startswith(prefix))


def test_vision_models_are_tracked_and_unloaded(client, tiny_sd, tiny_vision, tiny_clip):
    from gradient.ai.diffusion.loading import get_pipeline, loaded_pipelines
    from gradient.ai.lab import embed, features

    features.unload_feature_models()  # earlier tests may have loaded others
    embed.unload_clip_models()
    get_pipeline(tiny_sd)
    features.load_feature_model("dinov2-tiny")
    embed.load_clip(tiny_clip)
    assert _tracked("vision:") == ["vision:clip-tiny", "vision:dinov2-tiny"]

    report = client.get("/api/v1/system/memory").json()
    vision = {m["name"]: m for m in report["loaded"]["vision"]}
    assert set(vision) == {"dinov2-tiny", "clip-tiny"}
    assert vision["clip-tiny"]["detail"] == "metrics (CLIP/SigLIP)"
    assert tiny_sd in [m["name"] for m in report["loaded"]["diffusion"]]
    assert report["vram"] is None  # no CUDA in the tests
    assert set(report["caches"]) == {"explorer_paths", "word_vocabularies", "prompt_tokenizers"}

    # only the vision models
    r = client.post("/api/v1/system/unload", json={"targets": ["vision"]}).json()
    assert sorted(r["freed"]["vision"]) == ["clip-tiny", "dinov2-tiny"]
    assert r["unloaded"] == 0 and loaded_pipelines
    assert not features._cache and not embed._cache and _tracked("vision:") == []

    # no body: everything that holds VRAM (as before, `unloaded` counts image models)
    features.load_feature_model("dinov2-tiny")
    r = client.post("/api/v1/system/unload").json()
    assert r["unloaded"] >= 1 and tiny_sd in r["freed"]["diffusion"]
    assert r["freed"]["vision"] == ["dinov2-tiny"] and r["freed"]["llm"] == []
    assert not loaded_pipelines and not features._cache
    assert "caches" not in r["freed"]

    bad = client.post("/api/v1/system/unload", json={"targets": ["everything"]})
    assert bad.status_code == 422


def test_caches_are_cleared_on_request(client, tiny_sd):
    from gradient.ai.diffusion import prompt_syntax
    from gradient.ai.lab import explorer

    prompt_syntax._tokenizers["x"] = None
    explorer._vocab_cache[1] = (lambda: None, [], None)
    r = client.post("/api/v1/system/unload", json={"targets": ["caches"]}).json()
    assert r["freed"]["caches"]["prompt_tokenizers"] >= 1
    assert r["freed"]["caches"]["word_vocabularies"] >= 1
    assert not prompt_syntax._tokenizers and not explorer._vocab_cache


def test_word_vocabulary_is_rebuilt_for_a_new_encoder(tiny_sd):
    import weakref

    import torch

    from gradient.ai.diffusion.loading import get_pipeline
    from gradient.ai.lab import explorer

    pipe, _ = get_pipeline(tiny_sd)
    key = id(pipe.text_encoder)
    # an entry left by an unloaded encoder whose id was reused is not returned
    explorer._vocab_cache[key] = (lambda: None, ["stale"], torch.zeros(1, 4))
    assert explorer._vocabulary(pipe) is None  # the tiny tokenizer has no whole words
    # an entry for this very encoder is
    explorer._vocab_cache[key] = (weakref.ref(pipe.text_encoder), ["fresh"], torch.zeros(1, 4))
    assert explorer._vocabulary(pipe)[0] == ["fresh"]
    explorer.clear_vocabularies()
