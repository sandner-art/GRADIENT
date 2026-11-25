# tests/test_prompt_syntax.py
"""A1111 prompt syntax: weights, long prompts, BREAK, prompt editing, alternation."""

import numpy as np
import pytest
import torch

from gradient.ai.diffusion import prompt_syntax as ps

BASE = {"width": 64, "height": 64, "num_inference_steps": 4, "guidance_scale": 5.0, "seed": 11}
LONG = ", ".join(f"word{i}" for i in range(20))  # about 120 tokens for the tiny tokenizer


def test_weights_follow_a1111_rules():
    assert ps.parse_weights("a (red:1.5) dog") == [("a ", 1.0), ("red", 1.5), (" dog", 1.0)]
    runs = dict(ps.parse_weights("((big)) [small] (x:0.5)"))
    assert runs["big"] == pytest.approx(1.21) and runs["small"] == pytest.approx(1 / 1.1)
    assert runs["x"] == 0.5
    assert ps.parse_weights(r"\(not weighted\)") == [("(not weighted)", 1.0)]
    assert ps.parse_weights("(pixar|disney:1.2) monster")[0] == ("pixar|disney", 1.2)
    assert ps.parse_weights("a BREAK b") == [("a", 1.0), ("", None), ("b", 1.0)]
    assert ps.parse_weights("open (end") == [("open ", 1.0), ("end", 1.1)]
    assert ps.parse_weights("a) b] c:2") == [("a) b] c:2", 1.0)]
    assert not ps.uses_weights("plain text, ratio 3:4")
    assert ps.uses_weights("a (b)") and ps.uses_weights("a BREAK b") and ps.uses_weights(r"\(x\)")


def test_editing_and_alternation_schedules():
    assert ps.schedule("a [b:0.5]", 4) == [(0, "a "), (2, "a b")]
    assert ps.schedule("[cat:dog:2] x", 4) == [(0, "cat x"), (2, "dog x")]
    assert ps.schedule("[gone::0.25] stays", 4) == [(0, "gone stays"), (1, " stays")]
    assert [t for _, t in ps.schedule("[a|b|c]", 4)] == ["a", "b", "c", "a"]
    assert ps.schedule("[emphasis] and (x:1.2)", 10) == [(0, "[emphasis] and (x:1.2)")]
    assert ps.schedule("[(red:1.3):blue:0.5]", 2) == [(0, "(red:1.3)"), (1, "blue")]


def test_chunks_break_after_a_nearby_comma():
    tokens = [(i, 1.0) for i in range(80)]
    tokens[60] = (999, 1.0)  # a comma 14 tokens before the chunk ends
    assert [len(c) for c in ps.chunk_tokens(tokens, 75, 999)] == [61, 19]
    tokens[60] = (60, 1.0)
    tokens[40] = (999, 1.0)  # too far back: cut at 75
    assert [len(c) for c in ps.chunk_tokens(tokens, 75, 999)] == [75, 5]
    assert [len(c) for c in ps.chunk_tokens(tokens[:5] + [None] + tokens[:5], 75, 999)] == [5, 5]
    assert ps.chunk_tokens([], 75, 999) == [[]]


def _pipe(model_id):
    from gradient.ai.diffusion.loading import get_pipeline

    return get_pipeline(model_id)[0]


def test_plain_short_prompts_match_the_pipeline(tiny_sd, tiny_sdxl):
    prompt, negative = "a red dog on a hill", "blurry"
    pipe = _pipe(tiny_sd)
    enc = ps.PromptEncoder(pipe)
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(prompt, "cpu", 1, True, negative)
        mine, mine_neg = enc.encode(prompt), enc.encode(negative)
    assert mine.chunks == 1 and torch.allclose(mine.embeds, pe, atol=1e-6)
    assert torch.allclose(mine_neg.embeds, ne, atol=1e-6)

    pipe = _pipe(tiny_sdxl)
    enc = ps.PromptEncoder(pipe)
    with torch.no_grad():
        pe, ne, ppe, npe = pipe.encode_prompt(
            prompt=prompt, device="cpu", do_classifier_free_guidance=True, negative_prompt=negative
        )
        mine, mine_neg = enc.encode(prompt), enc.encode(negative)
    assert torch.allclose(mine.embeds, pe, atol=1e-6)
    assert torch.allclose(mine.pooled, ppe, atol=1e-6)
    assert torch.allclose(mine_neg.embeds, ne, atol=1e-6)
    assert torch.allclose(mine_neg.pooled, npe, atol=1e-6)


def test_long_prompts_and_weights(tiny_sd):
    pipe = _pipe(tiny_sd)
    with torch.no_grad():
        long = ps.PromptEncoder(pipe).encode(LONG)
        assert long.chunks == 2 and long.embeds.shape[1] == 154 and long.tokens > 75
        # the first chunk is the prompt up to its last comma that fits
        first = ps.PromptEncoder(pipe).encode(LONG[: LONG.index("word12") - 1])
        assert torch.allclose(long.embeds[:, :77], first.embeds, atol=1e-6)

        plain = ps.PromptEncoder(pipe).encode("a red dog")
        comfy = ps.PromptEncoder(pipe, "comfy").encode("a (red:1.5) dog")
        a1111 = ps.PromptEncoder(pipe, "a1111").encode("a (red:1.5) dog")
    assert comfy.weighted and a1111.weighted and not plain.weighted
    # comfy: tokens of weight 1 keep their embeddings; 'red' (tokens 3-5 after BOS 'a') moves
    changed = (comfy.embeds - plain.embeds).abs().amax(-1)[0]
    assert changed[:2].max() < 1e-6 and changed[2:5].min() > 1e-4
    # a1111: 'red' scaled by 1.5 relative to the rest (then the chunk's mean restored)
    ratio = (a1111.embeds / plain.embeds)[0]
    assert torch.allclose(ratio[2:5].median(), 1.5 * ratio[0].median(), rtol=1e-4)
    assert torch.allclose(ratio[0], ratio[0].median().expand(16), rtol=1e-4)


def _render(model_id, prompt, **extra):
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import generate_images

    req = Text2ImageRequest(model_id=model_id, prompt=prompt, **{**BASE, **extra})
    result = generate_images(req)
    return np.asarray(result.images[0]).astype(np.int16), result.params


def test_renders_unchanged_without_syntax(tiny_sd):
    image, params = _render(tiny_sd, "a red dog", negative_prompt="blurry")
    plain, _ = _render(tiny_sd, "a red dog", negative_prompt="blurry", prompt_syntax="plain")
    assert "prompt_syntax_applied" not in params
    assert np.array_equal(image, plain)


def test_long_prompt_renders_every_chunk(tiny_sd, tiny_sdxl):
    for model in (tiny_sd, tiny_sdxl):
        image, params = _render(model, LONG, negative_prompt="")
        assert params["prompt_chunks"] == 2 and params["prompt_tokens"] > 75
        cut, cut_params = _render(model, LONG, negative_prompt="", prompt_syntax="plain")
        assert cut_params["prompt_truncated"] is True
        assert not np.array_equal(image, cut)


def test_weights_change_the_image_and_modes_differ(tiny_sd):
    plain, _ = _render(tiny_sd, "a red dog")
    a1111, params = _render(tiny_sd, "a (red:1.6) dog")
    comfy, _ = _render(tiny_sd, "a (red:1.6) dog", prompt_syntax="comfy")
    assert params["prompt_weights_applied"] is True
    assert not np.array_equal(plain, a1111) and not np.array_equal(a1111, comfy)


def test_prompt_editing_switches_mid_render(tiny_sd, tiny_sdxl):
    for model in (tiny_sd, tiny_sdxl):
        cat, _ = _render(model, "a cat")
        dog, _ = _render(model, "a dog")
        late, params = _render(model, "a [cat:dog:0.99]")  # 'dog' would start at step 3.96
        assert np.abs(late - cat).max() <= 1, model
        swapped, params = _render(model, "a [cat:dog:0.25]")
        assert params["prompt_schedule"] == [[0, "a cat"], [1, "a dog"]]
        assert not np.array_equal(swapped, cat) and not np.array_equal(swapped, dog)
        _, params = _render(model, "a [cat|dog]", negative_prompt="[blurry::0.5]")
        assert len(params["prompt_schedule"]) == 4 and len(params["negative_prompt_schedule"]) == 2


def test_lab_tools_keep_plain_prompts(tiny_sd):
    from gradient.ai.lab.probes import AttentionRecorder, ModulationController

    assert AttentionRecorder.needs_plain_prompt
    from gradient.ai.diffusion.models import ModulationLane, Text2ImageRequest

    req = Text2ImageRequest(model_id=tiny_sd, prompt="a (red:1.3) dog", **BASE)
    lanes = [ModulationLane(type="prompt", prompt="a cat", start=0.5, end=1.0)]
    assert ModulationController(lanes, req).needs_plain_prompt
    assert not ModulationController([], req).needs_plain_prompt
