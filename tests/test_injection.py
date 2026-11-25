# tests/test_injection.py
"""prompt_injection lanes: another prompt in the cross-attention of chosen blocks."""

import numpy as np
import pytest

BASE = {"width": 64, "height": 64, "num_inference_steps": 4, "guidance_scale": 5.0, "seed": 9}


def _render(model, prompt, lanes=None, **extra):
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import generate_images

    req = Text2ImageRequest(model_id=model, prompt=prompt, modulation=lanes, **{**BASE, **extra})
    result = generate_images(req)
    return np.asarray(result.images[0]).astype(np.int16), result.params


def lane(**kw):
    return {"type": "prompt_injection", "prompt": "a blue cat", "targets": "sdxl-style", **kw}


def test_injection_changes_only_while_active(tiny_sdxl):
    plain, _ = _render(tiny_sdxl, "a red dog")
    injected, params = _render(tiny_sdxl, "a red dog", [lane()])
    assert params["prompt_injection_layers"] == {"sdxl-style": 1}
    assert params["modulation_applied"]["prompt_injection"] is True
    assert not np.array_equal(plain, injected)
    # off for every step that runs: the image is untouched
    late, _ = _render(tiny_sdxl, "a red dog", [lane(start=0.9, end=1.0)], num_inference_steps=2)
    plain2, _ = _render(tiny_sdxl, "a red dog", num_inference_steps=2)
    assert np.array_equal(late, plain2)
    # blending with weight 0 keeps the main prompt
    zero, _ = _render(tiny_sdxl, "a red dog", [lane(mode="blend", value=0.0)])
    assert np.abs(zero - plain).max() <= 1


def test_injecting_the_same_prompt_changes_nothing(tiny_sd):
    plain, _ = _render(tiny_sd, "a red dog")
    same, params = _render(
        tiny_sd, "a red dog", [lane(prompt="a red dog", targets="up_blocks.0.attentions.1*")]
    )
    assert params["prompt_injection_layers"] == {"up_blocks.0.attentions.1*": 1}
    assert np.abs(same - plain).max() <= 1
    other, _ = _render(
        tiny_sd, "a red dog", [lane(prompt="a long green lizard", targets="up_blocks.*")]
    )
    assert not np.array_equal(other, plain)


def test_injection_errors(tiny_sd, tiny_flux):
    from gradient.ai.diffusion.models import ModulationLane

    with pytest.raises(ValueError, match="need targets"):
        ModulationLane(type="prompt_injection", prompt="x")
    with pytest.raises(ValueError, match="matches no cross-attention|No attention layers"):
        _render(tiny_sd, "a dog", [lane(targets="down_blocks.2.attentions.1.*attn2")])
    with pytest.raises(ValueError, match="UNet model"):
        _render(tiny_flux, "a dog", [lane()], guidance_scale=1.0)


@pytest.fixture
def style_file():
    from gradient.ai.diffusion import styles

    path = styles.styles_dir() / "blocks.csv"
    path.write_text('name,prompt,negative_prompt\nInk,"ink wash of {prompt}, sepia",photo\n')
    yield
    path.unlink(missing_ok=True)


def test_style_blocks_position(tiny_sdxl, tiny_sd, style_file):
    plain, _ = _render(tiny_sdxl, "a red dog")
    blocks, params = _render(tiny_sdxl, "a red dog", styles=["Ink"], style_position="blocks")
    assert params["style_injection"] == "ink wash of a red dog, sepia"
    assert "styled_prompt" not in params  # the main prompt stays as it is
    assert params["styled_negative_prompt"] == "photo"
    assert params["prompt_injection_layers"] == {"sdxl-style": 1}
    assert not np.array_equal(plain, blocks)
    _, sd = _render(tiny_sd, "a red dog", styles=["Ink"], style_position="blocks")
    assert sd["style_blocks_applied"] is False
    assert sd["styled_prompt"] == "a red dog, ink wash of, sepia"  # placed after the prompt
