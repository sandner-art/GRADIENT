# tests/test_styles.py
"""Prompt styles: A1111 styles.csv, positions, wildcards, <lora:> tags, rendering, API."""

import random

import numpy as np
import pytest

from gradient.ai.diffusion import styles

CSV = """name,prompt,negative_prompt
Style: Photographic,"cinematic photo {prompt} . 35mm photograph, film, bokeh","drawing, painting"
CX: Dynamic,"dynamic pose, motion","static"
--------<,,
AX: Color,"{prompt}, {red background|blue background}",
EX: Detail,"{prompt} <lora:detailer:0.6>",
"""
BASE = {"width": 64, "height": 64, "num_inference_steps": 4, "guidance_scale": 5.0, "seed": 5}


@pytest.fixture
def style_files():
    folder = styles.styles_dir()
    made = []

    def put(name, text):
        path = folder / name
        path.write_text(text, encoding="utf-8")
        made.append(path)
        return path

    put("styles.csv", "﻿" + CSV)
    yield put
    for path in made:
        path.unlink(missing_ok=True)


def test_reads_a1111_csv_and_json(style_files):
    style_files(
        "more.json",
        '[{"name": "CX: Dynamic", "prompt": "dupe"}, {"name": "Ink", "prompt": "ink {prompt}"}]',
    )
    found, errors = styles.load_styles()
    names = [s.name for s in found]  # files in name order; a later clash gets the file name
    assert names == [
        "CX: Dynamic",
        "Ink",
        "Style: Photographic",
        "CX: Dynamic (styles)",
        "AX: Color",
        "EX: Detail",
    ]
    by_name = {s.name: s for s in found}
    assert by_name["CX: Dynamic"].prompt == "dupe" and by_name["CX: Dynamic"].group == "CX"
    assert by_name["Ink"].group == "more"
    assert errors == {}
    style_files("broken.json", "{not json")
    assert "broken.json" in styles.load_styles()[1]


def test_positions():
    photo = "cinematic photo {prompt} . 35mm photograph, film"
    assert styles.merge("a cat", photo) == "cinematic photo a cat . 35mm photograph, film"
    assert styles.merge("a cat", "motion blur") == "a cat, motion blur"
    assert styles.merge("a cat", photo, "before") == "cinematic photo, 35mm photograph, film, a cat"
    assert styles.merge("a cat", photo, "after") == "a cat, cinematic photo, 35mm photograph, film"
    assert styles.merge("a cat", photo, "chunk") == (
        "a cat BREAK cinematic photo, 35mm photograph, film"
    )
    assert styles.merge("a cat", photo, "chunk", chunks=False).startswith("a cat, cinematic")
    assert styles.merge("", "motion blur") == "motion blur"
    assert styles.merge("a cat", "") == "a cat"


def test_wildcards_and_tags():
    rng = random.Random(1)
    picked = {styles.resolve_wildcards("{red|blue} {a|{b|c}}", random.Random(s)) for s in range(30)}
    assert len(picked) > 2 and all("{" not in p for p in picked)
    assert styles.resolve_wildcards("{prompt} stays", rng) == "{prompt} stays"
    text, tags = styles.extract_tags("a cat, <lora:detailer:0.6>, <lyco:x>, <hypernet:h:1>")
    assert text == "a cat" and tags == [
        ("lora", "detailer", 0.6),
        ("lyco", "x", 1.0),
        ("hypernet", "h", 1.0),
    ]
    assert styles.resolve_lora("Detailer", ["detailer.safetensors"]) == "detailer.safetensors"
    assert styles.resolve_lora("other", ["detailer.safetensors"]) is None


def _req(**kw):
    from gradient.ai.diffusion.models import Text2ImageRequest

    return Text2ImageRequest(model_id="m", prompt=kw.pop("prompt", "a cat"), **kw)


def test_prepare(style_files):
    p = styles.prepare(
        _req(styles=["Style: Photographic", "CX: Dynamic", "gone"], negative_prompt="ugly"), 1
    )
    assert p.prompt == "cinematic photo a cat . 35mm photograph, film, bokeh, dynamic pose, motion"
    assert p.negative_prompt == "ugly, drawing, painting, static"
    assert p.params["styles_missing"] == ["gone"]
    later = styles.prepare(_req(styles=["CX: Dynamic"], style_start=0.4), 1)
    assert later.prompt == "[a cat:a cat, dynamic pose, motion:0.4]"
    flux = styles.prepare(_req(styles=["CX: Dynamic"], style_start=0.4), 1, clip_syntax=False)
    assert (
        flux.prompt == "a cat, dynamic pose, motion" and flux.params["style_start_applied"] is False
    )
    tagged = styles.prepare(_req(styles=["EX: Detail"]), 1, installed_loras=["detailer.pt"])
    assert tagged.prompt == "a cat" and tagged.loras == [{"name": "detailer.pt", "weight": 0.6}]
    same = {styles.prepare(_req(styles=["AX: Color"]), 7).prompt for _ in range(3)}
    assert len(same) == 1  # the same seed picks the same option
    plain = styles.prepare(_req(prompt="a {b|c} <lora:x:1>", prompt_syntax="plain"), 1)
    assert plain.prompt == "a {b|c} <lora:x:1>" and not plain.params
    # a single name (an XYZ axis value) is a list of one; 'none' is no style
    assert _req(styles="CX: Dynamic").styles == ["CX: Dynamic"]
    assert _req(styles="none").styles == []
    assert _req(styles="CX: Dynamic + Ink").styles == ["CX: Dynamic", "Ink"]


def test_render_with_styles(tiny_sd, style_files):
    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import generate_images
    from gradient.gallery.metadata import a1111_parameters

    plain = generate_images(Text2ImageRequest(model_id=tiny_sd, prompt="a cat", **BASE))
    styled = generate_images(
        Text2ImageRequest(model_id=tiny_sd, prompt="a cat", styles=["CX: Dynamic"], **BASE)
    )
    assert styled.params["prompt"] == "a cat"
    assert styled.params["styled_prompt"] == "a cat, dynamic pose, motion"
    assert styled.params["styled_negative_prompt"] == "static"
    assert not np.array_equal(np.asarray(plain.images[0]), np.asarray(styled.images[0]))
    text = a1111_parameters({**styled.params, "seed": 5})
    assert text.startswith("a cat, dynamic pose, motion\nNegative prompt: static\n")

    chunked = generate_images(
        Text2ImageRequest(
            model_id=tiny_sd, prompt="a cat", styles=["CX: Dynamic"], style_position="chunk", **BASE
        )
    )
    assert chunked.params["prompt_chunks"] == 2
    late = generate_images(
        Text2ImageRequest(
            model_id=tiny_sd, prompt="a cat", styles=["CX: Dynamic"], style_start=0.5, **BASE
        )
    )
    assert late.params["prompt_schedule"][1][0] == 2


def test_render_with_a_lora_tag(tiny_sd, workspace, style_files):
    import shutil

    from helpers import build_tiny_lora

    from gradient.ai.diffusion.models import Text2ImageRequest
    from gradient.ai.diffusion.pipelines import generate_images

    lora_dir = workspace / "models" / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    made = build_tiny_lora(
        workspace / "sources" / "tiny-sd", workspace / "sources" / "tag.safetensors", 4
    )
    shutil.copy(made, lora_dir / "detailer.safetensors")
    try:
        result = generate_images(
            Text2ImageRequest(model_id=tiny_sd, prompt="a cat", styles=["EX: Detail"], **BASE)
        )
        assert result.params["lora_tags"] == ["lora:detailer:0.6"]
        assert result.params["loras_used"] == [{"name": "detailer.safetensors", "weight": 0.6}]
        assert "loras_skipped" not in result.params
        missing = generate_images(
            Text2ImageRequest(model_id=tiny_sd, prompt="a cat <lora:nothere:1>", **BASE)
        )
        assert missing.params["lora_tags_missing"] == ["nothere"]
    finally:
        (lora_dir / "detailer.safetensors").unlink(missing_ok=True)


def test_api(client, tiny_sd, style_files):
    body = client.get("/api/v1/styles").json()
    assert body["folder"].endswith("styles")
    assert {"name": "styles.csv", "styles": 4, "error": None} in body["files"]
    bad = client.post("/api/v1/styles/import", json={"filename": "x.txt", "content": "a"})
    assert bad.status_code == 400
    empty = client.post("/api/v1/styles/import", json={"filename": "x.csv", "content": "a,b\n1,2"})
    assert empty.status_code == 400
    ok = client.post(
        "/api/v1/styles/import",
        json={"filename": "../mine.csv", "content": "name,prompt\nMine,ink {prompt}\n"},
    )
    assert ok.json() == {"file": "mine.csv", "styles": 1}
    style_files("mine.csv", "name,prompt\nMine,ink {prompt}\n")  # removed after the test
    preview = client.post(
        "/api/v1/styles/preview",
        json={"model_id": tiny_sd, "prompt": "a cat", "styles": ["Mine", "CX: Dynamic"]},
    ).json()
    assert preview["prompt"] == "ink a cat, dynamic pose, motion"
    assert preview["negative_prompt"] == "static"
    assert preview["tokens"]["prompt"]["tokens"] > 0 and preview["tokens"]["prompt"]["limit"] == 75
    assert preview["reads_syntax"] is True
