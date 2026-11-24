# tests/test_explorer.py
"""The explorer: trajectories, region-locked branches, settling map, word suggestions."""

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
    "num_inference_steps": 6,
    "guidance_scale": 5.0,
    "seed": 21,
}


def _img(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), np.float32)


def left_half(width=64, height=64) -> str:
    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (0, 0, width // 2, height))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _post(client, path, body):
    r = client.post(f"/api/v1/lab/explore/{path}", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_branch_keeps_outside_and_replays(client, tiny_sd):
    from gradient.ai.lab import explorer
    from gradient.ai.lab.models import ExploreBase, ExploreOp

    base = {"base_request": {**BASE, "model_id": tiny_sd}}
    start = _post(client, "start", {**base, "frames": 4})
    assert start["steps"] == 6 and start["seed"] == 21
    assert len(start["frames"]) == 4 and start["frames"][0]["still_changing"] == pytest.approx(1)
    assert start["frames"][-1]["still_changing"] < 0.5  # the last estimate is nearly final

    mask = left_half()
    out = _post(
        client,
        "branches",
        {**base, "mask_base64": mask, "step": 2, "branches": [{"kind": "seed"}, {"kind": "seed"}]},
    )
    assert [b["op"]["seed"] for b in out["branches"]] == [1021, 1022]
    first = out["branches"][0]
    assert first["op"]["step"] == 2 and first["op"]["kind"] == "seed"
    assert first["parameters"]["explore_path"][-1]["seed"] == 1021
    assert first["change"] > 0 and first["outside_change"] is not None

    # outside the region the latents are the recorded ones, exactly
    spec = ExploreBase.model_validate({**base, "path": [first["op"]]})
    branch, _req, _ = explorer.get_trajectory(spec)
    plain, _req, _ = explorer.get_trajectory(ExploreBase.model_validate(base))
    pipe_mask = explorer.latent_mask(None, plain.latents[0], mask, 64, 64)
    outside = pipe_mask < 1e-6
    assert outside.any()
    assert torch.equal(branch.latents[-1][outside], plain.latents[-1][outside])
    assert not torch.allclose(branch.latents[-1][~outside], plain.latents[-1][~outside])
    assert torch.equal(branch.latents[1], plain.latents[1])  # steps before the branch are shared

    # replaying the path (cold cache) gives the same image as the batched branch
    explorer.clear_cache()
    replay = _post(client, "start", {**base, "path": [first["op"]], "frames": 2})
    diff = np.abs(_img(replay["image_base64"]) - _img(first["image_base64"]))
    assert diff.max() <= 2, diff.max()

    # continuing from the branch: the next branch inherits its state
    op2 = ExploreOp.model_validate({**first["op"], "step": 4, "seed": 5})
    again = explorer.get_trajectory(
        ExploreBase.model_validate({**base, "path": [first["op"], op2.model_dump()]})
    )[0]
    assert len(again.latents) == 6 and torch.equal(again.latents[3], branch.latents[3])


def test_map_tokens_and_kinds(client, tiny_sd, tiny_vision):
    base = {"base_request": {**BASE, "model_id": tiny_sd}}
    mask = left_half()
    data = _post(
        client,
        "map",
        {
            **base,
            "mask_base64": mask,
            "num_steps": 3,
            "branches": 2,
            "feature_model": tiny_vision[0],
        },
    )
    points = data["points"]
    assert points[0]["step"] == 0 and points[0]["change"] == pytest.approx(1)
    assert [p["step"] for p in points] == sorted(p["step"] for p in points)
    assert all(len(p["images_base64"]) == 2 and p["heatmap_base64"] for p in points)
    assert points[-1]["change"] < points[0]["change"]
    assert set(data["suggested"]) == {"big", "variations", "details"}
    assert data["metric"] == "dino" and data["summary"].endswith(".")
    assert points[0]["estimate"] == pytest.approx(1) and data["estimate_error"] is not None
    assert data["image_base64"]

    words = _post(client, "tokens", {**base, "mask_base64": mask, "alternatives": 3})
    assert words["method"] == "attention" and words["words"]
    assert {w["word"] for w in words["words"]} <= {"red", "lighthouse", "rocky", "coast"}
    assert max(w["score"] for w in words["words"]) == pytest.approx(1)
    assert all(len(w["alternatives"]) <= 3 for w in words["words"])

    abl = _post(
        client,
        "tokens",
        {
            **base,
            "mask_base64": mask,
            "method": "ablation",
            "max_words": 2,
            "feature_model": tiny_vision[0],
        },
    )
    assert abl["method"] == "ablation" and len(abl["words"]) == 2

    out = _post(
        client,
        "branches",
        {
            **base,
            "mask_base64": mask,
            "step": 1,
            "branches": [
                {"kind": "prompt", "prompt": "a blue lighthouse on a rocky coast"},
                {"kind": "mix", "mix_prompt": "a castle", "mix_weight": 0.5},
            ],
        },
    )
    word, mix = out["branches"]
    assert (
        word["op"]["prompt"] == "a blue lighthouse on a rocky coast" and word["label"] == "+ blue"
    )
    assert mix["op"]["mix_prompt"] == "a castle" and mix["op"]["seed"] == word["op"]["seed"]
    assert not np.array_equal(_img(word["image_base64"]), _img(mix["image_base64"]))


def test_image_to_image_base(client, tiny_sd):
    img = Image.new("RGB", (64, 64), (200, 60, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    base = {
        "base_request": {**BASE, "model_id": tiny_sd},
        "init_image_base64": base64.b64encode(buf.getvalue()).decode(),
        "strength": 0.5,
    }
    start = _post(client, "start", {**base, "frames": 2})
    assert start["steps"] == 3  # half of the 6 steps
    out = _post(
        client, "branches", {**base, "mask_base64": left_half(), "step": 1, "branches": [{}]}
    )
    assert out["branches"][0]["parameters"]["explore_image_to_image"] == {"strength": 0.5}


@pytest.mark.parametrize("fixture", ["tiny_flux", "tiny_sd3"])
def test_transformer_models(client, request, fixture):
    from gradient.ai.lab import explorer
    from gradient.ai.lab.models import ExploreBase

    model_id = request.getfixturevalue(fixture)
    base = {"base_request": {**BASE, "model_id": model_id, "num_inference_steps": 4}}
    start = _post(client, "start", {**base, "frames": 3})
    assert start["steps"] == 4 and start["frames"]
    mask = left_half()
    out = _post(
        client,
        "branches",
        {
            **base,
            "mask_base64": mask,
            "step": 1,
            "branches": [{"kind": "seed"}, {"kind": "mix", "mix_prompt": "a castle"}],
        },
    )
    seed_branch = out["branches"][0]
    spec = ExploreBase.model_validate({**base, "path": [seed_branch["op"]]})
    branch = explorer.get_trajectory(spec)[0]
    plain = explorer.get_trajectory(ExploreBase.model_validate(base))[0]
    from gradient.ai.diffusion.loading import get_pipeline

    pipe, _ = get_pipeline(model_id)
    m = explorer.latent_mask(pipe, plain.latents[0], mask, 64, 64)
    outside = m < 1e-6
    assert outside.any() and torch.equal(branch.latents[-1][outside], plain.latents[-1][outside])
    words = _post(client, "tokens", {**base, "mask_base64": mask, "alternatives": 2})
    assert words["words"] and words["method"] in ("attention", "ablation")


def test_branches_saved_to_gallery(client, tiny_sd):
    base = {"base_request": {**BASE, "model_id": tiny_sd}}
    out = _post(
        client, "branches", {**base, "mask_base64": left_half(), "step": 3, "branches": [{}, {}]}
    )
    assert [b["label"] for b in out["branches"]] == ["seed 1021", "seed 1022"]
    assert out["branches"][0]["op"]["label"] == "seed 1021"  # shown in the path
    images = client.get("/api/v1/history/outputs", params={"limit": 2}).json()["items"]
    assert len(images) == 2 and all(i["run_type"] == "lab_explore_branches" for i in images)
    detail = client.get(f"/api/v1/history/outputs/{images[0]['id']}").json()["parameters"]
    assert detail["explore_path"][-1]["mask_base64"].startswith("data:image/png")
    assert detail["lab_tool"] == "explore_branches" and detail["explore_label"].startswith("seed")
    r = client.post("/api/v1/lab/explore/branches", json={**base, "step": 99, "branches": [{}]})
    assert r.status_code == 400 and "outside" in r.text
