# tests/test_pbr.py
"""PBR maps: the ported network and classical pieces, wrap-around, the map set and its API."""

import base64
import io
import json
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from gradient.ai.pbr import classic as C
from gradient.ai.pbr import maps
from gradient.ai.pbr.net import UNet, load_net, predict_all, save_net


def _periodic_texture(n=96, seed=0) -> np.ndarray:
    """A periodic height, lit from the upper left, with a periodic colour pattern."""
    y, x = np.mgrid[0:n, 0:n] / n
    h = (
        np.sin(2 * np.pi * 3 * x) * np.cos(2 * np.pi * 2 * y) + 0.5 * np.sin(2 * np.pi * (x + y))
    ) * 6
    p, q = C.height_slopes(h, (True, True))
    L = np.array([-0.5, 0.5, 0.7])
    shade = np.clip(C.slopes_to_normal(p, q) @ (L / np.linalg.norm(L)), 0, 1)
    tint = 0.6 + 0.3 * np.sin(2 * np.pi * 2 * x)[..., None] * np.array([1.0, 0.5, 0.2])
    return np.clip(shade[..., None] * tint, 0, 1).astype(np.float32)


def _seam(a: np.ndarray, axis: int) -> float:
    """Step across the wrap edge / step in the middle (about 1 = no seam)."""
    a = a if a.ndim == 2 else a.mean(-1)
    d = np.abs(np.diff(a, axis=axis, append=np.take(a, [0], axis=axis)))
    wrap = np.take(d, -1, axis=axis).mean()
    mid = np.take(d, a.shape[axis] // 2 - 1, axis=axis).mean()
    return float(wrap / (mid + 1e-9))


@pytest.fixture(scope="module")
def tiny_net(workspace):
    """A tiny random pbr_net with property heads, installed as <models>/pbr/tiny-pbr.pt."""
    torch.manual_seed(0)
    net = UNet(cues="both", width=(8, 8, 16, 16, 16), props=True)
    folder = workspace / "models" / "pbr"
    folder.mkdir(parents=True, exist_ok=True)
    save_net(net, folder / "tiny-pbr.pt")
    return "tiny-pbr"


def test_checkpoint_round_trip_and_outputs(tiny_net, workspace):
    net = load_net(workspace / "models" / "pbr" / "tiny-pbr.pt")
    assert net.props and not net.training
    img = _periodic_texture(80)  # not a multiple of 16: padded and cut back
    o = predict_all(net, img)
    assert o["normal"].shape == (80, 80, 3) and o["albedo"].shape == (80, 80, 3)
    assert o["flow"].shape == (80, 80, 2) and o["light"].shape == (3,)
    assert np.allclose(np.linalg.norm(o["normal"], axis=-1), 1, atol=1e-4)
    for k in ("ao", "rough", "metal", "emissive"):
        assert 0 <= o[k].min() and o[k].max() <= 1


def test_wrap_makes_the_network_a_torus(tiny_net):
    net = maps.get_net(tiny_net)
    img = _periodic_texture(96)
    a = predict_all(net, img, (True, True))
    b = predict_all(net, np.roll(img, (32, 48), (0, 1)), (True, True))
    assert np.abs(np.roll(a["normal"], (32, 48), (0, 1)) - b["normal"]).max() < 1e-4
    # without wrap-around the borders break the symmetry
    c = predict_all(net, img)
    d = predict_all(net, np.roll(img, (32, 48), (0, 1)))
    assert np.abs(np.roll(c["normal"], (32, 48), (0, 1)) - d["normal"]).max() > 1e-3


def test_poisson_recovers_a_periodic_height():
    n = 64
    y, x = np.mgrid[0:n, 0:n] / n
    h = np.sin(2 * np.pi * 2 * x) + np.cos(2 * np.pi * 3 * y) * 0.5
    p, q = C.height_slopes(h, (True, True))
    z = C.frankot_chellappa(p, q, (True, True))
    assert np.abs((z - z.mean()) - (h - h.mean())).max() < 0.02
    # an open domain still integrates a ramp (mirrored, so no wrap jump is invented)
    ramp = x * 10
    pr, qr = C.height_slopes(ramp)
    zr = C.frankot_chellappa(pr, qr)
    assert np.corrcoef(zr.ravel(), ramp.ravel())[0, 1] > 0.99


@pytest.mark.parametrize("use_net", [False, True])
def test_seamless_maps_have_no_seam(tiny_net, use_net):
    img = _periodic_texture(96)
    net = maps.get_net(tiny_net) if use_net else None
    tiled = maps.compute_maps(img, net=net, wrap=(True, True))
    normal = C.slopes_to_normal(tiled["p"], tiled["q"])
    for m in (tiled["height"], normal, tiled["ao"], tiled["albedo"]):
        assert _seam(m, 1) < 1.6 and _seam(m, 0) < 1.6
    open_ = maps.compute_maps(img, net=net, wrap=(False, False))
    assert _seam(open_["height"], 1) > 2 or _seam(open_["height"], 0) > 2
    # one axis only
    x_only = maps.compute_maps(img, net=net, wrap=(True, False))
    assert _seam(x_only["height"], 1) < 1.6


def test_albedo_keeps_the_picture_colour(tiny_net):
    img = _periodic_texture(96) * np.array([1.0, 0.45, 0.3], np.float32)  # reddish
    m = maps.compute_maps(img, net=maps.get_net(tiny_net))
    r, g, b = m["albedo"].reshape(-1, 3).mean(0)
    assert r > g > b
    # the light is gone: the albedo varies less than the picture
    lum = m["albedo"] @ C.LUMA
    assert lum.std() / lum.mean() < (img @ C.LUMA).std() / (img @ C.LUMA).mean()


def _b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_api_map_set(client, tiny_net):
    models = client.get("/api/v1/pbr/models").json()
    assert "tiny-pbr" in [m["name"] for m in models["models"]]
    r = client.post(
        "/api/v1/pbr/maps",
        json={
            "image_base64": _b64(_periodic_texture(96)),
            "name": "brick wall",
            "model": tiny_net,
            "seamless": "xy",
            "strength": 2,
        },
    )
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["name"] == "brick_wall" and s["model"] == "tiny-pbr" and s["method"] == "hybrid"
    assert s["seamless"] == {"x": True, "y": True}
    kinds = [m["kind"] for m in s["maps"]]
    assert kinds[:6] == ["albedo", "normal", "height", "ao", "curvature", "glow"]
    assert {"roughness", "metallic", "mask_map", "orm", "metallic_smoothness"} <= set(kinds)
    base = f"/api/v1/pbr/maps/{s['set_id']}"
    height = Image.open(io.BytesIO(client.get(f"{base}/files/brick_wall_height.png").content))
    assert height.mode.startswith("I") and height.size == (96, 96)  # 16-bit
    mask = Image.open(io.BytesIO(client.get(f"{base}/files/brick_wall_mask_map.png").content))
    assert mask.mode == "RGBA"
    side = client.get(f"{base}/files/{s['sidecar']}").json()
    assert side["schema"] == 2 and side["width"] == 96 and side["strength"] == 2
    assert side["packs"]["_mask_map"] == ["Metal", "AO", "DetailMask", "Smoothness"]
    assert "azimuth_deg" in side["light"]
    z = zipfile.ZipFile(io.BytesIO(client.get(f"{base}/zip").content))
    assert "brick_wall/brick_wall_normal.png" in z.namelist()
    assert len(z.namelist()) == len(s["maps"]) + 1
    # only the set's own files are served
    assert client.get(f"{base}/files/..%2F..%2Fsecret.png").status_code == 404
    assert client.get("/api/v1/pbr/maps/notanid/zip").status_code == 404
    # the network shows in the memory report and is freed with the vision models
    vision = client.get("/api/v1/system/memory").json()["loaded"]["vision"]
    assert any(v["name"] == "pbr:tiny-pbr" for v in vision)
    freed = client.post("/api/v1/system/unload", json={"targets": ["vision"]}).json()["freed"]
    assert "pbr:tiny-pbr" in freed["vision"]


def test_api_classical_and_errors(client):
    img = _periodic_texture(96)
    r = client.post("/api/v1/pbr/maps", json={"image_base64": _b64(img), "method": "classical"})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["method"] == "classical" and s["model"] is None and s["light"] is None
    assert "roughness" not in [m["kind"] for m in s["maps"]]
    small = _b64(_periodic_texture(96)[:40, :40])
    assert client.post("/api/v1/pbr/maps", json={"image_base64": small}).status_code == 400
    assert client.post("/api/v1/pbr/maps", json={}).status_code == 422


def test_api_from_a_gallery_image_follows_its_tiling(client, tiny_sd, tiny_net):
    r = client.post(
        "/api/v1/generate/text-to-image",
        json={
            "model_id": tiny_sd,
            "prompt": "stone wall",
            "width": 96,
            "height": 96,
            "num_inference_steps": 2,
            "seed": 3,
            "tiling": "x",
        },
    )
    assert r.status_code == 200, r.text
    output_id = r.json()["images"][0]["output_id"]
    s = client.post("/api/v1/pbr/maps", json={"output_id": output_id, "model": tiny_net}).json()
    assert s["seamless"] == {"x": True, "y": False} and s["source_output_id"] == output_id
    side = json.loads(maps.set_file(s["set_id"], s["sidecar"]).read_text())
    assert side["source"].endswith(".png")
    assert client.post("/api/v1/pbr/maps", json={"output_id": 999999}).status_code == 404
