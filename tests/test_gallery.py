# tests/test_gallery.py
"""Gallery/history API, PNG metadata, database upgrade, LoRA adapter cache."""

import io
import json
import sqlite3

import numpy as np
import pytest
from PIL import Image

BASE = {
    "prompt": "a blue castle on a hill",
    "negative_prompt": "blurry",
    "width": 64,
    "height": 64,
    "num_inference_steps": 3,
    "guidance_scale": 5.0,
    "seed": 21,
}


def render(client, model_id, **overrides):
    r = client.post(
        "/api/v1/generate/text-to-image", json={**BASE, "model_id": model_id, **overrides}
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_a1111_parameters_text():
    from gradient.gallery.metadata import a1111_parameters

    text = a1111_parameters(
        {
            "prompt": "a cat,\nsitting",
            "negative_prompt": "dog",
            "num_inference_steps": 20,
            "scheduler_id": "EulerAncestralDiscreteScheduler",
            "guidance_scale": 6.5,
            "seed": 7,
            "width": 512,
            "height": 768,
            "model_id": "sd15",
            "loras": [{"name": "detail.safetensors", "weight": 0.8}, {"name": "x", "weight": 1}],
            "strength": 0.6,
        }
    )
    lines = text.split("\n")
    assert lines[0] == "a cat, sitting" and lines[1] == "Negative prompt: dog"
    assert lines[2].startswith(
        "Steps: 20, Sampler: EulerAncestralDiscreteScheduler, CFG scale: 6.5, Seed: 7, "
        "Size: 512x768, Model: sd15, Lora: detail.safetensors:0.8; x:1, Denoising strength: 0.6"
    )


def test_browse_filter_edit_delete(client, tiny_sd, workspace):
    first = render(client, tiny_sd, num_outputs=2)
    second = render(client, tiny_sd, prompt="a green dragon in a cave", seed=5)
    a, b = (im["output_id"] for im in first["images"])
    c = second["images"][0]["output_id"]

    page = client.get("/api/v1/history/outputs", params={"limit": 2}).json()
    assert page["items"][0]["id"] == c and page["total"] >= 3 and page["next_offset"] == 2
    item = page["items"][0]
    assert item["prompt"] == "a green dragon in a cave" and item["seed"] == 5
    assert (item["width"], item["height"]) == (64, 64) and item["run_type"] == "text_to_image"

    def ids(**params):
        return [
            i["id"] for i in client.get("/api/v1/history/outputs", params=params).json()["items"]
        ]

    assert ids(q="dragon cave") == [c]
    assert set(ids(run=first["run_uuid"])) == {a, b}
    assert ids(run=first["run_uuid"], sort="oldest") == [a, b]

    # rate, favourite, tag, annotate
    r = client.patch(
        f"/api/v1/history/outputs/{a}",
        json={
            "rating": 4,
            "favorite": True,
            "tags": ["castle", " Blue  ", "castle"],
            "notes": "keeper",
        },
    )
    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail["rating"] == 4 and detail["favorite"] and detail["notes"] == "keeper"
    assert [t["tag"] for t in detail["tags"]] == ["Blue", "castle"]
    assert detail["parameters"]["model_id"] == tiny_sd and detail["exists"]
    assert ids(favorite=True) == [a] and ids(min_rating=4) == [a] and ids(tag="castle") == [a]
    assert ids(q="keeper") == [a]
    assert ids(sort="rating", run=first["run_uuid"])[0] == a
    assert client.patch(f"/api/v1/history/outputs/{a}", json={"rating": 0}).json()["rating"] is None

    # bulk
    r = client.post(
        "/api/v1/history/outputs/bulk",
        json={"ids": [a, b], "action": "add_tags", "value": ["set1"]},
    )
    assert r.json()["changed"] == 2 and set(ids(tag="set1")) == {a, b}
    client.post(
        "/api/v1/history/outputs/bulk",
        json={"ids": [a], "action": "remove_tags", "value": ["set1"]},
    )
    assert ids(tag="set1") == [b]
    client.post("/api/v1/history/outputs/bulk", json={"ids": [b, c], "action": "rate", "value": 5})
    assert set(ids(min_rating=5)) == {b, c}
    assert (
        client.post(
            "/api/v1/history/outputs/bulk", json={"ids": [a], "action": "rate", "value": 9}
        ).status_code
        == 400
    )

    facets = client.get("/api/v1/history/facets").json()
    assert {"tag": "set1", "count": 1} in facets["tags"]
    assert any(m["model_id"] == tiny_sd for m in facets["models"]) and facets["favorites"] >= 1

    # thumbnails
    t = client.get(f"/api/v1/history/outputs/{c}/thumb", params={"size": 128})
    assert t.status_code == 200 and t.headers["content-type"] == "image/webp"
    assert max(Image.open(io.BytesIO(t.content)).size) <= 128

    # runs
    runs = client.get("/api/v1/history/runs").json()["items"]
    run = next(r for r in runs if r["run_uuid"] == first["run_uuid"])
    assert run["count"] == 2 and run["cover_ids"] == [a, b] and run["status"] == "completed"
    assert run["prompt"] == BASE["prompt"] and run["model_id"] == tiny_sd

    # delete: files go to the trash, the gallery forgets them
    path = client.get(f"/api/v1/history/outputs/{c}").json()["file_path"]
    assert client.delete(f"/api/v1/history/outputs/{c}").json()["removed"] == 1
    assert client.get(f"/api/v1/history/outputs/{c}").status_code == 404
    trash = workspace / "experiments" / ".trash" / second["run_uuid"]
    assert (trash / path.rsplit("/", 1)[-1]).is_file()

    # a run with no images left drops out of the runs list (unless asked for)
    def run_ids(**params):
        items = client.get("/api/v1/history/runs", params=params).json()["items"]
        return {r["run_uuid"] for r in items}

    assert second["run_uuid"] not in run_ids()
    assert second["run_uuid"] in run_ids(include_empty=True)
    assert first["run_uuid"] in run_ids()
    assert client.delete(f"/api/v1/history/runs/{first['run_uuid']}").json()["removed"] == 2
    assert ids(run=first["run_uuid"]) == []
    assert client.delete(f"/api/v1/history/runs/{first['run_uuid']}").status_code == 404


def test_upgrade_of_older_database(tmp_path):
    """A v0.4 database gets the new columns, backfilled fields and migrated comma tags."""
    from sqlalchemy import create_engine

    from gradient.database.session import _add_missing_columns, _backfill_outputs

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE experiment_runs (id INTEGER PRIMARY KEY, run_uuid VARCHAR NOT NULL UNIQUE,
            run_type VARCHAR NOT NULL, start_time DATETIME NOT NULL, end_time DATETIME,
            status VARCHAR NOT NULL, base_parameters JSON, user_notes TEXT);
        CREATE TABLE generated_outputs (id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL,
            output_type VARCHAR NOT NULL, file_path VARCHAR NOT NULL UNIQUE, parameters JSON,
            metadata JSON, creation_time DATETIME NOT NULL, user_rating INTEGER, user_tags VARCHAR);
        INSERT INTO experiment_runs VALUES (1, 'r1', 'text_to_image', '2026-01-01', NULL, 'completed', '{}', NULL);
        """
    )
    con.execute(
        "INSERT INTO generated_outputs VALUES (1, 1, 'image', '/x.png', ?, NULL, '2026-01-01', 3, 'old, tags')",
        (
            json.dumps(
                {"model_id": "sd15", "prompt": "a cat", "seed": 9, "width": 512, "height": 512}
            ),
        ),
    )
    con.commit()
    con.close()

    from gradient.database.models import Base

    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        Base.metadata.create_all(conn)  # adds the new tables only
        added = _add_missing_columns(conn)
        _backfill_outputs(conn)
    assert {
        "generated_outputs.favorite",
        "generated_outputs.model_id",
        "generated_outputs.seed",
    } <= set(added)
    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT model_id, prompt, seed, width, favorite FROM generated_outputs"
    ).fetchone()
    assert row == ("sd15", "a cat", 9, 512, 0)
    tags = {t for (t,) in con.execute("SELECT tag FROM output_tags WHERE source='manual'")}
    assert tags == {"old", "tags"}
    with engine.begin() as conn:  # running again changes nothing
        assert _add_missing_columns(conn) == []


@pytest.fixture(scope="module")
def tiny_loras(workspace, tiny_sd_source):
    from helpers import build_tiny_lora

    root = workspace / "models" / "lora"
    return [
        build_tiny_lora(tiny_sd_source, root / f"tiny-lora-{i}.safetensors", seed=i).name
        for i in (1, 2)
    ]


def _pixels(data):
    return np.asarray(
        Image.open(io.BytesIO(__import__("base64").b64decode(data["images"][0]["image_base64"]))),
        dtype=np.int16,
    )


def test_loras_stay_loaded(client, tiny_sd, tiny_loras, monkeypatch):
    from gradient.ai.diffusion import loading_loras
    from gradient.ai.diffusion.loading import get_loaded_entry

    loads = []
    original = loading_loras.load_lora_adapter

    def counting(pipe, name, adapter):
        loads.append(name)
        return original(pipe, name, adapter)

    monkeypatch.setattr(loading_loras, "load_lora_adapter", counting)
    one, two = tiny_loras

    plain = _pixels(render(client, tiny_sd))
    with_one = _pixels(render(client, tiny_sd, loras=[{"name": one, "weight": 1.0}]))
    half_one = _pixels(render(client, tiny_sd, loras=[{"name": one, "weight": 0.5}]))
    both = _pixels(
        render(client, tiny_sd, loras=[{"name": one, "weight": 1.0}, {"name": two, "weight": 0.7}])
    )
    assert loads == [one, two]  # weight changes and re-use never reload from disk
    assert np.abs(with_one - plain).max() > 0
    assert np.abs(half_one - with_one).max() > 0 and np.abs(both - with_one).max() > 0
    # without LoRAs the adapters are only disabled: the image is the plain one again
    assert np.abs(_pixels(render(client, tiny_sd)) - plain).max() == 0
    assert (
        np.abs(
            _pixels(render(client, tiny_sd, loras=[{"name": one, "weight": 1.0}])) - with_one
        ).max()
        == 0
    )
    assert loads == [one, two]
    entry = get_loaded_entry(tiny_sd)
    assert list(entry.adapters) == [two, one] and len(entry.active_adapters) == 1

    # beyond performance.max_loaded_loras the least recently used LoRA is unloaded
    from gradient.core import config

    monkeypatch.setitem(config.settings.setdefault("performance", {}), "max_loaded_loras", 1)
    render(client, tiny_sd, loras=[{"name": two, "weight": 1.0}])
    assert list(entry.adapters) == [two]
    render(client, tiny_sd, loras=[{"name": one, "weight": 1.0}])
    assert loads == [one, two, one]


def test_lora_listing(client, tiny_loras):
    names = client.get("/api/v1/system/models/loras").json()
    assert set(tiny_loras) <= set(names)
