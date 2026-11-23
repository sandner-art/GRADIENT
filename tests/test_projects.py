# tests/test_projects.py
"""Project folders: export to a folder, let "other tools" change it, sync back."""

import json
import re
from pathlib import Path

from gradient.gallery.projects import build_sidecar, read_sidecar

BASE = {
    "prompt": "a red lighthouse on a rocky coast",
    "negative_prompt": "blurry",
    "width": 64,
    "height": 64,
    "num_inference_steps": 3,
    "guidance_scale": 5.0,
    "seed": 11,
}


def render(client, model_id, **overrides) -> int:
    r = client.post(
        "/api/v1/generate/text-to-image", json={**BASE, "model_id": model_id, **overrides}
    )
    assert r.status_code == 200, r.text
    return r.json()["images"][0]["output_id"]


def test_sidecar_round_trip():
    text = build_sidecar(
        ["castle", "Creatures/Dragons", "Projects/Set A", "GRADIENT/Model/sd15"],
        rating=4,
        description='a "castle" & <moat>',
        created="2026-11-23T10:00:00+00:00",
        attrs={"OutputId": 7, "Seed": 5, "Model": None},
    )
    assert "<rdf:li>Creatures|Dragons</rdf:li>" in text  # Lightroom / Kurator form
    assert "<rdf:li>Creatures/Dragons</rdf:li>" in text  # digiKam form
    assert 'gradient:OutputId="7"' in text and "gradient:Model" not in text
    data = read_sidecar(text)
    assert set(data["keywords"]) == {
        "castle",
        "Creatures/Dragons",
        "Projects/Set A",
        "GRADIENT/Model/sd15",
    }
    assert data["rating"] == 4 and data["description"] == 'a "castle" & <moat>'
    assert read_sidecar("<not xml") is None


def _add_keyword(sidecar: Path, keyword: str) -> None:
    """What Kurator does with 'Also update sidecars of other programs': add to the keyword lists."""
    text = sidecar.read_text(encoding="utf-8")
    for prop, value in (
        ("lr:hierarchicalSubject", keyword.replace("/", "|")),
        ("digiKam:TagsList", keyword),
        ("dc:subject", keyword.split("/")[-1]),
    ):
        text = re.sub(
            rf"(<{prop}>\s*<rdf:(?:Bag|Seq)>)",
            rf"\1\n     <rdf:li>{value}</rdf:li>",
            text,
            count=1,
        )
    text = text.replace("</rdf:Description>", "<dc:source>Kurator</dc:source>\n</rdf:Description>")
    sidecar.write_text(text, encoding="utf-8")


def _drop_keyword(sidecar: Path, keyword: str) -> None:
    text = sidecar.read_text(encoding="utf-8")
    for value in (keyword, keyword.replace("/", "|"), keyword.split("/")[-1]):
        text = text.replace(f"<rdf:li>{value}</rdf:li>", "")
    sidecar.write_text(text, encoding="utf-8")


def _set_rating(sidecar: Path, rating: int) -> None:
    text = sidecar.read_text(encoding="utf-8")
    text = re.sub(r'\s+xmp:Rating="\d+"', "", text)
    text = text.replace("<rdf:Description ", f'<rdf:Description xmp:Rating="{rating}" ', 1)
    sidecar.write_text(text, encoding="utf-8")


def test_export_and_sync_round_trip(client, tiny_sd, workspace):
    a = render(client, tiny_sd, prompt="a dragon over the sea")
    b = render(client, tiny_sd, seed=12)
    c = render(client, tiny_sd, seed=13)
    client.patch(f"/api/v1/history/outputs/{a}", json={"tags": ["keeper"], "rating": 3})

    # ---- export a and b into a new project (default folder: <data>/projects/<name>)
    r = client.post("/api/v1/projects/export", json={"ids": [a, b], "name": "Test set"})
    assert r.status_code == 200, r.text
    res = r.json()
    project = res["project"]
    folder = Path(project["path"])
    assert folder == workspace / "data" / "projects" / "Test-set"
    assert res["added"] == 2 and project["count"] == 2 and project["exists"]
    manifest = json.loads((folder / "gradient-project.json").read_text(encoding="utf-8"))
    files = {e["output_id"]: e["file"] for e in manifest["items"]}
    src = Path(client.get(f"/api/v1/history/outputs/{a}").json()["file_path"])
    assert (folder / files[a]).read_bytes() == src.read_bytes()
    side_a = folder / (files[a] + ".xmp")
    text = side_a.read_text(encoding="utf-8")
    for expected in (
        "<rdf:li>Projects|Test set</rdf:li>",
        f"<rdf:li>GRADIENT|Model|{tiny_sd}</rdf:li>",
        "<rdf:li>GRADIENT|Type|Text-to-image</rdf:li>",
        "<rdf:li>keeper</rdf:li>",
        'xmp:Rating="3"',
        "a dragon over the sea",
    ):
        assert expected in text, expected
    assert not list(folder.glob("*.txt"))  # no captions unless asked

    # the gallery can show the project
    ids = [
        i["id"]
        for i in client.get(f"/api/v1/history/outputs?project={project['uuid']}").json()["items"]
    ]
    assert set(ids) == {a, b}
    facets = client.get("/api/v1/history/facets").json()
    assert {"uuid": project["uuid"], "name": "Test set", "count": 2} in facets["projects"]

    # ---- other tools work on the folder
    _add_keyword(side_a, "Creatures/Dragons")  # Kurator
    (folder / files[a]).with_suffix(".txt").write_text(
        "a dragon flying over the sea, dramatic light", encoding="utf-8"
    )  # ArtAgents
    side_b = folder / (files[b] + ".xmp")
    _set_rating(side_b, 5)  # digiKam
    (folder / files[b]).rename(folder / "renamed.png")  # renamed, sidecar travels along
    side_b.rename(folder / "renamed.png.xmp")
    (folder / "stray.png").write_bytes(b"not really a png")

    r = client.post(
        f"/api/v1/projects/{project['uuid']}/sync", json={"caption_source": "artagents:test"}
    )
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["tags_added"] == 1 and rep["ratings_changed"] == 1 and rep["captions"] == 1
    assert rep["renamed"] == [{"from": files[b], "to": "renamed.png"}]
    assert rep["unknown_files"] == ["stray.png"] and not rep["missing"]

    da = client.get(f"/api/v1/history/outputs/{a}").json()
    assert {"tag": "Creatures/Dragons", "source": "xmp", "confidence": None} in da["tags"]
    assert da["captions"][0]["source"] == "artagents:test"
    assert da["captions"][0]["text"].startswith("a dragon flying")
    assert da["projects"][0]["name"] == "Test set"
    db = client.get(f"/api/v1/history/outputs/{b}").json()
    assert db["rating"] == 5 and db["projects"][0]["file_name"] == "renamed.png"

    # a second sync changes nothing
    rep = client.post(f"/api/v1/projects/{project['uuid']}/sync", json={}).json()
    assert rep["tags_added"] == rep["ratings_changed"] == rep["captions"] == 0

    # ---- re-export into the same project: c is added, a's sidecar keeps Kurator's additions
    r = client.post(
        "/api/v1/projects/export",
        json={"ids": [a, c], "project": project["uuid"], "captions": "prompt"},
    )
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["added"] == 1 and res["updated"] == 1 and res["kept_captions"] == 1
    text = side_a.read_text(encoding="utf-8")
    assert "<rdf:li>Creatures|Dragons</rdf:li>" in text and "<dc:source>Kurator</dc:source>" in text
    manifest = json.loads((folder / "gradient-project.json").read_text(encoding="utf-8"))
    files = {e["output_id"]: e["file"] for e in manifest["items"]}
    caption_c = (folder / files[c]).with_suffix(".txt")
    assert caption_c.read_text(encoding="utf-8") == BASE["prompt"]  # the prompt, dataset style
    rep = client.post(f"/api/v1/projects/{project['uuid']}/sync", json={}).json()
    assert rep["captions"] == 0  # GRADIENT's own prompt caption isn't read back as a caption

    # ---- a keyword removed in the other program removes the imported tag (not your own)
    _drop_keyword(side_a, "Creatures/Dragons")
    _drop_keyword(side_a, "keeper")
    rep = client.post(f"/api/v1/projects/{project['uuid']}/sync", json={}).json()
    assert rep["tags_removed"] == 1
    tags = {
        (t["tag"], t["source"]) for t in client.get(f"/api/v1/history/outputs/{a}").json()["tags"]
    }
    assert tags == {("keeper", "manual")}

    # ---- safety: a non-empty folder that isn't a project is refused
    other = workspace / "not-a-project"
    other.mkdir()
    (other / "notes.txt").write_text("mine", encoding="utf-8")
    r = client.post("/api/v1/projects/export", json={"ids": [a], "name": "X", "path": str(other)})
    assert r.status_code == 400 and "isn't a GRADIENT project" in r.json()["detail"]

    # ---- forget, then add the folder again (e.g. after moving it)
    assert client.delete(f"/api/v1/projects/{project['uuid']}").status_code == 200
    assert client.get(f"/api/v1/projects/{project['uuid']}").status_code == 404
    assert folder.is_dir()
    moved = workspace / "moved-project"
    folder.rename(moved)
    r = client.post("/api/v1/projects/open", json={"path": str(moved)})
    assert r.status_code == 200, r.text
    again = r.json()["project"]
    assert again["uuid"] == project["uuid"] and again["path"] == str(moved)
    assert again["count"] == 3

    # ---- deleting an image drops it from the project and its captions
    client.delete(f"/api/v1/history/outputs/{a}")
    assert client.get(f"/api/v1/projects/{project['uuid']}").json()["count"] == 2


def test_export_by_filter_and_errors(client, tiny_sd, workspace):
    render(client, tiny_sd, prompt="a purple octopus", seed=21)
    render(client, tiny_sd, prompt="a purple octopus", seed=22)
    target = workspace / "octopus-project"
    r = client.post(
        "/api/v1/projects/export",
        json={
            "filter": {"q": "octopus"},
            "name": "Octopus",
            "path": str(target),
            "sidecars": False,
            "model_keywords": False,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["added"] == 2
    assert not list(target.glob("*.xmp")) and len(list(target.glob("*.png"))) == 2

    bad = client.post("/api/v1/projects/export", json={"ids": [1], "filter": {}, "name": "Y"})
    assert bad.status_code == 422
    bad = client.post("/api/v1/projects/export", json={"filter": {"color": "red"}, "name": "Y"})
    assert bad.status_code == 422
    none = client.post(
        "/api/v1/projects/export", json={"filter": {"q": "no-such-words-zzz"}, "name": "Empty"}
    )
    assert none.status_code == 400 and "No images" in none.json()["detail"]
    names = [p["name"] for p in client.get("/api/v1/projects").json()["items"]]
    assert "Empty" not in names and "Octopus" in names
    bad = client.post(
        f"/api/v1/projects/{r.json()['project']['uuid']}/sync", json={"caption_source": "xmp"}
    )
    assert bad.status_code == 400
