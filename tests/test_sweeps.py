# tests/test_sweeps.py
"""Test sweep files from ArtAgents: preview, LoRAs by sha256, queue, results folder."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest
from PIL import Image


def sweep_text(model_id, loras=(), jobs=None, **model):
    """A sweep file as ArtAgents writes it (core/lab/sweep_file.py)."""
    jobs = jobs or [
        {
            "index": 0,
            "prompt_index": 0,
            "bucket": "held-out",
            "prompt": "txcl lighthouse",
            "negative": "",
            "seed": 5,
            "width": 63,
            "height": 64,
            "steps": 3,
            "cfg": 4.0,
            "lora": None,
            "strength": 0.0,
            "condition": "baseline (no LoRA)",
        }
    ]
    data = {
        "format": "artagents.lora-sweep",
        "version": 1,
        "run": {"name": "unit sweep", "folder": "20261124-120000-unit", "trigger": "txcl"},
        "model": {"model_id": model_id, **model},
        "loras": [{"key": k, "name": n, "sha256": s} for k, n, s in loras],
        "jobs": jobs,
    }
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def lora_jobs(key, seeds=(5, 6)):
    jobs, i = [], 0
    for seed in seeds:
        for lora, strength, cond in ((None, 0.0, "baseline (no LoRA)"), (key, 1.0, f"{key} @ 1")):
            jobs.append(
                {
                    "index": i,
                    "prompt_index": 0,
                    "bucket": "trigger-less",
                    "prompt": "a lighthouse",
                    "negative": "blurry",
                    "seed": seed,
                    "width": 64,
                    "height": 64,
                    "steps": 3,
                    "cfg": None,
                    "lora": lora,
                    "strength": strength,
                    "condition": cond,
                }
            )
            i += 1
    return jobs


@pytest.fixture(scope="module")
def sweep_loras(tiny_sd_source, workspace):
    from helpers import build_tiny_lora

    root = workspace / "models" / "lora"
    root.mkdir(parents=True, exist_ok=True)
    a = build_tiny_lora(tiny_sd_source, root / "sweep-a.safetensors", seed=11)
    b = build_tiny_lora(tiny_sd_source, workspace / "elsewhere" / "orig_b.safetensors", seed=12)
    renamed = root / "renamed-b.safetensors"
    shutil.copyfile(b, renamed)
    other = build_tiny_lora(tiny_sd_source, workspace / "elsewhere" / "other.safetensors", seed=13)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    return {"a": (a, sha(a)), "b": (b, sha(b)), "renamed": renamed, "other": (other, sha(other))}


def wait_written(client, sid, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/api/v1/sweeps/{sid}").json()
        if s.get("written"):
            return s
        time.sleep(0.1)
    raise AssertionError(f"sweep {sid} results not written: {s}")


def test_preview_finds_loras_by_name_and_hash(client, tiny_sd, sweep_loras):
    (a, sha_a), (_, sha_b) = sweep_loras["a"], sweep_loras["b"]
    text = sweep_text(
        tiny_sd,
        loras=[
            (
                "a",
                "C:/train/out/sweep-a.safetensors",
                sha_a,
            ),  # another machine's path, same name here
            ("b", "C:/train/out/orig_b.safetensors", sha_b),  # renamed here: found by its sha256
            ("plain", "sweep-a.safetensors", None),  # a name without hash: unverified
        ],
    )
    r = client.post("/api/v1/sweeps/preview", json={"sweep_text": text})
    assert r.status_code == 200, r.text
    info = r.json()
    states = {lo["key"]: lo for lo in info["loras"]}
    assert states["a"]["status"] == "ok" and states["a"]["resolved"] == str(a.resolve())
    assert states["b"]["status"] == "found_by_hash"
    assert Path(states["b"]["resolved"]).name == "renamed-b.safetensors"
    assert states["plain"]["status"] == "unverified"
    assert info["ready"] and info["model_found"] and info["images"] == 1
    assert info["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert info["id"] == info["sha256"][:16]
    assert info["results_dir"].endswith(f"20261124-120000-unit-{info['sha256'][:8]}")


def test_different_file_is_refused(client, tiny_sd, sweep_loras):
    (_, sha_a), (other, _) = sweep_loras["a"], sweep_loras["other"]
    text = sweep_text(tiny_sd, loras=[("a", "sweep-a.safetensors", sha_a)], jobs=lora_jobs("a"))
    info = client.post(
        "/api/v1/sweeps/preview", json={"sweep_text": text, "lora_map": {"a": str(other)}}
    ).json()
    assert info["loras"][0]["status"] == "hash_mismatch" and not info["ready"]
    assert any("different file" in p for p in info["problems"])
    r = client.post("/api/v1/sweeps", json={"sweep_text": text, "lora_map": {"a": str(other)}})
    assert r.status_code == 400 and "different file" in r.json()["detail"]


def test_missing_model_and_lora(client, tiny_sd):
    text = sweep_text(
        "no-such-model", loras=[("x", "gone.safetensors", "0" * 64)], jobs=lora_jobs("x")
    )
    info = client.post("/api/v1/sweeps/preview", json={"sweep_text": text}).json()
    assert not info["ready"] and not info["model_found"]
    assert info["loras"][0]["status"] == "missing"
    assert len(info["problems"]) == 2


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(format="something.else"), "Not an ArtAgents test sweep"),
        (lambda d: d.update(version=2), "version 2"),
        (lambda d: d["jobs"][0].update(lora="nope"), "LoRA keys the file doesn't list"),
        (lambda d: d["jobs"].append(dict(d["jobs"][0])), "indexes repeat"),
        (lambda d: d["jobs"][0].update(width=-5), "not valid"),
        (lambda d: d.update(jobs=[]), "not valid"),
    ],
)
def test_bad_files(client, tiny_sd, change, message):
    data = json.loads(sweep_text(tiny_sd))
    change(data)
    r = client.post("/api/v1/sweeps/preview", json={"sweep_text": json.dumps(data)})
    assert r.status_code == 400 and message in r.json()["detail"]
    assert client.post("/api/v1/sweeps/preview", json={"sweep_text": "{"}).status_code == 400


def test_queue_and_results_folder(client, tiny_sd, sweep_loras):
    (a, sha_a) = sweep_loras["a"]
    text = sweep_text(
        tiny_sd,
        loras=[("a", "C:/train/out/sweep-a.safetensors", sha_a)],
        jobs=lora_jobs("a"),
        guidance="6",
    )
    r = client.post("/api/v1/sweeps", json={"sweep_text": text})
    assert r.status_code == 202, r.text
    queued = r.json()
    assert (
        queued["queued"] == 4 and queued["group"] == "ArtAgents · LoRA test · 20261124-120000-unit"
    )
    sid = queued["id"]
    status = wait_written(client, sid)
    assert status["done"] and status["counts"] == {"completed": 4}

    folder = Path(status["results_dir"])
    results = json.loads((folder / "artagents-results.json").read_text(encoding="utf-8"))
    assert results["format"] == "artagents.lora-sweep-results" and results["version"] == 1
    assert results["sweep_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert results["renderer"]["app"] == "GRADIENT" and results["renderer"]["model_id"] == tiny_sd
    assert (folder / "sweep.lora-sweep.json").read_text(encoding="utf-8") == text
    by_index = {e["index"]: e for e in results["results"]}
    assert sorted(by_index) == [0, 1, 2, 3]
    for entry in by_index.values():
        assert entry["error"] is None and entry["output_id"] and entry["loras_skipped"] == []
        assert Image.open(folder / entry["file"]).size == (64, 64)
    assert by_index[1]["file"] == "0001_p00_s5_a_@_1.png" and by_index[0]["seed"] == 5

    # the jobs: baseline without LoRAs, LoRA by its local path, CFG override, group and origin
    jobs = [j for j in client.get("/api/v1/queue").json()["jobs"] if j["group"] == queued["group"]]
    assert len(jobs) == 4 and {j["run_uuid"] for j in jobs} == {queued["run_uuid"]}
    base = next(j for j in jobs if j["origin"]["job_index"] == 0)
    with_lora = next(j for j in jobs if j["origin"]["job_index"] == 1)
    assert base["payload"]["loras"] == [] and base["payload"]["guidance_scale"] == 6.0
    assert with_lora["payload"]["loras"] == [{"name": str(a.resolve()), "weight": 1.0}]
    assert with_lora["origin"]["sweep"] == sid and with_lora["tags"] == [
        "ArtAgents/LoRA test/unit sweep"
    ]

    listed = client.get("/api/v1/sweeps").json()["items"]
    assert any(s["id"] == sid and s["written"] for s in listed)


def test_cancelled_sweep_writes_errors(client, tiny_sd):
    jobs = lora_jobs("a")[:1]
    jobs[0].update(steps=400, prompt="cancel me", lora=None)
    text = sweep_text(tiny_sd, jobs=jobs + [dict(jobs[0], index=1, seed=9)])
    queued = client.post("/api/v1/sweeps", json={"sweep_text": text}).json()
    sid = queued["id"]
    again = client.post("/api/v1/sweeps/preview", json={"sweep_text": text}).json()
    assert not again["ready"] and any("already in the queue" in p for p in again["problems"])
    for j in client.get("/api/v1/queue").json()["jobs"]:
        if (j["origin"] or {}).get("sweep") == sid and j["status"] in ("queued", "running"):
            client.post(f"/api/v1/queue/{j['id']}/cancel")
    status = wait_written(client, sid)
    assert status["done"] and set(status["counts"]) == {"cancelled"}
    results = json.loads(
        (Path(status["results_dir"]) / "artagents-results.json").read_text(encoding="utf-8")
    )
    assert all(e["file"] is None and e["error"] for e in results["results"])
    # writing again on request works too
    r = client.post(f"/api/v1/sweeps/{sid}/results")
    assert r.status_code == 200 and r.json()["errors"] == 2


def test_unknown_sweep(client):
    assert client.get("/api/v1/sweeps/0123456789abcdef").status_code == 404
    assert client.get("/api/v1/sweeps/../../etc").status_code == 404
    assert client.post("/api/v1/sweeps/zzzz/results").status_code == 404
