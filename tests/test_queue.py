# tests/test_queue.py
"""The render queue: submit, progress, cancel (waiting and running), reorder, retry, events."""

import io
import json
import time

import pytest
import torch
from PIL import Image

BASE = {
    "prompt": "a red lighthouse on a rocky coast",
    "negative_prompt": "blurry",
    "width": 64,
    "height": 64,
    "num_inference_steps": 4,
    "guidance_scale": 5.0,
    "seed": 11,
}


def wait_for(client, job_id, statuses=("completed", "failed", "cancelled"), timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/queue/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stayed {job['status']}")


def submit(client, model_id, count=1, **overrides):
    r = client.post(
        "/api/v1/queue",
        json={
            "kind": "text_to_image",
            "payload": {**BASE, "model_id": model_id, **overrides},
            "count": count,
        },
    )
    assert r.status_code == 202, r.text
    return [j["id"] for j in r.json()["jobs"]]


def test_submit_runs_and_saves(client, tiny_sd):
    ids = submit(client, tiny_sd, count=2, num_outputs=2)
    jobs = [wait_for(client, i) for i in ids]
    assert [j["status"] for j in jobs] == ["completed", "completed"]
    # a fixed seed is stepped per job, so a batch doesn't repeat itself
    assert [j["result"]["outputs"][0]["seed"] for j in jobs] == [BASE["seed"], BASE["seed"] + 1]
    job = jobs[0]
    assert job["label"].startswith("a red lighthouse") and job["label"].endswith("×2")
    assert job["model_id"] == tiny_sd and job["run_uuid"]
    assert job["progress"]["step"] == job["progress"]["total"] == 4
    outputs = job["result"]["outputs"]
    assert len(outputs) == 2 and all(o["id"] for o in outputs)
    # the PNG carries the full JSON and the A1111-style summary
    png = client.get(f"/api/v1/history/outputs/{outputs[0]['id']}/image")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    text = Image.open(io.BytesIO(png.content)).text
    assert json.loads(text["grad_parameters"])["model_id"] == tiny_sd
    assert text["parameters"].startswith(
        "a red lighthouse on a rocky coast\nNegative prompt: blurry\n"
    )
    assert "Steps: 4, Sampler: " in text["parameters"] and f"Model: {tiny_sd}" in text["parameters"]


def test_invalid_payload_rejected(client, tiny_sd):
    r = client.post(
        "/api/v1/queue",
        json={"kind": "text_to_image", "payload": {**BASE, "model_id": tiny_sd, "width": -8}},
    )
    assert r.status_code == 422
    r = client.post("/api/v1/queue", json={"kind": "train_lora", "payload": {}})
    assert r.status_code == 422


def test_cancel_reorder_retry(client, tiny_sd):
    long_id, *waiting = submit(client, tiny_sd, count=1, num_inference_steps=400)
    a, b, c = (submit(client, tiny_sd)[0] for _ in range(3))
    running = wait_for(client, long_id, statuses=("running",))
    assert running["status"] == "running"

    # reorder the waiting jobs
    r = client.post(f"/api/v1/queue/{c}/move", json={"to": "top"})
    assert r.status_code == 200 and r.json()["order"][:3] == [c, a, b]
    assert client.get(f"/api/v1/queue/{c}").json()["position"] == 1
    assert client.post(f"/api/v1/queue/{long_id}/move", json={"to": "top"}).status_code == 409

    # cancel a waiting job, then the running one (between denoising steps)
    assert client.post(f"/api/v1/queue/{a}/cancel").json()["status"] == "cancelled"
    deadline = time.time() + 60
    while client.get(f"/api/v1/queue/{long_id}").json()["progress"].get("step", 0) < 2:
        assert time.time() < deadline
        time.sleep(0.02)
    client.post(f"/api/v1/queue/{long_id}/cancel")
    cancelled = wait_for(client, long_id)
    assert cancelled["status"] == "cancelled" and cancelled["progress"]["step"] < 400
    # the stopped run has no images: hidden from the runs list unless asked for
    params = {"limit": 50, "include_empty": True}
    runs = client.get("/api/v1/history/runs", params=params).json()["items"]
    assert any(r["status"] == "cancelled" for r in runs)
    shown = client.get("/api/v1/history/runs", params={"limit": 50}).json()["items"]
    assert not any(r["status"] == "cancelled" and r["count"] == 0 for r in shown)

    # the queue keeps going, in the new order
    done_c, done_b = wait_for(client, c), wait_for(client, b)
    assert done_c["status"] == done_b["status"] == "completed"
    assert done_c["started"] < done_b["started"]

    # retry the cancelled job
    r = client.post(f"/api/v1/queue/{a}/retry")
    assert r.status_code == 202
    assert wait_for(client, r.json()["id"])["status"] == "completed"
    assert client.post(f"/api/v1/queue/{c}/retry").status_code == 409  # completed: no retry


def test_api_calls_go_through_the_queue(client, tiny_sd):
    r = client.post("/api/v1/generate/text-to-image", json={**BASE, "model_id": tiny_sd})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["run_uuid"] and data["images"][0]["output_id"]
    jobs = client.get("/api/v1/queue").json()["jobs"]
    assert any(j["run_uuid"] == data["run_uuid"] and j["status"] == "completed" for j in jobs)
    assert client.delete("/api/v1/queue/finished").json()["removed"] >= 1
    assert all(
        j["status"] not in ("completed", "failed", "cancelled")
        for j in client.get("/api/v1/queue").json()["jobs"]
    )


def test_failed_job_reports_error(client):
    (job_id,) = submit(client, "no-such-model")
    job = wait_for(client, job_id)
    assert job["status"] == "failed" and "no-such-model" in job["error"]
    assert job["retryable"]


def test_event_stream(client, tiny_sd):
    with client.stream("GET", "/api/v1/queue/events") as stream:
        lines = stream.iter_lines()
        first = next(line for line in lines if line.startswith("data: "))
        assert json.loads(first[6:])["type"] == "snapshot"
        (job_id,) = submit(client, tiny_sd)
        statuses = []
        for line in lines:
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "job" and event["job"]["id"] == job_id:
                    statuses.append(event["job"]["status"])
                    if event["job"]["status"] == "completed":
                        break
        assert statuses[0] == "queued" and statuses[-1] == "completed" and "running" in statuses


def test_lab_results_are_saved_as_runs(client, tiny_sd, workspace):
    r = client.post(
        "/api/v1/lab/attention-maps",
        json={"base_request": {**BASE, "model_id": tiny_sd}, "words": ["lighthouse", "coast"]},
    )
    assert r.status_code == 200, r.text
    runs = client.get("/api/v1/history/runs", params={"run_type": "lab_*"}).json()["items"]
    run = runs[0]
    assert run["run_type"] == "lab_attention" and run["count"] == 1 and run["model_id"] == tiny_sd
    images = client.get("/api/v1/history/outputs", params={"run": run["run_uuid"]}).json()["items"]
    assert images[0]["seed"] == BASE["seed"] and images[0]["prompt"] == BASE["prompt"]
    folder = workspace / "experiments" / run["run_uuid"]
    report = json.loads((folder / "report.json").read_text())
    assert report["tool"] == "attention" and len(report["result"]["maps"]) == 2
    assert (folder / report["result"]["maps"][0]["heatmap_file"]).is_file()
    # the job shows in the queue with its run
    jobs = client.get("/api/v1/queue").json()["jobs"]
    assert any(j["kind"] == "lab_attention" and j["run_uuid"] == run["run_uuid"] for j in jobs)


@pytest.mark.parametrize("bad", [{"to": "sideways"}])
def test_move_validation(client, bad):
    assert client.post("/api/v1/queue/nope/move", json=bad).status_code == 422
    assert client.post("/api/v1/queue/nope/cancel").status_code == 404


def test_live_preview(client, tiny_sd):
    from gradient.ai.diffusion.preview import estimate_x0

    (job_id,) = submit(client, tiny_sd)
    job = wait_for(client, job_id)
    assert job["status"] == "completed"
    # the last step is always published
    assert job["progress"]["preview"] >= 1 and job["progress"]["preview_step"] == 4
    r = client.get(f"/api/v1/queue/{job_id}/preview")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size[0] > 0
    assert client.get("/api/v1/queue/nope/preview").status_code == 404

    # Euler step: x = x0 + sigma * eps; two consecutive latents give x0 back exactly
    x0, eps = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    x_prev, x_now = x0 + 5.0 * eps, x0 + 2.0 * eps
    assert torch.allclose(estimate_x0(x_now, x_prev, 2.0, 5.0), x0, atol=1e-5)
    assert torch.equal(estimate_x0(x_now, x_prev, 2.0, 2.0), x_now)


def test_preview_projection_matches_vae(tiny_sd):
    """The fitted latent→RGB map reproduces what the VAE decodes (roughly, it's linear)."""
    import numpy as np

    from gradient.ai.diffusion.loading import get_pipeline
    from gradient.ai.diffusion.preview import _chart, _normalize, fit_projection, to_rgb

    pipe, _ = get_pipeline(tiny_sd)
    proj = fit_projection(pipe)
    assert proj is not None and proj[0].shape == (pipe.vae.config.latent_channels, 3)
    chart = _chart(256)
    x = torch.from_numpy(chart).permute(2, 0, 1)[None] * 2 - 1
    with torch.no_grad():
        z = _normalize(pipe.vae.encode(x).latent_dist.mean, pipe.vae)[0]
    rgb = to_rgb(z, proj).astype(np.float32) / 255
    h, w = rgb.shape[:2]
    target = (
        torch.nn.functional.adaptive_avg_pool2d(
            torch.from_numpy(chart).permute(2, 0, 1)[None], (h, w)
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )
    # a random tiny VAE is far from linear, so only require clearly better than a flat grey
    assert np.abs(rgb - target).mean() < np.abs(0.5 - target).mean()
