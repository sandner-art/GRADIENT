# tests/test_quiet_startup.py
"""Version info and a quiet console: no warnings that say nothing useful."""

import json
import logging
import time
import tomllib
import warnings

import pytest

from gradient.core.config import PROJECT_ROOT


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_version_from_checkout(client):
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    info = client.get("/api/v1/system/info").json()
    assert info["version"] == expected
    assert "installed_version" in info and isinstance(info["install_is_stale"], bool)
    assert client.get("/api/v1/status").json()["version"] == expected


def test_ready_banner():
    from gradient.cli import _ready_banner

    class Opener:
        def __init__(self, info):
            self.info = info

        def open(self, url, timeout=0):
            import io
            import json

            return io.BytesIO(json.dumps(self.info).encode())

    gpu = {
        "version": "0.9.0",
        "install_is_stale": True,
        "installed_version": "0.7.0",
        "device": {"device": "cuda:0", "name": "RTX 4070", "total_vram_gb": 11.99, "cuda": "12.8"},
        "versions": {"torch": "2.15.0", "diffusers": "0.42.1"},
    }
    text = _ready_banner("http://127.0.0.1:8000", Opener(gpu))
    assert "GRADIENT 0.9.0 is ready: http://127.0.0.1:8000" in text
    assert "RTX 4070 · 11.99 GB VRAM · CUDA 12.8 · torch 2.15.0 · diffusers 0.42.1" in text
    assert "from 0.7.0" in text
    cpu = {"version": "0.9.0", "device": {"device": "cpu"}, "versions": {"torch": "2.15.0"}}
    assert "CPU only" in _ready_banner("http://x", Opener(cpu))


def test_no_safety_checker_warning(client, tiny_sd):
    """A model saved with requires_safety_checker=true (like many SD 1.5 fine-tunes without
    a checker) loads without the 'You have disabled the safety checker' warnings."""
    import json
    import shutil

    from diffusers import FluxPipeline, StableDiffusionPipeline

    from gradient.ai.diffusion import loading
    from gradient.ai.diffusion.loading import _without_safety_checker
    from gradient.model_manager.inventory import find_model_path

    assert _without_safety_checker(StableDiffusionPipeline) == {
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    assert _without_safety_checker(FluxPipeline) == {}

    src = find_model_path(tiny_sd, "diffusers")
    copy = src.parent / "tiny-sd-checker"
    if not copy.exists():
        shutil.copytree(src, copy)
        index = json.loads((copy / "model_index.json").read_text())
        index["requires_safety_checker"] = True
        (copy / "model_index.json").write_text(json.dumps(index))
    collect = _Collect()
    diffusers_logger = logging.getLogger("diffusers")
    diffusers_logger.addHandler(collect)
    try:
        entry = loading.get_loaded_entry("tiny-sd-checker", None, False, None)
        loading.get_pipeline("tiny-sd-checker", task="image2image")
    finally:
        diffusers_logger.removeHandler(collect)
        loading.unload_all_pipelines()
    assert entry.pipeline.config.requires_safety_checker is False
    assert not [m for m in collect.messages if "safety checker" in m], collect.messages


def test_loading_and_branches_are_quiet(client, tiny_sd):
    from gradient.ai.diffusion import loading
    from gradient.core.quiet import quiet_libraries

    quiet_libraries()
    loading.unload_all_pipelines()
    collect = _Collect()
    diffusers_logger = logging.getLogger("diffusers")
    diffusers_logger.addHandler(collect)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = client.post(
                "/api/v1/lab/explore/branches",
                json={
                    "base_request": {
                        "model_id": tiny_sd,
                        "prompt": "a lighthouse",
                        "width": 64,
                        "height": 64,
                        "num_inference_steps": 4,
                        "seed": 3,
                    },
                    "step": 1,
                    "branches": [{}, {}, {}],
                },
            )
        assert r.status_code == 200, r.text
        # the library's float32 notice: dropped when its list is empty, kept otherwise
        modeling = logging.getLogger("diffusers.models.modeling_utils")
        modeling.warning("There are modules in X that should be kept in float32: []. Casting...")
        modeling.warning("There are modules in X that should be kept in float32: ['norm']. ...")
    finally:
        diffusers_logger.removeHandler(collect)
    assert not [m for m in collect.messages if "safety checker" in m], collect.messages
    assert [m for m in collect.messages if "float32" in m] == [
        "There are modules in X that should be kept in float32: ['norm']. ..."
    ]
    future = [
        w for w in caught if issubclass(w.category, FutureWarning) and "image" in str(w.message)
    ]
    assert not future, [str(w.message) for w in future]
    assert loading.loaded_pipelines  # it did load (the check above is meaningful)


# ---- Ctrl+C
@pytest.fixture
def stopping():
    from gradient.core import shutdown

    yield shutdown
    shutdown.reset()


def test_ctrl_c_ends_streams_and_stops_the_running_job(client, tiny_sd, stopping):
    from tests.test_queue import submit, wait_for

    (long_id,) = submit(client, tiny_sd, num_inference_steps=400)
    (waiting_id,) = submit(client, tiny_sd)
    wait_for(client, long_id, statuses=("running",))
    with client.stream("GET", "/api/v1/queue/events") as stream:
        lines = stream.iter_lines()
        first = next(line for line in lines if line.startswith("data: "))
        assert json.loads(first[6:])["type"] == "snapshot"
        stopping.request()  # what the first Ctrl+C does
        rest = [line for line in lines if line.startswith("data: ")]  # ends by itself
    assert json.loads(rest[-1][6:]) == {"type": "shutdown"}
    job = wait_for(client, long_id)
    assert job["status"] == "failed" and "stopped" in job["error"]
    assert job["progress"]["step"] < 400  # stopped at a step, not run to the end
    time.sleep(0.3)
    assert client.get(f"/api/v1/queue/{waiting_id}").json()["status"] == "queued"  # not started

    stopping.reset()  # as after a restart: the queue runs again
    (next_id,) = submit(client, tiny_sd)
    assert wait_for(client, waiting_id)["status"] == "completed"
    assert wait_for(client, next_id)["status"] == "completed"


def test_second_ctrl_c_stops_at_once(monkeypatch, stopping):
    import signal

    import uvicorn

    from gradient.api import server as srv

    exits = []
    monkeypatch.setattr(srv.os, "_exit", exits.append)
    monkeypatch.setattr(srv, "_interrupts", 0)
    server = srv.GracefulServer(uvicorn.Config("gradient.api.main:app"))
    server.handle_exit(signal.SIGTERM, None)  # the reloader's SIGTERM: graceful, not counted
    assert stopping.requested() and server.should_exit and exits == []
    assert server._captured_signals == []  # nothing is raised again after the stop
    server.handle_exit(signal.SIGINT, None)  # the first Ctrl+C
    assert exits == []
    server.handle_exit(signal.SIGINT, None)  # the second: at once
    assert exits == [0]


class _FakeServer:
    should_exit = False


def test_stop_endpoint(client, monkeypatch, stopping):
    from gradient.api.endpoints import system

    # only from this computer (the test client isn't a loopback address)
    r = client.post("/api/v1/system/shutdown")
    assert r.status_code == 403 and "this computer" in r.text
    monkeypatch.setattr(system, "_is_local", lambda request: True)
    # a server not started by `gradient serve` can't be stopped from here
    r = client.post("/api/v1/system/shutdown")
    assert r.status_code == 409 and "gradient serve" in r.text
    fake = _FakeServer()
    stopping.attach(fake)
    try:
        r = client.post("/api/v1/system/shutdown")
        assert r.status_code == 200 and r.json() == {"stopping": True}
        deadline = time.time() + 5
        while not fake.should_exit and time.time() < deadline:
            time.sleep(0.05)
        assert fake.should_exit and stopping.requested()
    finally:
        stopping.detach()


def test_windows_console_events(monkeypatch, stopping):
    from gradient.api import server as cli

    exits = []
    monkeypatch.setattr(cli.os, "_exit", exits.append)
    monkeypatch.setattr(cli, "_interrupts", 0)
    fake = _FakeServer()
    stopping.attach(fake)
    try:
        assert cli.on_console_event(5) is False  # log-off: Windows' default handling
        assert cli.on_console_event(0) is True  # Ctrl+C: graceful
        assert fake.should_exit and stopping.requested() and exits == []
        assert cli.on_console_event(0) is True and exits == [0]  # again: at once
        stopping.detach()  # the server has finished: a console close returns right away
        started = time.time()
        assert cli.on_console_event(2) is True and time.time() - started < 1
    finally:
        stopping.detach()
    assert cli.install_windows_console_handler() is None  # not Windows here: nothing to do


def test_gradient_stop_without_a_server(capsys):
    import argparse

    from gradient import cli

    assert cli.cmd_stop(argparse.Namespace(port=1)) == 1
    assert "isn't running" in capsys.readouterr().out
