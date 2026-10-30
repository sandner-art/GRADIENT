# tests/test_model_manager.py
"""Catalog, inventory and downloads (against a local fake Hugging Face Hub)."""

import asyncio
import os

import pytest

from gradient.model_manager import inventory, manifest
from gradient.model_manager.catalog import load_catalog
from gradient.model_manager.downloader import (
    DownloadManager,
    infer_type_from_repo_files,
    parse_hf_reference,
    pick_gguf,
    sanitize_name,
)
from gradient.model_manager.schemas import DownloadRequest


# --------------------------------------------------------------------------- pure helpers
def test_catalog_is_valid_and_unique():
    entries = load_catalog()
    assert len(entries) >= 20
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))
    types = {e.type for e in entries}
    assert {"diffusers", "lora", "vae", "controlnet", "llm"} <= types
    for e in entries:
        if e.type in ("lora", "llm"):
            assert e.files, f"{e.id} should name its file(s)"
        assert e.repo_id and "/" in e.repo_id


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("stabilityai/sdxl-turbo", ("stabilityai/sdxl-turbo", None, None)),
        ("https://huggingface.co/stabilityai/sdxl-turbo", ("stabilityai/sdxl-turbo", None, None)),
        (
            "https://huggingface.co/ByteDance/SDXL-Lightning/blob/main/sdxl_lightning_4step_lora.safetensors",
            ("ByteDance/SDXL-Lightning", "main", "sdxl_lightning_4step_lora.safetensors"),
        ),
        ("https://hf.co/a/b/resolve/v1/sub/dir/x.gguf", ("a/b", "v1", "sub/dir/x.gguf")),
        ("https://huggingface.co/a/b/tree/dev", ("a/b", "dev", None)),
    ],
)
def test_parse_hf_reference(ref, expected):
    assert parse_hf_reference(ref) == expected


@pytest.mark.parametrize(
    "bad",
    ["not a repo", "https://example.com/a/b", "https://huggingface.co/datasets/a/b", "../etc"],
)
def test_parse_hf_reference_rejects(bad):
    with pytest.raises(ValueError):
        parse_hf_reference(bad)


def test_pick_gguf_prefers_q4_k_m():
    files = ["m-Q8_0.gguf", "m-Q4_K_M.gguf", "m-Q2_K.gguf", "mmproj-m-Q4_K_M.gguf", "README.md"]
    assert pick_gguf(files) == "m-Q4_K_M.gguf"
    assert pick_gguf(["README.md"]) is None


def test_infer_type():
    assert (
        infer_type_from_repo_files("a/b", ["model_index.json", "unet/config.json"]) == "diffusers"
    )
    assert infer_type_from_repo_files("a/b", ["x-Q4_K_M.gguf"]) == "llm"
    assert infer_type_from_repo_files("a/my-lora", ["w.safetensors"]) == "lora"
    assert (
        infer_type_from_repo_files(
            "a/sdxl-vae", ["config.json", "diffusion_pytorch_model.safetensors"]
        )
        == "vae"
    )
    assert infer_type_from_repo_files("a/b", ["model.safetensors"]) == "checkpoints"
    assert infer_type_from_repo_files("a/b", ["README.md"]) is None


def test_sanitize_name():
    assert sanitize_name("../../evil name!.safetensors") == "evil-name-.safetensors"
    assert sanitize_name("") == "model"


def test_find_model_path_blocks_traversal(workspace):
    (workspace / "models" / "lora").mkdir(parents=True, exist_ok=True)
    assert inventory.find_model_path("../diffusers", "lora") is None


def test_delete_rejects_bad_names():
    for bad in ("", "..", "a/b", ".staging"):
        with pytest.raises(ValueError):
            inventory.delete_local_model("lora", bad)


def test_request_needs_exactly_one_source():
    with pytest.raises(ValueError):
        DownloadRequest()
    with pytest.raises(ValueError):
        DownloadRequest(repo_id="a/b", url="https://x/y")


# --------------------------------------------------------------------------- end-to-end with fake hub
@pytest.fixture
def manager():
    return DownloadManager()


async def _download(manager: DownloadManager, req: DownloadRequest):
    job = await manager.submit(req)
    return await asyncio.wait_for(manager.wait(job.id), timeout=60)


async def test_snapshot_download_installs_diffusers_model(
    fake_hub, tiny_sd_source, manager, workspace
):
    fake_hub.add_folder("test/tiny-sd-remote", tiny_sd_source)
    job = await _download(manager, DownloadRequest(repo_id="test/tiny-sd-remote"))
    assert job.status == "completed", job.error
    assert job.model_type == "diffusers"
    target = workspace / "models" / "diffusers" / "tiny-sd-remote"
    assert (target / "model_index.json").is_file()
    assert job.total_bytes and job.downloaded_bytes == job.total_bytes
    assert not (workspace / "models" / "diffusers" / ".staging").exists() or not any(
        (workspace / "models" / "diffusers" / ".staging").iterdir()
    )
    names = [m.name for m in inventory.list_local_models("diffusers")]
    assert "tiny-sd-remote" in names
    meta = manifest.get_entry("diffusers/tiny-sd-remote")
    assert meta and meta["repo_id"] == "test/tiny-sd-remote"


async def test_gguf_repo_picks_one_file(fake_hub, manager, workspace):
    fake_hub.add_repo(
        "test/tiny-llm-GGUF",
        {"tiny-Q8_0.gguf": b"Q8" * 1000, "tiny-Q4_K_M.gguf": b"Q4" * 1000, "README.md": b"hi"},
    )
    job = await _download(manager, DownloadRequest(repo_id="test/tiny-llm-GGUF"))
    assert job.status == "completed", job.error
    assert job.model_type == "llm"
    assert (workspace / "models" / "llm" / "tiny-Q4_K_M.gguf").is_file()
    assert not (workspace / "models" / "llm" / "tiny-Q8_0.gguf").exists()


async def test_single_file_from_url_style_reference(fake_hub, manager, workspace):
    fake_hub.add_repo(
        "test/loras", {"sub/style.safetensors": b"\0" * 2048, "other.safetensors": b"\1" * 10}
    )
    url = "https://huggingface.co/test/loras/blob/main/sub/style.safetensors"
    job = await _download(manager, DownloadRequest(repo_id=url, model_type="lora"))
    assert job.status == "completed", job.error
    assert (workspace / "models" / "lora" / "style.safetensors").stat().st_size == 2048
    assert "style.safetensors" in inventory.list_model_names("lora")


async def test_gated_repo_reports_gated(fake_hub, manager):
    fake_hub.add_repo("test/gated", {"model_index.json": b"{}"}, gated=True)
    job = await _download(manager, DownloadRequest(repo_id="test/gated", model_type="diffusers"))
    assert job.status == "failed"
    assert job.error_code == "gated"
    assert "huggingface.co/test/gated" in job.error


async def test_missing_repo_reports_not_found(fake_hub, manager):
    job = await _download(manager, DownloadRequest(repo_id="test/missing", model_type="diffusers"))
    assert job.status == "failed"
    assert job.error_code == "not_found"


async def test_cancel_keeps_nothing_in_place(fake_hub, manager, workspace):
    # > 10 MB so huggingface_hub reports progress before the end (it updates per 10 MB chunk)
    fake_hub.add_repo("test/slow", {"big.safetensors": os.urandom(24 * 1024 * 1024)}, slow=True)
    job = await manager.submit(
        DownloadRequest(repo_id="test/slow", files=["big.safetensors"], model_type="checkpoints")
    )
    for _ in range(150):
        await asyncio.sleep(0.1)
        if job.downloaded_bytes > 0:
            break
    assert 0 < job.downloaded_bytes < 24 * 1024 * 1024
    assert job.total_bytes == 24 * 1024 * 1024
    await manager.cancel(job.id)
    assert job.status == "cancelled"
    assert not (workspace / "models" / "checkpoints" / "big.safetensors").exists()
    assert "big.safetensors" not in inventory.list_model_names("checkpoints")


async def test_direct_url_download_with_resume(fake_hub, manager, workspace, monkeypatch):
    import gradient.model_manager.downloader as dl

    data = os.urandom(300_000)
    fake_hub.direct_files["detail.safetensors"] = data
    url = f"{fake_hub.url}/files/detail.safetensors"
    # Pre-create a partial file to exercise HTTP Range resume
    plan = await manager.resolve(
        DownloadRequest(url="https://example.org/detail.safetensors", model_type="lora")
    )
    plan.url = url
    staging = plan.type_dir / dl.STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)
    import hashlib

    (staging / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".part")).write_bytes(data[:100_000])
    from gradient.model_manager.schemas import DownloadJob

    job = DownloadJob(id="resume1", name="detail", source=url, model_type="lora")
    await manager._run(job, plan)
    assert job.status == "completed", job.error
    assert (workspace / "models" / "lora" / "detail.safetensors").read_bytes() == data
    assert any("GET /files/detail.safetensors" in r for r in fake_hub.requests)


async def test_url_download_rejects_non_weight_files(fake_hub, manager):
    fake_hub.direct_files["evil.exe"] = b"MZ"
    plan = await manager.resolve(
        DownloadRequest(url="https://example.org/evil.exe", model_type="lora")
    )
    plan.url = f"{fake_hub.url}/files/evil.exe"
    from gradient.model_manager.schemas import DownloadJob

    job = DownloadJob(id="evil", name="evil", source=plan.url, model_type="lora")
    await manager._run(job, plan)
    assert job.status == "failed" and job.error_code == "invalid"
