# src/gradient/api/endpoints/models.py
"""Model manager API: catalog, local inventory, downloads."""

import logging
import os
import shutil

from fastapi import APIRouter, Body, HTTPException, Path, Query

from gradient.ai.diffusion import loading
from gradient.core.config import get_model_search_dirs, get_setting
from gradient.core.scheduling import run_gpu_task
from gradient.model_manager import inventory, manifest
from gradient.model_manager.catalog import load_catalog
from gradient.model_manager.downloader import download_manager
from gradient.model_manager.schemas import CatalogEntryView, DownloadRequest, LocalModel, ModelType

logger = logging.getLogger(__name__)
router = APIRouter()

_TYPES: tuple[ModelType, ...] = (
    "diffusers",
    "checkpoints",
    "vae",
    "lora",
    "controlnet",
    "embeddings",
    "llm",
)


@router.get(
    "/models/catalog", response_model=list[CatalogEntryView], summary="Curated downloadable models"
)
async def get_catalog(
    type: ModelType | None = Query(None, description="Filter by model type"),
    include_tests: bool = Query(False, description="Include tiny test models"),
):
    installed = {
        meta.get("catalog_id"): key
        for key, meta in manifest.all_entries().items()
        if meta.get("catalog_id")
    }
    base = get_model_search_dirs()[0]
    views = []
    for entry in load_catalog():
        if type and entry.type != type:
            continue
        if "test" in entry.tags and not include_tests:
            continue
        key = installed.get(entry.id)
        path = base / key if key else None
        exists = bool(path and path.exists())
        views.append(
            CatalogEntryView(
                **entry.model_dump(), installed=exists, installed_path=str(path) if exists else None
            )
        )
    return views


@router.get(
    "/models/local",
    response_model=list[LocalModel],
    summary="Installed models with size and source",
)
async def get_local_models(type: ModelType | None = Query(None)):
    types = [type] if type else list(_TYPES)
    result: list[LocalModel] = []
    for t in types:
        result.extend(inventory.list_local_models(t))
    return result


@router.delete("/models/local/{model_type}/{name}", summary="Delete an installed model")
async def delete_local_model(model_type: ModelType = Path(...), name: str = Path(...)):
    if model_type in ("diffusers", "checkpoints"):
        await run_gpu_task(loading.unload_model, name)
    try:
        path = inventory.delete_local_model(model_type, name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"deleted": str(path)}


@router.get("/models/storage", summary="Model folders, disk usage and token status")
async def get_storage():
    dirs = []
    for d in get_model_search_dirs():
        info = {"path": str(d), "exists": d.exists()}
        if d.exists():
            usage = shutil.disk_usage(d)
            info.update(
                free_gb=round(usage.free / 1024**3, 1), total_gb=round(usage.total / 1024**3, 1)
            )
        dirs.append(info)
    try:
        from huggingface_hub import get_token

        hf_token = bool(get_token() or get_setting("downloads.hf_token"))
    except Exception:
        hf_token = bool(get_setting("downloads.hf_token"))
    return {
        "search_dirs": dirs,
        "download_dir": str(get_model_search_dirs()[0]),
        "hf_token_configured": hf_token,
        "civitai_token_configured": bool(
            os.environ.get("CIVITAI_TOKEN") or get_setting("downloads.civitai_token")
        ),
    }


@router.post("/models/downloads", status_code=202, summary="Start a model download")
async def start_download(request: DownloadRequest = Body(...)):
    try:
        job = await download_manager.submit(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return job.model_dump_api()


@router.get("/models/downloads", summary="Download jobs (newest first)")
async def list_downloads():
    return [j.model_dump_api() for j in download_manager.list_jobs()]


@router.get("/models/downloads/{job_id}")
async def get_download(job_id: str):
    job = download_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown download job")
    return job.model_dump_api()


@router.delete(
    "/models/downloads/{job_id}",
    summary="Cancel a running download (partial files are kept for resume)",
)
async def cancel_download(job_id: str):
    try:
        job = await download_manager.cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown download job") from None
    return job.model_dump_api()


@router.post("/models/downloads/clear", summary="Forget finished/failed/cancelled jobs")
async def clear_downloads():
    return {"cleared": download_manager.clear_finished()}
