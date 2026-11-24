# src/gradient/api/job_kinds.py
"""What the render queue can run, and how jobs are stored between sessions."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy import desc, select

from gradient.ai.diffusion.models import (
    ImageToImageRequest,
    Text2ImageRequest,
    XYZGridRequest,
    XYZGridResponse,
)
from gradient.ai.diffusion.pipelines import run_image_to_image, run_text_to_image
from gradient.ai.diffusion.xyz_grid import run_xyz_grid
from gradient.core.jobs import Job, JobCancelled, job_queue
from gradient.core.scheduling import run_gpu_task
from gradient.database.models import QueueJob
from gradient.database.session import get_db_session
from gradient.gallery import groups

logger = logging.getLogger(__name__)

#: kinds the public POST /queue endpoint accepts, with their request models
SUBMITTABLE: dict[str, type[BaseModel]] = {
    "text_to_image": Text2ImageRequest,
    "image_to_image": ImageToImageRequest,
    "xyz_grid": XYZGridRequest,
}
#: payload keys that are not stored in the database (large inline images)
INLINE_KEYS = ("image_base64", "init_image_base64")


def validate_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    model = SUBMITTABLE.get(kind)
    if model is None:
        raise ValueError(f"Unknown job kind '{kind}'; use one of {', '.join(SUBMITTABLE)}")
    return model.model_validate(payload).model_dump(mode="json", exclude_unset=True)


def label_for(kind: str, payload: dict[str, Any]) -> str:
    def short(text: str | None, n: int = 48) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= n else text[: n - 1] + "…"

    if kind == "xyz_grid":
        base = payload.get("base_parameters") or {}
        return f"XYZ grid · {short(base.get('prompt'), 36)}"
    if kind.startswith("lab_"):
        base = payload.get("base_request") or {}
        return f"Lab {kind[4:].replace('_', ' ')} · {short(base.get('prompt'), 36)}"
    n = int(payload.get("num_outputs") or 1)
    prefix = "img2img · " if kind == "image_to_image" else ""
    return f"{prefix}{short(payload.get('prompt'))}" + (f" ×{n}" if n > 1 else "")


# --------------------------------------------------------------------------- handlers
def _image_summary(response) -> dict[str, Any]:
    summary = {
        "run_uuid": response.run_uuid,
        "outputs": [{"id": im.output_id, "seed": im.seed} for im in response.images],
        "performance_ms": response.performance_ms,
    }
    # LoRAs that matched no layer of the model (another model family) were not applied
    skipped = sorted(
        {n for im in response.images for n in (im.parameters or {}).get("loras_skipped") or []}
    )
    if skipped:
        summary["loras_skipped"] = skipped
    return summary


async def _render(job: Job, request, run):
    """A text-to-image or image-to-image job; jobs of one group share a gallery run."""
    group = job.meta.get("group")
    if not group:
        response = await run(request)
    else:
        run_uuid, run_id = await groups.open_run(group, request.model_id)
        job.run_uuid = run_uuid
        status = "failed"
        try:
            response = await run(request, parent_run_uuid=run_uuid, parent_run_id=run_id)
            status = "completed"
        except JobCancelled:
            status = "cancelled"
            raise
        finally:
            await groups.close_run(run_id, status)
    await groups.label_outputs([im.output_id for im in response.images], job.meta)
    return response


async def _text_to_image(job: Job):
    return await _render(job, Text2ImageRequest.model_validate(job.payload), run_text_to_image)


async def _image_to_image(job: Job):
    return await _render(job, ImageToImageRequest.model_validate(job.payload), run_image_to_image)


async def _xyz_grid(job: Job) -> XYZGridResponse:
    request = XYZGridRequest.model_validate(job.payload)
    run_uuid = str(uuid.uuid4())
    job.run_uuid = run_uuid
    cells = await run_xyz_grid(request, run_uuid=run_uuid)
    await groups.label_outputs(
        [c.image_result.output_id for c in cells if c.image_result is not None], job.meta
    )
    return XYZGridResponse(grid_results=cells, request_params=request, run_uuid=run_uuid)


def _xyz_summary(response: XYZGridResponse) -> dict[str, Any]:
    outputs = [
        {"id": c.image_result.output_id, "seed": c.image_result.seed}
        for c in response.grid_results
        if c.image_result is not None
    ]
    return {
        "run_uuid": response.run_uuid,
        "outputs": outputs,
        "cells": len(response.grid_results),
        "errors": sum(1 for c in response.grid_results if c.error),
    }


def register_lab_kind(tool: str, fn, request_model: type[BaseModel], save: bool = True) -> str:
    """Lab tools run as interactive jobs; their results are saved as gallery runs
    (unless `save` is False: previews and suggestions that make no new images)."""
    from gradient.gallery.store import record_lab_run

    kind = f"lab_{tool}"

    async def handler(job: Job):
        request = request_model.model_validate(job.payload)
        response = await run_gpu_task(fn, request)
        if not save:
            return response
        try:
            job.result = await record_lab_run(tool, request, response)
            job.run_uuid = job.result["run_uuid"]
        except Exception as e:  # saving must never lose the result itself
            logger.warning("Could not save %s result to the gallery: %s", tool, e)
        return response

    job_queue.register(kind, handler, summarize=lambda _response: None)
    return kind


def register_job_kinds() -> None:
    job_queue.register("text_to_image", _text_to_image, _image_summary)
    job_queue.register("image_to_image", _image_to_image, _image_summary)
    job_queue.register("xyz_grid", _xyz_grid, _xyz_summary)


# --------------------------------------------------------------------------- persistence
async def persist_job(job: Job) -> None:
    payload = {k: v for k, v in job.payload.items() if k not in INLINE_KEYS}
    async with get_db_session() as db:
        await db.merge(
            QueueJob(
                id=job.id,
                kind=job.kind,
                label=job.label,
                status=job.status,
                priority=job.priority,
                payload=payload,
                created=job.created,
                started=job.started,
                finished=job.finished,
                run_uuid=job.run_uuid,
                error=job.error,
                result=job.result,
                meta=job.meta or None,
            )
        )


async def load_recent_jobs(limit: int = 100) -> list[Job]:
    async with get_db_session() as db:
        rows = (
            await db.scalars(select(QueueJob).order_by(desc(QueueJob.created)).limit(limit))
        ).all()
    jobs = []
    for r in rows:
        payload = r.payload or {}
        job = Job(
            kind=r.kind,
            payload=payload,
            label=r.label,
            priority=r.priority,
            id=r.id,
            status=r.status,
            created=r.created,
            started=r.started,
            finished=r.finished,
            run_uuid=r.run_uuid,
            error=r.error,
            result=r.result,
            meta=r.meta or {},
        )
        # img2img inputs aren't stored, and Lab jobs are re-run from the Lab instead
        job.retryable = r.kind in SUBMITTABLE and not (
            r.kind == "image_to_image" and "image_base64" not in payload
        )
        jobs.append(job)
    return jobs
