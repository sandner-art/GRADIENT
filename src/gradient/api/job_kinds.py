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
from gradient.core.jobs import Job, job_queue
from gradient.core.scheduling import run_gpu_task
from gradient.database.models import QueueJob
from gradient.database.session import get_db_session

logger = logging.getLogger(__name__)

#: kinds the public POST /queue endpoint accepts, with their request models
SUBMITTABLE: dict[str, type[BaseModel]] = {
    "text_to_image": Text2ImageRequest,
    "image_to_image": ImageToImageRequest,
    "xyz_grid": XYZGridRequest,
}
#: payload keys that are not stored in the database (large inline images)
INLINE_KEYS = ("image_base64",)


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
    return {
        "run_uuid": response.run_uuid,
        "outputs": [{"id": im.output_id, "seed": im.seed} for im in response.images],
        "performance_ms": response.performance_ms,
    }


async def _text_to_image(job: Job):
    return await run_text_to_image(Text2ImageRequest.model_validate(job.payload))


async def _image_to_image(job: Job):
    return await run_image_to_image(ImageToImageRequest.model_validate(job.payload))


async def _xyz_grid(job: Job) -> XYZGridResponse:
    request = XYZGridRequest.model_validate(job.payload)
    run_uuid = str(uuid.uuid4())
    job.run_uuid = run_uuid
    cells = await run_xyz_grid(request, run_uuid=run_uuid)
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


def register_lab_kind(tool: str, fn, request_model: type[BaseModel]) -> str:
    """Lab tools run as interactive jobs; their results are saved as gallery runs."""
    from gradient.gallery.store import record_lab_run

    kind = f"lab_{tool}"

    async def handler(job: Job):
        request = request_model.model_validate(job.payload)
        response = await run_gpu_task(fn, request)
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
        )
        # img2img inputs aren't stored, and Lab jobs are re-run from the Lab instead
        job.retryable = r.kind in SUBMITTABLE and not (
            r.kind == "image_to_image" and "image_base64" not in payload
        )
        jobs.append(job)
    return jobs
