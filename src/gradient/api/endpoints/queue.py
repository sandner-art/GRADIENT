# src/gradient/api/endpoints/queue.py
"""The render queue: submit, watch (server-sent events), cancel, reorder, retry."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from gradient.api.job_kinds import label_for, validate_payload
from gradient.core import shutdown
from gradient.core.jobs import job_queue
from gradient.gallery.store import clean_tags

router = APIRouter()
logger = logging.getLogger(__name__)


class JobSubmit(BaseModel):
    kind: Literal["text_to_image", "image_to_image", "xyz_grid"]
    payload: dict[str, Any]
    label: str | None = None
    count: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Queue this many jobs. With a fixed seed, job i uses seed + i (see vary_seed); "
        "with a random seed every job draws its own.",
    )
    vary_seed: bool = Field(default=True, description="Step a fixed seed by one per job")
    group: str | None = Field(
        default=None,
        max_length=120,
        description="Jobs with the same group save their images into one gallery run "
        "(text_to_image and image_to_image)",
    )
    tags: list[str] = Field(
        default_factory=list, max_length=32, description="Gallery tags for the images"
    )
    origin: dict[str, Any] | None = Field(
        default=None,
        description="Where the job came from (app, run, item...), kept with each image; "
        "at most 4 KB as JSON",
    )

    @field_validator("group")
    @classmethod
    def _group(cls, value: str | None) -> str | None:
        return " ".join(value.split()) or None if value else None

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: list[str]) -> list[str]:
        return clean_tags(value)

    @field_validator("origin")
    @classmethod
    def _origin(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None and len(json.dumps(value, default=str)) > 4096:
            raise ValueError("origin is limited to 4 KB of JSON")
        return value or None

    def meta(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in (("group", self.group), ("tags", self.tags), ("origin", self.origin))
            if v
        }


class JobMove(BaseModel):
    to: Literal["top", "up", "down", "bottom"]


def _not_found(e: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(e).strip("'"))


@router.post("/queue", status_code=202, summary="Queue one or more jobs")
async def submit(body: JobSubmit = Body(...)):
    try:
        payload = validate_payload(body.kind, body.payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors(include_url=False)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    jobs = []
    seed = payload.get("seed")
    for i in range(body.count):
        job_payload = dict(payload)
        if i and body.vary_seed and isinstance(seed, int):
            job_payload["seed"] = (seed + i) % 2**32
        job = await job_queue.submit(
            body.kind,
            job_payload,
            body.label or label_for(body.kind, job_payload),
            meta=body.meta(),
        )
        jobs.append(job_queue.view(job))
    return {"jobs": jobs}


@router.get("/queue", summary="Running, waiting and recent jobs")
async def list_jobs():
    jobs = job_queue.list()
    return {
        "jobs": jobs,
        "running": job_queue.running.id if job_queue.running else None,
        "waiting": len(job_queue.order),
    }


@router.get("/queue/events", summary="Server-sent events with job updates")
async def events(request: Request):
    queue = job_queue.subscribe()

    async def stream():
        try:
            snapshot = {"type": "snapshot", "jobs": job_queue.list()}
            # browsers reconnect 2 s after the stream ends (GRADIENT stopped or restarted)
            yield f"retry: 2000\ndata: {json.dumps(snapshot)}\n\n"
            quiet = 0.0
            while True:
                if shutdown.requested():  # Ctrl+C: tell the page and end the stream now
                    yield 'data: {"type": "shutdown"}\n\n'
                    break
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                except TimeoutError:
                    quiet += 0.5
                    if quiet >= 15:
                        quiet = 0.0
                        yield ": keep-alive\n\n"
                    continue
                quiet = 0.0
                yield f"data: {json.dumps(event, default=str)}\n\n"
        finally:
            job_queue.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/queue/{job_id}", summary="One job")
async def get_job(job_id: str):
    try:
        return job_queue.view(job_queue.get(job_id))
    except KeyError as e:
        raise _not_found(e) from e


@router.get(
    "/queue/{job_id}/preview",
    summary="Latest live preview of a job (JPEG); progress.preview counts updates",
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "No preview yet"}},
)
async def job_preview(job_id: str):
    try:
        job = job_queue.get(job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\"")) from e
    if not job.preview:
        raise HTTPException(status_code=404, detail="No preview for this job")
    return Response(
        content=job.preview, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
    )


@router.post("/queue/{job_id}/cancel", summary="Cancel a waiting job or stop a running one")
async def cancel(job_id: str):
    try:
        return job_queue.view(await job_queue.cancel(job_id))
    except KeyError as e:
        raise _not_found(e) from e


@router.post("/queue/{job_id}/move", summary="Move a waiting job (top, up, down, bottom)")
async def move(job_id: str, body: JobMove = Body(...)):
    try:
        await job_queue.move(job_id, body.to)
        return {"order": list(job_queue.order)}
    except KeyError as e:
        raise _not_found(e) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.post(
    "/queue/{job_id}/retry", status_code=202, summary="Queue a failed or cancelled job again"
)
async def retry(job_id: str):
    try:
        return job_queue.view(await job_queue.retry(job_id))
    except KeyError as e:
        raise _not_found(e) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.delete("/queue/finished", summary="Remove finished jobs from the list")
async def clear_finished():
    return {"removed": await job_queue.clear_finished()}
