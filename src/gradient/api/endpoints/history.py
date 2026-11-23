# src/gradient/api/endpoints/history.py
"""Gallery and history: browse images and runs, rate, tag, favourite, delete."""

from __future__ import annotations

import datetime
from typing import Any, Literal

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from gradient.gallery import store

router = APIRouter()


class OutputUpdate(BaseModel):
    rating: int | None = Field(default=None, ge=0, le=5, description="0 clears the rating")
    favorite: bool | None = None
    notes: str | None = None
    tags: list[str] | None = Field(default=None, description="Replaces your own (manual) tags")


class BulkAction(BaseModel):
    ids: list[int] = Field(..., min_length=1, max_length=2000)
    action: Literal["rate", "favorite", "add_tags", "remove_tags", "delete"]
    value: Any = None


def _date(value: str | None, name: str) -> datetime.datetime | None:
    if not value:
        return None
    try:
        dt = datetime.datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{name} must be an ISO date") from None
    return dt if dt.tzinfo else dt.replace(tzinfo=datetime.UTC)


@router.get("/history/outputs", summary="Images, newest first, with filters")
async def list_outputs(
    run: str | None = None,
    model: str | None = None,
    run_type: str | None = Query(default=None, description="e.g. text_to_image, xyz_grid, lab_*"),
    q: str | None = Query(default=None, description="Words in the prompt or notes"),
    min_rating: int | None = Query(default=None, ge=1, le=5),
    favorite: bool | None = None,
    tag: str | None = None,
    since: str | None = None,
    until: str | None = None,
    sort: Literal["newest", "oldest", "rating"] = "newest",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=60, ge=1, le=store.MAX_PAGE),
):
    return await store.list_outputs(
        run=run,
        model=model,
        run_type=run_type,
        q=q,
        min_rating=min_rating,
        favorite=favorite,
        tag=tag,
        since=_date(since, "since"),
        until=_date(until, "until"),
        sort=sort,
        offset=offset,
        limit=limit,
    )


@router.get("/history/facets", summary="Models, run types and tags with counts")
async def facets():
    return await store.facets()


@router.get("/history/runs", summary="Runs (renders, grids, Lab results), newest first")
async def list_runs(
    run_type: str | None = None,
    model: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=40, ge=1, le=store.MAX_PAGE),
    include_empty: bool = Query(
        default=False, description="Also list finished runs that have no outputs left"
    ),
):
    return await store.list_runs(
        run_type=run_type, model=model, offset=offset, limit=limit, include_empty=include_empty
    )


@router.delete("/history/runs/{run_uuid}", summary="Delete a run and move its files to the trash")
async def delete_run(run_uuid: str):
    try:
        return {"removed": await store.delete_run(run_uuid)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'")) from e


@router.get("/history/outputs/{output_id}", summary="One image with all its parameters")
async def get_output(output_id: int):
    try:
        return await store.get_output(output_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'")) from e


@router.patch("/history/outputs/{output_id}", summary="Rate, favourite, tag or annotate an image")
async def update_output(output_id: int, body: OutputUpdate = Body(...)):
    try:
        return await store.update_output(
            output_id,
            rating=body.rating,
            favorite=body.favorite,
            notes=body.notes,
            tags=body.tags,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'")) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/history/outputs/{output_id}", summary="Delete an image (file goes to the trash)")
async def delete_output(output_id: int):
    removed = await store.delete_outputs([output_id])
    if not removed:
        raise HTTPException(status_code=404, detail=f"No image {output_id}")
    return {"removed": removed}


@router.post("/history/outputs/bulk", summary="Rate, favourite, tag or delete many images")
async def bulk(body: BulkAction = Body(...)):
    try:
        return await store.bulk(body.ids, body.action, body.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


async def _path(output_id: int):
    try:
        return await store.output_path(output_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'")) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=410, detail=str(e)) from e


@router.get("/history/outputs/{output_id}/image", summary="The image file")
async def image(output_id: int, download: bool = False):
    path = await _path(output_id)
    return FileResponse(
        path,
        media_type="image/png" if path.suffix.lower() == ".png" else None,
        filename=path.name if download else None,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/history/outputs/{output_id}/thumb", summary="A WebP thumbnail")
async def thumb(output_id: int, size: int = Query(default=256, ge=64, le=1024)):
    path = await _path(output_id)
    try:
        thumb_path = await run_in_threadpool(store.make_thumb, path, output_id, size)
    except OSError as e:
        raise HTTPException(status_code=415, detail=f"Can't read image {output_id}: {e}") from e
    return FileResponse(
        thumb_path, media_type="image/webp", headers={"Cache-Control": "private, max-age=604800"}
    )
