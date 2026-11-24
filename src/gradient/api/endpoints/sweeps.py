# src/gradient/api/endpoints/sweeps.py
"""Test sweeps from ArtAgents as files: preview, queue, status, results folder."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from gradient.gallery import sweeps

router = APIRouter()


class SweepImport(BaseModel):
    sweep_text: str = Field(
        max_length=sweeps.MAX_SWEEP_CHARS,
        description="The sweep file's exact text (its sha256 ties the results to it)",
    )
    lora_map: dict[str, str] = Field(
        default_factory=dict,
        description="LoRA key → local file (full path or a name in the LoRA folder)",
    )


def _bad(e: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


@router.post("/sweeps/preview", summary="Check a sweep file: model, LoRAs (by sha256), image count")
async def preview(body: SweepImport = Body(...)):
    try:
        return await asyncio.to_thread(sweeps.preview, body.sweep_text, body.lora_map)
    except sweeps.SweepError as e:
        raise _bad(e) from e


@router.post("/sweeps", status_code=202, summary="Queue every image of a sweep file")
async def queue(body: SweepImport = Body(...)):
    try:
        return await sweeps.queue_sweep(body.sweep_text, body.lora_map)
    except sweeps.SweepError as e:
        raise _bad(e) from e


@router.get("/sweeps", summary="Imported sweeps, newest first, with job counts")
async def list_sweeps():
    return {"items": await sweeps.list_sweeps()}


@router.get("/sweeps/{sweep_id}", summary="One imported sweep with job counts")
async def get_sweep(sweep_id: str):
    try:
        return await sweeps.status(sweep_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\"")) from e


@router.post("/sweeps/{sweep_id}/results", summary="Write the results folder now (also partial)")
async def write_results(sweep_id: str):
    try:
        return await sweeps.write_results(sweep_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\"")) from e
