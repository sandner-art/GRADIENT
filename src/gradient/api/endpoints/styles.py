# src/gradient/api/endpoints/styles.py
"""Prompt styles: list, import a styles file, preview the prompts a render will use."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from gradient.ai.diffusion import prompt_syntax, styles
from gradient.ai.diffusion.loading import model_profile
from gradient.ai.diffusion.pipelines import resolve_seed

router = APIRouter()

MAX_STYLES_FILE = 5_000_000  # characters


@router.get("/styles", summary="Prompt styles from the styles folder (A1111 styles.csv, JSON)")
async def list_styles():
    def run():
        found, errors = styles.load_styles()
        files: dict[str, int] = {}
        for style in found:
            files[style.file] = files.get(style.file, 0) + 1
        return {
            "folder": str(styles.styles_dir()),
            "files": [
                {"name": name, "styles": files.get(name, 0), "error": errors.get(name)}
                for name in sorted({*files, *errors})
            ],
            "styles": [style.as_dict() for style in found],
        }

    return await asyncio.to_thread(run)


class StylesFile(BaseModel):
    filename: str = Field(..., description="File name to save as (.csv or .json)")
    content: str = Field(..., max_length=MAX_STYLES_FILE, description="The file's text")


@router.post("/styles/import", summary="Save a styles file into the styles folder")
async def import_styles(body: StylesFile = Body(...)):
    try:
        path, count = await asyncio.to_thread(styles.save_styles_file, body.filename, body.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"file": path.name, "styles": count}


class StylePreviewRequest(BaseModel):
    model_id: str | None = None
    prompt: str = ""
    negative_prompt: str | None = ""
    styles: list[str] | None = None
    style_position: Literal["template", "before", "after", "chunk"] = "template"
    style_start: float = Field(default=0.0, ge=0.0, lt=1.0)
    prompt_syntax: Literal["a1111", "comfy", "plain"] = "a1111"
    seed: int | None = Field(default=None, description="Picks {a|b} wildcard options")


@router.post("/styles/preview", summary="The prompts a render will use, with token counts")
async def preview(body: StylePreviewRequest = Body(...)):
    def run():
        family = model_profile(body.model_id)[0] if body.model_id else None
        clip = body.prompt_syntax != "plain" and family in ("sd15", "sdxl")
        seed = resolve_seed(body.seed) if body.seed is not None else 0
        prepared = styles.prepare(body, seed, clip_syntax=clip)
        counts = {}
        if body.model_id and family in ("sd15", "sdxl"):
            counts = {
                "prompt": prompt_syntax.count_tokens(body.model_id, prepared.prompt),
                "negative_prompt": prompt_syntax.count_tokens(
                    body.model_id, prepared.negative_prompt or ""
                ),
            }
        return {
            "prompt": prepared.prompt,
            "negative_prompt": prepared.negative_prompt,
            "loras": prepared.loras,
            "notes": prepared.params,
            "model_family": family,
            "reads_syntax": clip,
            "tokens": counts or None,
        }

    return await asyncio.to_thread(run)
