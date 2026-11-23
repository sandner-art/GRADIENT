# src/gradient/api/endpoints/projects.py
"""Project folders: export gallery images for other tools (Kurator, ArtAgents, digiKam ...)
and sync their tags, ratings and captions back."""

from typing import Any, Literal

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, model_validator

from gradient.gallery import projects

router = APIRouter()


class ExportRequest(BaseModel):
    ids: list[int] | None = Field(default=None, description="Images to export")
    filter: dict[str, Any] | None = Field(
        default=None,
        description="Or: a gallery filter (run, model, run_type, q, min_rating, favorite, tag, "
        "project); every matching image is exported",
    )
    project: str | None = Field(default=None, description="Add to this project (uuid)")
    name: str | None = Field(default=None, description="Name of a new project")
    path: str | None = Field(
        default=None,
        description="Folder of a new project (default: data/projects/<name>); relative paths "
        "are inside data/projects",
    )
    sidecars: bool = Field(default=True, description="Write XMP sidecars (name.png.xmp)")
    model_keywords: bool = Field(
        default=True, description="Add GRADIENT/Model/... and GRADIENT/Type/... keywords"
    )
    captions: Literal["none", "prompt"] = Field(
        default="none",
        description="'prompt' writes the prompt as name.txt (dataset style); existing caption "
        "files from other tools are never overwritten",
    )
    link: bool = Field(default=False, description="Hard-link instead of copying (same drive)")

    @model_validator(mode="after")
    def _one_source(self):
        if (self.ids is None) == (self.filter is None):
            raise ValueError("Give either ids or filter")
        if not self.project and not (self.name or "").strip():
            raise ValueError("Give a project uuid or a name for a new project")
        return self


_FILTER_KEYS = {"run", "model", "run_type", "q", "min_rating", "favorite", "tag", "project"}


class SyncRequest(BaseModel):
    caption_source: str = Field(
        default="artagents",
        description="Label for captions read from name.txt files, e.g. 'artagents:qwen2.5vl'; "
        "one caption is kept per label",
    )
    captions_as_tags: bool = Field(
        default=False, description="Also add comma-separated captions as tags (label as source)"
    )


class OpenRequest(SyncRequest):
    path: str


def _error(e: Exception) -> HTTPException:
    if isinstance(e, KeyError):
        return HTTPException(status_code=404, detail=str(e).strip("'\""))
    return HTTPException(status_code=400, detail=str(e))


@router.get("/projects", summary="Project folders")
async def list_projects():
    return {"items": await projects.list_projects(), "root": str(projects.projects_root())}


@router.get("/projects/{uuid}", summary="One project")
async def get_project(uuid: str):
    try:
        return await projects.get_project(uuid)
    except KeyError as e:
        raise _error(e) from e


@router.post("/projects/export", summary="Export images into a project folder")
async def export(body: ExportRequest = Body(...)):
    filters = None
    if body.filter is not None:
        unknown = set(body.filter) - _FILTER_KEYS
        if unknown:
            raise HTTPException(status_code=422, detail=f"Unknown filter keys: {sorted(unknown)}")
        filters = {k: v for k, v in body.filter.items() if v not in (None, "")}
    try:
        return await projects.export(
            ids=body.ids,
            filters=filters,
            project=body.project,
            name=body.name,
            path=body.path,
            sidecars=body.sidecars,
            model_keywords=body.model_keywords,
            captions=body.captions,
            link=body.link,
        )
    except (KeyError, ValueError, OSError) as e:
        raise _error(e) from e


@router.post("/projects/{uuid}/sync", summary="Read tags, ratings and captions back")
async def sync(uuid: str, body: SyncRequest = Body(default_factory=SyncRequest)):
    try:
        return await projects.sync(uuid, body.caption_source, body.captions_as_tags)
    except (KeyError, ValueError, OSError) as e:
        raise _error(e) from e


@router.post("/projects/open", summary="Add an existing project folder and sync it")
async def open_folder(body: OpenRequest = Body(...)):
    try:
        return await projects.open_folder(body.path, body.caption_source)
    except (KeyError, ValueError, OSError) as e:
        raise _error(e) from e


@router.delete("/projects/{uuid}", summary="Forget a project (its folder stays)")
async def forget(uuid: str):
    try:
        await projects.forget(uuid)
    except KeyError as e:
        raise _error(e) from e
    return {"forgotten": uuid}
