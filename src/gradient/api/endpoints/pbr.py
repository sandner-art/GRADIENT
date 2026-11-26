# src/gradient/api/endpoints/pbr.py
"""PBR maps from an image: normal, height, AO, albedo, curvature, glow, roughness, metallic and
engine packs, written as a set that Unity, HDRP, glTF and Unreal can use."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from gradient.ai.lab.embed import decode_image
from gradient.ai.pbr import maps
from gradient.core.scheduling import run_gpu_task

router = APIRouter()


class MapsRequest(BaseModel):
    output_id: int | None = Field(default=None, description="A gallery image")
    image_base64: str | None = Field(default=None, description="Or an image (base64 or data URL)")
    name: str | None = Field(default=None, description="File name stem (default: the image's)")
    model: str = Field(default="auto", description="PBR network in <models>/pbr/, or auto")
    method: Literal["hybrid", "network", "classical"] = Field(
        default="hybrid",
        description="hybrid: network normals plus the picture's fine detail (default); network: "
        "the network's normals alone; classical: no network",
    )
    seamless: Literal["auto", "none", "x", "y", "xy"] = Field(
        default="auto",
        description="Wrap-around per axis, so the maps tile; auto: as the image was rendered "
        "(its tiling setting)",
    )
    detail: float = Field(default=0.35, ge=0, le=2, description="Hybrid: fine detail added")
    strength: float = Field(default=1.0, gt=0, le=8, description="Normal map strength")

    @model_validator(mode="after")
    def _one_source(self):
        if (self.output_id is None) == (self.image_base64 is None):
            raise ValueError("Send output_id or image_base64.")
        return self


class MapFile(BaseModel):
    kind: str
    label: str
    file: str
    color: bool = Field(description="Colour data (sRGB); the others are linear data")


class MapsResponse(BaseModel):
    set_id: str
    name: str
    width: int
    height: int
    method: str
    model: str | None
    seamless: dict[str, bool]
    maps: list[MapFile]
    sidecar: str
    light: dict | None = Field(default=None, description="The key light painted into the picture")
    height_range_px: float
    notes: list[str] = []
    seconds: float
    source_output_id: int | None = None


@router.get("/pbr/models", summary="PBR map networks in <models>/pbr/")
async def pbr_models():
    nets = await asyncio.to_thread(maps.installed_nets)
    return {
        "folder": str(maps.pbr_model_dirs()[0]),
        "models": [{k: v for k, v in n.items() if k != "path"} for n in nets],
        "default": maps.default_net(),
    }


@router.post("/pbr/maps", response_model=MapsResponse, summary="Make a PBR map set from an image")
async def make_maps(request: MapsRequest = Body(...)):
    """Runs between renders (it shares the GPU lock). The set is written to
    <outputs>/pbr_maps/<set_id>/ and served by the endpoints below."""
    from gradient.gallery.store import get_output, output_path

    name = request.name
    tiling = None
    source = ""
    try:
        if request.output_id is not None:
            info = await get_output(request.output_id)
            path = await output_path(request.output_id)
            tiling = (info.get("parameters") or {}).get("tiling")
            image = await asyncio.to_thread(lambda: Image.open(path).copy())
            name = name or path.stem
            source = path.name
        else:
            image = await asyncio.to_thread(decode_image, request.image_base64)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\"")) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    seamless = tiling if request.seamless == "auto" else request.seamless
    try:
        result = await run_gpu_task(
            lambda: maps.make_maps(
                image,
                name=name or "texture",
                model=request.model,
                method=request.method,
                wrap=maps.wrap_from_tiling(seamless),
                detail=request.detail,
                strength=request.strength,
                source=source,
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {**result, "source_output_id": request.output_id}


@router.get("/pbr/maps/{set_id}/files/{file}", summary="One file of a map set")
async def map_file(set_id: str, file: str):
    try:
        path = await asyncio.to_thread(maps.set_file, set_id, file)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="No such map file.") from e
    media = "application/json" if path.suffix == ".json" else "image/png"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "max-age=31536000"})


@router.get("/pbr/maps/{set_id}/zip", summary="A map set as a ZIP")
async def map_zip(set_id: str):
    try:
        data, filename = await asyncio.to_thread(maps.set_zip, set_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="No such map set.") from e
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
