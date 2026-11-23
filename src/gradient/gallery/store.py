# src/gradient/gallery/store.py
"""Queries and changes behind the gallery: images, runs, tags, ratings, thumbnails."""

from __future__ import annotations

import asyncio
import base64
import datetime
import io
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy import delete, desc, exists, func, or_, select
from sqlalchemy.orm import selectinload

from gradient.core.config import get_outputs_dir
from gradient.database.models import ExperimentRun, GeneratedOutput, OutputTag
from gradient.database.session import get_db_session

from .metadata import output_fields, png_info

logger = logging.getLogger(__name__)

THUMB_SIZES = (128, 256, 384, 512)
MAX_PAGE = 200
GALLERY_TYPES = ("image",)


def _iso(dt: datetime.datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return dt.isoformat()


def run_model_id(params: dict | None) -> str | None:
    p = params or {}
    return (
        p.get("model_id")
        or (p.get("base") or {}).get("model_id")
        or (p.get("base_request") or {}).get("model_id")
    )


def run_prompt(params: dict | None) -> str | None:
    p = params or {}
    return (
        p.get("prompt")
        or (p.get("base") or {}).get("prompt")
        or (p.get("base_request") or {}).get("prompt")
    )


def output_view(o: GeneratedOutput, detail: bool = False) -> dict[str, Any]:
    view: dict[str, Any] = {
        "id": o.id,
        "run_uuid": o.run.run_uuid if o.run else None,
        "run_type": o.run.run_type if o.run else None,
        "output_type": o.output_type,
        "created": _iso(o.creation_time),
        "model_id": o.model_id,
        "prompt": o.prompt,
        "seed": o.seed,
        "width": o.width,
        "height": o.height,
        "rating": o.user_rating,
        "favorite": bool(o.favorite),
        "notes": o.notes,
        "tags": [
            {"tag": t.tag, "source": t.source, "confidence": t.confidence}
            for t in sorted(o.tags, key=lambda t: (t.source != "manual", t.tag.lower()))
        ],
        "file_name": Path(o.file_path).name,
        "image_url": f"/api/v1/history/outputs/{o.id}/image",
        "thumb_url": f"/api/v1/history/outputs/{o.id}/thumb",
    }
    if detail:
        params = o.parameters or {}
        view.update(
            parameters=params,
            negative_prompt=params.get("negative_prompt"),
            file_path=o.file_path,
            exists=Path(o.file_path).is_file(),
            run_status=o.run.status if o.run else None,
        )
    return view


# --------------------------------------------------------------------------- listing
async def list_outputs(
    *,
    run: str | None = None,
    model: str | None = None,
    run_type: str | None = None,
    q: str | None = None,
    min_rating: int | None = None,
    favorite: bool | None = None,
    tag: str | None = None,
    since: datetime.datetime | None = None,
    until: datetime.datetime | None = None,
    output_type: str | None = "image",
    sort: str = "newest",
    offset: int = 0,
    limit: int = 60,
) -> dict[str, Any]:
    limit = max(1, min(limit, MAX_PAGE))
    stmt = select(GeneratedOutput).join(ExperimentRun, GeneratedOutput.run_id == ExperimentRun.id)
    if output_type:
        stmt = stmt.where(GeneratedOutput.output_type == output_type)
    if run:
        stmt = stmt.where(ExperimentRun.run_uuid == run)
    if model:
        stmt = stmt.where(GeneratedOutput.model_id == model)
    if run_type:
        if run_type.endswith("*"):
            stmt = stmt.where(ExperimentRun.run_type.like(run_type[:-1] + "%"))
        else:
            stmt = stmt.where(ExperimentRun.run_type == run_type)
    if q:
        for word in q.split():
            like = f"%{word}%"
            stmt = stmt.where(
                or_(GeneratedOutput.prompt.ilike(like), GeneratedOutput.notes.ilike(like))
            )
    if min_rating:
        stmt = stmt.where(GeneratedOutput.user_rating >= min_rating)
    if favorite is not None:
        stmt = stmt.where(GeneratedOutput.favorite.is_(favorite))
    if tag:
        stmt = stmt.where(
            GeneratedOutput.id.in_(select(OutputTag.output_id).where(OutputTag.tag == tag))
        )
    if since:
        stmt = stmt.where(GeneratedOutput.creation_time >= since)
    if until:
        stmt = stmt.where(GeneratedOutput.creation_time < until)

    order = {
        "oldest": [GeneratedOutput.creation_time, GeneratedOutput.id],
        "rating": [
            desc(func.coalesce(GeneratedOutput.user_rating, 0)),
            desc(GeneratedOutput.creation_time),
        ],
    }.get(sort, [desc(GeneratedOutput.creation_time), desc(GeneratedOutput.id)])

    async with get_db_session() as db:
        total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
        rows = (
            await db.scalars(
                stmt.options(selectinload(GeneratedOutput.tags), selectinload(GeneratedOutput.run))
                .order_by(*order)
                .offset(offset)
                .limit(limit)
            )
        ).all()
        items = [output_view(o) for o in rows]
    next_offset = offset + len(items)
    return {
        "items": items,
        "total": total or 0,
        "offset": offset,
        "next_offset": next_offset if next_offset < (total or 0) else None,
    }


async def get_output(output_id: int, detail: bool = True) -> dict[str, Any]:
    async with get_db_session() as db:
        o = await _load(db, output_id)
        return output_view(o, detail=detail)


async def _load(db, output_id: int) -> GeneratedOutput:
    o = await db.scalar(
        select(GeneratedOutput)
        .where(GeneratedOutput.id == output_id)
        .options(selectinload(GeneratedOutput.tags), selectinload(GeneratedOutput.run))
    )
    if o is None:
        raise KeyError(f"No image {output_id}")
    return o


async def output_path(output_id: int) -> Path:
    async with get_db_session() as db:
        path = await db.scalar(
            select(GeneratedOutput.file_path).where(GeneratedOutput.id == output_id)
        )
    if path is None:
        raise KeyError(f"No image {output_id}")
    p = Path(path)
    if not await asyncio.to_thread(p.is_file):
        raise FileNotFoundError(f"The file of image {output_id} is missing: {p}")
    return p


async def list_runs(
    *,
    run_type: str | None = None,
    model: str | None = None,
    offset: int = 0,
    limit: int = 40,
    include_empty: bool = False,
) -> dict[str, Any]:
    limit = max(1, min(limit, MAX_PAGE))
    stmt = select(ExperimentRun)
    if not include_empty:
        # finished runs whose outputs were all deleted (or never made) are just noise;
        # running and failed runs stay visible
        stmt = stmt.where(
            or_(
                ExperimentRun.status.in_(("running", "failed")),
                exists().where(GeneratedOutput.run_id == ExperimentRun.id),
            )
        )
    if run_type:
        stmt = stmt.where(
            ExperimentRun.run_type.like(run_type[:-1] + "%")
            if run_type.endswith("*")
            else ExperimentRun.run_type == run_type
        )
    async with get_db_session() as db:
        runs = (
            await db.scalars(
                stmt.order_by(desc(ExperimentRun.start_time), desc(ExperimentRun.id))
                .limit(limit * 3 if model else limit)
                .offset(offset)
            )
        ).all()
        if model:
            runs = [r for r in runs if run_model_id(r.base_parameters) == model][:limit]
        total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
        counts: dict[int, int] = {}
        covers: dict[int, list[int]] = {}
        if runs:
            ids = [r.id for r in runs]
            for run_id, n in (
                await db.execute(
                    select(GeneratedOutput.run_id, func.count())
                    .where(GeneratedOutput.run_id.in_(ids), GeneratedOutput.output_type == "image")
                    .group_by(GeneratedOutput.run_id)
                )
            ).all():
                counts[run_id] = n
            for run_id, out_id in (
                await db.execute(
                    select(GeneratedOutput.run_id, GeneratedOutput.id)
                    .where(GeneratedOutput.run_id.in_(ids), GeneratedOutput.output_type == "image")
                    .order_by(GeneratedOutput.id)
                )
            ).all():
                covers.setdefault(run_id, [])
                if len(covers[run_id]) < 4:
                    covers[run_id].append(out_id)
    items = [
        {
            "run_uuid": r.run_uuid,
            "run_type": r.run_type,
            "status": r.status,
            "started": _iso(r.start_time),
            "finished": _iso(r.end_time),
            "model_id": run_model_id(r.base_parameters),
            "prompt": run_prompt(r.base_parameters),
            "count": counts.get(r.id, 0),
            "cover_ids": covers.get(r.id, []),
            "error": r.user_notes[7:] if (r.user_notes or "").startswith("Error: ") else None,
        }
        for r in runs
    ]
    return {"items": items, "total": total or 0, "offset": offset}


async def facets() -> dict[str, Any]:
    async with get_db_session() as db:
        images = GeneratedOutput.output_type == "image"
        models = (
            await db.execute(
                select(GeneratedOutput.model_id, func.count())
                .where(images)
                .group_by(GeneratedOutput.model_id)
                .order_by(desc(func.count()))
            )
        ).all()
        types = (
            await db.execute(
                select(ExperimentRun.run_type, func.count(GeneratedOutput.id))
                .join(GeneratedOutput, GeneratedOutput.run_id == ExperimentRun.id)
                .where(images)
                .group_by(ExperimentRun.run_type)
            )
        ).all()
        tags = (
            await db.execute(
                select(OutputTag.tag, func.count())
                .group_by(OutputTag.tag)
                .order_by(desc(func.count()), OutputTag.tag)
                .limit(300)
            )
        ).all()
        total = await db.scalar(select(func.count()).where(images))
        favorites = await db.scalar(
            select(func.count()).where(images, GeneratedOutput.favorite.is_(True))
        )
    return {
        "models": [{"model_id": m or "", "count": n} for m, n in models],
        "run_types": [{"run_type": t, "count": n} for t, n in types],
        "tags": [{"tag": t, "count": n} for t, n in tags],
        "total": total or 0,
        "favorites": favorites or 0,
    }


# --------------------------------------------------------------------------- edits
def _clean_tags(tags: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for t in tags:
        t = " ".join(str(t).split())[:80]
        if t:
            seen.setdefault(t, None)
    return list(seen)


async def update_output(
    output_id: int,
    *,
    rating: int | None = None,
    clear_rating: bool = False,
    favorite: bool | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    async with get_db_session() as db:
        o = await _load(db, output_id)
        if clear_rating:
            o.user_rating = None
        elif rating is not None:
            if not 0 <= rating <= 5:
                raise ValueError("rating must be 0-5")
            o.user_rating = rating or None
        if favorite is not None:
            o.favorite = favorite
        if notes is not None:
            o.notes = notes.strip() or None
        if tags is not None:
            wanted = _clean_tags(tags)
            keep = [t for t in o.tags if t.source != "manual" or t.tag in wanted]
            have = {t.tag for t in keep if t.source == "manual"}
            o.tags = keep + [OutputTag(tag=t, source="manual") for t in wanted if t not in have]
        await db.flush()
        await db.refresh(o, ["tags"])
        return output_view(o, detail=True)


async def bulk(ids: list[int], action: str, value: Any = None) -> dict[str, Any]:
    if not ids:
        return {"changed": 0}
    if action == "delete":
        return {"changed": await delete_outputs(ids)}
    async with get_db_session() as db:
        rows = (
            await db.scalars(
                select(GeneratedOutput)
                .where(GeneratedOutput.id.in_(ids))
                .options(selectinload(GeneratedOutput.tags))
            )
        ).all()
        for o in rows:
            if action == "rate":
                rating = int(value or 0)
                if not 0 <= rating <= 5:
                    raise ValueError("rating must be 0-5")
                o.user_rating = rating or None
            elif action == "favorite":
                o.favorite = bool(value)
            elif action == "add_tags":
                have = {t.tag for t in o.tags if t.source == "manual"}
                o.tags.extend(
                    OutputTag(tag=t, source="manual")
                    for t in _clean_tags(value or [])
                    if t not in have
                )
            elif action == "remove_tags":
                drop = set(_clean_tags(value or []))
                o.tags = [t for t in o.tags if not (t.source == "manual" and t.tag in drop)]
            else:
                raise ValueError(f"Unknown action '{action}'")
    return {"changed": len(rows)}


def _trash(path: Path, run_uuid: str | None) -> None:
    if not path.is_file():
        return
    target_dir = get_outputs_dir() / ".trash" / (run_uuid or "misc")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    if target.exists():
        target = target_dir / f"{path.stem}-{uuid.uuid4().hex[:6]}{path.suffix}"
    shutil.move(str(path), target)


def _drop_thumbs(output_id: int) -> None:
    for size in THUMB_SIZES:
        (thumb_dir() / f"{output_id}-{size}.webp").unlink(missing_ok=True)


async def delete_outputs(ids: list[int]) -> int:
    """Move the files to <outputs>/.trash/<run>/ and remove the images from the gallery."""
    async with get_db_session() as db:
        rows = (
            await db.scalars(
                select(GeneratedOutput)
                .where(GeneratedOutput.id.in_(ids))
                .options(selectinload(GeneratedOutput.run), selectinload(GeneratedOutput.tags))
            )
        ).all()
        files = [(o.id, Path(o.file_path), o.run.run_uuid if o.run else None) for o in rows]
        for o in rows:
            await db.delete(o)
    await asyncio.to_thread(_trash_files, files)
    return len(rows)


def _trash_files(files: list[tuple[int, Path, str | None]]) -> None:
    for output_id, path, run_uuid in files:
        try:
            _trash(path, run_uuid)
        except OSError as e:
            logger.warning("Could not move %s to the trash: %s", path, e)
        _drop_thumbs(output_id)


async def delete_run(run_uuid: str) -> int:
    async with get_db_session() as db:
        run = await db.scalar(select(ExperimentRun).where(ExperimentRun.run_uuid == run_uuid))
        if run is None:
            raise KeyError(f"No run {run_uuid}")
        ids = list(
            await db.scalars(select(GeneratedOutput.id).where(GeneratedOutput.run_id == run.id))
        )
    removed = await delete_outputs(ids)
    async with get_db_session() as db:
        await db.execute(delete(ExperimentRun).where(ExperimentRun.run_uuid == run_uuid))
    await asyncio.to_thread(_trash_folder, run_uuid)
    return removed


def _trash_folder(run_uuid: str) -> None:
    """Move what is left of a run folder (reports, maps) to the trash."""
    folder = get_outputs_dir() / run_uuid
    if not folder.is_dir():
        return
    target = get_outputs_dir() / ".trash" / run_uuid
    target.mkdir(parents=True, exist_ok=True)
    for f in folder.iterdir():
        shutil.move(str(f), target / f.name)
    folder.rmdir()


# --------------------------------------------------------------------------- thumbnails
def thumb_dir() -> Path:
    d = get_outputs_dir() / ".thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_thumb(source: Path, output_id: int, size: int) -> Path:
    """WebP thumbnail (longest side `size`), cached next to the outputs; blocking."""
    size = min(THUMB_SIZES, key=lambda s: abs(s - size))
    target = thumb_dir() / f"{output_id}-{size}.webp"
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    with Image.open(source) as im:
        im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        im.save(target, "WEBP", quality=82, method=4)
    return target


# --------------------------------------------------------------------------- Lab runs
_MAIN_IMAGE_KEYS = ("image_base64", "base_image_base64")


def _ext(data: bytes) -> str:
    return ".webp" if data[:4] == b"RIFF" else ".png"


async def record_lab_run(tool: str, request: Any, response: Any) -> dict[str, Any]:
    """Save a Lab result as a run: the image goes to the gallery, maps and the JSON report
    into the run folder. Returns {"run_uuid", "outputs": [{"id"}]}."""
    started = time.time()
    run_uuid = str(uuid.uuid4())
    req = request.model_dump(mode="json")
    data = response.model_dump(mode="json")
    base_params = dict(req.get("base_request") or {})
    if "seed" in data:
        base_params["seed"] = data["seed"]
    folder, main = await asyncio.to_thread(_write_lab_files, tool, run_uuid, req, data, base_params)

    output_ids: list[int] = []
    async with get_db_session() as db:
        run = ExperimentRun(
            run_uuid=run_uuid,
            run_type=f"lab_{tool}",
            start_time=datetime.datetime.fromtimestamp(started, tz=datetime.UTC),
            end_time=datetime.datetime.now(tz=datetime.UTC),
            status="completed",
            base_parameters=req,
        )
        db.add(run)
        await db.flush()
        rows = [
            GeneratedOutput(
                run_id=run.id,
                output_type="report",
                file_path=str((folder / "report.json").resolve()),
                parameters={"tool": tool},
            )
        ]
        if main is not None:
            file, img = main
            params = {**base_params, "lab_tool": tool}
            rows.insert(
                0,
                GeneratedOutput(
                    run_id=run.id,
                    output_type="image",
                    file_path=str(file.resolve()),
                    parameters=params,
                    **{**output_fields(params), "width": img.width, "height": img.height},
                ),
            )
        db.add_all(rows)
        await db.flush()
        output_ids = [r.id for r in rows if r.output_type == "image"]
    return {"run_uuid": run_uuid, "outputs": [{"id": i} for i in output_ids]}


def _write_lab_files(
    tool: str, run_uuid: str, req: dict, data: dict, base_params: dict
) -> tuple[Path, tuple[Path, Image.Image] | None]:
    """Write the main image (with metadata), maps and report.json of a Lab result."""
    folder = get_outputs_dir() / run_uuid
    folder.mkdir(parents=True, exist_ok=True)
    main: tuple[Path, Image.Image] | None = None
    counter = 0

    def externalize(node: Any, top: bool) -> Any:
        nonlocal main, counter
        if isinstance(node, list):
            return [externalize(v, False) for v in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for k, v in node.items():
            if not (isinstance(v, str) and k.endswith("_base64")):
                out[k] = externalize(v, False)
                continue
            raw = base64.b64decode(v)
            stem = k.removesuffix("_base64")
            if main is None and top and k in _MAIN_IMAGE_KEYS:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                file = folder / f"lab_{tool}_seed{base_params.get('seed', 'x')}.png"
                img.save(file, "PNG", pnginfo=png_info({**base_params, "lab_tool": tool}))
                main = (file, img)
                out[stem + "_file"] = file.name
            else:
                counter += 1
                name = f"maps/{counter:03d}_{stem}{_ext(raw)}"
                (folder / "maps").mkdir(exist_ok=True)
                (folder / name).write_bytes(raw)
                out[stem + "_file"] = name
        return out

    report = externalize(data, True)
    (folder / "report.json").write_text(
        json.dumps({"tool": tool, "request": req, "result": report}, indent=2), encoding="utf-8"
    )
    return folder, main
