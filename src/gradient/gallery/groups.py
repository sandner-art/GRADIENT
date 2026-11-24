# src/gradient/gallery/groups.py
"""Job groups, tags and origin from POST /queue.

Jobs queued with the same `group` save their images into one gallery run (run type
`group`, one output folder), so a sweep from another app (e.g. an ArtAgents LoRA test)
shows as one run instead of hundreds. The run's uuid is derived from the group name,
so the group survives restarts and needs no table of its own.

`tags` become gallery tags (source `api`) on every image of the job; `origin` (where the
job came from: app, run, item...) is kept with each image.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from gradient.database.models import ExperimentRun, GeneratedOutput, OutputTag
from gradient.database.session import get_db_session

logger = logging.getLogger(__name__)

RUN_TYPE = "group"
TAG_SOURCE = "api"
_NAMESPACE = uuid.UUID("3f1c5a52-7d0e-4c55-9b7e-8e2f4a9d6c10")


def _now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)


def group_run_uuid(group: str) -> str:
    """The gallery run of a group (same name, same run)."""
    return str(uuid.uuid5(_NAMESPACE, group))


async def open_run(group: str, model_id: str | None) -> tuple[str, int | None]:
    """Find or create the group's gallery run and mark it running; returns (run_uuid, db id)."""
    run_uuid = group_run_uuid(group)
    try:
        async with get_db_session() as db:
            run = await db.scalar(select(ExperimentRun).where(ExperimentRun.run_uuid == run_uuid))
            if run is None:
                run = ExperimentRun(
                    run_uuid=run_uuid,
                    run_type=RUN_TYPE,
                    start_time=_now(),
                    status="running",
                    base_parameters={"group": group, "model_id": model_id},
                )
                db.add(run)
            else:
                run.status = "running"
            await db.flush()
            return run_uuid, run.id
    except Exception as e:  # the images are still made, just not grouped in the gallery
        logger.error("Group %r: could not open its gallery run: %s", group, e)
        return run_uuid, None


async def close_run(run_id: int | None, status_if_empty: str = "completed") -> None:
    """After each job: the run is completed once it has images (else failed or cancelled)."""
    if run_id is None:
        return
    try:
        async with get_db_session() as db:
            run = await db.get(ExperimentRun, run_id)
            if run is None:
                return
            count = await db.scalar(select(func.count()).where(GeneratedOutput.run_id == run_id))
            run.status = "completed" if count else status_if_empty
            run.end_time = _now()
    except Exception as e:
        logger.error("Group run %s: could not update its status: %s", run_id, e)


async def label_outputs(output_ids: list[int | None], meta: dict[str, Any]) -> None:
    """Add the job's tags and origin to its images (never fails the job)."""
    ids = [i for i in output_ids if i]
    tags = list(meta.get("tags") or [])
    extra = {k: meta[k] for k in ("group", "origin") if meta.get(k)}
    if not ids or not (tags or extra):
        return
    try:
        async with get_db_session() as db:
            rows = (
                await db.scalars(
                    select(GeneratedOutput)
                    .where(GeneratedOutput.id.in_(ids))
                    .options(selectinload(GeneratedOutput.tags))
                )
            ).all()
            for output in rows:
                have = {t.tag for t in output.tags}
                output.tags.extend(
                    OutputTag(tag=t, source=TAG_SOURCE) for t in tags if t not in have
                )
                if extra:
                    output.output_metadata = {**(output.output_metadata or {}), **extra}
    except Exception as e:
        logger.error("Could not tag outputs %s: %s", ids, e)
