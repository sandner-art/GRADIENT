# src/gradient/gallery/sweeps.py
"""Test sweeps from ArtAgents' LoRA Lab, as files: import, queue, write a results folder.

ArtAgents exports a sweep file (`<run>.lora-sweep.json`, format `artagents.lora-sweep` v1)
when it can't reach GRADIENT directly. Importing it here:

1. checks the file and finds each LoRA: the file named in the sweep, a file you pick
   (`lora_map`), or any file in the LoRA folders with the same sha256. A file whose hash
   differs from the sweep's is refused, so the test really uses the trained LoRA;
2. queues one text-to-image job per image (one gallery run for the sweep, tagged
   `ArtAgents/LoRA test/<name>`), filling steps/CFG the sweep leaves open from the
   model's recommended settings;
3. when the last job is done (or on request), writes a results folder under
   `<outputs>/sweeps/`: the images plus `artagents-results.json`, which ArtAgents
   imports as a LoRA Lab run (results match jobs by index; failed jobs carry their error).

Records live in `<data>/sweeps/<id>.json`, with the exact sweep text (its sha256 ties
the results to the file ArtAgents exported).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from gradient.core.config import get_data_dir, get_outputs_dir
from gradient.core.jobs import FINISHED, Job, job_queue
from gradient.database.models import QueueJob
from gradient.database.session import get_db_session
from gradient.model_manager.inventory import find_model_path, iter_type_dirs, list_local_models

logger = logging.getLogger(__name__)

FORMAT, RESULTS_FORMAT, VERSION = "artagents.lora-sweep", "artagents.lora-sweep-results", 1
RESULTS_FILE = "artagents-results.json"
SWEEP_COPY = "sweep.lora-sweep.json"
MAX_SWEEP_CHARS = 10 * 1024 * 1024
MAX_JOBS = 5000
OK_STATUSES = ("ok", "found_by_hash", "unverified")
_hash_cache: dict[str, tuple[float, int, str]] = {}


class SweepError(ValueError):
    pass


# --------------------------------------------------------------------------- the file
class SweepRun(BaseModel):
    name: str = Field(default="sweep", max_length=200)
    folder: str | None = Field(default=None, max_length=200)
    validation_set: str | None = None
    validation_version: int | None = None
    trigger: str | None = None


class SweepModel(BaseModel):
    model_id: str = Field(min_length=1, max_length=500)
    scheduler_id: str | None = None
    quantization: str | None = None
    vae_id: str | None = None
    guidance: str | float | None = None


class SweepLora(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=2000)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SweepJob(BaseModel):
    index: int = Field(ge=0)
    prompt_index: int = Field(default=0, ge=0)
    bucket: str = ""
    prompt: str = Field(max_length=10000)
    negative: str = Field(default="", max_length=10000)
    seed: int = Field(ge=0, lt=2**32)
    width: int = Field(ge=8, le=8192)
    height: int = Field(ge=8, le=8192)
    steps: int | None = Field(default=None, ge=1, le=1000)
    cfg: float | None = None
    lora: str | None = None
    strength: float = 0.0
    condition: str = ""


class SweepFile(BaseModel):
    format: str
    version: int
    run: SweepRun = Field(default_factory=SweepRun)
    model: SweepModel
    loras: list[SweepLora] = Field(default_factory=list, max_length=200)
    jobs: list[SweepJob] = Field(min_length=1, max_length=MAX_JOBS)


def parse(text: str) -> tuple[SweepFile, str]:
    """The sweep and the sha256 of its exact text (UTF-8)."""
    if len(text) > MAX_SWEEP_CHARS:
        raise SweepError("The sweep file is too large.")
    try:
        raw = json.loads(text)
    except ValueError as e:
        raise SweepError(f"Not a JSON file: {e}") from e
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise SweepError("Not an ArtAgents test sweep file (format artagents.lora-sweep).")
    if raw.get("version") != VERSION:
        raise SweepError(
            f"Sweep file version {raw.get('version')!r}; this GRADIENT reads version {VERSION}."
        )
    try:
        sweep = SweepFile.model_validate(raw)
    except ValidationError as e:
        first = e.errors(include_url=False)[0]
        where = ".".join(str(p) for p in first["loc"])
        raise SweepError(f"The sweep file is not valid ({where}: {first['msg']}).") from e
    keys = {lo.key for lo in sweep.loras}
    unknown = sorted({j.lora for j in sweep.jobs if j.lora and j.lora not in keys})
    if unknown:
        raise SweepError(f"Jobs use LoRA keys the file doesn't list: {', '.join(unknown)}")
    if len({j.index for j in sweep.jobs}) != len(sweep.jobs):
        raise SweepError("Job indexes repeat in the sweep file.")
    return sweep, hashlib.sha256(text.encode("utf-8")).hexdigest()


def sweep_id(sha: str) -> str:
    return sha[:16]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:60] or "sweep"


def run_label(sweep: SweepFile) -> str:
    return sweep.run.folder or sweep.run.name


def group_name(sweep: SweepFile) -> str:
    """Same group as ArtAgents' direct sweeps: one gallery run per test run."""
    return f"ArtAgents · LoRA test · {run_label(sweep)}"


# --------------------------------------------------------------------------- LoRAs
def file_sha256(path: Path) -> str:
    stat = path.stat()
    key = str(path)
    hit = _hash_cache.get(key)
    if hit and hit[0] == stat.st_mtime and hit[1] == stat.st_size:
        return hit[2]
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    _hash_cache[key] = (stat.st_mtime, stat.st_size, digest.hexdigest())
    return _hash_cache[key][2]


def _lora_files(limit: int = 4000) -> list[Path]:
    files: list[Path] = []
    for folder in iter_type_dirs("lora"):
        if folder.is_dir():
            files.extend(sorted(folder.rglob("*.safetensors"))[: limit - len(files)])
    return files


def _resolve(candidate: str) -> Path | None:
    try:
        return find_model_path(candidate, "lora")
    except Exception:
        return None


def resolve_loras(sweep: SweepFile, lora_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Find each LoRA here. Blocking (hashes files): run it in a thread."""
    lora_map = {k: v.strip() for k, v in (lora_map or {}).items() if v and v.strip()}
    out = []
    for lora in sweep.loras:
        picked = lora_map.get(lora.key)
        candidates = [picked] if picked else [lora.name, Path(lora.name.replace("\\", "/")).name]
        item: dict[str, Any] = {
            "key": lora.key,
            "name": lora.name,
            "sha256": lora.sha256,
            "resolved": None,
            "status": "missing",
            "message": "Not found here: pick the file.",
        }
        found = next((p for p in (_resolve(c) for c in candidates if c) if p is not None), None)
        if found is not None and found.is_file() and lora.sha256:
            if file_sha256(found) == lora.sha256:
                item.update(resolved=str(found), status="ok", message="Same file (sha256 matches).")
            else:
                item.update(
                    status="hash_mismatch",
                    message=f"{found.name} is a different file (its sha256 doesn't match the sweep's).",
                )
        elif found is not None and lora.sha256:
            item.update(
                status="hash_mismatch", message="A folder can't be checked against the sha256."
            )
        elif found is not None:
            item.update(
                resolved=str(found),
                status="unverified",
                message="Found by name; the sweep has no sha256 to check it against.",
            )
        if item["status"] == "missing" and lora.sha256 and not picked:
            match = next((p for p in _lora_files() if file_sha256(p) == lora.sha256), None)
            if match is not None:
                item.update(
                    resolved=str(match),
                    status="found_by_hash",
                    message=f"Found as {match.name} (same sha256).",
                )
        out.append(item)
    return out


# --------------------------------------------------------------------------- preview / queue
def model_defaults(model_id: str) -> dict[str, Any]:
    for kind in ("diffusers", "checkpoints"):
        for m in list_local_models(kind, with_sizes=False):
            if m.name == model_id:
                return m.defaults.model_dump(exclude_none=True) if m.defaults else {}
    return {}


def _model_found(model_id: str) -> bool:
    return any(
        m.name == model_id
        for kind in ("diffusers", "checkpoints")
        for m in list_local_models(kind, with_sizes=False)
    )


def preview(text: str, lora_map: dict[str, str] | None = None) -> dict[str, Any]:
    """What importing would do. Blocking: run it in a thread."""
    sweep, sha = parse(text)
    loras = resolve_loras(sweep, lora_map)
    found = _model_found(sweep.model.model_id)
    problems = []
    if not found:
        problems.append(f"Model {sweep.model.model_id!r} is not installed here.")
    problems += [
        f"LoRA {lo['key']}: {lo['message']}" for lo in loras if lo["status"] not in OK_STATUSES
    ]
    waiting = _waiting_jobs(sweep_id(sha))
    if waiting:
        problems.append(f"This sweep is already in the queue ({waiting} job(s) not finished).")
    defaults = model_defaults(sweep.model.model_id) if found else {}
    return {
        "id": sweep_id(sha),
        "sha256": sha,
        "name": sweep.run.name,
        "run": run_label(sweep),
        "model_id": sweep.model.model_id,
        "model_found": found,
        "model_defaults": defaults,
        "options": sweep.model.model_dump(exclude={"model_id"}, exclude_none=True),
        "images": len(sweep.jobs),
        "prompts": len({j.prompt_index for j in sweep.jobs}),
        "conditions": len({(j.lora, j.strength) for j in sweep.jobs}),
        "loras": loras,
        "ready": not problems,
        "problems": problems,
        "results_dir": str(results_dir_for(sweep, sha)),
    }


def results_dir_for(sweep: SweepFile, sha: str) -> Path:
    return get_outputs_dir() / "sweeps" / f"{_slug(run_label(sweep))}-{sha[:8]}"


def _record_path(sid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16}", sid or ""):
        raise KeyError(f"No sweep {sid}")
    return get_data_dir() / "sweeps" / f"{sid}.json"


def load_record(sid: str) -> dict[str, Any]:
    path = _record_path(sid)
    if not path.is_file():
        raise KeyError(f"No sweep {sid}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_record(record: dict[str, Any]) -> None:
    path = _record_path(record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _waiting_jobs(sid: str) -> int:
    return sum(
        1
        for j in job_queue.jobs.values()
        if j.status not in FINISHED and (j.meta.get("origin") or {}).get("sweep") == sid
    )


def _payload(sweep: SweepFile, job: SweepJob, loras: dict[str, str], defaults: dict) -> dict:
    m = sweep.model
    payload: dict[str, Any] = {
        "model_id": m.model_id,
        "prompt": job.prompt,
        "negative_prompt": job.negative,
        "width": max(8, job.width // 8 * 8),
        "height": max(8, job.height // 8 * 8),
        "seed": job.seed,
        "num_outputs": 1,
        "loras": [{"name": loras[job.lora], "weight": job.strength}] if job.lora else [],
    }
    steps = job.steps or defaults.get("steps")
    if steps:
        payload["num_inference_steps"] = int(steps)
    guidance: Any = m.guidance
    if isinstance(guidance, str):
        try:
            guidance = float(guidance)
        except ValueError:
            pass  # a schedule: 'linear(9,3)', '9@0,4@20'
    if guidance in (None, ""):
        guidance = job.cfg if job.cfg is not None else defaults.get("guidance_scale")
    if guidance is not None:
        payload["guidance_scale"] = guidance
    for key, value in (
        ("scheduler_id", m.scheduler_id),
        ("quantization", m.quantization),
        ("vae_id", m.vae_id),
    ):
        if value:
            payload[key] = value
    return payload


async def queue_sweep(text: str, lora_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Queue every image of the sweep as one group; returns the preview plus the group."""
    from gradient.api.job_kinds import validate_payload
    from gradient.gallery.groups import group_run_uuid

    info = await asyncio.to_thread(preview, text, lora_map)
    if not info["ready"]:
        raise SweepError(" ".join(info["problems"]))
    sweep, sha = parse(text)
    sid = info["id"]
    loras = {lo["key"]: lo["resolved"] for lo in info["loras"]}
    defaults = info["model_defaults"]
    name, run, group = sweep.run.name, run_label(sweep), group_name(sweep)
    # baseline first, then one LoRA at a time, so each LoRA loads once
    order = sorted(
        sweep.jobs, key=lambda j: (j.lora is not None, j.lora or "", j.strength, j.index)
    )
    payloads = []
    for job in order:
        try:
            payloads.append(
                (job, validate_payload("text_to_image", _payload(sweep, job, loras, defaults)))
            )
        except (ValidationError, ValueError) as e:
            raise SweepError(f"Job {job.index} can't be rendered: {e}") from e
    record = {
        "id": sid,
        "sha256": sha,
        "name": name,
        "run": run,
        "group": group,
        "run_uuid": group_run_uuid(group),
        "model_id": sweep.model.model_id,
        "imported": time.time(),
        "results_dir": info["results_dir"],
        "written": None,
        "loras": info["loras"],
        "jobs": {},
        "sweep_text": text,
    }
    await asyncio.to_thread(_save_record, record)  # before queueing: a fast job may finish at once
    for job, payload in payloads:
        queued = await job_queue.submit(
            "text_to_image",
            payload,
            f"ArtAgents · LoRA test · {name} · {job.condition or job.lora or 'baseline'} · seed {job.seed}",
            meta={
                "group": group,
                "tags": [f"ArtAgents/LoRA test/{name}"[:80]],
                "origin": {
                    "app": "ArtAgents",
                    "run": run,
                    "sweep": sid,
                    "job_index": job.index,
                    "prompt_index": job.prompt_index,
                    "bucket": job.bucket,
                    "condition": job.condition,
                },
            },
        )
        record["jobs"][str(job.index)] = queued.id
    await asyncio.to_thread(_save_record, record)
    logger.info("Queued test sweep %s (%s): %d jobs", sid, name, len(payloads))
    if not _waiting_jobs(sid):  # everything finished while queueing
        _schedule_write(sid)
    return {**info, "group": group, "run_uuid": record["run_uuid"], "queued": len(payloads)}


# --------------------------------------------------------------------------- results
async def _job_states(job_ids: list[str]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    async with get_db_session() as db:
        rows = (await db.scalars(select(QueueJob).where(QueueJob.id.in_(job_ids)))).all()
        for r in rows:
            states[r.id] = {"status": r.status, "result": r.result, "error": r.error}
    for jid in job_ids:  # the live queue is newer than the database
        job = job_queue.jobs.get(jid)
        if job is not None:
            states[jid] = {"status": job.status, "result": job.result, "error": job.error}
    return states


async def status(sid: str) -> dict[str, Any]:
    record = await asyncio.to_thread(load_record, sid)
    states = await _job_states(list(record["jobs"].values()))
    counts: dict[str, int] = {}
    for jid in record["jobs"].values():
        s = (states.get(jid) or {}).get("status", "missing")
        counts[s] = counts.get(s, 0) + 1
    finished = sum(n for s, n in counts.items() if s in FINISHED or s == "missing")
    return {
        **{k: v for k, v in record.items() if k not in ("sweep_text", "jobs")},
        "images": len(record["jobs"]),
        "counts": counts,
        "done": finished == len(record["jobs"]),
    }


async def list_sweeps() -> list[dict[str, Any]]:
    folder = get_data_dir() / "sweeps"
    if not folder.is_dir():
        return []
    items = []
    for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        try:
            items.append(await status(path.stem))
        except (KeyError, ValueError, OSError):
            continue
    return items


def _file_name(job: SweepJob) -> str:
    cond = re.sub(r"[^A-Za-z0-9.@_-]+", "_", job.condition or job.lora or "baseline")
    return f"{job.index:04d}_p{job.prompt_index:02d}_s{job.seed}_{cond}.png"


async def write_results(sid: str) -> dict[str, Any]:
    """Write (or rewrite) the results folder from the jobs' current state."""
    from gradient.core.version import app_version
    from gradient.gallery.store import output_path

    record = await asyncio.to_thread(load_record, sid)
    sweep, sha = parse(record["sweep_text"])
    states = await _job_states(list(record["jobs"].values()))
    folder = Path(record["results_dir"])
    await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
    results, copies = [], []
    counts = {"images": 0, "errors": 0}
    for job in sorted(sweep.jobs, key=lambda j: j.index):
        state = states.get(record["jobs"].get(str(job.index), ""), {})
        result = state.get("result") or {}
        entry: dict[str, Any] = {
            "index": job.index,
            "file": None,
            "seed": job.seed,
            "output_id": None,
            "run_uuid": result.get("run_uuid"),
            "error": None,
        }
        outputs = result.get("outputs") or []
        if state.get("status") == "completed" and outputs and outputs[0].get("id"):
            entry.update(output_id=outputs[0]["id"], seed=outputs[0].get("seed", job.seed))
            entry["loras_skipped"] = list(result.get("loras_skipped") or [])
            try:
                source = await output_path(outputs[0]["id"])
                entry["file"] = _file_name(job)
                copies.append((source, folder / entry["file"]))
            except (KeyError, FileNotFoundError) as e:
                entry["error"] = f"The image is gone from GRADIENT's gallery: {e}"
        elif state.get("status") in ("failed", "cancelled"):
            entry["error"] = state.get("error") or f"Job {state['status']}."
        else:
            entry["error"] = "Not rendered yet." if state else "The job is no longer known."
        counts["images" if entry["file"] else "errors"] += 1
        results.append(entry)

    def _write() -> None:
        for source, target in copies:
            shutil.copyfile(source, target)
        (folder / SWEEP_COPY).write_text(record["sweep_text"], encoding="utf-8")
        data = {
            "format": RESULTS_FORMAT,
            "version": VERSION,
            "sweep_sha256": sha,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "renderer": {
                "app": "GRADIENT",
                "version": app_version(),
                "model_id": sweep.model.model_id,
                "gallery_run": record.get("run_uuid"),
            },
            "results": results,
        }
        tmp = folder / (RESULTS_FILE + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(folder / RESULTS_FILE)

    await asyncio.to_thread(_write)
    record["written"] = time.time()
    await asyncio.to_thread(_save_record, record)
    logger.info("Sweep %s: results folder %s (%s)", sid, folder, counts)
    return {"results_dir": str(folder), **counts}


_writing: set[asyncio.Task] = set()


async def on_job_finished(job: Job) -> None:
    """Queue hook: when the last job of an imported sweep finishes, write its results folder."""
    sid = (job.meta.get("origin") or {}).get("sweep")
    if not sid or _waiting_jobs(sid):
        return
    try:
        record = await asyncio.to_thread(load_record, sid)
    except (KeyError, ValueError, OSError):
        return  # a direct sweep from ArtAgents, not an imported one
    if job.id in record["jobs"].values():
        _schedule_write(sid)


def _schedule_write(sid: str) -> None:
    task = asyncio.create_task(write_results(sid))
    _writing.add(task)
    task.add_done_callback(_writing.discard)
