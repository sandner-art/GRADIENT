# src/gradient/model_manager/downloader.py
"""Background model downloads (Hugging Face repos/files and direct URLs).

* Hugging Face downloads run in a child process (see hf_worker.py) so they can be
  cancelled and resumed; progress arrives as JSON lines.
* Direct URL downloads (e.g. Civitai) stream with httpx and resume via HTTP Range.
* Everything is first written to "<type dir>/.staging/" and moved into place when
  complete, so half-finished models never show up in the model lists.
"""

import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import httpx

from gradient.core.config import PROJECT_ROOT, get_primary_model_dir, get_setting

from . import manifest
from .catalog import get_catalog_entry
from .inventory import WEIGHT_SUFFIXES, manifest_key
from .schemas import CatalogEntry, DownloadJob, DownloadRequest, ModelType

logger = logging.getLogger(__name__)

STAGING_DIRNAME = ".staging"
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
# Preferred GGUF quantizations when a repo holds many (best quality/size trade-off first)
GGUF_PREFERENCE = ("Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K", "Q8_0", "Q4_0", "IQ4_XS")


class DownloadError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class DownloadPlan:
    kind: Literal["hf_snapshot", "hf_files", "url"]
    model_type: ModelType
    name: str  # final folder/file name inside the type dir
    display: str
    repo_id: str | None = None
    revision: str | None = None
    files: list[str] | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    token: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def type_dir(self) -> Path:
        return get_primary_model_dir(self.model_type)

    @property
    def target(self) -> Path:
        return self.type_dir / self.name


# --------------------------------------------------------------------------- helpers
def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return cleaned[:150] or "model"


def parse_hf_reference(ref: str) -> tuple[str, str | None, str | None]:
    """'owner/repo' or a huggingface.co URL -> (repo_id, revision, file_path)."""
    ref = ref.strip()
    if ref.startswith(("http://", "https://")):
        parsed = urlparse(ref)
        if parsed.hostname not in _HF_HOSTS:
            raise ValueError(f"Not a Hugging Face URL: {ref}")
        parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
        if parts and parts[0] in ("models",):
            parts = parts[1:]
        if parts and parts[0] in ("datasets", "spaces"):
            raise ValueError("Only model repositories are supported")
        if len(parts) < 2:
            raise ValueError(f"Cannot parse repository from URL: {ref}")
        repo_id = f"{parts[0]}/{parts[1]}"
        if len(parts) >= 5 and parts[2] in ("blob", "resolve"):
            return repo_id, parts[3], "/".join(parts[4:])
        if len(parts) >= 4 and parts[2] == "tree":
            return repo_id, parts[3], None
        return repo_id, None, None
    if not _REPO_ID_RE.match(ref):
        raise ValueError(f"Invalid Hugging Face repo id: {ref!r} (expected 'owner/name')")
    return ref, None, None


def pick_gguf(filenames: list[str]) -> str | None:
    ggufs = [f for f in filenames if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    if not ggufs:
        return None
    for quant in GGUF_PREFERENCE:
        for f in ggufs:
            if quant.lower() in Path(f).name.lower():
                return f
    return sorted(ggufs)[0]


def _hf_token(explicit: str | None) -> str | None:
    return explicit or get_setting("downloads.hf_token") or None


def _civitai_token() -> str | None:
    return os.environ.get("CIVITAI_TOKEN") or get_setting("downloads.civitai_token") or None


def infer_type_from_repo_files(repo_id: str, filenames: list[str]) -> ModelType | None:
    names = {f.split("/")[-1] for f in filenames}
    root = {f for f in filenames if "/" not in f}
    lower_repo = repo_id.lower()
    if "model_index.json" in root:
        return "diffusers"
    if any(k in lower_repo for k in ("dinov2", "dinov3", "siglip")) and "config.json" in root:
        return "vision"
    if any(f.lower().endswith(".gguf") for f in filenames):
        return "llm"
    if "adapter_config.json" in names or "lora" in lower_repo:
        return "lora"
    if "controlnet" in lower_repo and "config.json" in root:
        return "controlnet"
    if "vae" in lower_repo and "config.json" in root:
        return "vae"
    if "embedding" in lower_repo or "textual" in lower_repo:
        return "embeddings"
    root_weights = [f for f in root if Path(f).suffix.lower() in WEIGHT_SUFFIXES]
    if root_weights:
        return "checkpoints"
    return None


def _disk_check(path: Path, needed_bytes: int | None) -> None:
    if not needed_bytes:
        return
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    reserve = float(get_setting("downloads.min_free_disk_gb", 2)) * 1024**3
    if free - needed_bytes < reserve:
        raise DownloadError(
            "disk_space",
            f"Not enough disk space in {path}: need {needed_bytes / 1024**3:.1f} GB, "
            f"{free / 1024**3:.1f} GB free (keeping {reserve / 1024**3:.0f} GB in reserve).",
        )


def _filename_from_response(resp: httpx.Response) -> str | None:
    cd = resp.headers.get("content-disposition", "")
    match = re.search(r"filename\*\s*=\s*(?:UTF-8'')?([^;]+)", cd, re.I) or re.search(
        r'filename\s*=\s*"?([^";]+)"?', cd, re.I
    )
    if match:
        return Path(unquote(match.group(1).strip().strip('"'))).name
    name = Path(unquote(urlparse(str(resp.url)).path)).name
    return name or None


# --------------------------------------------------------------------------- manager
class DownloadManager:
    def __init__(self) -> None:
        self.jobs: dict[str, DownloadJob] = {}
        self._plans: dict[str, DownloadPlan] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    # ---- public API -------------------------------------------------------
    async def submit(self, request: DownloadRequest) -> DownloadJob:
        plan = await self.resolve(request)
        for other_id, other in self._plans.items():
            if other.target == plan.target and not self.jobs[other_id].done:
                raise ValueError(f"'{plan.name}' is already being downloaded (job {other_id})")
        job = DownloadJob(
            id=uuid.uuid4().hex[:12],
            name=plan.display,
            model_type=plan.model_type,
            source=plan.repo_id or plan.url or "",
            target_path=str(plan.target),
            catalog_id=plan.meta.get("catalog_id"),
            total_bytes=plan.meta.get("expected_bytes"),
        )
        self.jobs[job.id] = job
        self._plans[job.id] = plan
        self._tasks[job.id] = asyncio.create_task(self._run(job, plan), name=f"download-{job.id}")
        logger.info("Queued download %s: %s -> %s", job.id, job.source, plan.target)
        return job

    def list_jobs(self) -> list[DownloadJob]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> DownloadJob | None:
        return self.jobs.get(job_id)

    async def cancel(self, job_id: str) -> DownloadJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        return job

    def clear_finished(self) -> int:
        finished = [jid for jid, j in self.jobs.items() if j.done]
        for jid in finished:
            self.jobs.pop(jid, None)
            self._plans.pop(jid, None)
            self._tasks.pop(jid, None)
        return len(finished)

    async def wait(self, job_id: str) -> DownloadJob:
        task = self._tasks.get(job_id)
        if task:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return self.jobs[job_id]

    async def shutdown(self) -> None:
        for job_id, task in list(self._tasks.items()):
            if not task.done():
                await self.cancel(job_id)

    # ---- planning -----------------------------------------------------------
    async def resolve(self, req: DownloadRequest) -> DownloadPlan:
        if req.catalog_id:
            entry = get_catalog_entry(req.catalog_id)
            if entry is None:
                raise ValueError(f"Unknown catalog id: {req.catalog_id}")
            return self._plan_from_catalog(entry, req)
        if req.repo_id:
            return await self._plan_from_repo(req, req.repo_id)
        assert req.url
        parsed = urlparse(req.url)
        if parsed.scheme != "https":
            raise ValueError("Only https:// URLs are allowed")
        if parsed.hostname in _HF_HOSTS:
            return await self._plan_from_repo(req, req.url)
        return await self._plan_from_url(req)

    def _plan_from_catalog(self, entry: CatalogEntry, req: DownloadRequest) -> DownloadPlan:
        meta = {
            "catalog_id": entry.id,
            "family": entry.family,
            "pipeline": entry.pipeline,
            "variant": entry.variant,
            "torch_dtype": entry.torch_dtype,
            "license": entry.license,
            "defaults": entry.defaults.model_dump(exclude_none=True) if entry.defaults else None,
            "base_model_family": entry.base_model_family,
            "expected_bytes": int(entry.size_gb * 1e9) if entry.size_gb else None,
        }
        if entry.source == "url":
            return DownloadPlan(
                kind="url",
                model_type=entry.type,
                name=sanitize_name(req.target_name) if req.target_name else "",
                display=entry.name,
                url=entry.url,
                meta=meta,
            )
        assert entry.repo_id
        meta.update(repo_id=entry.repo_id, revision=entry.revision)
        if entry.files:
            first = Path(entry.files[0]).name
            name = (
                sanitize_name(req.target_name)
                if req.target_name and len(entry.files) == 1
                else first
            )
            return DownloadPlan(
                kind="hf_files",
                model_type=entry.type,
                name=name,
                display=entry.name,
                repo_id=entry.repo_id,
                revision=entry.revision,
                files=list(entry.files),
                token=_hf_token(req.token),
                meta=meta,
            )
        return DownloadPlan(
            kind="hf_snapshot",
            model_type=entry.type,
            name=self._unique_dir_name(
                entry.type, req.target_name or entry.repo_id.split("/")[-1], entry.repo_id
            ),
            display=entry.name,
            repo_id=entry.repo_id,
            revision=entry.revision,
            allow_patterns=entry.allow_patterns,
            ignore_patterns=entry.ignore_patterns,
            token=_hf_token(req.token),
            meta=meta,
        )

    async def _plan_from_repo(self, req: DownloadRequest, ref: str) -> DownloadPlan:
        repo_id, url_revision, url_file = parse_hf_reference(ref)
        revision = req.revision or url_revision
        files = list(req.files) if req.files else ([url_file] if url_file else None)
        model_type = req.model_type
        token = _hf_token(req.token)

        if model_type is None or (
            files is None and model_type in ("llm", "checkpoints", "embeddings")
        ):
            filenames = await asyncio.to_thread(self._list_repo_files, repo_id, revision, token)
            if model_type is None:
                if files:
                    suffix = Path(files[0]).suffix.lower()
                    model_type = (
                        "llm"
                        if suffix == ".gguf"
                        else infer_type_from_repo_files(repo_id, filenames)
                    )
                else:
                    model_type = infer_type_from_repo_files(repo_id, filenames)
                if model_type is None:
                    raise ValueError(
                        f"Could not infer what kind of model '{repo_id}' is. Please choose a model type."
                    )
            if files is None and model_type == "llm":
                choice = pick_gguf(filenames)
                if not choice:
                    raise ValueError(f"No .gguf files found in '{repo_id}'")
                files = [choice]
            elif files is None and model_type in ("checkpoints", "embeddings"):
                root_weights = sorted(
                    f
                    for f in filenames
                    if "/" not in f and Path(f).suffix.lower() in WEIGHT_SUFFIXES
                )
                if len(root_weights) != 1:
                    raise ValueError(
                        f"'{repo_id}' has {len(root_weights)} weight files at its root; pick one with `files`."
                    )
                files = root_weights

        meta = {"repo_id": repo_id, "revision": revision}
        display = f"{repo_id}" + (f" / {Path(files[0]).name}" if files else "")
        if files:
            name = (
                sanitize_name(req.target_name)
                if (req.target_name and len(files) == 1)
                else Path(files[0]).name
            )
            return DownloadPlan(
                kind="hf_files",
                model_type=model_type,
                name=name,
                display=display,
                repo_id=repo_id,
                revision=revision,
                files=files,
                token=token,
                meta=meta,
            )
        return DownloadPlan(
            kind="hf_snapshot",
            model_type=model_type,
            name=self._unique_dir_name(
                model_type, req.target_name or repo_id.split("/")[-1], repo_id
            ),
            display=display,
            repo_id=repo_id,
            revision=revision,
            allow_patterns=req.allow_patterns,
            ignore_patterns=req.ignore_patterns,
            token=token,
            meta=meta,
        )

    async def _plan_from_url(self, req: DownloadRequest) -> DownloadPlan:
        assert req.url
        url = req.url
        headers: dict[str, str] = {}
        model_type = req.model_type
        meta: dict[str, Any] = {"url": url}
        host = urlparse(url).hostname or ""
        if host.endswith("civitai.com"):
            url, inferred = await self._civitai_resolve(url)
            model_type = model_type or inferred
            if token := _civitai_token():
                headers["Authorization"] = f"Bearer {token}"
            meta["url"] = url
        if model_type is None:
            raise ValueError("Please choose a model type for direct URL downloads.")
        return DownloadPlan(
            kind="url",
            model_type=model_type,
            name=sanitize_name(req.target_name) if req.target_name else "",
            display=req.target_name or Path(urlparse(url).path).name or url,
            url=url,
            headers=headers,
            meta=meta,
        )

    @staticmethod
    def _list_repo_files(repo_id: str, revision: str | None, token: str | None) -> list[str]:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        try:
            info = HfApi(endpoint=os.environ.get("HF_ENDPOINT") or None).model_info(
                repo_id, revision=revision, token=token
            )
        except GatedRepoError:
            raise ValueError(
                f"'{repo_id}' is gated; accept its licence on huggingface.co and set a token."
            ) from None
        except RepositoryNotFoundError:
            raise ValueError(f"Repository '{repo_id}' not found (or private).") from None
        return [s.rfilename for s in (info.siblings or [])]

    async def _civitai_resolve(self, url: str) -> tuple[str, ModelType | None]:
        """Turn a civitai model page URL into an API download URL and guess the type."""
        parsed = urlparse(url)
        version_id: str | None = None
        if m := re.search(r"/api/download/models/(\d+)", parsed.path):
            version_id = m.group(1)
        elif m := re.search(r"modelVersionId=(\d+)", parsed.query):
            version_id = m.group(1)
        if version_id is None:
            return url, None
        download_url = f"https://civitai.com/api/download/models/{version_id}"
        type_map: dict[str, ModelType] = {
            "Checkpoint": "checkpoints",
            "LORA": "lora",
            "LoCon": "lora",
            "DoRA": "lora",
            "TextualInversion": "embeddings",
            "VAE": "vae",
            "Controlnet": "controlnet",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"https://civitai.com/api/v1/model-versions/{version_id}")
                r.raise_for_status()
                model_kind = (r.json().get("model") or {}).get("type")
                return download_url, type_map.get(model_kind)
        except Exception as e:  # metadata is optional
            logger.debug("Civitai metadata lookup failed: %s", e)
            return download_url, None

    def _unique_dir_name(self, model_type: ModelType, preferred: str, repo_id: str) -> str:
        name = sanitize_name(preferred)
        existing = get_primary_model_dir(model_type) / name
        if existing.exists():
            meta = manifest.get_entry(manifest_key(model_type, name)) or {}
            if meta.get("repo_id") not in (None, repo_id):
                return sanitize_name(repo_id.replace("/", "--"))
        return name

    # ---- execution ---------------------------------------------------------
    def _sem(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(int(get_setting("downloads.max_concurrent", 2)))
            self._semaphore_loop = loop
        return self._semaphore

    async def _run(self, job: DownloadJob, plan: DownloadPlan) -> None:
        try:
            async with self._sem():
                job.status = "resolving"
                job.started_at = dt.datetime.now(dt.UTC)
                if plan.kind == "url":
                    await self._run_url(job, plan)
                else:
                    await self._run_hf(job, plan)
                job.status = "completed"
                logger.info("Download %s completed: %s", job.id, job.target_path)
        except asyncio.CancelledError:
            job.status = "cancelled"
            logger.info("Download %s cancelled", job.id)
        except DownloadError as e:
            job.status, job.error_code, job.error = "failed", e.code, e.message
            logger.warning("Download %s failed (%s): %s", job.id, e.code, e.message)
        except Exception as e:
            job.status, job.error_code, job.error = "failed", "internal", f"{type(e).__name__}: {e}"
            logger.exception("Download %s crashed", job.id)
        finally:
            job.finished_at = dt.datetime.now(dt.UTC)
            job.speed_bps = 0.0

    def _record(self, plan: DownloadPlan, name: str, path: Path) -> None:
        meta = {k: v for k, v in plan.meta.items() if v is not None and k != "expected_bytes"}
        meta.update(
            type=plan.model_type,
            downloaded_at=dt.datetime.now(dt.UTC).isoformat(),
            size_bytes=sum(
                p.stat().st_size for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts
            )
            if path.is_dir()
            else path.stat().st_size,
        )
        manifest.set_entry(manifest_key(plan.model_type, name), meta)

    @staticmethod
    def _update_speed(job: DownloadJob, new_bytes: int, state: dict[str, float]) -> None:
        now = time.monotonic()
        last_t, last_b = state.get("t", now), state.get("b", float(new_bytes))
        if now - last_t >= 0.5:
            inst = (new_bytes - last_b) / (now - last_t)
            job.speed_bps = inst if not job.speed_bps else 0.7 * job.speed_bps + 0.3 * inst
            state["t"], state["b"] = now, float(new_bytes)
        state.setdefault("t", now)
        state.setdefault("b", float(new_bytes))
        job.downloaded_bytes = new_bytes

    async def _run_hf(self, job: DownloadJob, plan: DownloadPlan) -> None:
        type_dir = plan.type_dir
        staging_root = type_dir / STAGING_DIRNAME
        in_place = plan.kind == "hf_snapshot" and plan.target.is_dir()
        if in_place:
            local_dir = plan.target  # update/repair an existing download
        else:
            key = hashlib.sha1(f"{plan.repo_id}|{plan.revision}|{plan.name}".encode()).hexdigest()[
                :12
            ]
            local_dir = staging_root / f"{plan.name}-{key}"
        local_dir.mkdir(parents=True, exist_ok=True)

        spec = {
            "repo_id": plan.repo_id,
            "revision": plan.revision,
            "local_dir": str(local_dir),
            "allow_patterns": plan.files if plan.kind == "hf_files" else plan.allow_patterns,
            "ignore_patterns": None if plan.kind == "hf_files" else plan.ignore_patterns,
        }
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(PROJECT_ROOT / "src"), env.get("PYTHONPATH")])
        )
        env.pop("GRADIENT_HF_TOKEN", None)
        if plan.token:
            env["GRADIENT_HF_TOKEN"] = plan.token

        # A plain subprocess read from worker threads works with every asyncio loop
        # (asyncio subprocesses are unavailable on Windows' selector loop, e.g. with --reload).
        proc = await asyncio.to_thread(
            subprocess.Popen,
            [sys.executable, "-m", "gradient.model_manager.hf_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdin and proc.stdout and proc.stderr
        proc.stdin.write(json.dumps(spec))
        proc.stdin.close()
        stderr_lines: list[str] = []
        stderr_reader = threading.Thread(
            target=lambda: stderr_lines.extend(proc.stderr), daemon=True
        )
        stderr_reader.start()

        speed_state: dict[str, float] = {}
        error: DownloadError | None = None
        done = False
        try:
            while line := await asyncio.to_thread(proc.stdout.readline):
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = msg.get("event")
                if event == "plan":
                    job.status = "downloading"
                    job.total_bytes = int(msg.get("total_bytes") or 0) or None
                    _disk_check(type_dir, job.total_bytes)
                elif event == "progress":
                    self._update_speed(job, int(msg.get("bytes", 0)), speed_state)
                elif event == "done":
                    done = True
                elif event == "error":
                    error = DownloadError(
                        msg.get("code", "internal"), msg.get("message", "Download failed")
                    )
            await asyncio.to_thread(proc.wait)
        except BaseException:
            # Cancelled (or disk check failed): stop the worker; partial files stay for resume.
            if proc.poll() is None:
                proc.terminate()
                try:
                    await asyncio.to_thread(proc.wait, 10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            raise
        await asyncio.to_thread(stderr_reader.join, 5)
        stderr_tail = "".join(stderr_lines).strip().splitlines()[-5:]

        if error:
            raise error
        if not done:
            raise DownloadError(
                "internal",
                f"Download worker exited unexpectedly (code {proc.returncode}): "
                + " | ".join(stderr_tail),
            )
        if proc.returncode != 0:
            logger.warning(
                "Download worker reported success but exited with code %s", proc.returncode
            )

        if job.total_bytes:
            job.downloaded_bytes = job.total_bytes

        if plan.kind == "hf_snapshot":
            if not in_place:
                if plan.target.exists():
                    raise DownloadError(
                        "exists", f"{plan.target} appeared while downloading; not overwriting."
                    )
                local_dir.replace(plan.target)
            self._record(plan, plan.name, plan.target)
            job.target_path = str(plan.target)
            return

        # hf_files: flatten requested files into the type dir
        moved: list[Path] = []
        for rel in plan.files or []:
            src = local_dir / rel
            if not src.is_file():
                raise DownloadError("not_found", f"File '{rel}' was not downloaded")
            dest_name = plan.name if len(plan.files or []) == 1 else Path(rel).name
            if len(plan.files or []) == 1 and not Path(dest_name).suffix:
                dest_name += src.suffix
            dest = type_dir / dest_name
            if dest.exists():
                dest.unlink()
            shutil.move(str(src), dest)
            self._record(plan, dest.name, dest)
            moved.append(dest)
        shutil.rmtree(local_dir, ignore_errors=True)
        with contextlib.suppress(OSError):
            staging_root.rmdir()  # only if empty
        job.target_path = str(moved[0]) if moved else None

    async def _run_url(self, job: DownloadJob, plan: DownloadPlan) -> None:
        assert plan.url
        type_dir = plan.type_dir
        staging = type_dir / STAGING_DIRNAME
        staging.mkdir(parents=True, exist_ok=True)
        part = staging / (hashlib.sha1(plan.url.encode()).hexdigest()[:16] + ".part")
        headers = dict(plan.headers)
        offset = part.stat().st_size if part.exists() else 0
        if offset:
            headers["Range"] = f"bytes={offset}-"

        timeout = httpx.Timeout(30.0, read=300.0)
        speed_state: dict[str, float] = {}
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            try:
                async with client.stream("GET", plan.url, headers=headers) as resp:
                    if resp.status_code in (401, 403):
                        hint = (
                            " Set CIVITAI_TOKEN for Civitai downloads."
                            if "civitai" in plan.url
                            else ""
                        )
                        raise DownloadError(
                            "auth", f"Access denied (HTTP {resp.status_code}).{hint}"
                        )
                    if resp.status_code == 404:
                        raise DownloadError("not_found", "File not found (HTTP 404)")
                    if resp.status_code == 416:  # range not satisfiable -> restart
                        part.unlink(missing_ok=True)
                        raise DownloadError("network", "Resume failed; please retry the download.")
                    resp.raise_for_status()
                    if resp.status_code != 206:
                        offset = 0  # server ignored Range -> start over
                    filename = plan.name or sanitize_name(
                        _filename_from_response(resp) or "model.safetensors"
                    )
                    if not Path(filename).suffix:
                        filename += (
                            Path(_filename_from_response(resp) or ".safetensors").suffix
                            or ".safetensors"
                        )
                    if Path(filename).suffix.lower() not in WEIGHT_SUFFIXES:
                        raise DownloadError(
                            "invalid",
                            f"Refusing to save '{filename}': not a model weight file ({', '.join(sorted(WEIGHT_SUFFIXES))}).",
                        )
                    dest = type_dir / filename
                    if dest.exists():
                        raise DownloadError(
                            "exists", f"{dest.name} already exists. Delete it first to re-download."
                        )
                    length = resp.headers.get("content-length")
                    job.total_bytes = offset + int(length) if length else None
                    job.status = "downloading"
                    _disk_check(type_dir, (job.total_bytes or 0) - offset)
                    job.target_path = str(dest)
                    written = offset
                    # local-disk appends of 1 MB chunks are fast enough to do inline
                    with open(part, "ab" if offset else "wb") as fh:  # noqa: ASYNC230
                        async for chunk in resp.aiter_bytes(1024 * 1024):
                            fh.write(chunk)
                            written += len(chunk)
                            self._update_speed(job, written, speed_state)
            except httpx.HTTPError as e:
                raise DownloadError("network", f"{type(e).__name__}: {e}") from e
        part.replace(dest)
        with contextlib.suppress(OSError):
            staging.rmdir()
        job.downloaded_bytes = job.total_bytes or written
        self._record(plan, dest.name, dest)


download_manager = DownloadManager()
