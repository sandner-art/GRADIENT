# src/gradient/core/jobs.py
"""The render queue: every GPU job (renders, grids, Lab tools) runs through one worker.

* `submit()` queues a job and returns at once (the UI's "Queue" button);
  `run()` queues it and waits for the result (the synchronous API endpoints), so API
  calls also show up in the queue panel and can be cancelled.
* Interactive jobs (Lab tools) get priority 1 and run before queued batch renders.
* Running jobs are cancelled between denoising steps: the generation callback calls
  `check_cancelled()`, which raises `JobCancelled` in the worker thread.
* Progress and status changes are pushed to subscribers (the SSE endpoint) and the
  jobs are kept in SQLite so the panel shows recent history after a restart.

The current job travels in a ContextVar; asyncio.to_thread copies the context, so code
running in the GPU worker thread (progress callbacks) sees it.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from . import shutdown

logger = logging.getLogger(__name__)

QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED = (
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
)
FINISHED = (COMPLETED, FAILED, CANCELLED)
PRIORITY_NORMAL, PRIORITY_INTERACTIVE = 0, 1


class JobCancelled(BaseException):
    """Raised inside a job when the user cancels it.

    A BaseException, like asyncio.CancelledError, so generic `except Exception`
    handlers (e.g. per-cell error handling of XYZ grids) don't swallow it.
    """


class JobCancelledError(RuntimeError):
    """What a caller waiting on a job (`JobQueue.run`) receives when the job was cancelled."""


Handler = Callable[["Job"], Awaitable[Any]]
Summarizer = Callable[[Any], dict[str, Any]]


@dataclass
class Job:
    kind: str
    payload: dict[str, Any]
    label: str = ""
    priority: int = PRIORITY_NORMAL
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    run_uuid: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    retryable: bool = True
    #: from POST /queue: group (one gallery run for all its jobs), tags, origin
    meta: dict[str, Any] = field(default_factory=dict)
    # not part of the public view
    preview: bytes | None = field(default=None, repr=False)  # latest live preview (JPEG)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    waiter: asyncio.Future | None = field(default=None, repr=False)

    def public(self, position: int | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "priority": self.priority,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "progress": dict(self.progress),
            "run_uuid": self.run_uuid,
            "error": self.error,
            "result": self.result,
            "position": position,
            "retryable": self.retryable and self.status in (FAILED, CANCELLED),
            "group": self.meta.get("group"),
            "tags": list(self.meta.get("tags") or []),
            "origin": self.meta.get("origin"),
            # the settings (without input images, which can be megabytes)
            "payload": _without_images(self.payload),
            "model_id": (
                self.payload.get("model_id")
                or self.payload.get("base_request", {}).get("model_id")
                or self.payload.get("base_parameters", {}).get("model_id")
            ),
        }


def _without_images(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _without_images(v)
            for k, v in value.items()
            if not (k.endswith("image_base64") and isinstance(v, str))
        }
    if isinstance(value, list):
        return [_without_images(v) for v in value]
    return value


_current: contextvars.ContextVar[Job | None] = contextvars.ContextVar("gradient_job", default=None)


def current_job() -> Job | None:
    return _current.get()


def check_cancelled() -> None:
    """Call between steps of long work; raises JobCancelled if the user cancelled the job
    or GRADIENT is stopping."""
    job = _current.get()
    if job is not None and (job.cancel_event.is_set() or shutdown.requested()):
        raise JobCancelled()


def report_preview(jpeg: bytes, step: int) -> None:
    """Publish a live preview of the running job (fetched from /queue/{id}/preview)."""
    job = _current.get()
    if job is not None:
        job.preview = jpeg
        job.progress = {
            **job.progress,
            "preview": int(job.progress.get("preview", 0)) + 1,
            "preview_step": step,
        }


def report_progress(**values: Any) -> None:
    """Update the running job's progress (step, total, item, items, phase); thread-safe enough."""
    job = _current.get()
    if job is not None:
        job.progress = {**job.progress, **values}


@dataclass
class _Kind:
    handler: Handler
    summarize: Summarizer | None = None
    keep_payload_keys: tuple[str, ...] = ()


class JobQueue:
    def __init__(self, keep_finished: int = 200) -> None:
        self.kinds: dict[str, _Kind] = {}
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []  # queued job ids, next first
        self.keep_finished = keep_finished
        self.running: Job | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self._wake: asyncio.Event | None = None
        self._worker: asyncio.Task | None = None
        self._loop_ref: asyncio.AbstractEventLoop | None = None
        self._persist: Callable[[Job], Awaitable[None]] | None = None

    # ---- setup
    def register(self, kind: str, handler: Handler, summarize: Summarizer | None = None) -> None:
        self.kinds[kind] = _Kind(handler, summarize)

    def set_persistence(self, fn: Callable[[Job], Awaitable[None]] | None) -> None:
        self._persist = fn

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop_ref is not loop:  # first start, or a new event loop (tests, reload)
            self._loop_ref = loop
            self._wake = asyncio.Event()
            self._worker = None
            self.running = None
            self.subscribers.clear()
            for jid in list(self.order):  # jobs queued on a dead loop can't be awaited any more
                job = self.jobs[jid]
                if job.waiter is not None:
                    self.order.remove(jid)
                    self._finish(job, FAILED, "Event loop restarted")
                    job.waiter = None
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._loop(), name="gradient-job-worker")
            if self.order:
                assert self._wake is not None
                self._wake.set()

    async def stop(self) -> None:
        if self._worker is not None:
            if self.running is not None:
                self.running.cancel_event.set()
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
            self._worker = None

    def restore(self, jobs: list[Job]) -> None:
        """Show jobs from the previous session (unfinished ones are marked interrupted)."""
        for job in jobs:
            if job.status not in FINISHED:
                job.status = FAILED
                job.error = "GRADIENT was restarted before this job finished"
                job.finished = job.finished or time.time()
            self.jobs.setdefault(job.id, job)
        self._trim()

    # ---- queries
    def position(self, job_id: str) -> int | None:
        try:
            return self.order.index(job_id) + 1
        except ValueError:
            return None

    def view(self, job: Job) -> dict[str, Any]:
        return job.public(self.position(job.id))

    def list(self) -> list[dict[str, Any]]:
        running = [self.running] if self.running else []
        queued = [self.jobs[i] for i in self.order]
        done = sorted(
            (j for j in self.jobs.values() if j.status in FINISHED),
            key=lambda j: j.finished or j.created,
            reverse=True,
        )
        return [self.view(j) for j in (*running, *queued, *done)]

    def get(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError:
            raise KeyError(f"No job {job_id}") from None

    # ---- commands
    async def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        label: str = "",
        priority: int = PRIORITY_NORMAL,
        wait: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> Job:
        if kind not in self.kinds:
            raise ValueError(f"Unknown job kind '{kind}'")
        job = Job(
            kind=kind,
            payload=payload,
            label=label or kind,
            priority=priority,
            meta=dict(meta or {}),
        )
        if wait:
            job.waiter = asyncio.get_running_loop().create_future()
        self.jobs[job.id] = job
        # higher priority goes before lower; FIFO within a priority
        index = next(
            (i for i, jid in enumerate(self.order) if self.jobs[jid].priority < priority),
            len(self.order),
        )
        self.order.insert(index, job.id)
        await self._changed(job)
        await self.start()
        assert self._wake is not None
        self._wake.set()
        return job

    async def run(
        self, kind: str, payload: dict[str, Any], label: str = "", priority: int = PRIORITY_NORMAL
    ):
        """Submit and wait for the handler's return value (raises its exception)."""
        job = await self.submit(kind, payload, label, priority, wait=True)
        assert job.waiter is not None
        return await job.waiter

    async def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == QUEUED:
            self.order.remove(job.id)
            self._finish(job, CANCELLED)
            await self._changed(job)
        elif job.status == RUNNING:
            job.cancel_event.set()
            job.progress = {**job.progress, "phase": "cancelling"}
            await self._changed(job)
        return job

    async def move(self, job_id: str, to: str) -> Job:
        job = self.get(job_id)
        if job.status != QUEUED:
            raise ValueError("Only waiting jobs can be moved")
        i = self.order.index(job_id)
        self.order.pop(i)
        target = {"top": 0, "up": max(0, i - 1), "down": i + 1, "bottom": len(self.order)}.get(to)
        if target is None:
            self.order.insert(i, job_id)
            raise ValueError("to must be top, up, down or bottom")
        self.order.insert(min(target, len(self.order)), job_id)
        await self._broadcast({"type": "order", "order": list(self.order)})
        return job

    async def retry(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status not in (FAILED, CANCELLED) or not job.retryable:
            raise ValueError(
                "Only failed or cancelled jobs with their full settings can be retried"
            )
        return await self.submit(job.kind, job.payload, job.label, job.priority, meta=job.meta)

    async def clear_finished(self) -> int:
        ids = [j.id for j in self.jobs.values() if j.status in FINISHED and j.waiter is None]
        for jid in ids:
            del self.jobs[jid]
        await self._broadcast({"type": "removed", "ids": ids})
        return len(ids)

    # ---- events
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    async def _broadcast(self, event: dict[str, Any]) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # a stalled client; it resyncs on reconnect
                self.subscribers.discard(q)

    async def _changed(self, job: Job, persist: bool = True) -> None:
        await self._broadcast({"type": "job", "job": self.view(job)})
        if persist and self._persist is not None:
            try:
                await self._persist(job)
            except Exception as e:  # never let bookkeeping break rendering
                logger.warning("Could not store job %s: %s", job.id, e)

    # ---- worker
    def _finish(self, job: Job, status: str, error: str | None = None) -> None:
        job.status = status
        job.error = error
        job.finished = time.time()

    def _trim(self) -> None:
        done = sorted(
            (j for j in self.jobs.values() if j.status in FINISHED and j.waiter is None),
            key=lambda j: j.finished or j.created,
        )
        for job in done[: max(0, len(done) - self.keep_finished)]:
            del self.jobs[job.id]

    async def _loop(self) -> None:
        assert self._wake is not None
        while True:
            if shutdown.requested():  # stopping: waiting jobs stay queued (marked on restart)
                return
            if not self.order:
                self._wake.clear()
                await self._wake.wait()
                continue
            job = self.jobs[self.order.pop(0)]
            await self._execute(job)
            self._trim()

    async def _watch_progress(self, job: Job) -> None:
        last: dict[str, Any] = {}
        while True:
            await asyncio.sleep(0.3)
            if job.progress != last:
                last = dict(job.progress)
                await self._changed(job, persist=False)

    async def _execute(self, job: Job) -> None:
        kind = self.kinds[job.kind]
        job.status, job.started = RUNNING, time.time()
        self.running = job
        await self._changed(job)
        token = _current.set(job)
        watcher = asyncio.create_task(self._watch_progress(job))
        outcome: Any = None
        error: BaseException | None = None
        try:
            outcome = await kind.handler(job)
            summary = kind.summarize(outcome) if kind.summarize else None
            if summary is not None:  # handlers may also set job.result themselves
                job.result = summary
            if job.result and job.result.get("run_uuid"):
                job.run_uuid = job.result["run_uuid"]
            self._finish(job, COMPLETED)
        except JobCancelled as e:
            if shutdown.requested():
                self._finish(
                    job, FAILED, "GRADIENT was stopped during this job (Retry runs it again)"
                )
            else:
                self._finish(job, CANCELLED)
            error = e
        except Exception as e:
            self._finish(job, FAILED, _describe(e))
            error = e
        finally:
            _current.reset(token)
            watcher.cancel()
            self.running = None
        job.progress = {k: v for k, v in job.progress.items() if k != "phase"}
        await self._changed(job)
        if job.waiter is not None and not job.waiter.done():
            if error is None:
                job.waiter.set_result(outcome)
            else:
                job.waiter.set_exception(
                    error
                    if isinstance(error, Exception)
                    else JobCancelledError("Cancelled by the user")
                )
            job.waiter = None


def _describe(e: BaseException) -> str:
    detail = getattr(e, "detail", None)
    return str(detail) if detail else f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


job_queue = JobQueue()
