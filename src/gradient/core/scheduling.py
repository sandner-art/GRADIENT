# gradient-project-root/src/gradient/core/scheduling.py
"""Minimal task scheduling for GPU-bound work.

All heavy model work (loading, denoising, decoding) is synchronous PyTorch code.
Running it directly inside `async def` handlers blocks the event loop, which
freezes every other request (status polling, downloads, the UI). `run_gpu_task`
executes such work in a worker thread while a single asyncio lock serializes
access to the accelerator, so jobs queue instead of fighting over VRAM.

This is the foundation for the planned TaskScheduler (queue + job status).
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_gpu_lock: asyncio.Lock | None = None
_gpu_lock_loop: asyncio.AbstractEventLoop | None = None
_queue_depth = 0


def _lock() -> asyncio.Lock:
    """The GPU lock for the running event loop (recreated if the loop changes, e.g. in tests)."""
    global _gpu_lock, _gpu_lock_loop
    loop = asyncio.get_running_loop()
    if _gpu_lock is None or _gpu_lock_loop is not loop:
        _gpu_lock = asyncio.Lock()
        _gpu_lock_loop = loop
    return _gpu_lock


def gpu_queue_depth() -> int:
    """Number of GPU jobs currently running or waiting."""
    return _queue_depth


async def run_gpu_task(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking GPU function in a thread, one at a time."""
    global _queue_depth
    _queue_depth += 1
    name = getattr(fn, "__name__", repr(fn))
    try:
        waited = time.perf_counter()
        async with _lock():
            waited = time.perf_counter() - waited
            if waited > 0.5:
                logger.info("GPU task %s waited %.1fs in queue", name, waited)
            return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))
    finally:
        _queue_depth -= 1


class ProgressTracker:
    """Step progress of the currently running generation (read by /system/status)."""

    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None

    def start(self, task: str, model_id: str, total_steps: int) -> None:
        self.state = {
            "task": task,
            "model_id": model_id,
            "step": 0,
            "total_steps": total_steps,
            "started_at": time.time(),
        }

    def update(self, step: int) -> None:
        if self.state is not None:
            self.state["step"] = step

    def finish(self) -> None:
        self.state = None

    def snapshot(self) -> dict[str, Any] | None:
        return dict(self.state) if self.state else None


progress_tracker = ProgressTracker()


class TaskScheduler:
    """Placeholder for a future persistent job queue (submit/status/cancel)."""

    async def submit_task(self, task_fn, *args, **kwargs):
        return await run_gpu_task(task_fn, *args, **kwargs)


task_scheduler = TaskScheduler()
