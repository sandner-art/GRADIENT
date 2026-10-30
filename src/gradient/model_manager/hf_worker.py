# src/gradient/model_manager/hf_worker.py
"""Hugging Face download worker, run as a subprocess by the DownloadManager.

    python -m gradient.model_manager.hf_worker  < spec.json

Running downloads in a child process means a download can be cancelled by
terminating the process (huggingface_hub's worker threads cannot be stopped
from outside), and heavy hashing/IO never competes with the API server.
Partially downloaded files are kept, so a later retry resumes.

Protocol: the spec is a JSON object on stdin; the worker writes JSON lines to
stdout:  {"event": "plan" | "progress" | "done" | "error", ...}
The access token (if any) is passed in the GRADIENT_HF_TOKEN env variable.
"""

import errno
import json
import os
import sys
import threading
import time
from typing import Any

from tqdm.std import tqdm as _tqdm

_DEVNULL = open(os.devnull, "w")  # noqa: SIM115 - lives for the process lifetime
_emit_lock = threading.Lock()


def emit(event: str, **payload: Any) -> None:
    with _emit_lock:
        sys.stdout.write(json.dumps({"event": event, **payload}) + "\n")
        sys.stdout.flush()


class _Progress:
    """Aggregates byte counters from all progress bars huggingface_hub creates."""

    def __init__(self) -> None:
        self.bars: list[_tqdm] = []
        self.lock = threading.Lock()
        self.last_emit = 0.0
        self.last_bytes = -1

    def current(self) -> int:
        with self.lock:
            return int(max((b.n for b in self.bars), default=0))

    def maybe_emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_emit < 0.25:
            return
        value = self.current()
        if value != self.last_bytes or force:
            self.last_emit = now
            self.last_bytes = value
            emit("progress", bytes=value)


PROGRESS = _Progress()


class ReportingTqdm(_tqdm):
    """A silent tqdm that forwards byte progress to PROGRESS."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("name", None)
        kwargs["file"] = _DEVNULL
        kwargs["disable"] = False
        super().__init__(*args, **kwargs)
        if self.unit == "B":
            with PROGRESS.lock:
                PROGRESS.bars.append(self)

    def update(self, n: float | None = 1) -> bool | None:
        result = super().update(n)
        if self.unit == "B":
            PROGRESS.maybe_emit()
        return result


def classify_error(exc: BaseException, repo_id: str) -> tuple[str, str]:
    # Look through wrapped exceptions (huggingface_hub chains the HTTP cause)
    chain: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in chain:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    for candidate in chain:
        result = _classify_one(candidate, repo_id)
        if result[0] != "internal":
            return result
    return _classify_one(exc, repo_id)


def _classify_one(exc: BaseException, repo_id: str) -> tuple[str, str]:
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(exc, GatedRepoError):
        return "gated", (
            f"'{repo_id}' is a gated model. Accept its licence at https://huggingface.co/{repo_id} "
            "and provide a Hugging Face token (HF_TOKEN env var, `hf auth login`, or the token field)."
        )
    if isinstance(exc, RepositoryNotFoundError):
        return (
            "not_found",
            f"Repository '{repo_id}' not found (or private and your token has no access).",
        )
    if isinstance(exc, (RevisionNotFoundError, EntryNotFoundError)):
        return "not_found", str(exc).splitlines()[0]
    if isinstance(exc, LocalEntryNotFoundError):
        return "network", "Could not reach Hugging Face (offline, proxy or firewall?)."
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return "auth", f"Access denied by Hugging Face (HTTP {status}). Check your token."
        return "network", str(exc).splitlines()[0]
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "disk_space", "Not enough disk space."
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "network", str(exc)
    return "internal", f"{type(exc).__name__}: {exc}"


def run(spec: dict[str, Any]) -> int:
    from huggingface_hub import snapshot_download

    repo_id: str = spec["repo_id"]
    token = os.environ.get("GRADIENT_HF_TOKEN") or None  # None -> HF_TOKEN env / cached login
    common = {
        "repo_id": repo_id,
        "revision": spec.get("revision"),
        "local_dir": spec["local_dir"],
        "allow_patterns": spec.get("allow_patterns"),
        "ignore_patterns": spec.get("ignore_patterns"),
        "token": token,
    }
    from huggingface_hub.errors import DryRunError

    try:
        try:
            plan = snapshot_download(**common, dry_run=True)
        except DryRunError:
            # The dry run hides the real cause (gated / not found / offline); the real call
            # below raises the specific error, which we classify for the user.
            plan = None
        if plan is not None:
            if not plan:
                emit(
                    "error",
                    code="invalid",
                    message="No files in the repository match the requested files/patterns.",
                )
                return 2
            pending = [p for p in plan if p.will_download]
            emit(
                "plan",
                total_bytes=sum(p.file_size or 0 for p in pending),
                num_files=len(plan),
                files_to_download=len(pending),
                files=[p.filename for p in plan],
            )
        path = snapshot_download(
            **common, tqdm_class=ReportingTqdm, max_workers=int(spec.get("max_workers", 8))
        )
        PROGRESS.maybe_emit(force=True)
        emit("done", path=str(path))
        return 0
    except KeyboardInterrupt:
        emit("error", code="cancelled", message="Cancelled")
        return 130
    except BaseException as exc:  # noqa: BLE001 - report everything to the parent
        code, message = classify_error(exc, repo_id)
        emit("error", code=code, message=message)
        return 1


def main() -> int:
    try:
        spec = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        emit("error", code="invalid", message=f"Bad worker spec: {e}")
        return 2
    return run(spec)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter finalization: native download threads (hf_xet / httpx) can crash
    # during teardown after the work is complete, which would turn success into an error.
    os._exit(code)
