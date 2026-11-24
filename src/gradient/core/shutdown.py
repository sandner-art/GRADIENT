# src/gradient/core/shutdown.py
"""A clean stop: Ctrl+C, closing the console, `gradient stop` or the web UI's Stop button.

The first request asks GRADIENT to stop: event streams to browsers end (so the server isn't
left "waiting for connections to close"), the running job stops at its next step, and no
new job starts. A second Ctrl+C stops at once (see cli.py)."""

import threading

_requested = threading.Event()
_finished = threading.Event()
_server = None  # the running uvicorn server, when started by `gradient serve`


def attach(server) -> None:
    global _server
    _server = server
    _finished.clear()


def detach() -> None:
    global _server
    _server = None
    _finished.set()


def can_stop() -> bool:
    """A server this process can stop (started by `gradient serve`)."""
    return _server is not None


def begin(reason: str) -> bool:
    """Start a graceful stop (from any thread). Returns False if one was already under way."""
    first = not _requested.is_set()
    _requested.set()
    if first:
        print(
            f"\nStopping GRADIENT ({reason}): a running job stops at its next step. "
            "Press Ctrl+C again to stop immediately.",
            flush=True,
        )
    if _server is not None:
        _server.should_exit = True
    return first


def request() -> None:
    _requested.set()


def requested() -> bool:
    return _requested.is_set()


def wait_stopped(timeout: float) -> bool:
    return _finished.wait(timeout)


def reset() -> None:
    """For tests (and an embedding app that starts the server again)."""
    _requested.clear()
    _finished.clear()
