# src/gradient/core/shutdown.py
"""A clean stop on Ctrl+C.

The first Ctrl+C asks GRADIENT to stop: event streams to browsers end (so the server isn't
left "waiting for connections to close"), the running job stops at its next step, and no
new job starts. A second Ctrl+C stops at once (see cli.py)."""

import threading

_requested = threading.Event()


def request() -> None:
    _requested.set()


def requested() -> bool:
    return _requested.is_set()


def reset() -> None:
    """For tests (and an embedding app that starts the server again)."""
    _requested.clear()
