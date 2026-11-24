# src/gradient/api/server.py
"""The uvicorn server GRADIENT runs, with a clean stop from Ctrl+C, a closing console,
`gradient stop` or the web UI's Stop button, also in auto-reload mode.

With auto-reload, uvicorn runs this server in a child process (the parent only watches the
source files), so everything that makes the stop clean lives in the server itself:

- it registers with gradient.core.shutdown, so event streams end, the running job stops at
  its next step and /system/shutdown can stop it;
- on Windows it handles console events itself (in their own thread, so they also work while
  the main thread is busy) and switches Ctrl+C back on if a launcher had switched it off;
- it does not re-raise the signal after stopping (uvicorn's default), which otherwise ends
  in KeyboardInterrupt / CancelledError tracebacks;
- a second Ctrl+C in the same process stops at once.
The reloader stops too when the server stopped by itself (Stop button, `gradient stop`)."""

from __future__ import annotations

import os
import signal
import sys

import uvicorn

from gradient.core import shutdown

#: Windows console events: Ctrl+C, Ctrl+Break, the console window (or terminal) closing
CONSOLE_EVENTS = {0: "Ctrl+C", 1: "Ctrl+Break", 2: "the console was closed"}
_interrupts = 0  # Ctrl+C presses seen by this process
_console_handler = None  # the registered callback: must live as long as the process


def _interrupt(reason: str) -> None:
    """One more Ctrl+C: the first stops gracefully, the second at once."""
    global _interrupts
    _interrupts += 1
    if _interrupts > 1:
        print("\nStopped immediately.", flush=True)
        os._exit(0)
    shutdown.begin(reason)


def on_console_event(event: int) -> bool:
    """Windows console control handler. Returns True when handled."""
    reason = CONSOLE_EVENTS.get(event)
    if reason is None:  # log-off / system shutdown: Windows' default handling
        return False
    if event == 2:  # Windows ends the process about 5 s after the console closes
        shutdown.begin(reason)
        shutdown.wait_stopped(4.0)
        return True
    _interrupt(reason)
    return True


def install_windows_console_handler():
    """Handle console events ourselves on Windows (once per process). Returns the callback,
    which stays registered and referenced for the life of the process, or None."""
    global _console_handler
    if sys.platform != "win32":
        return None
    if _console_handler is not None:
        return _console_handler
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        routine_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        routine = routine_type(lambda event: bool(on_console_event(int(event))))
        kernel32.SetConsoleCtrlHandler(None, False)  # undo an inherited "ignore Ctrl+C"
        if not kernel32.SetConsoleCtrlHandler(routine, True):
            return None
        _console_handler = routine
        return routine
    except Exception:  # no console (e.g. started without one): signals still work
        return None


class GracefulServer(uvicorn.Server):
    def run(self, sockets=None) -> None:
        shutdown.attach(self)
        install_windows_console_handler()
        try:
            super().run(sockets=sockets)
        finally:
            shutdown.detach()
            after_stop()

    def handle_exit(self, sig, frame) -> None:
        if sig == signal.SIGINT:
            _interrupt("Ctrl+C")
        else:  # SIGTERM (e.g. from the reloader) or SIGBREAK: graceful, never "at once"
            shutdown.begin(signal.Signals(sig).name)
        self.should_exit = True  # not super(): it would raise the signal again afterwards


def after_stop() -> None:
    """Once stopped, Ctrl+C ends the process at once and quietly (Python may still be
    finishing a worker thread's current step)."""
    global _interrupts
    _interrupts = max(_interrupts, 1)  # so a console Ctrl+C is "the second" (Windows)
    try:
        signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    except ValueError:  # not the main thread
        pass


def reload_supervisor(config: uvicorn.Config, target, sockets):
    """uvicorn's file-watching reloader that also ends when the server stopped by itself
    with success (Stop button, `gradient stop`); a crash keeps it watching for a fix."""
    from uvicorn.supervisors import ChangeReload

    class Reload(ChangeReload):
        def should_restart(self):
            process = getattr(self, "process", None)
            if process is not None and not process.is_alive() and process.exitcode == 0:
                raise StopIteration
            return super().should_restart()

    return Reload(config, target=target, sockets=sockets)
