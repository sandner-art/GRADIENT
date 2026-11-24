# src/gradient/core/quiet.py
"""Library messages that say nothing useful to GRADIENT users.

- diffusers casts single-file components with `.to(dtype)` itself and then warns that
  modules "should be kept in float32" even when that list is empty;
- loading progress bars: GRADIENT logs each load and shows progress in the web UI.
Real warnings (non-empty lists, deprecations in our own calls) still come through."""

import logging


class _Drop(logging.Filter):
    def __init__(self, *needles: str) -> None:
        super().__init__()
        self.needles = needles

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(n in message for n in self.needles)


def quiet_libraries() -> None:
    logging.getLogger("diffusers.models.modeling_utils").addFilter(
        _Drop("should be kept in float32: []")
    )
    try:
        from diffusers.utils import logging as diffusers_logging

        diffusers_logging.disable_progress_bar()
    except Exception:  # diffusers missing or changed: nothing to quiet
        pass
