# src/gradient/ai/diffusion/guidance.py
"""Scheduled classifier-free guidance.

A schedule is turned into one guidance value per denoising step and applied
through diffusers' `callback_on_step_end`, which works for every pipeline that
reads `self._guidance_scale` each step (SD1.x, SDXL, SD3.x, PixArt, Kolors).

Accepted forms:
    7.5                      constant
    "7.5"                    constant
    "linear(9, 3)"           linear ramp from 9 (first step) to 3 (last step)
    "cosine(9, 3)"           cosine ease from 9 to 3
    "9@0, 5@15, 3@30"        keyframes value@step, linearly interpolated
    [[0, 9.0], [15, 5.0]]    keyframes as [step, value] pairs
"""

import math
import re
from collections.abc import Sequence

_FUNC_RE = re.compile(
    r"^\s*(linear|cosine)\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*$", re.I
)


def _keyframes(points: Sequence[tuple[float, float]], num_steps: int) -> list[float]:
    pts = sorted((int(step), float(val)) for step, val in points)
    if not pts:
        raise ValueError("Empty guidance schedule")
    out: list[float] = []
    for i in range(num_steps):
        if i <= pts[0][0]:
            out.append(pts[0][1])
            continue
        if i >= pts[-1][0]:
            out.append(pts[-1][1])
            continue
        for (s0, v0), (s1, v1) in zip(pts, pts[1:], strict=False):
            if s0 <= i <= s1:
                t = (i - s0) / (s1 - s0) if s1 != s0 else 0.0
                out.append(v0 + t * (v1 - v0))
                break
    return out


def parse_guidance_schedule(spec: float | int | str | Sequence, num_steps: int) -> list[float]:
    """Return a list of `num_steps` guidance values. Raises ValueError on bad input."""
    num_steps = max(1, int(num_steps))
    if isinstance(spec, (int, float)):
        return [float(spec)] * num_steps
    if isinstance(spec, str):
        text = spec.strip()
        try:
            return [float(text)] * num_steps
        except ValueError:
            pass
        if m := _FUNC_RE.match(text):
            kind, a, b = m.group(1).lower(), float(m.group(2)), float(m.group(3))
            if num_steps == 1:
                return [a]
            vals = []
            for i in range(num_steps):
                t = i / (num_steps - 1)
                if kind == "cosine":
                    t = (1 - math.cos(math.pi * t)) / 2
                vals.append(a + t * (b - a))
            return vals
        if "@" in text:
            points = []
            for part in text.split(","):
                if not part.strip():
                    continue
                try:
                    val, step = part.split("@")
                    points.append((int(step), float(val)))
                except ValueError:
                    raise ValueError(
                        f"Bad keyframe {part!r}; expected value@step, e.g. '7@10'"
                    ) from None
            return _keyframes(points, num_steps)
        raise ValueError(f"Unrecognised guidance schedule {spec!r}")
    if isinstance(spec, Sequence):
        try:
            return _keyframes([(p[0], p[1]) for p in spec], num_steps)
        except (TypeError, IndexError):
            raise ValueError("Guidance keyframes must be [step, value] pairs") from None
    raise ValueError(f"Unsupported guidance_scale type: {type(spec).__name__}")


def is_constant(values: Sequence[float]) -> bool:
    return max(values) - min(values) < 1e-9
