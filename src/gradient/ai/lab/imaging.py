# src/gradient/ai/lab/imaging.py
"""Small image helpers shared by the Lab tools (no matplotlib needed)."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image

# 6-stop "magma-like" colour ramp (black -> purple -> orange -> pale yellow)
_RAMP = np.array(
    [[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]],
    dtype=np.float32,
)


def normalize(values: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    v = values.astype(np.float32)
    lo, hi = float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))
    return np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)


def colorize(values: np.ndarray, already_normalized: bool = False) -> Image.Image:
    """Map a 2-D array to an RGB heatmap image."""
    v = (values if already_normalized else normalize(values)) * (len(_RAMP) - 1)
    idx = np.floor(v).astype(int).clip(0, len(_RAMP) - 2)
    frac = (v - idx)[..., None]
    rgb = _RAMP[idx] * (1 - frac) + _RAMP[idx + 1] * frac
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def overlay(
    image: Image.Image, values: np.ndarray, alpha: float = 0.55, absolute: bool = False
) -> Image.Image:
    """Blend a heatmap of `values` (any resolution) over `image`.

    absolute=True takes values as already in [0, 1] (e.g. probabilities) instead of
    stretching them between their own percentiles, so maps stay comparable.
    """
    v = np.clip(values, 0, 1) if absolute else normalize(values)
    heat = colorize(v, already_normalized=True).resize(image.size, Image.Resampling.BILINEAR)
    weight = Image.fromarray((v * 255 * alpha).astype(np.uint8), "L").resize(
        image.size, Image.Resampling.BILINEAR
    )
    return Image.composite(heat, image.convert("RGB"), weight)


def to_b64(img: Image.Image, fmt: str = "PNG", **kw) -> str:
    buf = BytesIO()
    img.save(buf, format=fmt, **kw)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def thumbnail_b64(img: Image.Image, max_side: int = 256) -> str:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    return to_b64(thumb, "WEBP", quality=80)


def pil_to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
