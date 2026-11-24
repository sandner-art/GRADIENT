# src/gradient/ai/lab/quant_report.py
"""Quantization fidelity report: what does loading a model quantized change?

The same settings and seeds are rendered with the reference (full precision by default) and
each quantized variant, one model load at a time. Per variant:

* cost: weight size of each part, load time, seconds per image, peak GPU memory;
* image fidelity: PSNR, SSIM and mean difference against the reference image of the same
  seed, and the DINO patch distance, normalised by the distance between two reference seeds
  ("relative change": 0 = identical, 1 = as different as a new seed);
* analysis fidelity: the cosine similarity of ∂image/∂guidance (the Taylor tangent) with the
  reference's, i.e. whether Lab measurements made on the quantized model still hold.
"""

from __future__ import annotations

import logging
import time
from itertools import combinations

import numpy as np
import torch
from PIL import Image, ImageDraw

from gradient.ai.diffusion import quantization as quant
from gradient.ai.diffusion.loading import get_pipeline
from gradient.ai.diffusion.pipelines import generate_images, resolve_seed
from gradient.core.jobs import check_cancelled, report_progress

from .features import distances, load_feature_model
from .imaging import colorize, pil_to_array, to_b64
from .models import QuantReportRequest, QuantReportResponse, QuantVariantReport
from .taylor import compute_tangent

logger = logging.getLogger(__name__)
THUMB = 192


# ------------------------------------------------------------------------------ metrics
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: float images in [0, 1]."""
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on luminance with an 11×11 Gaussian window (σ = 1.5)."""

    def gray(x):
        return torch.from_numpy(x @ np.array([0.299, 0.587, 0.114], np.float32))[None, None]

    x, y = gray(a.astype(np.float32)), gray(b.astype(np.float32))
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    k = torch.outer(g, g).float()[None, None]  # (1, 1, 11, 11)

    def blur(t):
        return torch.nn.functional.conv2d(t, k, padding=5)

    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx**2
    vy = blur(y * y) - my**2
    cov = blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(s.mean())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else 0.0


def verdict(rel: float | None, taylor: float | None, mad: float) -> str:
    if rel is None:
        image = (
            "nearly identical images"
            if mad < 2
            else "small pixel differences"
            if mad < 8
            else "visible differences"
        )
    elif rel < 0.1:
        image = "nearly identical images"
    elif rel < 0.35:
        image = "same composition, small detail changes"
    elif rel < 0.75:
        image = "noticeable changes (part way to a different seed)"
    else:
        image = "changes as large as a different seed"
    if taylor is None:
        return image.capitalize() + "."
    analysis = (
        "Lab analyses agree"
        if taylor > 0.95
        else "Lab analyses mostly agree"
        if taylor > 0.8
        else "Lab analyses differ: measure on the reference"
    )
    return f"{image.capitalize()}; {analysis} (tangent cosine {taylor:.2f})."


# --------------------------------------------------------------------------- the report
def quantization_report(request: QuantReportRequest) -> QuantReportResponse:
    """Blocking; run through the queue (Lab job)."""
    start = time.time()
    base = request.base_request.model_copy(deep=True)
    first = resolve_seed(base.seed)
    seeds = [first + i for i in range(request.num_seeds)]
    specs = [quant.parse(request.reference)]
    for v in request.variants:
        s = quant.parse(v)
        if s.key not in [x.key for x in specs]:
            specs.append(s)
    if len(specs) < 2:
        raise ValueError("Pick at least one quantization that differs from the reference")

    renders_per = request.num_seeds + (3 if request.taylor else 0)
    total = renders_per * len(specs)
    done = 0
    rows: list[dict] = []
    for spec in specs:
        label = quant.BUILTIN.get(spec.name, {}).get("label") or f"GGUF {spec.gguf_file}"
        row: dict = {"spec": spec, "report": QuantVariantReport(quantization=spec.key, label=label)}
        rows.append(row)
        reason = quant.unavailable_reason(spec)
        if reason:
            row["report"].error = reason
            done += renders_per
            continue
        try:
            check_cancelled()
            t0 = time.time()
            pipe, _entry = get_pipeline(
                base.model_id,
                vae_id=base.vae_id,
                loras=base.loras,
                embeddings=base.embeddings,
                quantization=spec.key,
            )
            row["report"].load_seconds = round(time.time() - t0, 2)
            sizes = quant.describe(pipe)
            row["report"].weights_mb = sizes
            row["report"].denoiser_mb = sizes.get("transformer") or sizes.get("unet")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            images, seconds = [], []
            for s in seeds:
                check_cancelled()
                report_progress(item=done + 1, items=total)
                t1 = time.time()
                one = base.model_copy(update={"seed": s, "quantization": spec.key})
                images.append(generate_images(one).images[0])
                seconds.append(time.time() - t1)
                done += 1
            row["images"] = images
            row["report"].seconds_per_image = round(float(np.median(seconds)), 3)
            if torch.cuda.is_available():
                row["report"].peak_memory_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if request.taylor:
                report_progress(item=done + 1, items=total)
                one = base.model_copy(update={"seed": seeds[0], "quantization": spec.key})
                _, tangent, method, _note = compute_tangent(one)
                row["tangent"], row["report"].taylor_method = tangent, method
                done += 3
        except Exception as e:
            if type(e).__name__ == "JobCancelled":
                raise
            logger.warning("Quantization %s failed for %s", spec.key, base.model_id, exc_info=True)
            row["report"].error = str(e).splitlines()[0][:300]
            done = sum(renders_per for _ in rows)

    ref = rows[0]
    if "images" not in ref:
        raise ValueError(f"The reference ({ref['spec'].key}) failed: {ref['report'].error}")

    # feature distances (DINOv3 / DINOv2); the report still works without them
    fm, feature_note = None, None
    try:
        fm, feature_note = load_feature_model(request.feature_model)
    except Exception as e:
        feature_note = f"No feature model ({str(e)[:200]}); feature distances left out."
    spread = None
    if fm is not None:
        pairs = list(combinations(ref["images"], 2))
        spread = float(np.mean([distances(fm, a, b)[1].mean() for a, b in pairs]))

    ref_arrays = [pil_to_array(im) for im in ref["images"]]
    for row in rows[1:]:
        rep: QuantVariantReport = row["report"]
        if "images" not in row:
            rep.verdict = "Not available."
            continue
        arrays = [pil_to_array(im) for im in row["images"]]
        rep.psnr = round(
            float(np.mean([psnr(a, b) for a, b in zip(ref_arrays, arrays, strict=True)])), 2
        )
        rep.ssim = round(
            float(np.mean([ssim(a, b) for a, b in zip(ref_arrays, arrays, strict=True)])), 4
        )
        mad = (
            float(np.mean([np.abs(a - b).mean() for a, b in zip(ref_arrays, arrays, strict=True)]))
            * 255
        )
        rep.mean_abs_diff = round(mad, 3)
        if fm is not None:
            d = float(
                np.mean(
                    [
                        distances(fm, a, b)[1].mean()
                        for a, b in zip(ref["images"], row["images"], strict=True)
                    ]
                )
            )
            rep.feature_distance = round(d, 5)
            rep.relative_change = round(d / spread, 4) if spread and spread > 1e-9 else None
        if "tangent" in row and "tangent" in ref:
            rep.taylor_cosine = round(cosine(ref["tangent"], row["tangent"]), 4)
        rep.verdict = verdict(rep.relative_change, rep.taylor_cosine, mad)
        rep.strip_base64 = to_b64(_strip(ref_arrays, arrays))
    ref["report"].verdict = "Reference."

    overview = _overview([r for r in rows if "images" in r], len(seeds))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return QuantReportResponse(
        model_id=base.model_id,
        seeds=seeds,
        reference=ref["spec"].key,
        seed_spread=round(spread, 5) if spread is not None else None,
        feature_model=fm.name if fm is not None else None,
        feature_note=feature_note,
        variants=[r["report"] for r in rows],
        image_base64=to_b64(overview),
        summary=_summary(rows),
        performance_ms=(time.time() - start) * 1000,
    )


# ------------------------------------------------------------------------------- images
def _thumb(arr: np.ndarray | Image.Image) -> Image.Image:
    img = (
        arr
        if isinstance(arr, Image.Image)
        else Image.fromarray((arr * 255).round().astype(np.uint8))
    )
    img = img.convert("RGB")
    img.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (THUMB, THUMB), (24, 24, 27))
    canvas.paste(img, ((THUMB - img.width) // 2, (THUMB - img.height) // 2))
    return canvas


def _strip(ref: list[np.ndarray], var: list[np.ndarray]) -> Image.Image:
    """One row per seed: reference | variant | |difference| (amplified heat map)."""
    out = Image.new("RGB", (THUMB * 3, THUMB * len(ref)))
    for i, (a, b) in enumerate(zip(ref, var, strict=True)):
        diff = np.abs(a - b).mean(-1)
        heat = colorize(np.clip(diff * 8, 0, 1), already_normalized=True)
        for j, img in enumerate((a, b, heat)):
            out.paste(_thumb(img), (j * THUMB, i * THUMB))
    return out


def _overview(rows: list[dict], n: int) -> Image.Image:
    label_h = 18
    out = Image.new("RGB", (THUMB * n, (THUMB + label_h) * len(rows)), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    for r, row in enumerate(rows):
        y = r * (THUMB + label_h)
        draw.text((4, y + 3), row["spec"].key, fill=(20, 20, 20))
        for c, img in enumerate(row["images"]):
            out.paste(_thumb(img), (c * THUMB, y + label_h))
    return out


def _summary(rows: list[dict]) -> str:
    parts = []
    ref = rows[0]["report"]
    for row in rows[1:]:
        rep = row["report"]
        if rep.error:
            parts.append(f"{rep.quantization}: not run ({rep.error})")
            continue
        size = ""
        if rep.denoiser_mb and ref.denoiser_mb:
            size = f", denoiser {rep.denoiser_mb:.0f} MB vs {ref.denoiser_mb:.0f} MB"
        parts.append(f"{rep.quantization}: {rep.verdict[:-1].lower()}{size}")
    return "; ".join(parts) + "."
