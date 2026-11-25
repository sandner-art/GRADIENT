# src/gradient/ai/lab/style_check.py
"""Style check: which styles change the image, and does the subject survive?

The Lab's settings are rendered without a style and with each style (× position × start),
on the same seeds. Per variant:

* style strength: DINO distance from the unstyled image of the same seed, also relative to
  the distance between two unstyled seeds (1 = the style changes the image as much as a new
  seed would);
* content kept: CLIP (or SigLIP) similarity between the image and the unstyled prompt,
  relative to the unstyled images (1 = the prompt is as recognisable as without the style);
* consistency: how alike the style's images are across seeds (a strong signature makes
  them more alike than the unstyled ones).

Effective styles are strong and keep the content. Blocking: run with `run_gpu_task`.
"""

from __future__ import annotations

import logging
import time
from itertools import combinations

import torch
from PIL import Image, ImageDraw

from gradient.ai.diffusion.loading import model_profile
from gradient.ai.diffusion.pipelines import generate_images, resolve_seed
from gradient.ai.diffusion.prompt_syntax import count_tokens, parse_weights
from gradient.core.jobs import check_cancelled, report_progress
from gradient.model_manager.inventory import list_model_names

from . import embed
from .features import load_feature_model
from .imaging import to_b64
from .models import StyleCheckCell, StyleCheckRequest, StyleCheckResponse, StyleCheckRow

logger = logging.getLogger(__name__)
MAX_VARIANTS = 32
THUMB = 160


def _clip_name(requested: str | None) -> tuple[str | None, str | None]:
    """(model name or None, note) for 'content kept'."""
    if requested and requested.lower() == "none":
        return None, None
    if requested:
        return requested, None
    for name in list_model_names("vision"):
        if embed.model_type(name) in embed.TEXT_MODEL_TYPES:
            return name, None
    return None, (
        "Content kept needs a CLIP or SigLIP model: download CLIP ViT-B/32 in the Models tab "
        "(vision)."
    )


def _plain_text(prompt: str) -> str:
    """The prompt's words without A1111 weights or BREAK (what CLIP should recognise)."""
    return " ".join(t for t, w in parse_weights(prompt) if w is not None).strip()


def _row_notes(params: dict, position: str, clip_family: bool) -> list[str]:
    notes = []
    if params.get("styles_missing"):
        notes.append("style not found")
    if params.get("lora_tags_missing"):
        notes.append("LoRA tags not found: " + ", ".join(params["lora_tags_missing"]))
    if params.get("loras_skipped"):
        notes.append("LoRAs not applied: " + ", ".join(params["loras_skipped"]))
    if params.get("prompt_truncated"):
        notes.append("prompt cut at 75 tokens")
    if params.get("style_start_applied") is False:
        notes.append("style start needs SD/SDXL")
    if position == "chunk" and not clip_family:
        notes.append("own chunk is the same as after for this model")
    if params.get("style_blocks_applied") is False:
        notes.append("style blocks need SDXL: placed after the prompt")
    return notes


def _sheet(cells: list[StyleCheckCell], images: list[Image.Image], rows, baseline, seeds):
    """Contact sheet: a row per variant (unstyled first), a column per seed, with labels."""
    label_w, pad = 220, 6
    lines = [("no style", baseline)] + [
        (f"{r.style} · {r.position}" + (f" · from {r.start:g}" if r.start else ""), r.cells)
        for r in rows
    ]
    cols = len(seeds)
    sheet = Image.new(
        "RGB",
        (label_w + cols * (THUMB + pad) + pad, len(lines) * (THUMB + pad) + pad),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    for r, (label, indexes) in enumerate(lines):
        y = pad + r * (THUMB + pad)
        draw.text((pad, y + THUMB // 2 - 6), label[:34], fill=(30, 30, 30))
        for c, index in enumerate(indexes):
            thumb = images[index].copy()
            thumb.thumbnail((THUMB, THUMB))
            sheet.paste(thumb, (label_w + pad + c * (THUMB + pad), y))
    return sheet


def style_check(request: StyleCheckRequest) -> StyleCheckResponse:
    started = time.time()
    variants = [
        (s, p, t) for s in request.styles for p in request.positions for t in request.style_starts
    ]
    if len(variants) > MAX_VARIANTS:
        raise ValueError(
            f"{len(variants)} variants (styles × positions × starts); at most {MAX_VARIANTS}"
        )
    base = request.base_request.model_copy(deep=True)
    base.seed = resolve_seed(base.seed)
    base.num_outputs = 1
    base.styles = None
    base.style_start = 0.0
    seeds = [(base.seed + i) % (2**32) for i in range(request.num_seeds)]
    clip_family = model_profile(base.model_id)[0] in ("sd15", "sdxl")
    notes: list[str] = []

    fm, fm_note = load_feature_model(request.feature_model)
    if fm_note:
        notes.append(fm_note)
    clip_name, clip_note = _clip_name(request.clip_model)
    if clip_note:
        notes.append(clip_note)

    total = len(seeds) * (1 + len(variants))
    done = 0
    cells: list[StyleCheckCell] = []
    images: list[Image.Image] = []
    params_of: list[dict] = []

    def render(req, style, position, start, seed) -> int:
        nonlocal done
        check_cancelled()
        done += 1
        report_progress(item=done, items=total, phase=style or "no style")
        result = generate_images(req.model_copy(update={"seed": seed}))
        image, params = result.images[0], {**result.params, "seed": seed}
        images.append(image)
        params_of.append(params)
        cells.append(
            StyleCheckCell(
                style=style,
                position=position,
                start=start,
                seed=seed,
                image_base64=to_b64(image),
                parameters=params,
            )
        )
        return len(cells) - 1

    baseline = [render(base, None, None, None, seed) for seed in seeds]
    groups: list[tuple[str, str, float, list[int]]] = []
    for style, position, start in variants:
        req = base.model_copy(
            update={"styles": [style], "style_position": position, "style_start": start}
        )
        groups.append(
            (style, position, start, [render(req, style, position, start, s) for s in seeds])
        )

    # measurements (features once per image)
    with torch.no_grad():
        dino = torch.stack([fm.encode(img)[0].float().cpu() for img in images])
        clip_sims = None
        if clip_name:
            try:
                model = embed.load_clip(clip_name)
                text = model.texts([_plain_text(base.prompt)])[0].cpu()
                clip_sims = (model.images(images).cpu() @ text).tolist()
            except ValueError as e:
                notes.append(str(e))

    def dist(a: int, b: int) -> float:
        return float(1 - (dino[a] * dino[b]).sum())

    def similarity(indexes: list[int]) -> float | None:
        pairs = list(combinations(indexes, 2))
        return sum(1 - dist(a, b) for a, b in pairs) / len(pairs) if pairs else None

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    seed_distance = mean([dist(a, b) for a, b in combinations(baseline, 2)])
    base_content = mean([clip_sims[i] for i in baseline]) if clip_sims else None
    rows = []
    for style, position, start, indexes in groups:
        strength = mean([dist(i, b) for i, b in zip(indexes, baseline, strict=True)]) or 0.0
        content = mean([clip_sims[i] for i in indexes]) if clip_sims else None
        first = params_of[indexes[0]]
        counted = first.get("prompt_tokens"), first.get("prompt_chunks")
        if counted[0] is None and clip_family:  # short plain prompts aren't counted by renders
            count = count_tokens(base.model_id, first.get("styled_prompt") or base.prompt)
            counted = (count["tokens"], count["chunks"]) if count else counted
        rows.append(
            StyleCheckRow(
                style=style,
                position=position,
                start=start,
                strength=round(strength, 6),
                strength_rel=round(strength / seed_distance, 3) if seed_distance else None,
                content=round(content, 6) if content is not None else None,
                content_rel=(
                    round(content / base_content, 3)
                    if content is not None and base_content
                    else None
                ),
                consistency=round(c, 6) if (c := similarity(indexes)) is not None else None,
                tokens=counted[0],
                chunks=counted[1],
                notes=_row_notes(first, position, clip_family),
                cells=indexes,
            )
        )
    sheet = _sheet(cells, images, rows, baseline, seeds)
    logger.info(
        "Style check: %d variants × %d seeds in %.1f s",
        len(rows),
        len(seeds),
        time.time() - started,
    )
    return StyleCheckResponse(
        model_id=base.model_id,
        prompt=base.prompt,
        seeds=seeds,
        feature_model=fm.name,
        clip_model=clip_name if clip_sims else None,
        seed_distance=round(seed_distance, 6) if seed_distance is not None else None,
        baseline_content=round(base_content, 6) if base_content is not None else None,
        baseline_consistency=(round(c, 6) if (c := similarity(baseline)) is not None else None),
        baseline=baseline,
        rows=rows,
        cells=cells,
        image_base64=to_b64(sheet),
        notes=notes,
        elapsed_s=round(time.time() - started, 2),
    )
