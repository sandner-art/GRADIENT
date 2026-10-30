# src/gradient/ai/lab/ablation.py
"""Token ablation: how much does each word change the image?

For every word we re-render with the same seed after removing it (from the prompt text,
or by replacing its token embeddings with the empty prompt's), then measure the change
with self-supervised vision features (DINOv3, or DINOv2 as fallback): a global distance
and a per-patch distance map showing *where* the word acted. This is a causal,
model-agnostic counterpart to attention maps.
"""

import re
import time

import torch

from gradient.ai.diffusion.loading import get_pipeline
from gradient.ai.diffusion.pipelines import generate_images, resolve_seed

from .features import distances, load_feature_model
from .imaging import colorize, overlay, thumbnail_b64, to_b64
from .models import AblationRequest, AblationResponse, AblationResult
from .probes import word_token_positions

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "with", "by", "for", "from", "is", "are",
    "as", "its", "it", "this", "that", "very", "into", "over", "under",
}  # fmt: skip


def content_words(prompt: str, limit: int) -> list[str]:
    seen: list[str] = []
    for word in re.findall(r"[\w'-]+", prompt):
        lw = word.lower()
        if lw in STOPWORDS or lw.isdigit() or lw in (w.lower() for w in seen):
            continue
        seen.append(word)
        if len(seen) >= limit:
            break
    return seen


def remove_word(prompt: str, word: str) -> str:
    text = re.sub(rf"(?i)(?<![\w'-]){re.escape(word)}(?![\w'-])", "", prompt)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,")


def _masked_embeddings(pipe, request, positions: list[int]) -> dict:
    """Prompt embeddings with `positions` replaced by the empty prompt's embeddings (SD/SDXL)."""
    device = pipe._execution_device
    cfg = isinstance(request.guidance_scale, (int, float)) and request.guidance_scale > 1
    if type(pipe).__name__ == "StableDiffusionXLPipeline":
        pe, ne, ppe, npe = pipe.encode_prompt(
            prompt=request.prompt, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=cfg, negative_prompt=request.negative_prompt or None,
        )  # fmt: skip
        empty = pipe.encode_prompt(
            prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
        )[0]
        pe = pe.clone()
        pe[:, positions] = empty[:, positions]
        out = {"prompt_embeds": pe, "pooled_prompt_embeds": ppe}
        if cfg:
            out.update(negative_prompt_embeds=ne, negative_pooled_prompt_embeds=npe)
        return out
    if type(pipe).__name__ == "StableDiffusionPipeline":
        pe, ne = pipe.encode_prompt(request.prompt, device, 1, cfg, request.negative_prompt or None)
        empty = pipe.encode_prompt("", device, 1, False)[0]
        pe = pe.clone()
        pe[:, positions] = empty[:, positions]
        out = {"prompt_embeds": pe}
        if cfg:
            out["negative_prompt_embeds"] = ne
        return out
    raise ValueError(
        f"mask_embedding mode supports SD 1.x/SDXL; use remove_word for {type(pipe).__name__}"
    )


def token_ablation(request: AblationRequest) -> AblationResponse:
    """Blocking; run through run_gpu_task."""
    start = time.time()
    base = request.base_request.model_copy(deep=True)
    base.seed = resolve_seed(base.seed)
    words = request.words or content_words(base.prompt, request.max_words)
    words = words[: request.max_words]
    if not words:
        raise ValueError("No words to ablate")

    fm, note = load_feature_model(request.feature_model)
    base_image = generate_images(base).images[0]
    pipe, _ = get_pipeline(
        base.model_id, vae_id=base.vae_id, loras=base.loras, embeddings=base.embeddings
    )
    positions = (
        {w.lower(): p for w, p in word_token_positions(pipe.tokenizer, base.prompt)}
        if request.mode == "mask_embedding"
        else {}
    )

    results = []
    for word in words:
        if request.mode == "remove_word":
            ablated_prompt = remove_word(base.prompt, word)
            if ablated_prompt == base.prompt:
                continue
            image = generate_images(
                base.model_copy(update={"prompt": ablated_prompt or " "})
            ).images[0]
        else:
            ablated_prompt = None
            pos = positions.get(word.lower())
            if not pos:
                continue
            with torch.no_grad():
                overrides = _masked_embeddings(pipe, base, pos)
            image = generate_images(base, call_overrides=overrides).images[0]
        global_d, patch_map = distances(fm, base_image, image)
        results.append(
            AblationResult(
                word=word,
                ablated_prompt=ablated_prompt,
                global_distance=global_d,
                patch_distance=float(patch_map.mean()),
                heatmap_base64=to_b64(colorize(patch_map).resize(base_image.size)),
                overlay_base64=to_b64(overlay(base_image, patch_map)),
                ablated_image_base64=thumbnail_b64(image, 384),
            )
        )
    results.sort(key=lambda r: r.patch_distance, reverse=True)
    return AblationResponse(
        base_image_base64=to_b64(base_image),
        seed=base.seed,
        mode=request.mode,
        feature_model=fm.name,
        feature_note=note,
        results=results,
        performance_ms=(time.time() - start) * 1000,
    )
