# src/gradient/ai/lab/attention.py
"""Per-word cross-attention maps for UNet models (DAAM-style aggregation).

Attention probabilities are computed explicitly only for the selected layers and steps
(everything else keeps the fast fused kernel), averaged over heads, layers and steps,
upsampled to latent resolution and summed over each word's tokens.
"""

import time

from gradient.ai.diffusion.loading import get_pipeline
from gradient.ai.diffusion.pipelines import generate_images, resolve_seed

from .imaging import colorize, overlay, to_b64
from .models import AttentionMapRequest, AttentionMapResponse, WordMap
from .probes import AttentionRecorder, word_token_positions


def attention_maps(request: AttentionMapRequest) -> AttentionMapResponse:
    """Blocking; run through run_gpu_task."""
    start = time.time()
    if request.end <= request.start:
        raise ValueError("end must be greater than start")
    base = request.base_request.model_copy(deep=True)
    base.seed = resolve_seed(base.seed)

    recorder = AttentionRecorder(request.layers, request.start, request.end)
    result = generate_images(base, probes=[recorder])
    image = result.images[0]
    pipe, _ = get_pipeline(
        base.model_id, vae_id=base.vae_id, loras=base.loras, embeddings=base.embeddings
    )

    f = getattr(pipe, "vae_scale_factor", 8)
    maps = recorder.token_maps(base.height // f, base.width // f)
    if maps is None:
        raise ValueError("No attention was recorded (check the step range and layer pattern)")

    words = word_token_positions(
        pipe.tokenizer, base.prompt, getattr(pipe.tokenizer, "model_max_length", 77)
    )
    if request.words:
        wanted = {w.lower() for w in request.words}
        words = [(w, pos) for w, pos in words if w.lower() in wanted]
        if not words:
            raise ValueError("None of the requested words occur in the prompt")

    out = []
    for word, positions in words:
        positions = [p for p in positions if p < maps.shape[-1]]
        if not positions:
            continue
        m = maps[..., positions].sum(-1).numpy()
        out.append(
            WordMap(
                word=word,
                token_positions=positions,
                attention_share=float(m.mean()),
                heatmap_base64=to_b64(colorize(m).resize(image.size)),
                overlay_base64=to_b64(overlay(image, m)),
            )
        )
    return AttentionMapResponse(
        image_base64=to_b64(image),
        seed=base.seed,
        maps=out,
        layers_recorded=sorted(recorder.sums),
        performance_ms=(time.time() - start) * 1000,
    )
