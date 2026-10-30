# src/gradient/ai/lab/concept_attention.py
"""ConceptAttention for multi-modal DiTs: FLUX.1, FLUX.2 and SD3/3.5.

Method (Helbling et al., "ConceptAttention: Diffusion Transformers Learn Highly
Interpretable Features", ICML 2025):

* Every concept word becomes one extra token that runs through the double-stream
  blocks with the *text stream's own weights* (modulation, projections, QK-norms,
  output projection, MLP).
* Concept queries attend to the image tokens and to the other concepts. Image and
  prompt tokens never attend to concepts, so the generated image is unchanged.
* The saliency map of one block is softmax over concepts of  o_x · o_c,  the dot
  product between image and concept attention outputs taken *before* the output
  projection. Maps are averaged over the selected blocks and steps.

Implementation notes for GRADIENT:

* Nothing in the model is replaced. Forward hooks read the image keys/values
  (`attn.to_k` / `attn.to_v` outputs) and the image attention output (the input of
  `attn.to_out[0]`) of each double-stream block, and run the small concept stream
  next to it. The image is bit-identical to a normal render, and the extra cost is
  a few tokens per block.
* All concept computation happens inside the attention module's forward, while the
  block's weights are on the device (works with model/group/sequential offload).
* Concept embeddings come from the pipeline's own `encode_prompt` on each word
  alone. The token is located from the input ids the text encoder actually saw:
  the first token for T5 (as in the reference code), the last token of the word
  for causal encoders (CLIP, Qwen3, Mistral).
* FLUX.1 modulates the concept stream with a zero pooled-text vector, as the
  reference code does; SD3 uses the prompt's conditioning, FLUX.2's modulation is
  text-independent. Concept RoPE uses the text position of the concept's token.
* Only the first transformer call of each step is recorded (the conditional pass);
  with batched CFG (SD3) the conditional half of the batch is used.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gradient.ai.diffusion.loading import get_pipeline
from gradient.ai.diffusion.pipelines import generate_images, resolve_seed

from .ablation import content_words
from .imaging import colorize, overlay, to_b64
from .models import ConceptAttentionRequest, ConceptAttentionResponse, ConceptMap
from .probes import Probe, step_range

#: double-stream block class -> family adapter
SUPPORTED_BLOCKS = {
    "FluxTransformerBlock": "flux1",
    "JointTransformerBlock": "sd3",
    "Flux2TransformerBlock": "flux2",
}
# Tableau 10, cycled when there are more concepts
PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]
MAX_CONCEPTS = 20


# --------------------------------------------------------------------------- model support
def concept_support(pipe) -> tuple[str, int] | None:
    """(adapter kind, number of double-stream blocks), or None if unsupported."""
    blocks = getattr(getattr(pipe, "transformer", None), "transformer_blocks", None)
    if not blocks:
        return None
    kind = SUPPORTED_BLOCKS.get(type(blocks[0]).__name__)
    return (kind, len(blocks)) if kind else None


def parse_blocks(spec: str | None, n: int) -> list[int]:
    """'9-18', '12,15,18', 'all'; default (None/'') = the deeper half of the blocks."""
    text = (spec or "").strip().lower()
    if not text or text == "default":
        return list(range(n // 2, n))
    if text == "all":
        return list(range(n))
    chosen: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                chosen.update(range(min(a, b), max(a, b) + 1))
            else:
                chosen.add(int(part))
        except ValueError:
            raise ValueError(
                f"Invalid block list '{spec}'; use e.g. '9-18' or '12,15,18'"
            ) from None
    bad = sorted(i for i in chosen if not 0 <= i < n)
    if bad:
        raise ValueError(f"Blocks {bad} don't exist; this model has double-stream blocks 0-{n - 1}")
    return sorted(chosen)


# --------------------------------------------------------------------------- concept embeddings
def _token_source(pipe, kind: str):
    """Text encoder + tokenizer whose token positions line up with prompt_embeds, and which token to use."""
    if kind == "flux1":  # prompt_embeds = T5 output
        return pipe.text_encoder_2, pipe.tokenizer_2, "first"
    if kind == "sd3":  # prompt_embeds = [CLIP-L|CLIP-G (77 tokens), T5]; use the CLIP part
        return pipe.text_encoder, pipe.tokenizer, "last"
    return pipe.text_encoder, pipe.tokenizer, "last"  # flux2: causal LLM (Qwen3 / Mistral)


def _find_span(ids: list[int], tokenizer, word: str) -> tuple[int, int] | None:
    for variant in dict.fromkeys((word, " " + word, word.lower(), " " + word.lower())):
        sub = tokenizer(variant, add_special_tokens=False).input_ids
        if not sub:
            continue
        for i in range(len(ids) - len(sub) + 1):
            if ids[i : i + len(sub)] == sub:
                return i, i + len(sub)
    return None


def _encode(pipe, text: str) -> torch.Tensor:
    params = inspect.signature(pipe.encode_prompt).parameters
    kwargs: dict[str, Any] = {"prompt": text}
    for name in params:  # e.g. SD3 requires prompt_2 / prompt_3
        if name.startswith("prompt_") and name[7:].isdigit():
            kwargs[name] = text
    if "do_classifier_free_guidance" in params:
        kwargs["do_classifier_free_guidance"] = False
    if "device" in params:
        kwargs["device"] = pipe._execution_device
    return pipe.encode_prompt(**kwargs)[0]


@torch.inference_mode()
def encode_concepts(pipe, kind: str, concepts: list[str]) -> tuple[torch.Tensor, list[int]]:
    """One embedding per concept, in the transformer's text-input space: (1, C, D) and token positions."""
    encoder, tokenizer, which = _token_source(pipe, kind)
    seen: list[torch.Tensor] = []

    def grab(_module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is not None:
            seen.append(ids.detach().cpu())

    vectors, positions = [], []
    handle = encoder.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        for concept in concepts:
            seen.clear()
            embeds = _encode(pipe, concept)
            if not seen:
                raise RuntimeError(
                    f"{type(encoder).__name__} was not called while encoding '{concept}'"
                )
            ids = seen[0][0].tolist()
            span = _find_span(ids, tokenizer, concept)
            if span is None:
                raise ValueError(
                    f"Could not find the tokens of '{concept}' in its own encoding; try a simpler word"
                )
            pos = span[0] if which == "first" else span[1] - 1
            pos = min(pos, embeds.shape[1] - 1)
            vectors.append(embeds[0, pos])
            positions.append(pos)
    finally:
        handle.remove()
    return torch.stack(vectors)[None], positions


# --------------------------------------------------------------------------- probe
class ConceptAttentionProbe(Probe):
    """Runs the concept stream next to the double-stream blocks and records concept maps."""

    def __init__(
        self,
        kind: str,
        concept_embeds: torch.Tensor,
        positions: list[int],
        blocks: list[int],
        start: float = 0.0,
        end: float = 1.0,
        zero_pooled: bool = True,
    ) -> None:
        self.kind = kind
        #: FLUX.1 only: modulate concepts with a zero pooled vector (reference) or the prompt's
        self.zero_pooled = zero_pooled
        #: set to {} to keep each block's concept attention output of the first recorded call (tests)
        self.trace: dict[int, torch.Tensor] | None = None
        self.concept_embeds = concept_embeds
        self.positions = positions
        self.record_blocks = set(blocks)
        self.start, self.end = start, end
        self.handles: list[Any] = []
        self.sum: torch.Tensor | None = None
        self.count = 0
        self.blocks_seen: set[int] = set()
        self.steps_seen: set[int] = set()
        self.tokens = 0

    # ---- lifecycle
    def attach(self, pipe, entry, total_steps, kwargs):
        self.transformer = pipe.transformer
        self.blocks = list(self.transformer.transformer_blocks)
        self.total = max(1, total_steps)
        self.step = 0
        self.calls = 0
        self.active = False
        self.handles.append(
            self.transformer.register_forward_pre_hook(self._on_transformer, with_kwargs=True)
        )
        for i, block in enumerate(self.blocks):
            attn = block.attn
            if getattr(attn, "fused_projections", False):
                raise ValueError("ConceptAttention needs unfused QKV projections")
            self.handles += [
                block.register_forward_pre_hook(partial(self._on_block, i), with_kwargs=True),
                attn.to_k.register_forward_hook(partial(self._grab, i, "k")),
                attn.to_v.register_forward_hook(partial(self._grab, i, "v")),
                attn.to_out[0].register_forward_pre_hook(partial(self._on_attn_out, i)),
            ]

    def on_step_end(self, pipe, step, timestep, cb_kwargs):
        self.step = step + 1
        self.calls = 0
        return cb_kwargs

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.stream = None
        self.k = self.v = None

    def report(self):
        return {
            "concept_attention_blocks": sorted(self.blocks_seen),
            "concept_attention_steps": len(self.steps_seen),
        }

    # ---- hooks
    def _on_transformer(self, _module, _args, kwargs):
        lo, hi = step_range(self.start, self.end, self.total)
        self.active = self.calls == 0 and lo <= self.step < hi
        self.calls += 1
        self.stream = None  # (re)initialised lazily inside the first block
        self.temb_inputs = {
            k: kwargs.get(k) for k in ("timestep", "guidance", "pooled_projections")
        }
        self.current = None

    def _on_block(self, i, _module, _args, kwargs):
        if self.active:
            self.current = i
            self.block_kwargs = kwargs
            self.k = self.v = None

    def _grab(self, i, which, _module, _inputs, output):
        if self.active and self.current == i:
            setattr(self, which, output)

    def _on_attn_out(self, i, _module, args):
        if not (self.active and self.current == i):
            return
        self.current = None
        if self.k is None or self.v is None:
            raise RuntimeError(
                "ConceptAttention could not read the image keys/values of this model"
            )
        self._concept_block(i, args[0])
        self.k = self.v = None

    # ---- the concept stream
    def _init_stream(self, device, dtype) -> None:
        embeds = self.concept_embeds.to(device=device, dtype=dtype)
        self.stream = self.transformer.context_embedder(embeds)
        if self.kind == "flux1" and self.zero_pooled:
            # reference implementation: concepts get the time/guidance embedding with a zero pooled vector
            t = self.temb_inputs["timestep"][-1:].to(dtype) * 1000
            pooled = torch.zeros_like(self.temb_inputs["pooled_projections"][-1:]).to(dtype)
            g = self.temb_inputs.get("guidance")
            embed = self.transformer.time_text_embed
            self.concept_temb = (
                embed(t, pooled) if g is None else embed(t, g[-1:].to(dtype) * 1000, pooled)
            )

    def _modulate(self, block) -> tuple[torch.Tensor, Callable | None]:
        """Normalised concept tokens for attention, and the residual update to apply afterwards."""
        c = self.stream
        kw = self.block_kwargs
        if self.kind == "flux2":
            from diffusers.models.transformers.transformer_flux2 import Flux2Modulation

            mod = kw["temb_mod_txt"]
            mod = mod[-1:] if mod.shape[0] > 1 else mod
            (sh_msa, sc_msa, g_msa), (sh_mlp, sc_mlp, g_mlp) = Flux2Modulation.split(mod, 2)
            normed = (1 + sc_msa) * block.norm1_context(c) + sh_msa

            def update(ctx):
                x = c + g_msa * ctx
                n = block.norm2_context(x) * (1 + sc_mlp) + sh_mlp
                return x + g_mlp * block.ff_context(n)

            return normed, update

        use_own = self.kind == "flux1" and self.zero_pooled
        temb = self.concept_temb if use_own else kw["temb"][-1:]
        if getattr(block, "context_pre_only", False):  # last SD3 block: no context update
            return block.norm1_context(c, temb), None
        normed, g_msa, sh_mlp, sc_mlp, g_mlp = block.norm1_context(c, emb=temb)

        def update(ctx):
            x = c + g_msa.unsqueeze(1) * ctx
            n = block.norm2_context(x) * (1 + sc_mlp[:, None]) + sh_mlp[:, None]
            x = x + g_mlp.unsqueeze(1) * block.ff_context(n)
            return x.clip(-65504, 65504) if x.dtype == torch.float16 else x

        return normed, update

    def _concept_block(self, i: int, image_out: torch.Tensor) -> None:
        from diffusers.models.embeddings import apply_rotary_emb

        block = self.blocks[i]
        attn = block.attn
        if self.stream is None:
            self._init_stream(image_out.device, image_out.dtype)
        heads = attn.heads
        head_dim = self.k.shape[-1] // heads

        def split(x):
            return x.unflatten(-1, (heads, head_dim))  # (B, S, H, D)

        k_x, v_x = split(self.k[-1:]), split(self.v[-1:])
        if getattr(attn, "norm_k", None) is not None:
            k_x = attn.norm_k(k_x)

        normed, update = self._modulate(block)
        q_c, k_c, v_c = (
            split(attn.add_q_proj(normed)),
            split(attn.add_k_proj(normed)),
            split(attn.add_v_proj(normed)),
        )
        if getattr(attn, "norm_added_q", None) is not None:
            q_c = attn.norm_added_q(q_c)
        if getattr(attn, "norm_added_k", None) is not None:
            k_c = attn.norm_added_k(k_c)

        rope = self.block_kwargs.get("image_rotary_emb")
        if rope is not None:  # FLUX: rows are [text tokens, image tokens]
            cos, sin = rope
            n_txt = self.block_kwargs["encoder_hidden_states"].shape[1]
            k_x = apply_rotary_emb(k_x, (cos[n_txt:], sin[n_txt:]), sequence_dim=1)
            pos = torch.tensor([min(p, n_txt - 1) for p in self.positions], device=cos.device)
            c_rope = (cos[pos], sin[pos])
            q_c = apply_rotary_emb(q_c, c_rope, sequence_dim=1)
            k_c = apply_rotary_emb(k_c, c_rope, sequence_dim=1)

        out = F.scaled_dot_product_attention(
            q_c.transpose(1, 2),
            torch.cat([k_x, k_c], dim=1).transpose(1, 2),
            torch.cat([v_x, v_c], dim=1).transpose(1, 2),
        )
        concept_out = out.transpose(1, 2).flatten(2).to(normed.dtype)  # (1, C, H*D)

        if self.trace is not None and i not in self.trace:
            self.trace[i] = concept_out.detach().clone()
        if i in self.record_blocks:
            scores = torch.einsum("pd,cd->cp", image_out[-1].float(), concept_out[0].float())
            probs = scores.softmax(dim=0).cpu()
            self.sum = probs if self.sum is None else self.sum + probs
            self.count += 1
            self.blocks_seen.add(i)
            self.steps_seen.add(self.step)
            self.tokens = probs.shape[1]

        if update is not None and getattr(attn, "to_add_out", None) is not None:
            self.stream = update(attn.to_add_out(concept_out))

    # ---- result
    def concept_maps(self) -> torch.Tensor | None:
        """(C, P) concept probabilities per image token, averaged over blocks and steps."""
        return None if self.sum is None else self.sum / self.count


# --------------------------------------------------------------------------- endpoint logic
def _grid(tokens: int, height: int, width: int) -> tuple[int, int]:
    h = max(1, round((tokens * height / width) ** 0.5))
    if tokens % h:
        raise ValueError(f"Can't arrange {tokens} image tokens as a {height}x{width} grid")
    return h, tokens // h


def _hex(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def default_concepts(prompt: str) -> list[str]:
    words = content_words(prompt, MAX_CONCEPTS - 1)[:7]
    return [*words, "background"] if "background" not in words else words


def concept_attention_maps(request: ConceptAttentionRequest) -> ConceptAttentionResponse:
    """Blocking; run through run_gpu_task."""
    started = time.time()
    if request.end <= request.start:
        raise ValueError("end must be greater than start")
    base = request.base_request.model_copy(deep=True)
    base.seed = resolve_seed(base.seed)
    base.num_outputs = 1

    concepts = [c.strip() for c in (request.concepts or default_concepts(base.prompt)) if c.strip()]
    concepts = list(dict.fromkeys(concepts))
    if not concepts:
        raise ValueError("Give at least one concept")
    if len(concepts) > MAX_CONCEPTS:
        raise ValueError(f"At most {MAX_CONCEPTS} concepts")

    pipe, entry = get_pipeline(
        base.model_id, vae_id=base.vae_id, loras=base.loras, embeddings=base.embeddings
    )
    support = concept_support(pipe)
    if support is None:
        block = getattr(getattr(pipe, "transformer", None), "transformer_blocks", [None])
        raise ValueError(
            "ConceptAttention needs a multi-modal DiT with double-stream blocks (FLUX.1, FLUX.2, "
            f"SD3/3.5); {entry.pipeline_class} ({type(block[0]).__name__ if block else 'UNet'}) "
            "is not supported. For SD 1.x/SDXL use the cross-attention maps."
        )
    kind, n_blocks = support
    blocks = parse_blocks(request.blocks, n_blocks)
    embeds, positions = encode_concepts(pipe, kind, concepts)

    probe = ConceptAttentionProbe(kind, embeds, positions, blocks, request.start, request.end)
    result = generate_images(base, probes=[probe])
    image = result.images[0].convert("RGB")
    maps = probe.concept_maps()
    if maps is None:
        raise ValueError("Nothing was recorded (check the step range)")

    gh, gw = _grid(maps.shape[1], image.height, image.width)
    grid = maps.reshape(len(concepts), gh, gw)
    up = F.interpolate(grid[None], size=(image.height, image.width), mode="bilinear")[0]
    winner = up.argmax(dim=0).numpy()
    winner_grid = grid.argmax(dim=0).numpy()

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(concepts))]
    lut = np.array([_hex(c) for c in colors], dtype=np.uint8)
    seg = Image.fromarray(lut[winner], "RGB")
    segmentation = Image.blend(image, seg, 0.55)

    out = []
    for i, concept in enumerate(concepts):
        m = grid[i].numpy()
        out.append(
            ConceptMap(
                concept=concept,
                color=colors[i],
                area=float((winner_grid == i).mean()),
                mean_probability=float(m.mean()),
                heatmap_base64=to_b64(
                    colorize(np.clip(m, 0, 1), already_normalized=True).resize(
                        image.size, Image.Resampling.BILINEAR
                    )
                ),
                overlay_base64=to_b64(overlay(image, m, absolute=True)),
            )
        )
    return ConceptAttentionResponse(
        image_base64=to_b64(image),
        seed=base.seed,
        segmentation_base64=to_b64(segmentation),
        concepts=out,
        blocks_recorded=sorted(probe.blocks_seen),
        num_blocks=n_blocks,
        steps_recorded=len(probe.steps_seen),
        performance_ms=(time.time() - started) * 1000,
    )
