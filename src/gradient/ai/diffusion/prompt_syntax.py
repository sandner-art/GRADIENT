# src/gradient/ai/diffusion/prompt_syntax.py
"""A1111 prompt syntax for models whose text encoders are all CLIP (SD 1.x, SD 2.x, SDXL).

* Weights: `(word)` ×1.1, `((word))` ×1.21, `[word]` ÷1.1, `(word:1.3)`; `\\(` `\\)` `\\[`
  `\\]` are plain brackets.
* Long prompts: CLIP reads 75 tokens at a time. A longer prompt continues in another 75-token
  chunk; the chunks are encoded one by one and joined, as in A1111 and ComfyUI (diffusers
  alone drops everything after the 77th token). A chunk ends after a comma when there is one
  among its last 20 tokens, so words aren't cut apart.
* `BREAK` starts a new chunk.
* Prompt editing: `[to:when]` adds `to` from step `when`, `[from:to:when]` swaps `from` for
  `to`, `[from::when]` removes `from`. `when` below 1 is a fraction of the steps (0.5 =
  halfway), else a step number.
* Alternation: `[a|b]` uses `a` on one step and `b` on the next.

Two ways to apply weights (`prompt_syntax`): "a1111" multiplies a token's embedding by its
weight and then restores the chunk's mean, as A1111 does; "comfy" scales the token's
difference from the empty prompt's embedding at that position, as ComfyUI does. "plain"
sends the text to the pipeline as it is.

Prompts without any of this syntax, of at most 75 tokens, go through the pipeline's own
encoding unchanged, so they render exactly as before.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import torch

from gradient.ai.lab.probes import Probe

from .loading_embeddings import clip_encoders

logger = logging.getLogger(__name__)

ROUND, SQUARE = 1.1, 1 / 1.1
COMMA_BACKTRACK = 20  # tokens: break a full chunk after a comma this close to its end
MODES = ("a1111", "comfy", "plain")
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_WEIGHT_END = re.compile(rf":\s*({_NUM})\s*\)")
_WHEN = re.compile(rf"\s*({_NUM})\s*")
_BREAK = re.compile(r"\s*\bBREAK\b\s*")


class PromptSyntaxError(ValueError):
    pass


# --------------------------------------------------------------------------- parsing
def parse_weights(text: str) -> list[tuple[str, float | None]]:
    """(text, weight) runs by A1111's emphasis rules; ("", None) marks a BREAK."""
    runs: list[list[Any]] = []
    round_open: list[int] = []
    square_open: list[int] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        for k, part in enumerate(_BREAK.split("".join(buf))):
            if k:
                runs.append(["", None])
            if part:
                runs.append([part, 1.0])
        buf.clear()

    def scale(start: int, factor: float) -> None:
        for run in runs[start:]:
            if run[1] is not None:
                run[1] *= factor

    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n and text[i + 1] in "()[]\\":
            buf.append(text[i + 1])
            i += 2
            continue
        if c == "(":
            flush()
            round_open.append(len(runs))
        elif c == "[":
            flush()
            square_open.append(len(runs))
        elif c == ":" and round_open and (m := _WEIGHT_END.match(text, i)):
            flush()
            scale(round_open.pop(), float(m.group(1)))
            i = m.end()
            continue
        elif c == ")" and round_open:
            flush()
            scale(round_open.pop(), ROUND)
        elif c == "]" and square_open:
            flush()
            scale(square_open.pop(), SQUARE)
        else:
            buf.append(c)
        i += 1
    flush()
    for start in round_open:  # unclosed brackets weigh up to the end, as in A1111
        scale(start, ROUND)
    for start in square_open:
        scale(start, SQUARE)
    merged: list[list[Any]] = []
    for part, weight in runs:
        if merged and weight is not None and merged[-1][1] == weight:
            merged[-1][0] += part
        else:
            merged.append([part, weight])
    return [(part, weight) for part, weight in merged] or [("", 1.0)]


def uses_weights(text: str) -> bool:
    """True when the text has weights, BREAK or escaped brackets (not plain text)."""
    if not any(ch in text for ch in "()[]\\") and "BREAK" not in text:
        return False
    return parse_weights(text) != [(text, 1.0)]


def _group_end(text: str, i: int) -> int:
    """Index of the ']' closing the '[' at i (nesting and escapes respected), or -1."""
    depth, j = 0, i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def _split_top(inner: str, sep: str) -> list[str]:
    parts, depth, start, j = [], 0, 0, 0
    while j < len(inner):
        c = inner[j]
        if c == "\\":
            j += 2
            continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        elif c == sep and depth == 0:
            parts.append(inner[start:j])
            start = j + 1
        j += 1
    parts.append(inner[start:])
    return parts


def prompt_at_step(text: str, step: int, total: int) -> str:
    """The prompt as it reads at `step` (0-based) of `total`: editing and alternation resolved."""
    if "[" not in text:
        return text
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            out.append(text[i : i + 2])
            i += 2
            continue
        if c == "[" and (j := _group_end(text, i)) > 0:
            inner = text[i + 1 : j]
            alternatives = _split_top(inner, "|")
            parts = _split_top(inner, ":")
            if len(alternatives) > 1:
                chosen = alternatives[step % len(alternatives)]
            elif len(parts) in (2, 3) and _WHEN.fullmatch(parts[-1]):
                when = float(parts[-1])
                at = when * total if when < 1 else when
                before, after = ("", parts[0]) if len(parts) == 2 else (parts[0], parts[1])
                chosen = before if step < at else after
            else:  # emphasis: keep the brackets for the weights
                out.append("[" + prompt_at_step(inner, step, total) + "]")
                i = j + 1
                continue
            out.append(prompt_at_step(chosen, step, total))
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def schedule(text: str, total: int) -> list[tuple[int, str]]:
    """[(first step, prompt text)] for each change over `total` steps."""
    entries: list[tuple[int, str]] = []
    for step in range(max(1, total)):
        current = prompt_at_step(text, step, total)
        if not entries or entries[-1][1] != current:
            entries.append((step, current))
        if "[" not in text:
            break
    return entries


# --------------------------------------------------------------------------- tokens and chunks
def _token_ids(pipe, tokenizer, text: str) -> list[int]:
    convert = getattr(pipe, "maybe_convert_prompt", None)
    if callable(convert):  # multi-vector embeddings: "tok" -> "tok tok_1 tok_2"
        text = convert(text, tokenizer)
    return list(tokenizer(text, add_special_tokens=False, truncation=False).input_ids)


def token_count(pipe, text: str | None) -> int | None:
    """Tokens of `text` for the model's first CLIP tokenizer (None for other models)."""
    encoders = clip_encoders(pipe)
    if not encoders or not text:
        return None if not encoders else 0
    tokens = _weighted_tokens(pipe, encoders[0][1], parse_weights(text))
    return sum(1 for t in tokens if t is not None)


def _weighted_tokens(pipe, tokenizer, runs) -> list[tuple[int, float] | None]:
    out: list[tuple[int, float] | None] = []
    for text, weight in runs:
        if weight is None:
            out.append(None)
        elif text:
            out.extend((t, weight) for t in _token_ids(pipe, tokenizer, text))
    return out


def chunk_tokens(
    tokens: list[tuple[int, float] | None], size: int, comma_id: int | None
) -> list[list[tuple[int, float]]]:
    """Split into chunks of at most `size` tokens (BREAK = None ends a chunk)."""
    chunks: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for token in tokens:
        if token is None:
            chunks.append(current)
            current = []
            continue
        if len(current) == size:
            commas = [k for k, (t, _) in enumerate(current) if t == comma_id]
            if commas and size - 1 - commas[-1] < COMMA_BACKTRACK and commas[-1] < size - 1:
                cut = commas[-1] + 1
                chunks.append(current[:cut])
                current = current[cut:]
            else:
                chunks.append(current)
                current = []
        current.append(token)
    if current or not chunks:
        chunks.append(current)
    return chunks


def _ids(tokenizer, chunk: list[tuple[int, float]]) -> tuple[list[int], list[float], list[int]]:
    length = tokenizer.model_max_length
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    real = [tokenizer.bos_token_id, *(t for t, _ in chunk), tokenizer.eos_token_id]
    ids = real + [pad] * (length - len(real))
    weights = [1.0, *(w for _, w in chunk), 1.0] + [1.0] * (length - len(real))
    mask = [1] * len(real) + [0] * (length - len(real))
    return ids, weights, mask


# --------------------------------------------------------------------------- encoding
@dataclass
class Encoded:
    embeds: torch.Tensor  # [1, 77 * chunks, D]
    pooled: torch.Tensor | None  # [1, P] (SDXL), from the first chunk
    chunks: int
    tokens: int
    weighted: bool


class PromptEncoder:
    """Encodes prompts the A1111 way with a pipeline's own CLIP text encoders."""

    def __init__(self, pipe, mode: str = "a1111") -> None:
        self.pipe = pipe
        self.mode = mode
        self.encoders = clip_encoders(pipe)
        if not self.encoders:
            raise PromptSyntaxError(f"{type(pipe).__name__} has no CLIP-only text encoders")
        self.sdxl = len(self.encoders) == 2  # hidden_states[-2] of both, pooled from the second
        self.device = getattr(pipe, "_execution_device", None) or torch.device("cpu")
        self._empty: list[torch.Tensor] | None = None

    def _device(self, encoder) -> torch.device:
        try:  # the device the encoder runs on, also with CPU offload hooks
            from diffusers.utils.torch_utils import get_module_execution_device

            return get_module_execution_device(encoder)
        except Exception:
            return self.device

    def _run(self, encoder, ids, mask) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = self._device(encoder)
        ids_t = torch.tensor(ids, device=device)
        if self.sdxl:
            out = encoder(ids_t, output_hidden_states=True)
            pooled = out[0] if out[0].ndim == 2 else None
            return out.hidden_states[-2], pooled
        use_mask = getattr(encoder.config, "use_attention_mask", False)
        mask_t = torch.tensor(mask, device=device) if use_mask else None
        return encoder(ids_t, attention_mask=mask_t)[0], None

    def _empty_chunk(self, index: int) -> torch.Tensor:
        """The empty prompt's chunk embedding for encoder `index` ([77, D])."""
        if self._empty is None:
            self._empty = []
            for encoder, tokenizer in self.encoders:
                ids, _, mask = _ids(tokenizer, [])
                self._empty.append(self._run(encoder, [ids], [mask])[0][0])
        return self._empty[index]

    def chunks(self, text: str) -> list[list[list[tuple[int, float]]]]:
        """Per encoder, the chunks of (token, weight); equal chunk lengths across encoders."""
        runs = parse_weights(text)
        per_encoder = []
        for _, tokenizer in self.encoders:
            comma = tokenizer(",", add_special_tokens=False).input_ids
            per_encoder.append(
                chunk_tokens(
                    _weighted_tokens(self.pipe, tokenizer, runs),
                    tokenizer.model_max_length - 2,
                    comma[0] if len(comma) == 1 else None,
                )
            )
        if len({tuple(len(c) for c in chunks) for chunks in per_encoder}) > 1:
            raise PromptSyntaxError("the two text encoders split the prompt differently")
        return per_encoder

    def encode(self, text: str, min_chunks: int = 1) -> Encoded:
        per_encoder = self.chunks(text)
        n = max(len(per_encoder[0]), min_chunks)
        hidden: list[torch.Tensor] = []
        pooled = None
        weighted = False
        for index, ((encoder, tokenizer), chunks) in enumerate(
            zip(self.encoders, per_encoder, strict=True)
        ):
            chunks = chunks + [[]] * (n - len(chunks))
            rows = [_ids(tokenizer, chunk) for chunk in chunks]
            z, pool = self._run(encoder, [r[0] for r in rows], [r[2] for r in rows])
            w = torch.tensor([r[1] for r in rows], device=z.device, dtype=z.dtype)
            if bool((w != 1).any()):
                weighted = True
                z = self._weigh(z, w, index)
            hidden.append(z)
            if pool is not None:
                pooled = pool[:1]
        z = torch.cat(hidden, dim=-1) if len(hidden) > 1 else hidden[0]
        tokens = sum(len(c) for c in per_encoder[0])
        return Encoded(z.reshape(1, -1, z.shape[-1]), pooled, n, tokens, weighted)

    def _weigh(self, z: torch.Tensor, w: torch.Tensor, index: int) -> torch.Tensor:
        if self.mode == "comfy":
            empty = self._empty_chunk(index).to(z.device, z.dtype)
            return (z - empty) * w[..., None] + empty
        mean = z.mean(dim=(1, 2), keepdim=True)
        z = z * w[..., None]
        new_mean = z.mean(dim=(1, 2), keepdim=True)
        # restore each chunk's mean (skipped where a mean is ~0, which would blow up)
        ok = (mean.abs() > 1e-6) & (new_mean.abs() > 1e-6)
        return torch.where(ok, z * (mean / torch.where(ok, new_mean, 1.0)), z)

    @staticmethod
    def zeros_like(other: Encoded) -> Encoded:
        pooled = torch.zeros_like(other.pooled) if other.pooled is not None else None
        return Encoded(torch.zeros_like(other.embeds), pooled, other.chunks, 0, False)


# --------------------------------------------------------------------------- the render probe
def plan(
    pipe, prompt: str, negative: str | None, mode: str, total_steps: int
) -> PromptSyntaxProbe | None:
    """A probe that encodes (and schedules) the prompts, or None when the pipeline's own
    encoding gives the same result (plain prompts of at most 75 tokens) or can't be used."""
    if mode == "plain" or not clip_encoders(pipe):
        return None
    negative = negative or ""
    pos = schedule(prompt, total_steps)
    neg = schedule(negative, total_steps)
    texts = {t for _, t in pos} | {t for _, t in neg}
    limit = clip_encoders(pipe)[0][1].model_max_length - 2
    if (
        pos == [(0, prompt)]
        and neg == [(0, negative)]
        and not any(uses_weights(t) for t in texts)
        and all((token_count(pipe, t) or 0) <= limit for t in texts)
    ):
        return None
    return PromptSyntaxProbe(mode, pos, neg)


class PromptSyntaxProbe(Probe):
    """Encodes the prompts with the A1111 syntax and switches them where they change."""

    tensor_inputs = ("prompt_embeds", "add_text_embeds")

    def __init__(self, mode: str, pos: list[tuple[int, str]], neg: list[tuple[int, str]]):
        self.mode = mode
        self.pos = pos
        self.neg = neg
        self.encoded: dict[str, Encoded] = {}
        self.zero_negative = False
        self.current: tuple[str, str] | None = None

    @staticmethod
    def _at(entries: list[tuple[int, str]], step: int) -> str:
        text = entries[0][1]
        for start, value in entries:
            if start <= step:
                text = value
        return text

    def attach(self, pipe, entry, total_steps, kwargs):
        encoder = PromptEncoder(pipe, self.mode)
        texts = list(dict.fromkeys([t for _, t in self.pos] + [t for _, t in self.neg]))
        n = max(len(encoder.chunks(t)[0]) for t in texts)
        self.encoded = {t: encoder.encode(t, min_chunks=n) for t in texts}
        # SDXL renders an empty negative prompt as zeros (force_zeros_for_empty_prompt)
        self.zero_negative = encoder.sdxl and bool(
            getattr(pipe.config, "force_zeros_for_empty_prompt", False)
        )
        pos, neg = self._pair(0)
        kwargs.pop("prompt", None)
        kwargs.pop("negative_prompt", None)
        kwargs["prompt_embeds"] = pos.embeds
        kwargs["negative_prompt_embeds"] = neg.embeds
        if pos.pooled is not None:
            kwargs["pooled_prompt_embeds"] = pos.pooled
            kwargs["negative_pooled_prompt_embeds"] = neg.pooled
        self.current = (self._at(self.pos, 0), self._at(self.neg, 0))

    def _pair(self, step: int) -> tuple[Encoded, Encoded]:
        pos = self.encoded[self._at(self.pos, step)]
        neg_text = self._at(self.neg, step)
        if self.zero_negative and not neg_text:
            return pos, PromptEncoder.zeros_like(pos)
        return pos, self.encoded[neg_text]

    def on_step_end(self, pipe, step, timestep, cb_kwargs):
        nxt = step + 1
        key = (self._at(self.pos, nxt), self._at(self.neg, nxt))
        if key == self.current or "prompt_embeds" not in cb_kwargs:
            return cb_kwargs
        self.current = key
        pos, neg = self._pair(nxt)
        cfg = bool(getattr(pipe, "do_classifier_free_guidance", False))
        old = cb_kwargs["prompt_embeds"]
        images = old.shape[0] // (2 if cfg else 1)

        def joined(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            p = p.expand(images, *p.shape[1:])
            return torch.cat([q.expand(images, *q.shape[1:]), p]) if cfg else p

        new = joined(pos.embeds, neg.embeds)
        if new.shape == old.shape:
            cb_kwargs["prompt_embeds"] = new.to(old.device, old.dtype)
        old_pooled = cb_kwargs.get("add_text_embeds")
        if pos.pooled is not None and torch.is_tensor(old_pooled):
            new_pooled = joined(pos.pooled, neg.pooled)
            if new_pooled.shape == old_pooled.shape:
                cb_kwargs["add_text_embeds"] = new_pooled.to(old_pooled.device, old_pooled.dtype)
        return cb_kwargs

    def report(self) -> dict[str, Any]:
        first = self.encoded.get(self.pos[0][1])
        out: dict[str, Any] = {"prompt_syntax_applied": True}
        if first is not None:
            out["prompt_tokens"] = first.tokens
            out["prompt_chunks"] = first.chunks
        if any(e.weighted for e in self.encoded.values()):
            out["prompt_weights_applied"] = True
        if len(self.pos) > 1 or len(self.neg) > 1:
            out["prompt_schedule"] = [[s, t] for s, t in self.pos]
            if len(self.neg) > 1:
                out["negative_prompt_schedule"] = [[s, t] for s, t in self.neg]
        return out
