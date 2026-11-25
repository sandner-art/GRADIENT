# src/gradient/ai/diffusion/loading_embeddings.py
"""Textual inversion embeddings for SD 1.x, SD 2.x and SDXL.

You use an embedding as in A1111: write its token, the file name without extension, in the
prompt or the negative prompt (ComfyUI's `embedding:name` works too). GRADIENT loads the
installed embeddings named in either prompt by itself; `request.embeddings` can name more.
A loaded token stays in the model's tokenizer until the model is unloaded, so switching
between embeddings costs nothing.

SDXL embeddings hold two tensors (`clip_l` and `clip_g`), one per text encoder. An embedding
for another model family is not loaded; the render reports it in `embeddings_skipped`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from gradient.model_manager.inventory import find_model_path, list_model_names

logger = logging.getLogger(__name__)

EMBEDDING_SUFFIXES = (".safetensors", ".pt", ".bin", ".pth")
FAMILY_LABELS = {"sd1": "SD 1.x", "sd2": "SD 2.x", "sdxl": "SDXL"}
# which model families (gradient.core.families) each embedding family fits
MODEL_FAMILIES = {"sd1": {"sd15"}, "sd2": {"sd15", "sd2"}, "sdxl": {"sdxl"}}
_WIDTH_FAMILY = {768: "sd1", 1024: "sd2"}
_COMFY = re.compile(r"\bembedding:([^\s,()\[\]{}<>|:]+)")


@dataclass(frozen=True)
class EmbeddingInfo:
    name: str  # file name in the embeddings folder
    token: str  # what you write in the prompt
    family: str | None  # "sd1", "sd2", "sdxl", or None when the file can't be read
    vectors: int  # prompt tokens the embedding uses
    error: str | None = None


# --------------------------------------------------------------------------- files
def read_tensors(path: Path) -> dict[str, torch.Tensor]:
    """The embedding's tensors: {"clip_l", "clip_g"} for SDXL, else {"": vectors × width}."""
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        return {"": _matrix(state)}
    if not isinstance(state, dict):
        raise ValueError("not an embedding file")
    if "clip_l" in state and "clip_g" in state:
        return {"clip_l": _matrix(state["clip_l"]), "clip_g": _matrix(state["clip_g"])}
    if isinstance(state.get("string_to_param"), dict):  # A1111 .pt
        params = state["string_to_param"]
        tensor = params.get("*", next(iter(params.values()), None))
    elif "emb_params" in state:  # A1111 .safetensors
        tensor = state["emb_params"]
    else:  # diffusers learned_embeds: {token: tensor}
        tensors = [v for v in state.values() if isinstance(v, torch.Tensor)]
        tensor = tensors[0] if len(tensors) == 1 else None
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("not an embedding file")
    return {"": _matrix(tensor)}


def _matrix(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float()
    if tensor.ndim == 1:
        tensor = tensor[None]
    if tensor.ndim != 2:
        raise ValueError(f"unexpected embedding shape {tuple(tensor.shape)}")
    return tensor


def family_of(tensors: dict[str, torch.Tensor]) -> str | None:
    if "clip_g" in tensors:
        return "sdxl"
    return _WIDTH_FAMILY.get(int(tensors[""].shape[-1]))


_info_cache: dict[tuple[str, int, int], EmbeddingInfo] = {}


def embedding_info(name: str) -> EmbeddingInfo:
    path = find_model_path(name, "embeddings")
    token = Path(name).stem
    if path is None or not path.is_file():
        return EmbeddingInfo(name, token, None, 0, "not found")
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _info_cache:
        try:
            tensors = read_tensors(path)
            vectors = int(next(iter(tensors.values())).shape[0])
            _info_cache[key] = EmbeddingInfo(name, path.stem, family_of(tensors), vectors)
        except Exception as e:
            _info_cache[key] = EmbeddingInfo(name, path.stem, None, 0, f"unreadable: {e}")
    return _info_cache[key]


def installed_embeddings() -> list[str]:
    return [n for n in list_model_names("embeddings") if n.lower().endswith(EMBEDDING_SUFFIXES)]


def fits(family: str | None, model_family: str | None) -> bool | None:
    """True/False when both families are known, else None."""
    if not family or not model_family:
        return None
    return model_family in MODEL_FAMILIES.get(family, set())


# --------------------------------------------------------------------------- prompts
def normalize_prompt(text: str | None) -> str | None:
    """ComfyUI's `embedding:name(.ext)` becomes the plain token `name`."""
    if not text or "embedding:" not in text:
        return text

    def plain(m: re.Match) -> str:
        word = m.group(1)
        stem, dot, ext = word.rpartition(".")
        return stem if dot and f".{ext.lower()}" in EMBEDDING_SUFFIXES else word

    return _COMFY.sub(plain, text)


def token_pattern(token: str) -> re.Pattern:
    """The token as a whole word (names often contain '-', '_' or '.')."""
    return re.compile(rf"(?<![\w\-.]){re.escape(token)}(?![\w\-])")


def embeddings_in_prompts(texts: Iterable[str | None], installed: list[str]) -> list[str]:
    """Installed embeddings whose token appears in any of `texts` (first file per token)."""
    joined = "\n".join(normalize_prompt(t) or "" for t in texts)
    found: dict[str, str] = {}
    for name in installed:
        token = Path(name).stem
        if token not in found and token in joined and token_pattern(token).search(joined):
            found[token] = name
    return list(found.values())


# --------------------------------------------------------------------------- loading
def clip_encoders(pipeline) -> list[tuple]:
    """(text_encoder, tokenizer) pairs when every text encoder is CLIP (SD 1.x/2.x, SDXL).

    Models with a T5 or LLM text encoder (SD3, FLUX, ...) get an empty list: their CLIP part
    alone barely steers the image, so an embedding would look applied but do little.
    """
    pairs = []
    for suffix in ("", "_2", "_3"):
        encoder = getattr(pipeline, f"text_encoder{suffix}", None)
        if encoder is None:
            continue
        tokenizer = getattr(pipeline, f"tokenizer{suffix}", None)
        if tokenizer is None or not type(encoder).__name__.startswith("CLIPText"):
            return []
        pairs.append((encoder, tokenizer))
    return pairs


def _width(encoder) -> int:
    return int(encoder.get_input_embeddings().weight.shape[-1])


def _plan(pipeline, tensors: dict[str, torch.Tensor]) -> list[tuple]:
    """(tensor, text_encoder, tokenizer) per encoder, or ValueError with a plain reason."""
    encoders = clip_encoders(pipeline)
    family = family_of(tensors)
    label = FAMILY_LABELS.get(family or "", "this")
    if "clip_g" in tensors:
        if len(encoders) != 2:
            raise ValueError("an SDXL embedding; this model has one CLIP text encoder")
        plan = [(tensors["clip_l"], *encoders[0]), (tensors["clip_g"], *encoders[1])]
    else:
        if len(encoders) != 1:
            raise ValueError(
                f"a {label} embedding; this model needs SDXL embeddings (clip_l and clip_g)"
            )
        plan = [(tensors[""], *encoders[0])]
    for tensor, encoder, _ in plan:
        if int(tensor.shape[-1]) != _width(encoder):
            raise ValueError(
                f"a {label} embedding (width {int(tensor.shape[-1])}); "
                f"this model's text encoder has width {_width(encoder)}"
            )
    return plan


def load_textual_inversions(
    pipeline, embeddings: Iterable[str] | None
) -> tuple[set[str], dict[str, str]]:
    """Load embeddings by name. Returns (loaded names, {skipped name: reason})."""
    loaded: set[str] = set()
    skipped: dict[str, str] = {}
    names = list(dict.fromkeys(embeddings or []))
    if not names:
        return loaded, skipped
    if not callable(getattr(pipeline, "load_textual_inversion", None)) or not clip_encoders(
        pipeline
    ):
        reason = f"{type(pipeline).__name__} does not use embeddings (SD 1.x, SD 2.x and SDXL do)"
        return loaded, dict.fromkeys(names, reason)
    for name in names:
        path = find_model_path(name, "embeddings")
        if path is None or not path.is_file():
            skipped[name] = "not found in the embeddings folders"
            continue
        token = path.stem
        try:
            plan = _plan(pipeline, read_tensors(path))
        except Exception as e:
            skipped[name] = str(e)
            continue
        missing = [step for step in plan if token not in step[2].added_tokens_encoder]
        if not missing:
            loaded.add(name)  # loaded earlier (e.g. under another name for the same file)
            continue
        if any(token in tokenizer.get_vocab() for _, _, tokenizer in missing):
            skipped[name] = f"'{token}' is an ordinary word for this tokenizer; rename the file"
            continue
        try:
            for tensor, encoder, tokenizer in missing:
                dtype = encoder.get_input_embeddings().weight.dtype
                if dtype.is_floating_point:
                    tensor = tensor.to(dtype)
                pipeline.load_textual_inversion(
                    tensor, token=token, text_encoder=encoder, tokenizer=tokenizer
                )
        except Exception as e:
            skipped[name] = f"could not be loaded: {e}"
            logger.error("Embedding %s could not be loaded: %s", name, e)
            continue
        loaded.add(name)
        logger.info("Loaded embedding %s as token %s", path.name, token)
    for name, reason in skipped.items():
        logger.warning("Embedding %s not used: %s", name, reason)
    return loaded, skipped
