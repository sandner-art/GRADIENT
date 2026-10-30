# src/gradient/ai/diffusion/loading_embeddings.py
"""Textual Inversion embeddings (SD1.x / SD2.x style pipelines)."""

import logging

from diffusers import DiffusionPipeline

from gradient.model_manager.inventory import find_model_path

logger = logging.getLogger(__name__)


def load_textual_inversions(pipeline: DiffusionPipeline, embeddings: list[str] | None) -> set[str]:
    """Load embeddings by file name; the trigger token is the file stem. Returns loaded names."""
    loaded: set[str] = set()
    if not embeddings:
        return loaded
    if not callable(getattr(pipeline, "load_textual_inversion", None)):
        logger.warning(
            "%s does not support textual inversion; ignoring %s",
            type(pipeline).__name__,
            embeddings,
        )
        return loaded
    for name in embeddings:
        path = find_model_path(name, "embeddings")
        if path is None or not path.is_file():
            logger.warning("Embedding '%s' not found in embeddings folders", name)
            continue
        token = path.stem
        try:
            pipeline.load_textual_inversion(str(path), token=token)
            loaded.add(name)
            logger.info("Loaded embedding %s as token <%s>", path.name, token)
        except ValueError as e:
            # Typically "token already in tokenizer" -> already loaded, fine.
            logger.info("Embedding %s not (re)loaded: %s", name, e)
            loaded.add(name)
        except Exception as e:
            logger.error("Failed to load embedding %s: %s", name, e)
    return loaded
