# src/gradient/ai/diffusion/loading_embeddings.py (NEW FILE)

import logging
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)

# Assume find_model_path helper exists elsewhere and can find embeddings
# from .loading import find_model_path # Or from gradient.core.paths import find_model_path
# Placeholder implementation if not imported:
def find_embedding_path(embedding_name: str) -> Optional[Path]:
     from .loading import find_model_path # Temporary import
     return find_model_path(embedding_name, "embeddings")


async def load_textual_inversions(
    pipeline: DiffusionPipeline,
    embeddings: Optional[List[str]] = None
):
    """
    Explicitly loads Textual Inversion embeddings into a pipeline.
    Often unnecessary if embeddings are correctly placed and named,
    but provides robustness.

    Args:
        pipeline: The loaded DiffusionPipeline object.
        embeddings: List of embedding filenames or trigger tokens.
    """
    if not embeddings:
        return

    if not hasattr(pipeline, 'load_textual_inversion') or not callable(pipeline.load_textual_inversion):
        logger.warning("Pipeline does not support `load_textual_inversion`. Embeddings must be loaded via prompt triggers or model's internal loading.")
        return

    loaded_count = 0
    logger.info(f"Attempting to explicitly load {len(embeddings)} embedding(s)...")
    for emb_name in embeddings:
        emb_path_or_id = emb_name # Assume it might be a Hub ID if not found locally
        resolved_emb_path = find_embedding_path(emb_name) # Use helper

        if resolved_emb_path and resolved_emb_path.is_file():
            emb_path_or_id = str(resolved_emb_path)
            logger.debug(f"Found local embedding path: {emb_path_or_id}")
            try:
                # Diffusers load_textual_inversion often needs the token(s).
                # Simple approach: assume filename without suffix is the token.
                # More robust: require user to provide { "name": "file.pt", "token": "<trigger>" }
                token = resolved_emb_path.stem
                pipeline.load_textual_inversion(emb_path_or_id, token=token)
                logger.info(f"Loaded embedding '{emb_name}' with assumed token '{token}'")
                loaded_count += 1
            except ValueError as ve:
                 # Catch specific errors like token mismatch if possible
                 logger.error(f"Failed to load embedding '{emb_name}' (token issue?): {ve}", exc_info=False) # Less verbose log
            except Exception as e:
                logger.error(f"Failed to load embedding '{emb_name}' from path '{emb_path_or_id}': {e}", exc_info=True)
        else:
            logger.warning(f"Embedding file not found locally for '{emb_name}'. Cannot load explicitly.")
            # TODO: Add support for loading from Hub by ID? Requires repo_id info.

    logger.info(f"Explicitly loaded {loaded_count} embeddings.")