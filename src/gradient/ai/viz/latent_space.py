# gradient-project-root/src/gradient/ai/viz/latent_space.py

import logging
import time
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

# Import UMAP
try:
    import umap
except ImportError:
    umap = None
    logging.getLogger(__name__).error("umap-learn not found. Latent space visualization will be unavailable. Install it using 'poetry add umap-learn'.")

# --- CORRECTED IMPORTS --- vvv
# Import necessary components from diffusion loading
from gradient.ai.diffusion.loading import load_diffusion_pipeline, find_model_path # Use find_model_path now
# --- CORRECTED IMPORTS --- ^^^
from diffusers import AutoencoderKL # Need VAE class

# Import Pydantic models
from .models import LatentSpaceVizRequest, LatentSpaceVizResponse, LatentPoint

logger = logging.getLogger(__name__)

# --- Helper Functions (decode_latents_batch, encode_image_to_base64 - Unchanged) ---
def decode_latents_batch(vae: AutoencoderKL, latents: torch.Tensor, resolution: int) -> List[Image.Image]:
    # ... (implementation as before) ...
    if latents.ndim == 3: latents = latents.unsqueeze(0)
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215) # Use getattr for safety
    latents = latents / scaling_factor

    images_pil = []
    with torch.no_grad():
        batch_size = 16
        for i in range(0, latents.shape[0], batch_size):
            latents_batch = latents[i:i + batch_size].to(vae.device, dtype=vae.dtype)
            image_batch = vae.decode(latents_batch).sample
            image_batch = (image_batch / 2 + 0.5).clamp(0, 1)
            image_batch = image_batch.cpu().permute(0, 2, 3, 1).float().numpy()
            for img_np in image_batch:
                img = Image.fromarray((img_np * 255).astype(np.uint8))
                img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
                images_pil.append(img)
    return images_pil


def encode_image_to_base64(img: Image.Image, format="WEBP", quality=75) -> str:
    # ... (implementation as before) ...
    buffered = BytesIO()
    img.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- Main Generator ---

async def generate_latent_space_visualization(request: LatentSpaceVizRequest) -> Optional[LatentSpaceVizResponse]:
    """Generates data points for latent space visualization."""
    if umap is None:
        logger.error("Cannot generate visualization: umap-learn library is not available.")
        return None

    start_time = time.time()
    logger.info(f"Starting latent space visualization generation: {request.num_samples} samples, model={request.model_id}, vae={request.vae_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampling_dtype = torch.float32
    vae_dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Get VAE & Latent Dimensions
    vae: Optional[AutoencoderKL] = None
    latent_channels = 4
    latent_scale_factor = 8

    # --- Load VAE (MODIFIED logic) --- vvv
    vae_load_id_or_path = None # The final ID/path string passed to from_pretrained

    if request.vae_id:
        vae_load_id_or_path = request.vae_id # Start with the requested ID/path
        # Try to find it locally first using the correct helper
        resolved_vae_path = find_model_path(request.vae_id, "vae")
        if resolved_vae_path and (resolved_vae_path.is_file() or resolved_vae_path.is_dir()):
            logger.info(f"Found local VAE path: {resolved_vae_path}")
            vae_load_id_or_path = str(resolved_vae_path)
        else:
             logger.info(f"Local VAE path not found for '{request.vae_id}'. Assuming Hub ID or incorrect path.")
             # Keep vae_load_id_or_path as request.vae_id

        try:
            # TODO: Integrate with resource manager for VAE loading/unloading
            logger.info(f"Attempting to load VAE: '{vae_load_id_or_path}'")
            vae = AutoencoderKL.from_pretrained(vae_load_id_or_path, torch_dtype=vae_dtype).to(device)
            logger.info(f"Successfully loaded specified VAE: {request.vae_id}")
            if hasattr(vae.config, 'latent_channels'): latent_channels = vae.config.latent_channels
            # Calculate scale factor based on blocks if possible
            if hasattr(vae.config, 'block_out_channels') and isinstance(vae.config.block_out_channels, (list, tuple)) and len(vae.config.block_out_channels) > 0 :
                 latent_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            else: # Fallback if block info missing
                 logger.warning("Could not determine latent scale factor from VAE config, using default 8.")
                 latent_scale_factor = 8

        except Exception as e:
            logger.error(f"Failed to load specified VAE '{vae_load_id_or_path}': {e}. Cannot generate previews.", exc_info=True)
            if request.generate_previews: return None
            else: vae = None # Continue without VAE/previews
    # --- End Modified VAE Loading --- ^^^
    else:
        # Try to load VAE from the base model pipeline (as before)
        logger.info(f"No specific VAE requested, attempting to load from base model: {request.model_id}")
        pipeline = await load_diffusion_pipeline(
            model_path_or_id=request.model_id,
            device=device,
            dtype=vae_dtype
        )
        if pipeline and hasattr(pipeline, 'vae') and pipeline.vae is not None:
            vae = pipeline.vae
            logger.info(f"Using VAE from pipeline: {request.model_id}")
            if hasattr(vae.config, 'latent_channels'): latent_channels = vae.config.latent_channels
            if hasattr(vae.config, 'block_out_channels') and isinstance(vae.config.block_out_channels, (list, tuple)) and len(vae.config.block_out_channels) > 0 :
                 latent_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            else: # Fallback
                 logger.warning("Could not determine latent scale factor from pipeline VAE config, using default 8.")
                 latent_scale_factor = 8
            # TODO: Unload pipeline if only VAE was needed? Needs resource manager coordination.
        else:
            logger.error(f"Could not load VAE from pipeline: {request.model_id}. Cannot generate previews.")
            if request.generate_previews: return None
            else: vae = None

    if request.generate_previews and not vae:
         logger.error("Cannot generate previews because VAE loading failed.")
         return None

    # Determine latent dimensions
    latent_height = request.preview_resolution // latent_scale_factor
    latent_width = request.preview_resolution // latent_scale_factor
    if latent_height <= 0 or latent_width <= 0:
         logger.error(f"Calculated invalid latent dimensions ({latent_height}x{latent_width}) for resolution {request.preview_resolution} and scale factor {latent_scale_factor}. Check VAE/resolution.")
         return None
    logger.info(f"Target latent dimensions: {latent_channels}x{latent_height}x{latent_width}")

    # 2. Sample Latent Space (Unchanged)
    logger.info(f"Sampling {request.num_samples} points in latent space...")
    # ... (sampling logic as before) ...
    latents_np = np.random.randn(request.num_samples, latent_channels, latent_height, latent_width).astype(np.float32)
    flattened_latents = latents_np.reshape(request.num_samples, -1)
    logger.info(f"Sampled and flattened latents shape: {flattened_latents.shape}")

    # 3. Perform Dimensionality Reduction (UMAP) (Unchanged)
    logger.info("Running UMAP projection...")
    # ... (UMAP logic as before) ...
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=request.seed, n_jobs=1)
    try:
        projected_points = reducer.fit_transform(flattened_latents)
        logger.info(f"UMAP projection completed. Output shape: {projected_points.shape}")
    except Exception as e:
        logger.error(f"Error during UMAP processing: {e}", exc_info=True)
        return None

    # 4. Generate Previews & Format Output (Unchanged)
    logger.info("Processing results and generating previews (if enabled)...")
    # ... (preview generation and point formatting as before) ...
    output_points: List[LatentPoint] = []
    decoded_images: Optional[List[Image.Image]] = None
    if request.generate_previews and vae:
        try:
            # logger.info("Decoding latent samples...") # Already logged earlier
            latents_torch = torch.from_numpy(latents_np).to(device, dtype=sampling_dtype)
            decoded_images = decode_latents_batch(vae, latents_torch, request.preview_resolution)
            # logger.info(f"Decoded {len(decoded_images)} images.") # Already logged earlier
        except Exception as e:
            logger.error(f"Error decoding latent previews: {e}. Proceeding without previews.", exc_info=True)
            decoded_images = None

    for i in range(request.num_samples):
        preview_base64 = None
        if decoded_images and i < len(decoded_images):
            try:
                preview_base64 = encode_image_to_base64(decoded_images[i], format="WEBP", quality=70)
            except Exception as e:
                logger.warning(f"Failed to encode preview image {i}: {e}")
        point = LatentPoint(
            id=i,
            x=float(projected_points[i, 0]),
            y=float(projected_points[i, 1]),
            z=float(projected_points[i, 2]) if projected_points.shape[1] > 2 else None,
            image_preview_base64=preview_base64
        )
        output_points.append(point)


    # 5. Cleanup (Simplified - relies on future RM integration)
    if request.vae_id and vae:
         # TODO: Properly coordinate with ResourceManager to release the VAE if loaded standalone
         logger.debug("Standalone VAE cleanup relies on future ResourceManager integration.")
         # For now, just deleting the reference if not needed elsewhere
         # del vae # Might delete VAE used by cached pipeline if not careful
         # if torch.cuda.is_available(): torch.cuda.empty_cache()
    # If VAE came from pipeline, assume pipeline cache handles it

    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Latent space visualization generated in {performance_ms:.2f} ms.")

    return LatentSpaceVizResponse(
        request_params=request,
        points=output_points,
        projection_method=request.projection_method, # Should probably get from UMAP params?
        performance_ms=performance_ms
    )