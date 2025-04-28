# gradient-project-root/src/gradient/ai/viz/latent_space.py (NEW FILE)

import logging
import time
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

# Import UMAP - ensure umap-learn is installed
try:
    import umap
except ImportError:
    umap = None
    logging.getLogger(__name__).error("umap-learn not found. Latent space visualization will be unavailable. Install it using 'poetry add umap-learn'.")

# Import necessary components from diffusion loading
# Adjust imports based on final structure
from gradient.ai.diffusion.loading import load_diffusion_pipeline, get_vae_base_path, loaded_components # Assuming load_vae might be added there later or we load via pipeline
from diffusers import AutoencoderKL # Need VAE class

# Import Pydantic models
from .models import LatentSpaceVizRequest, LatentSpaceVizResponse, LatentPoint

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def decode_latents_batch(vae: AutoencoderKL, latents: torch.Tensor, resolution: int) -> List[Image.Image]:
    """Decodes a batch of latents into PIL images using the VAE."""
    if latents.ndim == 3: # Add batch dimension if missing
         latents = latents.unsqueeze(0)

    # Scale latents based on VAE config if needed (often needed for SDXL VAE)
    if hasattr(vae.config, 'scaling_factor'):
         latents = latents / vae.config.scaling_factor
    else: # Standard SD 1.5 scaling factor
         latents = 1 / 0.18215 * latents

    images_pil = []
    with torch.no_grad():
        # Process in smaller batches to avoid OOM during decoding if many samples
        batch_size = 16 # Adjust based on VRAM
        for i in range(0, latents.shape[0], batch_size):
            latents_batch = latents[i:i + batch_size].to(vae.device, dtype=vae.dtype)
            image_batch = vae.decode(latents_batch).sample
            image_batch = (image_batch / 2 + 0.5).clamp(0, 1) # Map to [0, 1] range
            image_batch = image_batch.cpu().permute(0, 2, 3, 1).float().numpy() # Move to CPU, float32 for PIL

            for img_np in image_batch:
                img = Image.fromarray((img_np * 255).astype(np.uint8))
                # Resize to preview resolution *after* decoding for better quality
                img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
                images_pil.append(img)
    return images_pil

def encode_image_to_base64(img: Image.Image, format="WEBP", quality=75) -> str:
     """Encodes a PIL image to a base64 string."""
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
    # Use float32 for latents/sampling unless VRAM is extremely limited? Float16 might affect quality here.
    # Let's use float32 for generation/sampling, float16 for VAE if on GPU
    sampling_dtype = torch.float32
    vae_dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Get VAE & Latent Dimensions
    vae: Optional[AutoencoderKL] = None
    latent_channels = 4 # Default for SD 1.5/2.x
    latent_scale_factor = 8 # Default for SD 1.5/2.x

    # Load VAE: Either specified one or from the base model pipeline
    if request.vae_id:
        # TODO: Implement separate VAE loading/caching (similar to pipeline loading)
        # This is simplified for now - assumes VAE path or Hub ID
        vae_load_path_str = None
        potential_vae_path_local = get_vae_base_path() / request.vae_id
        if potential_vae_path_local.is_file():
             vae_load_path_str = str(potential_vae_path_local)
        else:
             vae_load_path_str = request.vae_id # Assume Hub ID

        try:
            # TODO: Integrate with resource manager
            vae = AutoencoderKL.from_pretrained(vae_load_path_str, torch_dtype=vae_dtype).to(device)
            logger.info(f"Loaded specified VAE: {request.vae_id}")
            if hasattr(vae.config, 'latent_channels'): latent_channels = vae.config.latent_channels
            if hasattr(vae.config, 'scaling_factor'): latent_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) # Calculate from blocks
        except Exception as e:
            logger.error(f"Failed to load specified VAE '{request.vae_id}': {e}. Cannot generate previews.", exc_info=True)
            # Decide if we should proceed without previews or fail
            if request.generate_previews:
                 return None # Fail if previews requested but VAE failed
            else:
                 vae = None # Continue without VAE/previews
    else:
        # Try to load VAE from the base model pipeline (less efficient as it loads full pipeline)
        # TODO: Optimize this by loading only the VAE component later
        logger.info(f"No specific VAE requested, attempting to load from base model: {request.model_id}")
        pipeline = await load_diffusion_pipeline(
            model_path_or_id=request.model_id,
            device=device,
            dtype=vae_dtype # Load pipeline with VAE dtype for consistency
        )
        if pipeline and hasattr(pipeline, 'vae'):
            vae = pipeline.vae
            logger.info(f"Using VAE from pipeline: {request.model_id}")
            if hasattr(vae.config, 'latent_channels'): latent_channels = vae.config.latent_channels
            if hasattr(vae.config, 'scaling_factor'): latent_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            # We don't need the full pipeline anymore, maybe unload it? This needs resource manager coordination.
            # For now, keep it loaded, which uses extra VRAM.
        else:
            logger.error(f"Could not load VAE from pipeline: {request.model_id}. Cannot generate previews.")
            if request.generate_previews:
                return None
            else:
                 vae = None # Continue without previews

    if request.generate_previews and not vae:
         logger.error("Cannot generate previews because VAE loading failed.")
         return None

    # Determine latent dimensions based on preview resolution and scale factor
    latent_height = request.preview_resolution // latent_scale_factor
    latent_width = request.preview_resolution // latent_scale_factor
    logger.info(f"Target latent dimensions: {latent_channels}x{latent_height}x{latent_width}")


    # 2. Sample Latent Space
    logger.info(f"Sampling {request.num_samples} points in latent space...")
    if request.seed is not None:
        torch.manual_seed(request.seed)
        np.random.seed(request.seed)

    # Generate random latents (standard normal distribution often works well)
    # Shape: (num_samples, channels, height, width)
    latents_np = np.random.randn(request.num_samples, latent_channels, latent_height, latent_width).astype(np.float32)
    # Optional: Clamp or scale latents if needed based on typical ranges? Usually -3 to 3 covers most variance.
    # latents_np = np.clip(latents_np, -3, 3)

    # Flatten for UMAP: (num_samples, channels*height*width)
    flattened_latents = latents_np.reshape(request.num_samples, -1)
    logger.info(f"Sampled and flattened latents shape: {flattened_latents.shape}")


    # 3. Perform Dimensionality Reduction (UMAP)
    logger.info("Running UMAP projection...")
    reducer = umap.UMAP(
        n_components=2, # TODO: Make configurable (2 or 3)
        n_neighbors=15, # TODO: Make configurable
        min_dist=0.1, # TODO: Make configurable
        metric='cosine', # Common choice for high-dimensional embeddings
        random_state=request.seed,
        n_jobs=1 # Limit CPU usage if needed
    )
    try:
        projected_points = reducer.fit_transform(flattened_latents)
        logger.info(f"UMAP projection completed. Output shape: {projected_points.shape}")
    except Exception as e:
        logger.error(f"Error during UMAP processing: {e}", exc_info=True)
        return None


    # 4. Generate Previews (Optional) & Format Output
    logger.info("Processing results and generating previews (if enabled)...")
    output_points: List[LatentPoint] = []
    decoded_images: Optional[List[Image.Image]] = None

    if request.generate_previews and vae:
        try:
            logger.info("Decoding latent samples to generate previews...")
            latents_torch = torch.from_numpy(latents_np).to(device, dtype=sampling_dtype) # Use sampling dtype here
            decoded_images = decode_latents_batch(vae, latents_torch, request.preview_resolution)
            logger.info(f"Decoded {len(decoded_images)} images.")
        except Exception as e:
            logger.error(f"Error decoding latent previews: {e}. Proceeding without previews.", exc_info=True)
            decoded_images = None # Ensure it's None if decoding fails

    for i in range(request.num_samples):
        preview_base64 = None
        if decoded_images and i < len(decoded_images):
            try:
                # Use WEBP for smaller preview size
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

    # 5. Cleanup (if VAE was loaded separately or pipeline kept loaded)
    # TODO: Implement proper unloading based on how VAE was obtained
    if request.vae_id and vae:
         logger.debug("Cleaning up separately loaded VAE...")
         del vae
         if torch.cuda.is_available(): torch.cuda.empty_cache()
    # If loaded via pipeline, rely on pipeline caching/unloading for now

    end_time = time.time()
    performance_ms = (end_time - start_time) * 1000
    logger.info(f"Latent space visualization generated in {performance_ms:.2f} ms.")

    return LatentSpaceVizResponse(
        request_params=request,
        points=output_points,
        projection_method=request.projection_method,
        performance_ms=performance_ms
    )