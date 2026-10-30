# GRADIENT API Documentation

This document provides an overview of the GRADIENT backend API. For detailed, interactive documentation, please refer to the automatically generated **Swagger UI** available at the `/docs` endpoint (e.g., `http://localhost:8000/docs`) or the **ReDoc** interface at `/redoc` when the backend server is running.

## Base URL

All API endpoints described here are relative to the base URL `/api/v1`.

## Authentication

Currently, the API does not implement authentication or authorization. Access is unrestricted on the local network where the server is run. This will be addressed in future versions if public deployment is considered.

## Common Conventions

*   **Request Body:** `POST` and `PUT` requests typically expect a JSON request body conforming to the Pydantic models defined for the endpoint.
*   **Response Body:** Successful responses (`2xx`) usually return a JSON body. Error responses (`4xx`, `5xx`) return a JSON object typically containing a `detail` field with an error message or validation details.
*   **Identifiers:** Models (diffusion, LLM, LoRA, etc.) are primarily identified by strings which can be:
    *   Hugging Face Hub IDs (e.g., `runwayml/stable-diffusion-v1-5`).
    *   Local filenames relative to configured subdirectories (e.g., `my_model.safetensors` located in `data/models/checkpoints/`).
    *   Absolute paths on the server's filesystem (use with caution, less portable). The backend attempts to resolve these first if they exist.
*   **Image Format:** When images are required as input (e.g., Img2Img), they are expected as base64 encoded strings within the JSON payload. Output images are returned as base64 encoded strings within the JSON response.

## API Endpoints

Endpoints are grouped by functionality.

### Status (`/status`)

*   **`GET /status`**
    *   **Description:** Checks the basic operational status of the API.
    *   **Response:** `StatusResponse` JSON object containing `status` (e.g., "OK"), `timestamp`, and `version`.

### Generation (`/generate`)

Handles image generation tasks using diffusion models.

*   **`POST /generate/text-to-image`**
    *   **Description:** Generates image(s) from a text prompt and parameters.
    *   **Request Body:** `Text2ImageRequest`
        *   Includes `model_id`, `prompt`, `negative_prompt`, `height`, `width`, `num_inference_steps`, `guidance_scale`, `seed`, `scheduler_id`, `num_outputs`, `custom_sigmas`, `loras`, and optional `modulation` lanes (see the [user guide](user_guide.md#modulation-lanes)).
    *   **Response (Success - 200):** `Text2ImageResponse`
        *   Contains a list of generated `images` (each with `image_base64`, `seed`, `parameters`), echoed `request_params`, and `performance_ms`.
    *   **Response (Error):** `422` (Validation Error), `500` (Internal Server Error), `507` (Insufficient Memory).

*   **`POST /generate/image-to-image`**
    *   **Description:** Generates image variations based on an initial image and a text prompt.
    *   **Request Body:** `ImageToImageRequest`
        *   Includes all `Text2ImageRequest` fields plus `image_base64` (initial image) and `strength`.
    *   **Response (Success - 200):** `ImageToImageResponse`
        *   Contains `images`, `request_params`, `performance_ms`.
    *   **Response (Error):** `422`, `500`, `507`.

*   **`POST /generate/xyz-grid`**
    *   **Description:** Generates a grid of images by varying specified parameters across X, Y, and potentially Z axes. Calls underlying generation pipelines (currently T2I) for each cell.
    *   **Request Body:** `XYZGridRequest`
        *   Contains `base_parameters` (`DiffusionTaskBase`) and optional `x_axis`, `y_axis`, `z_axis` definitions (`AxisParameter`).
    *   **Response (Success - 200):** `XYZGridResponse`
        *   Contains `grid_results` (a list of `XYZGridResult` for each cell, including its parameters, image, or error) and echoed `request_params`.
    *   **Response (Error):** `400` (Invalid grid config), `422`, `500`, `507`.

*   **`POST /generate/inpainting`** (Future)
    *   **Description:** Generates image content within a masked region based on an initial image, mask, and prompt.
    *   **Request Body:** (To be defined - likely `InpaintingRequest` with `image_base64`, `mask_image_base64`, prompt, etc.)
    *   **Response:** (To be defined)

### Lab (`/lab`)

Research tools; see the [user guide](user_guide.md#lab) for what they measure. All take a `base_request` (a `Text2ImageRequest`) and return base64 PNG images.

*   **`POST /lab/taylor-preview`**: image plus ∂image/∂(guidance scale). `method`: `auto` (exact JVP on SD 1.x/SDXL, else finite differences), `jvp`, `finite_difference`; optional `delta`. Returns `base_image_base64`, `tangent_png_base64` (value = 128 + 127·t / `tangent_scale`), `tangent_scale`, `magnitude_heatmap_base64`, `mean_abs_tangent`.
*   **`POST /lab/attention-maps`**: per-word cross-attention maps (SD 1.x/SDXL). Fields `words` (default: every word), `layers` (glob, default `*attn2`), `start`/`end` (fractions of the steps). Returns `image_base64` and `maps` (`word`, `token_positions`, `attention_share`, `heatmap_base64`, `overlay_base64`).
*   **`POST /lab/concept-attention`**: ConceptAttention maps for multi-modal DiTs (FLUX.1, FLUX.2, SD3/3.5). Fields `concepts` (default: content words of the prompt plus "background"), `blocks` (e.g. `"9-18"`, `"all"`; default: the deeper half of the double-stream blocks), `start`/`end`. Returns `image_base64`, `segmentation_base64` and `concepts` (`concept`, `color`, `area`, `mean_probability`, `heatmap_base64`, `overlay_base64`), plus `blocks_recorded`, `num_blocks`, `steps_recorded`. UNet models get a `400` pointing to `/lab/attention-maps`.
*   **`POST /lab/token-ablation`**: leave-one-word-out influence. Fields `mode` (`remove_word` or `mask_embedding`), `words` (default: content words), `max_words`, `feature_model` (default DINOv3 ViT-S/16, falling back to DINOv2-small). Returns `results` ranked by `patch_distance`, each with `global_distance`, `heatmap_base64`, `overlay_base64`, `ablated_image_base64`.
*   **`GET /lab/layers?model_id=...`**: the model's attention layers (`name`, `kind`: self/cross/joint), suggested glob patterns for `modulation` targets and attention maps, and `concept_blocks` (double-stream blocks for ConceptAttention; 0 when the model has none). Loads the model.

Errors: `400` for requests the model can't do (e.g. JVP on a DiT model, unknown words, no matching layers), `422` for invalid fields, `507` when out of memory.

### LLM (`/llm`)

Handles interactions with Large Language Models.

*   **`POST /llm/chat/completions`**
    *   **Description:** Processes a chat history and generates a completion from a specified LLM (currently GGUF). Supports streaming (Server-Sent Events) and non-streaming responses.
    *   **Request Body:** `ChatCompletionRequest`
        *   Includes `messages` (history), `llm_config` (`model_id`, context size, layers, etc.), generation parameters (`temperature`, `top_p`, etc.), and `stream` flag.
    *   **Response (Non-Streaming - 200):** `ChatCompletionResponse`
        *   Contains the generated `choices` (including the message), `usage` statistics.
    *   **Response (Streaming - 200):** `text/event-stream`
        *   Sends JSON chunks (`ChatCompletionChunk`) via SSE.
    *   **Response (Error):** `422`, `500`, `507`.

### Presets (`/presets`)

Manages saved parameter presets for models.

*   **`POST /presets`**
    *   **Description:** Creates a new user-defined preset for a specific model.
    *   **Request Body:** `PresetCreate`
        *   Includes `model_identifier`, `preset_name`, `settings` (`PresetSetting` dictionary), `description`.
    *   **Response (Success - 201):** `PresetRead` (the created preset with its ID).
    *   **Response (Error):** `409` (Conflict - Preset name exists for model), `422`, `500`.

*   **`GET /presets`**
    *   **Description:** Lists all presets, optionally filtered by `model_identifier`.
    *   **Query Parameters:** `model_identifier` (string, optional).
    *   **Response (Success - 200):** `List[PresetRead]`.
    *   **Response (Error):** `500`.

*   **`GET /presets/{preset_id}`**
    *   **Description:** Retrieves a specific preset by its database ID.
    *   **Path Parameters:** `preset_id` (integer).
    *   **Response (Success - 200):** `PresetRead`.
    *   **Response (Error):** `404` (Not Found), `422`, `500`.

*   **`DELETE /presets/{preset_id}`**
    *   **Description:** Deletes a specific *user-defined* preset by its database ID. System presets cannot be deleted.
    *   **Path Parameters:** `preset_id` (integer).
    *   **Response (Success - 204):** No content.
    *   **Response (Error):** `403` (Forbidden - trying to delete system preset), `404` (Not Found), `422`, `500`.

*   **`PUT /presets/{preset_id}`** (Future)
    *   **Description:** Updates an existing user-defined preset.

### System Info (`/system`)

Provides information about the system environment and available resources.

*   **`GET /system/models/diffusion`**
    *   **Description:** Lists diffusion models (checkpoints and diffusers formats) found by scanning configured model directories.
    *   **Response (Success - 200):** `List[str]` (list of model filenames/directory names).
    *   **Response (Error):** `500`.

*   **`GET /system/models/loras`**
    *   **Description:** Lists LoRA files found by scanning the configured LoRA directory.
    *   **Response (Success - 200):** `List[str]`.
    *   **Response (Error):** `500`.

*   **`GET /system/models/embeddings`**
    *   **Description:** Lists Textual Inversion embedding files found by scanning the configured embeddings directory.
    *   **Response (Success - 200):** `List[str]`.
    *   **Response (Error):** `500`.

*   **`GET /system/schedulers`**
    *   **Description:** Attempts to dynamically list available `diffusers` scheduler class names. Includes "Default". May fall back to a static list if dynamic discovery fails.
    *   **Response (Success - 200):** `List[str]`.
    *   **Response (Error):** `500`.

### Visualization (`/visualize`) (Experimental/Basic)

Endpoints related to generating data for visualizations.

*   **`POST /visualize/latent-space`**
    *   **Description:** Generates data points for visualizing latent space using UMAP projection. Optionally decodes latents for image previews.
    *   **Request Body:** `LatentSpaceVizRequest`
        *   Includes `model_id`, `vae_id`, `num_samples`, `seed`, `generate_previews`, `preview_resolution`, `projection_method` (currently UMAP).
    *   **Response (Success - 200):** `LatentSpaceVizResponse`
        *   Contains list of `points` (each with `x`, `y`, `z`?, `image_preview_base64`?) and metadata.
    *   **Response (Error):** `422`, `500`, `507`.

*   **`POST /visualize/parameter-sensitivity`**
    *   **Description:** Analyzes sensitivity to a specific parameter (placeholder/experimental).
    *   **Request Body:** `SensitivityAnalysisRequest`
    *   **Response:** `SensitivityAnalysisResponse`
    *   **Response (Error):** `400`, `422`, `500`, `507`.

## Future Endpoints

*   ControlNet generation endpoints.
*   Endpoints for managing background tasks (queue status, results).
*   Endpoints for managing model downloads or configuration.
*   Endpoints related to specific research experiments.

