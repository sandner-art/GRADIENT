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

Handles image generation tasks using diffusion models. These endpoints wait for the result, but the work still goes through the render queue (it shows in the Queue panel and can be stopped there; a stopped request returns `409`). To queue work without waiting, use `POST /queue`.

*   **`POST /generate/text-to-image`**
    *   **Description:** Generates image(s) from a text prompt and parameters.
    *   **Request Body:** `Text2ImageRequest`
        *   Includes `model_id`, `prompt`, `negative_prompt`, `height`, `width`, `num_inference_steps`, `guidance_scale`, `seed`, `scheduler_id`, `num_outputs`, `custom_sigmas`, `loras`, and optional `modulation` lanes (see the [user guide](user_guide.md#modulation-lanes)).
    *   **Response (Success - 200):** `Text2ImageResponse`
        *   Contains a list of generated `images` (each with `image_base64`, `seed`, `parameters` and `output_id`, the image's id in the gallery), echoed `request_params`, `performance_ms` and `run_uuid`.
    *   **Response (Error):** `409` (stopped from the queue), `422` (Validation Error), `500` (Internal Server Error), `507` (Insufficient Memory).

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

### Queue (`/queue`)

One worker renders jobs in order; Lab requests run ahead of waiting renders. Jobs (without input images) are kept in the database so the list survives a restart; jobs that were unfinished at shutdown come back as failed and can be retried.

*   **`POST /queue`**: queue jobs and return at once (`202`). Body: `kind` (`text_to_image`, `image_to_image` or `xyz_grid`), `payload` (the request body of the matching `/generate` endpoint), optional `label`, `count` (1–100, default 1) and `vary_seed` (default `true`: with a fixed seed, job *i* uses `seed + i`). Returns `{"jobs": [...]}`.
*   **`GET /queue`**: `jobs` (running, waiting in order, then recent finished jobs), `running` (id or null) and `waiting` (count). A job has `id`, `kind`, `label`, `status` (`queued`, `running`, `completed`, `failed`, `cancelled`), `position` (when waiting), `progress` (`step`/`total`, and `item`/`items` for grids), `result` (`run_uuid`, `outputs` with gallery ids and seeds, `performance_ms`), `error` and `retryable`.
*   **`POST /system/shutdown`** (`by`: `web` or `cli`): stops GRADIENT gracefully, like the first Ctrl+C (a running job stops at its next step). Only from this computer (403 otherwise), and only when it was started by `gradient serve` (409 otherwise). `GET /system/info` also returns `version`, `installed_version` and `install_is_stale`.
*   Every generation request takes `quantization` (`none`, `bnb-8bit`, `bnb-4bit`, `torchao-int8`, `torchao-int8-dynamic`, `torchao-fp8`, `gguf:<file>`, optionally with `+text`); empty uses the model's default (`performance.default_quantization`). **`GET /system/quantization`** lists the options with `available` and `reason` (backend missing, needs an NVIDIA GPU) and the GGUF denoiser files found; `/system/models/diffusion` leaves those files out. `/system/info` shows each loaded pipeline's `quantization`.
*   **`GET /queue/{id}/preview`**: the newest live preview of a running render as a JPEG (`404` until there is one). `progress.preview` counts previews (use it to refresh the image) and `progress.preview_step` is the step shown. Previews are an estimate of the finished image, mapped to RGB with a projection fitted to the model's VAE; set `performance.live_preview: false` to turn them off. Jobs also carry their settings as `payload` (input images left out).
*   **`GET /queue/events`**: server-sent events: a `snapshot` on connect, then `job` (a job changed), `removed` and `order` events.
*   **`GET /queue/{id}`**, **`POST /queue/{id}/cancel`** (removes a waiting job or stops a running one at the next step), **`POST /queue/{id}/move`** (`{"to": "top" | "up" | "down" | "bottom"}`), **`POST /queue/{id}/retry`** (queues a copy of a failed or cancelled job), **`DELETE /queue/finished`**.

### History / gallery (`/history`)

*   **`GET /history/outputs`**: images, newest first. Filters: `run` (run uuid), `model`, `run_type` (e.g. `text_to_image`, `xyz_grid`, `lab_*`), `q` (words in prompt or notes), `min_rating`, `favorite`, `tag`, `since`/`until` (ISO dates); `sort` (`newest`, `oldest`, `rating`), `offset`, `limit` (≤ 200). Returns `items`, `total` and `next_offset`.
*   **`GET /history/outputs/{id}`**: one image with its full `parameters`, `file_path` and `exists`. **`PATCH`** changes `rating` (1–5, 0 clears), `favorite`, `notes` and `tags` (replaces your own tags; tags from other sources stay). **`DELETE`** moves the file to `<outputs>/.trash/<run>/` and forgets it.
*   **`POST /history/outputs/bulk`**: `{"ids": [...], "action": "rate" | "favorite" | "add_tags" | "remove_tags" | "delete", "value": ...}`.
*   **`GET /history/outputs/{id}/image`** (`?download=true` for a download) and **`/thumb?size=256`** (WebP thumbnails, cached in `<outputs>/.thumbs`).
*   **`GET /history/facets`**: models, run types and tags with counts, plus the totals.
*   **`GET /history/runs`**: runs with status, model, prompt, image count, cover image ids and error. Finished runs with no images left are hidden unless `include_empty=true`. **`DELETE /history/runs/{uuid}`** moves the run's folder to the trash.

Tags record their `source` (`manual` for tags set in GRADIENT, `xmp` for keywords read from project sidecars, or a caption label) and an optional `confidence`. The bulk `remove_tags` action removes a tag from every source. `GET /history/outputs` also takes `project` (a project uuid); the single-image view adds `captions` (`source`, `text`, `updated`) and `projects` (`uuid`, `name`, `file_name`).

### Projects (`/projects`)

Project folders for other tools; see the [user guide](user_guide.md#project-folders-kurator-artagents-digikam).

*   **`GET /projects`**: `items` (`uuid`, `name`, `path`, `count`, `created`, `updated`, `last_sync`, `exists`, `options`) and `root` (the default folder for new projects: `paths.projects`, or `<data>/projects`; env `GRADIENT_PROJECTS_DIR`). **`GET /projects/{uuid}`**: one project.
*   **`POST /projects/export`**: `ids` *or* `filter` (gallery filter keys: `run`, `model`, `run_type`, `q`, `min_rating`, `favorite`, `tag`, `project`), plus `project` (uuid, to add to it) *or* `name` (and optional `path`) for a new one. Options: `sidecars` (default true), `model_keywords` (default true), `captions` (`none` or `prompt`), `link` (hard links). Returns `added`, `updated`, `kept_captions`, `kept_foreign_sidecars`, `missing_source`, `errors`, `project`, and `sync` when the project existed. A non-empty folder that isn't a project is refused (`400`).
*   **`POST /projects/{uuid}/sync`**: `caption_source` (label for `name.txt` captions, default `artagents`), `captions_as_tags`. Returns counts (`tags_added`, `tags_removed`, `ratings_changed`, `captions`), `renamed`, `missing`, `unknown_files`, `unreadable_sidecars`, `project`.
*   **`POST /projects/open`**: `{"path": ...}` adds an existing project folder (moved, or forgotten) and syncs it. **`DELETE /projects/{uuid}`** forgets a project; its folder stays.

### Lab (`/lab`)

Research tools; see the [user guide](user_guide.md#lab) for what they measure. All take a `base_request` (a `Text2ImageRequest`) and return base64 PNG images. Each result is also saved to the gallery as a run (`run_type` `lab_taylor`, `lab_attention`, `lab_concept_attention` or `lab_token_ablation`; the run folder holds the image, the maps and `report.json`), and responses include `run_uuid`.

*   **`POST /lab/taylor-preview`**: image plus ∂image/∂(guidance scale). `method`: `auto` (exact JVP on SD 1.x/SDXL, else finite differences), `jvp`, `finite_difference`; optional `delta`. Returns `base_image_base64`, `tangent_png_base64` (value = 128 + 127·t / `tangent_scale`), `tangent_scale`, `magnitude_heatmap_base64`, `mean_abs_tangent`, and `note` (set when `auto` fell back from the JVP to finite differences, with the reason).
*   **Explorer** (`/lab/explore/*`). Every request carries the render to explore: `base_request` (give a `seed`; a random one is returned by `start`), optionally `init_image_base64` + `strength` for an image-to-image render, and `path`: the branches taken so far (`ExploreOp`: `step`, `mask_base64` (PNG, white or opaque = region; null = whole image), `kind` (`seed`, `prompt`, `mix`, `lora`), `seed`, `prompt`, `mix_prompt`, `mix_weight`, `lora_weight`, `label`). Trajectories are cached by settings and path, and a path replays exactly.
    *   **`POST /lab/explore/start`** (`frames`): the image at the end of the path, `steps`, the effective `scheduler`, the `prompt` in effect, and `frames` (the denoised estimate at a few steps and where it still changes). Not saved.
    *   **`POST /lab/explore/map`** (`mask_base64`, `steps` or `num_steps`, `branches` per step, `seed`, `feature_model`): `points` per step (`change` relative to step 0, `distance`, `spread`, the free `estimate`, branch thumbnails, a heatmap), `window`, `suggested` steps (`big`, `variations`, `details`), `estimate_error`, `estimate_correlation`, `metric` (`dino` or `pixels`), `summary`, an overview `image_base64`. Saved as a `lab_explore_map` run.
    *   **`POST /lab/explore/tokens`** (`mask_base64`, `method`: `auto`, `attention`, `ablation`; `max_words`, `alternatives`, `feature_model`): `words` with `score` (0–1), `strength`, `lift` (concentration in the region), `alternatives` and an overlay. Not saved.
    *   **`POST /lab/explore/branches`** (`mask_base64`, `step`, `branches`: 1–12 specs with `kind` and its fields; seeds default to seed+1000, +1001… for `seed` branches and seed+1000 for the others): `branches` with `label`, `op` (append it to `path` to continue), `image_base64`, `parameters` (incl. `explore_path`), `change` and `outside_change`. Saved as a `lab_explore_branches` run; each branch is a gallery image whose parameters hold its path.
*   **`POST /lab/quantization-report`**: `base_request`, `variants` (1–4 quantizations), `reference` (default `none`), `num_seeds` (2–8), `taylor` (default true), `feature_model`. Returns per variant `weights_mb`, `denoiser_mb`, `load_seconds`, `seconds_per_image`, `peak_memory_mb`, `psnr`, `ssim`, `mean_abs_diff`, `feature_distance`, `relative_change` (feature distance / distance between two reference seeds), `taylor_cosine`, `verdict`, `strip_base64`, `error`; plus `seeds`, `seed_spread`, `summary` and the overview `image_base64`. Runs as a normal-priority queue job and is saved as a `lab_quantization` run.
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

