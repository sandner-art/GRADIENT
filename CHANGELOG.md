# Changelog

## 0.10.0 (unreleased)

### Memory
- **Freeing VRAM covers everything**: `POST /system/unload` used to free only the image models; the DINO and CLIP/SigLIP models loaded for metrics (`/lab/embed`, ablation, Explorer, quantization check) stayed in VRAM. It now frees image models, vision models and LLMs (after the running GPU job), and takes `targets` (`diffusion`, `vision`, `llm`, `caches`) to free one kind. The answer lists what was freed and the free VRAM before and after. Vision models are tracked like the others, so `GET /system/info` lists them too.
- **Memory** button in the header: VRAM in use, the loaded models by kind with their size, **Free VRAM**, *Unload* per kind and *Clear caches* (Explorer paths, word vocabularies, tokenizers). `GET /system/memory` gives the same report.
- Fixed: the Explorer's word suggestions could reuse the vocabulary of an unloaded model when a new text encoder got the same memory address.

### Lab
- **Style check**: styles × positions (template, before, after, own chunk) × starts × seeds, rendered next to the unstyled image with the same seeds. Each variant gets a *strength* (DINO distance; 1 = as much change as a new seed), *content kept* (CLIP/SigLIP similarity with the unstyled prompt, relative to the unstyled images), *consistency* across seeds, the styled prompt's tokens and notes (missing styles or LoRA tags, a cut prompt). A scatter of strength against content kept, a sortable table, the images per variant, and a gallery run with every image, a contact sheet and the report. `POST /lab/style-check`.
- The automatic feature model (Lab tools, `/lab/embed`) now also uses a DINO model you copied into the vision folder when DINOv3 and DINOv2 small aren't installed.

### Prompts
- **Prompt injection** (SD 1.x/2.x, SDXL): a modulation lane `prompt_injection` gives chosen cross-attention layers another prompt (`replace`, or `blend` with the main one) for a step range. Presets `sdxl-style` (`up_blocks.0.attentions.1`, where InstantStyle found SDXL's style) and `sdxl-layout` (`down_blocks.2.attentions.1`), or any layer pattern. In the Lab's lane editor, with the preset *Style into SDXL style blocks*. Styles have a matching position, **Style blocks only (SDXL)**: your prompt stays as it is and only the style block reads the styled prompt (`style_injection` in the image's settings). The Style check compares it with the other positions.
- **Tokens view**: under the styles, *Tokens: 92 · 2 chunks* for SD/SDXL models, updated as you type (the tokenizer only, a few milliseconds). *Show tokens* lists each 75-token chunk, weights as colour (`red^1.3`), embedding vectors, and with plain text what is cut off after 75 tokens. With styles it shows the prompt as rendered. API: `POST /prompt/tokens`.
- **Styles** (A1111 `styles.csv`): a *Styles* row under the negative prompt picks styles by group (`CX: …`, `PX: …`; search by name or words) and applies them in order. **Import styles…** adds an A1111/Forge `styles.csv` or a JSON styles file (also the ComfyUI stylers' format) to the styles folder (`<data>/styles`, setting `paths.styles`). *Position*: **Template** (A1111: into `{prompt}`, else after the prompt), **Before** or **After** the prompt, or **Own chunk** (after a `BREAK`, so on SD/SDXL the style is encoded apart from your words). *Style from … % of the steps* adds the styles only later in the render (prompt editing), so the composition comes from your prompt. *Show the prompt as rendered* previews the final prompts with their token count and chunks; **Apply to prompt** writes them into the prompt boxes. `{red|blue}` options pick one by the seed (dynamic prompts), and `<lora:name:0.8>` tags load that LoRA (by file name or name without extension) instead of being read as words; tags for missing LoRAs are listed as `lora_tags_missing`. Images keep your prompt in `prompt` and the rendered one in `styled_prompt` / `styled_negative_prompt` (the A1111 `parameters` text has the rendered prompts, as A1111 writes them); *Use settings* restores the styles. Request fields `styles`, `style_position`, `style_start`; XYZ axes `styles` (`none`, a name, `A + B`), `style_position`, `style_start`. API: `GET /styles`, `POST /styles/import`, `POST /styles/preview`.
- **Long prompts and A1111 weights** (SD 1.x, SD 2.x, SDXL): prompts longer than 75 tokens used to be cut off after the 77th token (diffusers' limit), which silently dropped the end of long prompts and styles. They now continue in further 75-token chunks, as in A1111 and ComfyUI, broken after a comma where one is near. `(word:1.2)`, `(word)`, `[word]`, `\(` escapes and `BREAK` work as in A1111, and so do prompt editing (`[cat:dog:0.5]`, `[to:when]`, `[from::when]`) and alternation (`[cat|dog]`), switched at the right steps during the render (also in the negative prompt). `prompt_syntax` (form: *Advanced → Prompt syntax*; XYZ axis) picks how weights are applied: `a1111` (default: scale, then restore the chunk's mean), `comfy` (scale away from the empty prompt, as ComfyUI) or `plain` (the text as it is, as before). Plain prompts of at most 75 tokens render exactly as before. Images record `prompt_tokens`, `prompt_chunks`, `prompt_weights_applied` and `prompt_schedule`; a prompt that was cut has `prompt_truncated`. *Use settings* and the Explorer treat images saved before 0.10 as `plain`, so they render as they did. Lab tools that map words to token positions (attention maps, token ablation, the Explorer's word attention) read prompts as plain text.
- **Embeddings picker** (textual inversion): an *Embeddings* row under the LoRAs lists the installed embeddings with their model family (SD 1.x, SD 2.x, SDXL) and adds one to the prompt or the negative prompt. As in A1111 you can also just type an embedding's name in a prompt (ComfyUI's `embedding:name` works too): GRADIENT loads every installed embedding it finds there. SDXL embeddings (`clip_l` + `clip_g`) now load into both text encoders; A1111 `.pt` and `.safetensors` (`emb_params`) and diffusers files are read. Chips show the embeddings in use; *✕* takes one out of the prompt. An embedding for another model family is not loaded: the job's result has `embeddings_skipped` (the image's settings also the reasons), and the queue shows it, like LoRAs that don't fit, as *Not applied*. Models with a T5 or LLM text encoder (SD3, FLUX…) don't use embeddings. API: `GET /system/models/embeddings/details?model_id=`.

## 0.9.1 (unreleased)

### Working with other apps (ArtAgents LoRA tests)
- **Job groups**: `POST /queue` takes `group`: text-to-image and image-to-image jobs with the same group save their images into one gallery run (type *Group*, one output folder, the group's name as the title in *Runs*), so a sweep of hundreds of images from another app is one run. Queueing more jobs to the same group later adds to it.
- **Tags and origin**: `tags` become gallery tags on the job's images (shown as *from the app that queued the job*), and `origin` (app, run, item…, up to 4 KB of JSON) is kept with each image (`origin` and `group` in the image's details). Retried jobs keep all three, and they survive restarts in the queue's history.
- **LoRAs that don't fit are reported**: a LoRA that matches no layer of the model (one for another model family) used to be skipped with only a console warning, so the image looked like a no-LoRA render. Now the job's result has `loras_skipped`, and the image's parameters record it.

- **Import test sweep…** (Queue panel): a sweep file exported by ArtAgents' LoRA Lab (when ArtAgents can't reach GRADIENT, or for overnight renders) is checked (model installed, each LoRA found by name or by sha256, a different file refused, a path for missing ones), queued as one group, and when the last image is done GRADIENT writes a results folder (images + `artagents-results.json`) that ArtAgents imports as a normal LoRA Lab run. *Test sweeps* in the Queue panel shows progress, the results path and *Write results now*. API: `/sweeps/preview`, `/sweeps`, `/sweeps/{id}`, `/sweeps/{id}/results`.

- **Embeddings for other apps**: `POST /lab/embed` returns L2-normalized image embeddings (DINO) or image and text embeddings (CLIP, SigLIP) from the `vision` models, for images sent as base64 or gallery ids. ArtAgents scores LoRA tests with it (fidelity, drift, leakage, and CLIP-T with a CLIP model) without installing PyTorch. *CLIP ViT-B/32* is in the model catalog (vision). Vision models copied in by hand now show their family (from `config.json`).

### API token for other computers
- `app.api_token` (or `GRADIENT_API_TOKEN`): when set, requests from other computers need `Authorization: Bearer <token>`; a browser opens GRADIENT once with `?token=<token>` (kept in an HttpOnly cookie, removed from the address). Requests from this computer never need it, so the web UI and `gradient stop` work as before; requests through a proxy count as remote.
- The console warns when GRADIENT listens beyond this computer (`--host 0.0.0.0`) without a token.

## 0.9.0 (2026-11-24)

### Explorer: travel through the model, region by region
- **Region branching**: mark a region (rectangle, brush, eraser) and branch it from any step. The model's denoised estimate at that step is noised back with new noise (or under another prompt) and only the remaining steps run, through the model's own image-to-image pipeline; outside the region the latents are copied from the original after every step, so it stays identical. Works for SD 1.x/SDXL (UNet) and FLUX/SD 3.5 (packed or plain latents); image-to-image renders too.
- **Settling map**: branches at several steps measure how much the region can still change (DINO features or pixels, relative to a completely new region), with the step window where it settles and suggested steps for big changes, variations and details. A **free estimate** from the render alone is shown next to it, with its error and spatial correlation, so its precision is measured per model.
- **Word suggestions**: the prompt's words ranked by how much they act in the region (cross-attention on UNets, ConceptAttention on FLUX/SD 3.5, ablation for any model), with alternatives from the text encoder's vocabulary, typed in, or from the local LLM.
- **Branch kinds**: new noise, word swaps (same noise, only the word differs), prompt mixes (embedding average) and LoRA weights.
- **Paths**: continue from any branch and go back along the path; trajectories are cached and paths replay exactly. Branches are saved as gallery images with their path, and *Explore…* on one continues it.
- **Where**: the Lab's new **Explorer** tool (formation strip, settling chart with a table view, words, branches); **Explore** mode in the preview panel beside the render forms; **Explore…** in every image's right-click menu.
- **Branch viewer**: opens after rendering; ← → step through the image and its branches, hold Space to flip to the reference, W wipe, D difference, M region outline, R pins the shown branch as the reference (compare branches with each other), Enter continues, Esc closes.

### Precision must fit the model
- GGUF files are matched to the model by the architecture in their header (SD 1.5, SDXL, SD 3, FLUX.1/2, Chroma, Qwen-Image, HiDream): the Precision list greys out a GGUF for another model (*other model*), and bitsandbytes/GGUF on a single-file checkpoint (*needs a model folder*). Switching to a model the chosen precision doesn't fit resets it to the model default, with a note. The Lab's quantization check offers only what fits its model.
- An explicit precision that doesn't fit is refused before loading, with the reason; a default precision (setting or manifest) that can't be used falls back to full precision (logged) instead of failing every render.
- fp8's description: faster only on GPUs with fp8, elsewhere memory savings at the same speed.

### Startup and version
- The console says which version is ready, the device (or *CPU only*), and the torch/diffusers versions; the web UI shows the version in its header. The version comes from the checkout, so it's right after `git am`/`git pull`; if the installed packages are older, both say to run the update or install script.
- **Clean Ctrl+C**: the first Ctrl+C ends the browsers' live connections (no more "waiting for connections to close"), stops a running job at its next step (marked *stopped*, retryable) and starts no new job; GRADIENT then exits with code 0 and no tracebacks. A second Ctrl+C stops at once.
- **Auto-reload mode stops cleanly too** (`GRADIENT_APP_RELOAD`): the server in the reloader's child process gets the same clean stop, Windows console handling and Stop button; the reloader ends when the server was stopped on request. uvicorn no longer re-raises Ctrl+C after stopping (the KeyboardInterrupt/CancelledError tracebacks), and once stopped GRADIENT exits right away (a Ctrl+C during Python's slow PyTorch teardown used to end with an error code).
- **Stop without Ctrl+C**: **⏻ Stop** in the web UI's header and `gradient stop` (from another terminal) stop it the same way, from this computer only. On Windows, GRADIENT handles console events itself (in their own thread) and switches Ctrl+C back on if the launcher had switched it off, which can happen in IDE terminals; closing the console window also stops it cleanly.
- **Offline notice in the browser**: open pages say *GRADIENT was stopped* or *No connection to GRADIENT* (after 2 s, so short hiccups don't flash), reconnect by themselves, say *Connected to GRADIENT again*, and offer a **Reload** when it came back as another version. Requests that fail because GRADIENT isn't running say so.
- Quieter console: the safety checker isn't loaded (and its two warnings are gone), diffusers' empty "float32 modules" notice and loading progress bars are off, the single-file config fetch shows no progress bars, and explorer batches no longer trigger diffusers' image-duplication FutureWarning.

### Other
- Lab results can hold several gallery images (the explorer's branches); data-URL images in reports are stored correctly.
- Setting `lab.explore_batch` (default 4): branches rendered together.
- `/system/info` returns `version`, `installed_version` and `install_is_stale`.
- `install.ps1`/`update.ps1` forward `-WithQuant`.

### API
- `POST /lab/explore/start`, `/lab/explore/map`, `/lab/explore/tokens`, `/lab/explore/branches`; run types `lab_explore_map`, `lab_explore_branches`.

### Tests
- Exact background lock and replay on SD; FLUX and SD3 branching with packed and plain latents; settling map, word suggestions (attention and ablation), word and mix branches; image-to-image bases; gallery saving with paths; strength rounding for both img2img conventions.

## 0.8.0 (2026-11-24)

### Quantized models
- **Precision** per render (`quantization` on every request; selector next to the model in the render forms and the Lab): bitsandbytes 8-bit / 4-bit NF4 (CUDA), torchao int8 weight-only, int8 dynamic and fp8 weight-only, or a **GGUF** denoiser file with the rest of the pipeline from the selected model. `+text` also quantizes the text encoders. Diffusers-format models are quantized while loading; single-file checkpoints are quantized in place with torchao.
- The precision is part of the loaded pipeline's identity (switching reloads), shapes the memory-mode choice, is recorded in each image's settings and PNG metadata (`Quantization: …`) and restored by *Use settings*.
- `quantization` works as an **XYZ grid axis**; cells are grouped so each precision loads once.
- LoRAs load on top of quantized models; the exact Taylor JVP runs through torchao and GGUF weights.
- Unavailable options say why (backend not installed, needs an NVIDIA GPU). GGUF files in the checkpoints folder are offered as precisions and no longer listed as models.

### Lab: Quantization check
- Renders the Lab settings with the reference and up to four precisions on the same seeds and reports weight sizes, load time, seconds per image, peak GPU memory, PSNR, SSIM, DINO feature distance, **relative change** (feature distance divided by the distance between two reference seeds) and the **Taylor tangent agreement** with the reference, with a verdict per precision, reference | quantized | difference strips and an overview grid. Saved to the gallery as a Lab run.

### Install
- `install.ps1 -WithQuant` / `./install.sh --with-quant` (also forwarded by the update scripts) install bitsandbytes, torchao and gguf.
- New setting `performance.default_quantization` (default `none`).

### API
- `quantization` on generation and Lab requests; `GET /system/quantization`; `POST /lab/quantization-report`.

### Tests
- Spec parsing, availability, torchao renders with LoRA and exact JVP, a GGUF denoiser built from the test model, quantized weight sizes, and the full report.

## 0.7.0 (2026-11-24)

### Rendering
- **Preview panel** beside the Text-to-Image and Image-to-Image forms (below them on narrow screens): **Live** shows the running render step by step and then the result large, **Compare** wipes between two results (click a thumbnail for A, Shift+click for B), **Batch** shows every image of the newest job. Thumbnails: click to show, double-click for the viewer, `[` / `]` to step.
- **Live previews**: each step's latents become an estimate of the finished image (from two consecutive steps of Euler-type samplers, k-diffusion and flow matching alike), mapped to RGB with a linear projection fitted once per model by encoding a colour chart with its own VAE, so SD 1.5, SDXL, SD3, FLUX.1 and custom VAEs need no hand-made constants. Published at most every 0.25 s plus the last step; `GET /queue/{id}/preview`; setting `performance.live_preview`.
- **Send to Lab** from the render forms, the preview panel, the viewer and right-click menus.

### Desktop-style interaction
- **Right-click menus** for gallery tiles (and selections), thumbnails, the large preview, the viewer image and queue jobs: open, use settings (with or without the seed), use as input, send to Lab, rating, favourite, copy prompt/seed, download, show the run, export to a project, delete (two clicks); queue jobs: run next, retry, stop/cancel, use settings.
- **Back button**: the mouse's back button, Alt+Left and the browser's Back/Forward move through the app's own history: closing the viewer or the queue drawer, previous tab, previous gallery view. The tab is in the address (`#gallery`), so a reload keeps it.

### Fixes
- **Taylor preview with custom models**: "expected mat1 and mat2 to have the same dtype (float vs Half)". The JVP sampler now keeps every UNet input in the UNet's weight dtype (a float32 text encoder next to an fp16 UNet caused it), decodes in the VAE's actual weight dtype, and doesn't try to upcast an offloaded VAE. If the exact JVP still fails, `auto` falls back to finite differences and says why (`note`); the log has the traceback.
- The Lab's DINOv3 fallback message explains how to get access (request it on Hugging Face, then download it in the Models tab with a token).

### API
- `GET /queue/{id}/preview`; `progress.preview` / `progress.preview_step`; jobs include their `payload` (input images left out).
- Taylor responses include `note`.

### Tests
- Mixed-dtype JVP (fails with the old sampler), the Taylor fallback, live preview plumbing, the denoised estimate, the fitted projection; frontend tests for navigation history, Send to Lab and the job menu.

## 0.6.0 (2026-11-23)

### Project folders (Kurator, ArtAgents, digiKam)
- **Export** gallery images (a selection, or everything the current filters show) into a project folder: copies of the images, an XMP sidecar per image (`name.png.xmp`: tags, rating, prompt, `Projects/<name>`, optional `GRADIENT/Model/...` and `GRADIENT/Type/...` keywords, in the layout Lightroom, digiKam and Kurator use) and `gradient-project.json`. Optional prompt caption files (`name.txt`); caption files from other tools are never overwritten. Hard links instead of copies on request.
- **Sync** reads a project folder back: keywords added by another program become tags with source `xmp` (and are removed again when removed there; your own tags are never touched), changed `xmp:Rating` values become ratings, and caption files become captions under a label of your choice, one per label, so several captioners can be kept and compared. Files renamed by another tool are found by their content. Exporting again into a project syncs first.
- Gallery: a *Projects* view (sync with a caption label, show a project's images, forget, add an existing folder), a *Project* filter, and *Export* for selections and for the current view.
- Viewer: imported tags show their source and can be removed; captions and the projects an image is in are listed.
- Toasts appear bottom-left while the viewer is open, so they don't cover its details.

### API
- `/api/v1/projects`: list, export, sync, open, forget.
- `GET /history/outputs?project=<uuid>`; image details include `captions` and `projects`; facets include `projects`. Bulk `remove_tags` removes a tag from every source.
- New setting `paths.projects` (default `<data>/projects`), env `GRADIENT_PROJECTS_DIR`.

### Database
- New tables `projects`, `project_items` and `output_captions` (created on start).

### Tests
- Sidecar round trip; export → edits as Kurator/ArtAgents/digiKam would make them → sync → re-export → removal → forget/reopen; export by filter and the error cases.

## 0.5.0 (2026-11-23)

### Rendering and queue
- **Render queue**: Text-to-Image and Image-to-Image queue their work and return at once, so you can keep changing settings and queueing. *Batch count* repeats a job (with a fixed seed, job *i* uses `seed + i`); *One job per line* renders a list of prompts. XYZ grids and Lab tools go through the same queue (Lab requests first), so one GPU job runs at a time.
- **Queue panel** in the header: progress of the running job with **Stop** (between sampler steps), waiting jobs with run next / up / down / remove, finished jobs with **Gallery** and **Retry**. Live over server-sent events, with a polling fallback.
- The queue survives restarts: jobs are kept in the database (without input images), and jobs that were unfinished at shutdown come back as failed and retryable.
- New render form for both tabs: prompt with Ctrl+Enter, LoRAs with weight sliders, size presets with swap, seed dice, images per job and batch count, custom sigmas and embeddings under *Advanced*. Finished renders show as tiles under the form; a notice with **View** appears when a render finishes while you're on another tab.
- **LoRA hot-swap**: LoRAs stay loaded as named adapters per model. Changing weights, switching back to a LoRA you've used or rendering without LoRAs no longer reloads anything (the image without LoRAs is bit-identical to a plain render). Least recently used LoRAs are unloaded beyond `performance.max_loaded_loras` (default 6).

### Gallery
- **Gallery** tab for everything GRADIENT has rendered: search prompts and notes, filter by model, type, rating, tag and favourites, sort by date or rating, adjustable tile size, infinite scrolling. Star ratings, favourites, tags and notes; Shift/Ctrl selection with bulk rate, favourite, tag and delete. A *Runs* view lists renders, grids and Lab runs with status and errors.
- **Viewer** with keyboard shortcuts (←/→, 1–5, F, I, Delete, Esc), details, tags and notes. **Use settings** (optionally with a new seed) restores the model, prompt, LoRAs and every other setting into the form; **Use as input** sends the image to Image-to-Image; **Open in Lab** copies the settings to the Lab.
- **Lab results are saved** as runs (image, maps and `report.json`) and appear in the gallery.
- Deleting moves files to `<outputs>/.trash/<run>/`. Thumbnails are WebP files cached in `<outputs>/.thumbs`.
- PNGs also carry an A1111-style `parameters` text (prompt, negative prompt, steps, sampler, CFG, seed, size, model, LoRAs), which image managers and taggers can read.

### API
- `POST/GET /api/v1/queue`, `GET /queue/events` (SSE), `/queue/{id}` with `cancel`, `move`, `retry`; `DELETE /queue/finished`.
- `/api/v1/history/...`: outputs with filters and paging, single output with `PATCH`/`DELETE`, bulk actions, image and thumbnail files, facets, runs.
- Generation responses include `run_uuid` and each image's `output_id`; a job stopped from the queue returns `409`.

### Database
- Older databases upgrade on start: new gallery columns (`favorite`, `notes`, `model_id`, `prompt`, `seed`, `width`, `height`) are added and filled from the stored parameters; old comma-separated tags move to the new `output_tags` table (each tag has a source and an optional confidence, ready for tags from other tools). New `queue_jobs` table.
- SQLite runs in WAL mode with a busy timeout, and sessions roll back cleanly when a job is cancelled.

### Tests
- 13 new backend tests: queue submit, seed steps, cancel, reorder, retry, SSE, Lab runs in the gallery, PNG metadata, gallery browse/filter/edit/delete, the upgrade of an older database, and LoRA caching (no reloads, eviction, LoRA off gives an identical image). Frontend unit tests for the queue state, gallery filters and *Use settings*.

## 0.4.0 (2026-10-30)

### Lab
- **ConceptAttention** for multi-modal DiTs (FLUX.1, FLUX.2, SD3/3.5), after Helbling et al., ICML 2025. Concept words run as extra tokens through the double-stream blocks with the text stream's weights. Maps are the softmax over concepts of image·concept attention outputs, averaged over chosen blocks and steps, with a segmentation view. Forward hooks only read activations, so images are bit-identical to normal renders. `POST /api/v1/lab/concept-attention`; `GET /api/v1/lab/layers` reports `concept_blocks`.
- The Lab's attention tool picks ConceptAttention or cross-attention from the model family, with a manual switch.
- Fixed: the cross-attention results crashed the page when the prompt contained a word twice (e.g. "a ... a ...").

### Other
- Models added by hand (diffusers folders) now report their pipeline and family in the Models API.
- Log files are written to the project's `logs/` folder even when the server starts from another directory (previously logging fell back to the console with an error).

### Tests
- Tiny random FLUX.1, SD3.5-style (QK-norm, dual attention) and FLUX.2 [klein] (Qwen3) pipelines for offline tests.
- A consistency test feeds the prompt's own tokens in as concepts and checks that the concept stream reproduces the model's text-stream attention outputs in every block (fp32, within 2e-5), for all three families.

## 0.3.0 (2026-10-30)

### Lab
- New **Lab** tab with shared base settings and four tools; also available through the API (`/api/v1/lab/*`).
- **Taylor preview**: exact derivative of the image with respect to CFG by forward-mode autodiff through the whole sampler (SD 1.x/SDXL), or finite differences for any family. A slider previews nearby CFG values instantly, and *Render exact* reports how much of the real change the prediction captured.
- **Structural modulation lanes** (`modulation` field on every generation request): switch or blend the prompt, change guidance, scale chosen attention layers, or change a LoRA weight for part of the denoising run. The UI has presets, a lane editor with a timeline, and a baseline comparison with the same seed.
- **Attention maps**: a recording attention processor computes explicit probabilities only on the selected cross-attention layers and steps (SD 1.x/SDXL); other layers and steps keep the fast kernels.
- **Token ablation**: leave-one-word-out (word removal for any model, embedding masking for SD 1.x/SDXL) scored by DINOv3 ViT-S/16 patch and CLS distances, falling back to DINOv2-small when DINOv3 isn't available.
- `GET /api/v1/lab/layers` lists a model's attention layers and suggests patterns.
- New model type **vision** (folder `vision/`) with DINOv3 ViT-S/16 (gated) and DINOv2-small in the catalog (37 entries).

### Updates and UI
- `update.sh` / `update.ps1`: update to the newest release tag (or `--main` / `-Main`), then re-run the installer. `--check` / `-Check` only reports. If tracked files have local changes, the scripts list them and stop instead of overwriting them.
- `config/config.yaml` is no longer tracked, so updates never touch personal settings; start from `config/config.example.yaml`.
- XYZ grid: the model is picked from a dropdown (with *Use these* recommended settings) like the other tabs, `model_id` can be an axis (cells are rendered grouped by model so each model loads once), and the scheduler is a dropdown.
- Restored the Tailwind 3 default border color (Tailwind 4 had made unstyled borders black).

### Tests
- 16 new backend tests: JVP against finite differences, sampler parity with the pipeline, all lane types and their errors, the Lab endpoints, DINOv2/DINOv3 features on tiny random encoders. Frontend unit tests for the Taylor composition, lane rules and layer globs.

## 0.2.0 (2026-10-30)

### Install & models
- One-command installers (`install.sh`, `install.ps1`) that detect the GPU and install the matching PyTorch build (CUDA 13.2, legacy CUDA 12.6 for older GPUs/drivers, MPS, CPU); `run.sh` / `run.bat`.
- New **Model Manager**: verified catalog (`config/model_catalog.yaml`, 35 entries), downloads from Hugging Face repos/files and direct URLs (Civitai) with progress, cancel and resume, gated-model guidance, local inventory with delete. UI tab, REST API (`/api/v1/models/*`) and CLI (`gradient models ...`).
- `gradient` command (`serve`, `info`, `models`); the API serves the built web UI, so the app runs on one port.
- `gradient serve --open` opens the browser once the server is ready; without a built UI the start page explains how to build it. `run.bat` keeps its window open on errors (PowerShell: `.\run.bat`).

### Generation
- Any diffusers family (SDXL, SD 3.5, FLUX.1/2, Qwen-Image, Z-Image, Chroma, Sana, Krea 2, ...) instead of SD 1.5 only; single-file checkpoints are auto-detected.
- Image-to-image shares weights with the loaded text-to-image pipeline.
- Real custom sigma schedules; scheduled CFG (`linear`, `cosine`, keyframes); scheduler compatibility checks.
- Per-image seeded generators: every logged seed reproduces its image.
- Memory modes (auto / gpu / balanced / low / minimal) and optional attention backends.
- GPU work no longer blocks the server; LLM calls run off the event loop.
- Parameter sensitivity rewritten as central finite differences with a heatmap (the old prototype returned a constant map).
- XYZ grids keep one seed across cells and log cell outputs under the grid run.

### Stack
- Python ≥ 3.11; torch 2.15, diffusers 0.42.1, transformers 5.22, huggingface_hub 2.4, accelerate 1.15, peft 0.21, FastAPI 0.162, SQLAlchemy 2.1, llama-cpp-python 0.3.36 (optional `[llm]` extra).
- PEP 621 `pyproject.toml` (works with uv, pip and Poetry 2); stale `poetry.lock` removed.
- Frontend: SvelteKit 3, Svelte 5.57, Vite 8, Tailwind 4.3, TypeScript 6, ESLint 10, Storybook 10, Vitest 5.
- Fixed: startup crash from the removed `pkg_resources` module; presets endpoints under FastAPI 0.162; ResourceManager eviction crash; fp16 variant requested for models that have none.
- Tests: 52 offline backend tests (tiny random SD pipeline + fake Hugging Face Hub) and frontend unit/story tests; CI runs both.
- The `patch_peft*.py` scripts are obsolete and moved to `scripts/legacy/`.
