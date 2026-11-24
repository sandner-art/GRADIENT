# GRADIENT User Guide

Installation and model downloads are covered in the [README](../README.md). This guide covers using the workbench.

## Rendering, the queue and the gallery

**Queueing.** *Text-to-Image* and *Image-to-Image* don't wait for the render: **Queue** (or Ctrl+Enter in the prompt) adds jobs and you can change settings and queue more straight away. *Batch count* queues the same settings several times (with a fixed seed, each job uses the next seed; with an empty seed each gets a random one). Tick *One job per line* to render a list of prompts, one job per line. *Images per job* makes several images in one pass, which is faster on a big GPU but needs more memory.

**Preview panel.** On wide screens the render tabs show a large panel beside the form (below it on narrow ones):
- **Live** follows the newest job: while it renders you see the image develop (an estimate of the finished image at each step, cheap enough for every step; turn it off with `performance.live_preview: false`), then the result. Click a thumbnail in the strip to show it instead (*Follow newest* goes back), double-click for the viewer, and `[` / `]` step through the results.
- **Compare** shows two results with a wipe slider: click a thumbnail for A, Shift+click for B.
- **Batch** shows every image of the newest finished job.
- **Fit** switches between fitting the panel and full size.

**Send to Lab** (next to *Queue*, in the preview panel, the viewer and right-click menus) copies the settings into the Lab's shared settings and opens the Lab. The Lab uses one LoRA and a constant guidance value, so extra LoRAs, scheduled guidance and custom sigmas are left out (the notice says so).

**Right-click menus.** Right-click an image (gallery tile, thumbnail, the large preview, the viewer) for *Open*, *Use settings* (with or without its seed), *Use as input*, *Send to Lab*, a star rating, favourite, *Copy prompt* / *Copy seed*, *Download*, *Show its run*, *Export to a project…* and *Delete* (click twice). Right-click a selection in the gallery for the same actions on all selected images, and a queue job for *Run next*, *Retry*, *Stop*/*Cancel*, *Use settings* and *Send to Lab*. Esc or a click elsewhere closes the menu.

**Back button.** The mouse's back button, Alt+Left and the browser's Back go back inside GRADIENT: they close the image viewer or the queue drawer and return to the previous tab or gallery view (Forward works too). The tab is part of the address (`#gallery`), so reloading keeps it.

**The Queue panel** (header button; the dot is green while connected, a spinner and a thin bar show the running job) lists the running job with **Stop**, the waiting jobs (⤒ run next, ▲/▼ move, ✕ remove) and finished jobs (**Gallery** opens their images, **Retry** for failed or stopped ones). Lab tools and XYZ grids also go through the queue, and Lab requests run before waiting renders. Finished renders show as tiles under the form; when you're on another tab, a notice with **View** pops up.

**Gallery.** Every image GRADIENT saves is here: renders, grid cells and Lab results. Search prompts and notes, filter by model, type, rating, tag or favourites, sort by date or rating, and change the tile size. Hover a tile to favourite or rate it. Click a tile to open the viewer; with the checkbox, Ctrl+click or Shift+click you select several, then rate, favourite, tag, view or delete them together. *Runs* lists the renders, grids and Lab runs with their status; *Open* shows a run's images.

**Viewer keys:** ←/→ previous/next, 1–5 rate (0 clears), F favourite, I details, Delete twice to delete, Esc close. From the viewer:
- **Use settings** puts the image's model, prompt, size, steps, guidance, sampler, LoRAs and seed into *Text-to-Image* (**…with new seed** leaves the seed empty).
- **Use as input** loads the image into *Image-to-Image* with its settings.
- **Open in Lab** copies the settings into the Lab.

Deleted images go to `<outputs>/.trash/<run>/`, not straight to the recycle bin; empty that folder when you're sure.

**Metadata.** Each PNG carries the full settings as JSON (`grad_parameters`) and an A1111-style `parameters` text (prompt, negative prompt, steps, sampler, CFG, seed, size, model, LoRAs). Image managers, taggers and other UIs that read A1111 metadata can show and search them.

### Project folders: Kurator, ArtAgents, digiKam

A project folder hands gallery images to other tools and takes their work back. **Export** in the gallery (in the bulk bar for selected images, or *Export N…* in the toolbar for everything the current filters show) copies the images into a folder, by default `data/projects/<name>`:

- `0012_t2i_…_seed5.png`: the image, with its PNG metadata;
- `0012_t2i_…_seed5.png.xmp`: an XMP sidecar with your tags, the rating, the prompt, `Projects/<name>` and (optional) `GRADIENT/Model/<model>` and `GRADIENT/Type/<kind>` keywords;
- `gradient-project.json`: which file is which gallery image and what GRADIENT wrote.

The folder is flat, so tools that read one folder work directly. A caption file (`name.txt`) with the prompt is written only if you ask for it (dataset style), and a caption file another tool wrote is never overwritten.

**Kurator.** Add the folder (or a folder above it) as a source. *Settings → XMP sidecars → Import keywords* turns GRADIENT's keywords into tags, and `Projects/<name>` puts the images in that Kurator project. To send tags back, tick *Also update sidecars of other programs* and use *Write sidecars now*: Kurator then adds its keywords to GRADIENT's sidecars and keeps everything else. (Kurator has no ratings; digiKam and Lightroom write `xmp:Rating`, which GRADIENT reads.)

**ArtAgents.** In the *Captions* tab, load the folder and generate captions. They are saved next to the images as `name.txt`.

**Sync** (Gallery → *Projects*) reads the folder back:
- keywords another program added become tags marked `xmp`; keywords removed there remove those tags again. Your own GRADIENT tags are never removed by a sync;
- a changed rating becomes the image's rating;
- caption files become captions under the *Caption label* (default `artagents`). One caption is kept per label, so to compare captioners use one label each, for example sync with `artagents:llava`, caption again with another model, then sync with `artagents:qwen2.5vl`. *Also add comma-separated captions as tags* turns tag-style captions into tags under the same label;
- files another tool renamed are found again by their content; missing files and unknown images are listed.

Exporting again into the same project syncs first, then updates the copies and sidecars, so nothing the other tools added is lost. Tags show where they came from in the viewer (your tags are editable; imported ones can be removed with ×), and the viewer lists captions and the projects an image is in. *Show images* in the Projects view (or the *Project* filter) shows a project's images. *Forget* removes a project from the list but leaves the folder; *Add an existing project folder* brings it back (also after moving it).

### Quantized models (Precision)

**Precision** (next to the model in the render forms and in the Lab's base settings) loads the model quantized: smaller in memory, often a little faster, slightly different images. It is part of each render's settings, so you can render the same seed at full precision and quantized, and use `quantization` as an XYZ grid axis (`none, torchao-int8, bnb-4bit`); the grid renders all cells of one precision before loading the next.

| Precision | Memory (denoiser) | Notes |
|---|---|---|
| Full precision | 100 % | the reference |
| 8-bit (bitsandbytes) | ~55 % | NVIDIA GPUs; very close to full quality |
| 4-bit NF4 (bitsandbytes) | ~30 % | NVIDIA GPUs; a common choice for FLUX/Qwen-Image on 12–16 GB |
| int8 weights (torchao) | ~55 % | any device, also single-file checkpoints |
| int8 weights + activations (torchao) | ~55 % | faster on GPUs with int8 kernels, a little lossier |
| fp8 weights (torchao) | ~55 % | faster only on GPUs with fp8 (RTX 40xx, Ada/Hopper and newer); elsewhere it saves memory at the same speed |
| GGUF: *file* | depends on the file | pre-quantized denoiser (e.g. city96's FLUX GGUFs); put the `.gguf` in the checkpoints folder and pick the matching base model |

*Text encoders too* also quantizes T5, Mistral or Qwen text encoders (for GGUF, those get torchao int8). Precisions are checked against the selected model: GGUF files are matched by the architecture written in them (a FLUX GGUF is greyed out as *other model* for an SDXL model), and bitsandbytes and GGUF need a model folder (single-file checkpoints: torchao). If you switch to a model the chosen precision doesn't fit, it goes back to the model default and says why. A precision asked for explicitly (API, XYZ axis) that doesn't fit is refused with the reason; a *default* precision (the setting `performance.default_quantization` or a model's manifest) that can't be used falls back to full precision, with a line in the log. Unavailable options are greyed out with the reason; install the backends with `install.ps1 -WithQuant` (or `./install.sh --with-quant`). bitsandbytes needs a diffusers-format model; for single-file checkpoints use torchao. LoRAs load on top of every precision, and the Lab's exact JVP runs through torchao and GGUF weights (if it fails on another setup, *auto* uses finite differences and says so). Images record their precision in their settings and PNG metadata (`Quantization: bnb-4bit`), and *Use settings* restores it.

**Quantization check** (Lab) renders the Lab settings at full precision and with up to four precisions on the same seeds, one model load each, and reports per precision:
- weight size of each part, load time, seconds per image and peak GPU memory;
- PSNR, SSIM and the DINO feature distance to the reference; *relative change* divides that distance by the distance between two seeds of the reference, so 0.05 means "5 % of a new seed";
- the Taylor tangent's cosine with the reference's (∂image/∂guidance): near 1 means Lab measurements made on the quantized model hold for the full model.

A verdict sums it up per precision, the strips show reference | quantized | difference per seed, and the result is saved to the gallery as a Lab run.

### Fast rendering with LoRAs

LoRAs stay loaded on the model, so changing a weight, switching between LoRAs you've used, or rendering without any (*✕* removes them) needs no reloading. Up to `performance.max_loaded_loras` (default 6) LoRAs are kept per model; the least recently used one is unloaded when you go over. A LoRA must match the model family (an SD 1.5 LoRA won't apply to SDXL): a LoRA that matches no layers of the model is skipped with a warning in the log, and a file that can't be loaded at all stops the job with an error.

For quick everyday work on modest GPUs, SD 1.5 at 512×512 with 20–25 steps, or SDXL-Turbo / LCM-LoRA models at 1–8 steps, render in seconds. The XYZ grid's `loras` axis compares LoRAs or weights side by side with the same seed.

## Picking settings for a model

Modern models need very different settings. When you select a model that came from the catalog, a blue bar offers **Use these** (recommended size, steps, guidance). Rough guide:

| Family | Size | Steps | Guidance (CFG) |
|---|---|---|---|
| SD 1.5 | 512×512 | 25–30 | 6–8 |
| SDXL | 1024×1024 | 25–30 | 5–7 |
| SDXL-Turbo / Lightning / LCM LoRA | 512–1024 | 1–8 | 0–1.5 |
| SD 3.5 | 1024×1024 | 28–40 | 3.5–4.5 |
| FLUX.1 [dev], FLUX.2 [dev] | 1024×1024 | 28 | 3.5–4 (embedded guidance) |
| FLUX.1 [schnell], FLUX.2 [klein], Z-Image-Turbo, Krea 2 Turbo | 1024×1024 | 4–9 | 0–1 (distilled) |
| Qwen-Image | 1328×1328 | 40–50 | 4 |

Distilled models ignore negative prompts and CFG; that is expected.

## Research controls

### Scheduled guidance (CFG)
The *Guidance* field (API: `guidance_scale`) accepts a number or a schedule, applied per step:

- `linear(9, 3)`: ramp from 9 on the first step to 3 on the last
- `cosine(9, 3)`: the same with a cosine ease
- `9@0, 5@15, 3@30`: keyframes `value@step`, linearly interpolated
- `[[0, 9], [15, 5]]`: keyframes as JSON (API)

Schedules apply to classic-CFG models (SD 1.5, SDXL, SD 3.5, PixArt, Kolors). On guidance-distilled models (FLUX, Z-Image) the first value is used and the image metadata records `guidance_schedule_applied: false`.

### Custom sigma schedules
*Custom Sigmas* (API: `custom_sigmas`) are passed to the scheduler's `set_timesteps(sigmas=...)` and override *Steps*. Use a scheduler that accepts sigmas: `EulerDiscreteScheduler` for UNet models (Karras-style values such as `14.6, 6.0, 2.0, 0.5, 0.0`, the last value is the final sigma), or the default `FlowMatchEulerDiscreteScheduler` for flow-matching models (values in (0, 1]).

### Sensitivity maps
`POST /api/v1/visualize/parameter-sensitivity` renders the same seed at *p − δ* and *p + δ* and returns a heatmap of where the image changes:

```json
{
  "base_request": {"model_id": "sdxl-turbo", "prompt": "a lighthouse at dusk", "width": 512, "height": 512,
                   "num_inference_steps": 4, "guidance_scale": 1.5, "seed": 1},
  "target_parameter": {"name": "guidance_scale", "delta": 0.25}
}
```

Use `"name": "lora_weight:0"` to measure the first LoRA's weight instead.

### Modulation lanes
A lane changes one thing for part of the denoising run. `start` and `end` are fractions of the steps; a lane is active on steps `floor(start·N)` to `ceil(end·N) − 1`. Early steps decide composition, late steps decide texture and detail, so the same change does very different things depending on when it runs. Lanes are a field of every generation request (`modulation`), so they also work in the API, scripts and XYZ grids; the Lab tab has an editor with a timeline.

| `type` | Fields | Effect while active |
|---|---|---|
| `prompt` | `prompt`, `mode` (`replace` or `blend`), `value` (blend weight, default 0.5) | Uses another prompt's text embeddings, or mixes them with the main prompt's |
| `guidance` | `value` | Sets the CFG scale (SD 1.5, SDXL, SD 3.5, PixArt, Kolors) |
| `attention_scale` | `targets` (glob over attention layer names), `value` | Multiplies the output of the matching attention layers; `0` switches them off |
| `lora_weight` | `lora_index`, `value` | Sets the weight of one of the request's `loras` |

```json
"modulation": [
  {"type": "prompt", "start": 0.0, "end": 0.3, "prompt": "a castle on a hill"},
  {"type": "attention_scale", "start": 0.0, "end": 0.5, "targets": "up_blocks.*attn2", "value": 0.0},
  {"type": "guidance", "start": 0.7, "end": 1.0, "value": 2.0}
]
```

Notes:

- `GET /api/v1/lab/layers?model_id=...` lists a model's attention layers and suggests patterns (`*attn2` = all cross-attention in a UNet; DiT models use names like `transformer_blocks.12.attn`). Several patterns can be comma-separated.
- Guidance lanes can't switch CFG on or off mid-run, because the batch layout is fixed at the first step. Values stay above 1 if the run started with CFG, and at or below 1 if it didn't.
- Prompt lanes need embeddings of equal length. That is always true for CLIP/T5-padded families (SD 1.5, SDXL, SD 3.5, FLUX.1); for models with variable-length text encoders, keep the prompts a similar length. A blend that is active on the very first step takes effect from step 2 (the image metadata notes this).
- The image metadata records which lane types were applied (`modulation_applied`) and how many layers each pattern matched (`modulation_layers`).

### Gated feature models (DINOv3)

Token ablation measures changes with DINOv3 ViT-S/16 when it is available, and falls back to the ungated DINOv2 small (and says so). DINOv3 is gated by Meta. To use it:
1. Sign in on Hugging Face, open [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) and submit the access form (Meta reviews requests, so approval can take a while or be declined).
2. Create a *read* token at huggingface.co → Settings → Access Tokens.
3. Give GRADIENT the token: run `hf auth login` in GRADIENT's environment, or set `HF_TOKEN` before starting it, or paste it in the Models tab for the session.
4. Download **DINOv3 ViT-S/16** in the Models tab (Lab measurements use the local copy).

### Reproducibility
Every image is saved under `experiments/<run id>/` with its full parameters in the PNG text chunk `grad_parameters` (including the resolved seed, scheduler, pipeline class, dtype and memory mode), and logged to `data/database/gradient_data.db`. Databases from earlier versions are upgraded automatically on start (new columns are added and filled from the stored parameters). Image *i* of a batch uses seed `seed + i` and regenerates identically on its own.

## Lab

The **Lab** tab shares one set of base settings (model, prompt, size, steps, guidance, seed) between its tools. Keep a fixed seed so the tools look at the same image. Progress and the current sampler step are shown under the settings.

### Explorer

The Explorer travels through the model one region at a time. Open it from the Lab (*Explore the Lab settings*), from any image (right-click → **Explore…**), or beside the render forms (preview panel → **Explore**, a compact version; *Open in Lab* continues there with the full tools).

1. **Mark a region** on the image (rectangle, brush, eraser). Nothing marked = the whole image. The strip below shows what the model predicted at each step (*How it formed*); the percentage is how much that prediction still changes before the end.
2. **Where can it move?** (*Measure*) branches the region with new noise at several steps and measures how much it changes, in DINO features (or pixels without a vision model). The curve is 1 at step 0 (a completely new region) and falls to 0 where the region has settled; the grey band marks where it settles. *Big changes*, *Variations* and *Details* pick the steps where about 75 %, 45 % and 15 % of the change is left. The dashed line is a **free estimate** from the render alone (how far the prediction still was from the final image at that step); the summary says how far it is off and how well its map matches the measured one, so you can judge whether to trust it for a model.
3. **What drives it?** (*Suggest words*) ranks the prompt's words by how much they act in the region: attention maps on SD 1.x/SDXL (cross-attention) and FLUX/SD 3.5 (ConceptAttention) with one render, or ablation (one render per word) for any model. *× here* is how concentrated a word is in the region (1 = spread evenly). Each word gets alternatives from the text encoder's vocabulary; type your own, or *ask the LLM* if one is installed.
4. **Branch** from the chosen step: *New noise* makes variations; picked words make word branches (all with the same noise, so only the word differs); *Mix in a prompt* blends the prompt's embeddings with another prompt's (like conditioning *average*); with a LoRA, *LoRA weights to try* branches its weight. Only the region changes; everything outside it is copied from the original after every step, so it stays exactly the same (a soft edge of about 16 pixels blends the border).
5. **Compare the branches**: after rendering, the branch viewer opens. **← →** step through the image and its branches (↑ ↓ too, Home = the image), **hold Space** to flip to the reference, **W** a wipe between the two, **D** their difference (bright = changed), **M** the region's outline. The reference is *This image* until you pin the shown branch with **R** (**O** goes back to the original), so you can also compare branches with each other. **Enter** continues from the shown branch, **Esc** closes the viewer and shows the grid. Keys are ignored while you type in a field.
6. **Continue from here** makes a branch the new image; the path above the image lets you go back to any point. Branches are saved to the gallery with their path, so *Explore…* on a saved branch continues where you left off. *Use as input* sends a branch to image-to-image.

How it works: a branch takes the model's denoised estimate at that step, noises it back to the step's noise level (with new noise, or under another prompt) and denoises only the remaining steps through the model's own image-to-image pipeline, so late branches are cheap. The explorer needs a σ-based sampler (Euler, Euler a, DPM++ 2M, UniPC, DEIS, LMS, flow-matching Euler); others (DDIM, PNDM, Heun) are replaced by Euler, with a note. Image-to-image renders can be explored from the Image-to-Image preview (it has the input image). Custom sigmas and modulation lanes are not used by the explorer yet. `lab.explore_batch` (default 4) sets how many branches render together; lower it if a large model runs out of memory.

### Taylor preview
Computes the derivative of the whole image with respect to the guidance scale, ∂image/∂w. The browser then previews any nearby CFG value instantly as image(w + Δ) ≈ image(w) + Δ·∂image/∂w, while you drag the slider.

- **Exact JVP** (SD 1.x/SDXL): one forward-mode autodiff pass through every sampling step, the text-conditioned UNet and the VAE. It costs about as much as 2–3 normal renders and needs CFG above 1. Attention runs in plain math mode during this pass, so it uses more VRAM than a normal render.
- **Finite differences** (any family): renders at w − δ, w and w + δ (δ = 0.5 by default).
- **Auto** uses the exact JVP where it can. If the JVP fails for a model (some custom checkpoints, offloading setups), it uses finite differences instead and says why under the result; the server log has the details.
- **Render exact at this CFG** renders the real image and reports how much of the change the prediction captured. The prediction is first-order: good for small Δ, it drifts once the composition starts to change.
- **Sensitivity map** shows |∂image/∂w| per pixel: where the guidance scale matters.

API: `POST /api/v1/lab/taylor-preview` with `{"base_request": {...}, "method": "auto" | "jvp" | "finite_difference", "delta": 0.5}`. The tangent comes back as an 8-bit PNG: value = 128 + 127·t / `tangent_scale`, with t in image units (0–1) per unit of CFG.

### Attention maps
Shows where each word or concept is in the image. The tool picks the method from the model family; the switch above it overrides that.

**ConceptAttention** (FLUX.1, FLUX.2, SD 3.5; Helbling et al., ICML 2025) works on multi-modal DiTs, where text and image share one attention:

- Each concept word becomes an extra token that runs through the double-stream blocks with the text stream's own weights. Concepts attend to the image and to each other; the image and prompt never attend to the concepts, so the image is bit-identical to a normal render. The extra cost is a few tokens per block.
- The map for a block is the softmax over concepts of the dot product between image and concept attention outputs, averaged over the chosen blocks and steps. In every patch the probabilities add up to 1, so all maps share one scale, and the segmentation view colours each patch by its strongest concept.
- Concepts don't have to occur in the prompt. List the things you expect plus "background", since a concept that is absent still takes probability where nothing else fits. Single words work best: each concept is one token (the word's first T5 token on FLUX.1, its last token on SD 3.5's CLIP and on FLUX.2's Qwen3/Mistral encoder).
- *Blocks* takes indices such as `9-18`, `12,15,18` or `all`; the default is the deeper half, as in the paper's experiments (FLUX.1 has 19 double-stream blocks, FLUX.2 [dev] 8). Deeper blocks localize better than early ones, and the paper found averaging all blocks best for segmentation, so `all` is worth a try. *Count blocks* shows the number for the selected model.
- Only the double-stream blocks are used; FLUX's single-stream blocks are not. Other families (Qwen-Image, Z-Image, Chroma) are not supported yet; token ablation works for every model.

**Cross-attention** (SD 1.x/SDXL) shows where each word of the prompt is attended to, averaged over the chosen cross-attention layers and steps. Only the selected layers compute explicit attention probabilities, and only on the selected steps; everything else keeps the fast attention kernels, and the image matches a normal render up to rounding. The bar under each word compares how much attention it receives; each map is normalized on its own.

For both, use an early step range (0–30%) to see layout and a late one for detail.

API: `POST /api/v1/lab/concept-attention` with `{"base_request": {...}, "concepts": ["lighthouse", "sea", "sky", "background"], "blocks": null, "start": 0.0, "end": 1.0}`, and `POST /api/v1/lab/attention-maps` with `{"base_request": {...}, "words": ["lighthouse"], "layers": "*attn2", "start": 0.0, "end": 1.0}`.

### Token ablation
Removes one word at a time, re-renders with the same seed and measures how far the image moved in the feature space of a vision encoder. Unlike attention maps it measures the causal effect of the word, and it works for every model family. It costs one render per word, plus one.

- **Remove word**: drops the word from the prompt (any model). Later words shift position, which is part of the effect.
- **Mask embedding** (SD 1.x/SDXL): keeps every other token in place and replaces only the word's token embeddings with those of the empty prompt.
- Results are ranked by mean per-patch feature distance (1 − cosine similarity); *global* is the distance of the image-level (CLS) feature. The map shows where the image changed.
- The default encoder is **DINOv3 ViT-S/16**, which is gated: accept Meta's licence on Hugging Face, then download it under *Models → Vision*. Without it GRADIENT falls back to **DINOv2-small** (Apache-2.0) and says so. Set `lab.feature_model` in `config/config.yaml` to change the default.

API: `POST /api/v1/lab/token-ablation` with `{"base_request": {...}, "mode": "remove_word", "words": null, "max_words": 12, "feature_model": null}`.

## Memory modes

`performance.memory_mode` in `config/config.yaml` (or `GRADIENT_MEMORY_MODE`):

| Mode | VRAM | Speed |
|---|---|---|
| `gpu` | whole model on the GPU | fastest |
| `balanced` | components moved to the GPU when used (model CPU offload) | small slowdown |
| `low` | layer groups streamed to the GPU with CUDA streams (group offload) | slower, very low VRAM |
| `minimal` | sequential CPU offload | slowest |
| `auto` (default) | chosen from model size and free VRAM | |

Large models (FLUX, Qwen-Image) need plenty of **system RAM** when offloaded (roughly the model's download size). `performance.max_cached_pipelines` (default 1) controls how many models stay loaded; *POST /api/v1/system/unload* frees everything.

## Troubleshooting

- **"... is a gated model"**: accept the licence on the model's Hugging Face page, then set `HF_TOKEN`, run `.venv/bin/hf auth login`, or paste a token in the Models tab.
- **Out of memory (HTTP 507)**: lower the resolution or batch size, or set `memory_mode: low`.
- **PyTorch doesn't see the GPU**: run `.venv/bin/gradient info`. PyTorch 2.15 needs an NVIDIA driver with CUDA 13 support (R580+); with older drivers the installer falls back to PyTorch 2.14 + CUDA 12.6.
- **Model not in the list**: check it is in the right sub-folder (`diffusers/` needs `model_index.json`; single files go to `checkpoints/`). Hidden folders (starting with `.`) are ignored.
- **Stopping GRADIENT**: press **Ctrl+C** once in its window, click **⏻ Stop** in the web UI's header, or run `gradient stop` (`.venv\Scripts\gradient stop` on Windows) in another terminal. A running job stops at its next step (it shows as *stopped*, and *Retry* runs it again); waiting jobs stay in the queue for the next start. Press Ctrl+C a second time to stop at once, without waiting. On Windows, closing the console window also stops it cleanly.
- **Auto-reload**: `GRADIENT_APP_RELOAD=true` (for development; VS Code's Python extension loads a `.env` file into its terminals, so it can be on there without you noticing) runs the server in a second process that restarts when code changes. The console shows *Started reloader process*. Stopping works the same way in this mode.
- **Ctrl+C does nothing (e.g. in a VS Code terminal)**: if text is selected in VS Code's terminal, Ctrl+C copies it instead: click into the terminal to clear the selection and press it again. Some launchers start programs with Ctrl+C switched off; GRADIENT switches it back on at startup on Windows. If it still doesn't react, use **⏻ Stop** in the web UI or `gradient stop`. Open pages show *GRADIENT was stopped* (or *No connection to GRADIENT* if it closed or crashed), reconnect by themselves when it runs again, and offer a reload when it comes back as a new version. On Windows, `cmd` may still ask *Terminate batch job (Y/N)?* after GRADIENT stopped: either answer closes it.
- **Console at startup**: GRADIENT prints its version, the device it found (a *CPU only* line means PyTorch doesn't see a GPU) and the torch/diffusers versions. *The installed packages are from …* means the code is newer than the last install (e.g. after `git am` or `git pull`): run `update.ps1`/`./update.sh` or the installer, so new dependencies and the web UI are installed. The web UI shows the same note. Diffusers' own loading notices that say nothing (the empty float32 list, the safety-checker reminder, progress bars) are switched off; a one-time *Fetching N files* when a model's configuration is first downloaded is normal.
- **LLM chat disabled**: install the extra with `./install.sh --with-llm` (needs a C++ compiler), then download a GGUF from the Models tab (LLM filter).
