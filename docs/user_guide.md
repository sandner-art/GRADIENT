# GRADIENT User Guide

Installation and model downloads are covered in the [README](../README.md). This guide covers using the workbench.

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

### Reproducibility
Every image is saved under `experiments/<run id>/` with its full parameters in the PNG text chunk `grad_parameters` (including the resolved seed, scheduler, pipeline class, dtype and memory mode), and logged to `data/database/gradient_data.db`. Image *i* of a batch uses seed `seed + i` and regenerates identically on its own.

## Lab

The **Lab** tab shares one set of base settings (model, prompt, size, steps, guidance, seed) between four tools. Keep a fixed seed so the tools look at the same image. Progress and the current sampler step are shown under the settings.

### Taylor preview
Computes the derivative of the whole image with respect to the guidance scale, ∂image/∂w. The browser then previews any nearby CFG value instantly as image(w + Δ) ≈ image(w) + Δ·∂image/∂w, while you drag the slider.

- **Exact JVP** (SD 1.x/SDXL): one forward-mode autodiff pass through every sampling step, the text-conditioned UNet and the VAE. It costs about as much as 2–3 normal renders and needs CFG above 1. Attention runs in plain math mode during this pass, so it uses more VRAM than a normal render.
- **Finite differences** (any family): renders at w − δ, w and w + δ (δ = 0.5 by default).
- **Render exact at this CFG** renders the real image and reports how much of the change the prediction captured. The prediction is first-order: good for small Δ, it drifts once the composition starts to change.
- **Sensitivity map** shows |∂image/∂w| per pixel: where the guidance scale matters.

API: `POST /api/v1/lab/taylor-preview` with `{"base_request": {...}, "method": "auto" | "jvp" | "finite_difference", "delta": 0.5}`. The tangent comes back as an 8-bit PNG: value = 128 + 127·t / `tangent_scale`, with t in image units (0–1) per unit of CFG.

### Attention maps
Shows where each word of the prompt is attended to, averaged over the chosen cross-attention layers and steps (SD 1.x/SDXL). Only the selected layers compute explicit attention probabilities, and only on the selected steps; everything else keeps the fast attention kernels, and the image matches a normal render up to rounding. Use an early step range (0–30%) to see layout, a late one for detail. The bar under each word compares how much attention it receives; each map is normalized on its own. Maps for DiT models (FLUX, SD 3.5) are planned with ConceptAttention.

API: `POST /api/v1/lab/attention-maps` with `{"base_request": {...}, "words": ["lighthouse"], "layers": "*attn2", "start": 0.0, "end": 1.0}`.

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
- **LLM chat disabled**: install the extra with `./install.sh --with-llm` (needs a C++ compiler), then download a GGUF from the Models tab (LLM filter).
