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

### Reproducibility
Every image is saved under `experiments/<run id>/` with its full parameters in the PNG text chunk `grad_parameters` (including the resolved seed, scheduler, pipeline class, dtype and memory mode), and logged to `data/database/gradient_data.db`. Image *i* of a batch uses seed `seed + i` and regenerates identically on its own.

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
