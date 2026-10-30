# GRADIENT: Generative Research Lab

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/sandner-art/gradient/actions/workflows/python-app.yml/badge.svg)](https://github.com/sandner-art/gradient/actions/workflows/python-app.yml)

**Author:** Daniel Sandner

> GRADIENT: Generative Research & Architectural Diffusion Interactive Exploration Navigation Toolkit

## Abstract

GRADIENT is an open-source (Apache 2.0) desktop application designed as an advanced workbench for generative AI research, experimental art, and in-depth exploration of diffusion models. It provides a unified, highly interactive, and visually-driven environment that moves beyond the capabilities of standard UIs like Automatic1111 or ComfyUI. Key differentiators include interactive latent space navigation, direct modulation of diffusion architecture components, integrated research tooling, and agentic assistance, fostering deeper understanding and novel creative workflows.

## Motivation

While powerful, existing generative AI tools often present fragmented workflows, limited introspection into model internals, and lack intuitive interfaces for non-standard experimental manipulation. GRADIENT aims to provide a cohesive platform focused on:

1.  **Deep Visualization:** Enabling researchers and artists to "see inside" the diffusion process and latent space.
2.  **Interactive Experimentation:** Facilitating novel methods of control and exploration beyond static prompting.
3.  **Reproducible Research:** Integrating tools for systematic evaluation, logging, and analysis.
4.  **Structural Creativity:** Offering capabilities to manipulate the generative process itself for unique artistic outcomes.

## What works today (v0.4.0)

*   **Any diffusers model family:** SD 1.5, SDXL, SD 3.5, FLUX.1, FLUX.2, Qwen-Image, Z-Image, Chroma, Sana, Krea 2 and more (diffusers folders, Hub repos, and SD/SDXL/SD3/FLUX single-file checkpoints).
*   **Built-in Model Manager:** a verified catalog of base models, LoRAs, VAEs, ControlNets and GGUF LLMs; one-click downloads from Hugging Face, any repo/file link, or direct URLs (e.g. Civitai), with progress, cancel and resume. Also on the command line.
*   **Text-to-image, image-to-image, XYZ grids** with presets, LoRAs, embeddings, scheduler choice, **custom sigma schedules** and **scheduled CFG** (`linear(9,3)`, keyframes `9@0,4@20`).
*   **Reproducibility:** every image is generated with its own seeded generator, saved with full parameters in PNG metadata and logged to a local SQLite experiment database.
*   **Parameter sensitivity maps** (where in the image does CFG or a LoRA weight matter?) by central finite differences.
*   **Lab tab** for looking inside a generation:
    *   **Taylor preview:** one exact derivative of the image with respect to CFG (forward-mode autodiff through the whole sampler on SD 1.x/SDXL, finite differences elsewhere). A slider then previews nearby CFG values instantly, and you can check the prediction against a real render.
    *   **Structural modulation lanes:** switch or blend the prompt, change guidance, scale chosen attention layers or a LoRA weight for part of the denoising run. Lanes also work in every API generation request.
    *   **Attention maps:** where each word or concept is in the image. **ConceptAttention** on FLUX.1, FLUX.2 and SD 3.5 locates any concept word (it doesn't have to be in the prompt) and shows a segmentation. Per-word cross-attention covers SD 1.x/SDXL. Both only read activations, so the image doesn't change.
    *   **Token ablation:** leave one word out, re-render with the same seed and measure the change in DINOv3 (or DINOv2) feature space. This causal influence measure works for every model family.
*   **Memory modes** for small GPUs: full GPU, model offload, group offload with CUDA streams, sequential offload, picked automatically from model size and free VRAM.
*   **Local LLM chat** (GGUF via llama.cpp) for prompt work.

Further research features (latent-space navigator, attribute directions, agentic art director) are reviewed against the current state of the art in [docs/research/research_review_2026-10.md](docs/research/research_review_2026-10.md).

## Installation

Requirements: **Python 3.11–3.13**, **Node.js ≥ 22.17** (to build the web UI), git. An NVIDIA GPU is recommended; Apple Silicon and CPU also work (slowly for large models).

```bash
git clone https://github.com/sandner-art/gradient.git
cd gradient
./install.sh                # Linux / macOS
```

On Windows (PowerShell):

```powershell
git clone https://github.com/sandner-art/gradient.git
cd gradient
powershell -ExecutionPolicy Bypass -File install.ps1
```

The installer:

1. installs [uv](https://docs.astral.sh/uv/) if needed and creates `.venv` (Python 3.12),
2. detects your hardware and installs the matching PyTorch build,
3. installs GRADIENT (`pip install -e .`),
4. builds the web UI (if Node.js is available).

Options: `--cpu`, `--legacy-gpu`, `--with-llm` (adds llama-cpp-python, compiled locally), `--dev`, `--no-frontend` (PowerShell: `-Cpu`, `-LegacyGpu`, `-WithLlm`, `-Dev`, `-NoFrontend`).

### Updating

```bash
./update.sh                 # newest release
./update.sh --main          # latest development version (main branch)
./update.sh --check         # only show whether an update is available
```

On Windows: `powershell -ExecutionPolicy Bypass -File update.ps1` (options `-Main`, `-Check`).

The update script fetches the new version with git, then re-runs the installer, so dependencies and the web UI are rebuilt too. Installer options pass through (`./update.sh --legacy-gpu`; PowerShell `-LegacyGpu`). Models, outputs, the experiment database and `config/config.yaml` are not tracked by git, so updates keep them. If you edited files that are tracked, the script lists them and stops, so nothing is overwritten.

### PyTorch and your GPU

| Hardware | What the installer does |
|---|---|
| NVIDIA RTX 20xx and newer, driver R580+ (CUDA 13) | PyTorch 2.15 + CUDA 13.2 |
| NVIDIA with an older driver (CUDA 12.x) | PyTorch 2.14 + CUDA 12.6 (update the driver to get 2.15) |
| NVIDIA GTX 9xx/10xx, Titan V (Maxwell/Pascal/Volta) | PyTorch 2.14 + CUDA 12.6, the last build that supports them (`--legacy-gpu`) |
| Apple Silicon | PyPI build with MPS |
| AMD (ROCm) | pick a wheel index at [pytorch.org](https://pytorch.org/get-started/locally/) and run `TORCH_INDEX_URL=<url> ./install.sh` |
| No GPU | CPU build |

Manual install (any platform): install PyTorch for your hardware from [pytorch.org](https://pytorch.org/get-started/locally/), then `pip install -e ".[dev]"` and `cd frontend && npm ci && npm run build`.

## Quick start

```bash
./run.sh                    # Windows PowerShell: .\run.bat   (cmd.exe: run.bat)
```

The first start can take a minute while PyTorch loads; the browser opens when the server is ready. Otherwise open **http://127.0.0.1:8000**. With no models installed, GRADIENT opens the **Models** tab: pick something from the catalog (good first choices: *SDXL-Turbo* for 8 GB GPUs, *FLUX.2 [klein] 4B* or *Z-Image-Turbo* for 16 GB+) and press **Download**. When it finishes, select it in *Text-to-Image* and click **Use these** to apply the model's recommended size, steps and guidance.

API docs live at http://127.0.0.1:8000/docs.

## Models

### Downloading

*   **In the UI:** *Models* tab → catalog → *Download*. Or paste a Hugging Face repo (`stabilityai/sdxl-turbo`), a huggingface.co file link, or a direct https link into *Add from Hugging Face or URL*.
*   **On the command line:**

    ```bash
    .venv/bin/gradient models catalog                    # list the curated catalog
    .venv/bin/gradient models download sdxl-turbo        # by catalog id
    .venv/bin/gradient models download stabilityai/sdxl-turbo
    .venv/bin/gradient models download unsloth/Qwen3.5-4B-GGUF --type llm     # picks the Q4_K_M file
    .venv/bin/gradient models download "https://civitai.com/models/123?modelVersionId=456" --type lora
    .venv/bin/gradient models list
    .venv/bin/gradient models delete diffusers sdxl-turbo
    ```

On Windows use `.venv\Scripts\gradient.exe` instead of `.venv/bin/gradient`.

Downloads resume after a cancel or a dropped connection. Files are staged in a hidden `.staging` folder and only appear in model lists once complete.

**Gated models** (FLUX.1 [schnell]/[dev], SD 3.5, FLUX.2 [dev]/[klein 9B], Krea 2): accept the licence on the model's Hugging Face page, create a read token at https://huggingface.co/settings/tokens, then either `export HF_TOKEN=hf_...` before starting GRADIENT, run `.venv/bin/hf auth login`, or paste the token into the Models tab for the session. For Civitai links that need an account, set `CIVITAI_TOKEN`.

The catalog lives in [`config/model_catalog.yaml`](config/model_catalog.yaml); add your own entries with the same fields.

### Folder layout (bring your own models)

```
data/models/
  diffusers/     diffusers-format folders (model_index.json)
  checkpoints/   single-file .safetensors checkpoints (SD1.x, SDXL, SD3, FLUX, ...)
  lora/          LoRA files or PEFT folders
  vae/           VAE files or folders
  controlnet/    ControlNet folders
  embeddings/    textual inversion files
  llm/           GGUF chat models
```

Add more search folders (e.g. a shared models disk) in `config/config.yaml` under `paths.models.search_dirs`, or set `GRADIENT_MODEL_DIR`. The first folder is where downloads go.

## Configuration

Defaults are in [`config/default_settings.yaml`](config/default_settings.yaml); put overrides in `config/config.yaml` (start from [`config/config.example.yaml`](config/config.example.yaml)) or environment variables (see [`.env.example`](.env.example)). Most useful:

| Setting | Env var | Purpose |
|---|---|---|
| `performance.memory_mode` | `GRADIENT_MEMORY_MODE` | `auto` (default), `gpu`, `balanced`, `low`, `minimal` |
| `performance.max_cached_pipelines` | | how many models stay loaded (default 1) |
| `performance.attention_backend` | | e.g. `flash`, `sage` (default: PyTorch SDPA) |
| `paths.models.search_dirs` | `GRADIENT_MODEL_DIR` | model folders |
| `app.port` | `GRADIENT_APP_PORT` | server port |

## Development

```bash
./install.sh --dev
make run            # or: make dev  (API with auto-reload)
make frontend-dev   # Vite dev server on :5173, proxies /api to :8000
make check          # ruff, pytest, svelte-check, vitest
```

Backend tests run offline on CPU: they build a tiny random-weight Stable Diffusion pipeline and a fake Hugging Face Hub, so generation and downloads are tested end to end without network or GPU.

Stack: Python 3.11+, FastAPI, diffusers 0.42, transformers 5, PyTorch 2.15, SQLAlchemy 2.1 · SvelteKit 3, Svelte 5, Vite 8, Tailwind CSS 4, Storybook 10, Vitest 5.

## Core Concepts

*   **Latent Space Navigation:** Instead of relying solely on text prompts, GRADIENT allows direct visual exploration and interaction within the high-dimensional latent space where images are formed.
*   **Structural Modulation:** GRADIENT exposes controls to modify aspects of the diffusion model's *architecture* or *sampling dynamics* during generation, enabling unique artistic effects unachievable through standard parameter tuning.
*   **Experiment-Centric Workflow:** Research tasks are first-class citizens, with integrated tools for batching, systematic parameter sweeps, and automated logging for reproducibility.

## Architecture Overview

A FastAPI backend runs all model work (diffusers, llama.cpp) in a worker thread behind a GPU lock, so the API stays responsive during generation and downloads. The SvelteKit UI is built as a static single-page app and served by the same process. See [docs/architecture.md](docs/architecture.md).

## Roadmap

See the prioritized milestones in [docs/research/research_review_2026-10.md](docs/research/research_review_2026-10.md#9-roadmap-next-three-milestones): an instrumented, architecture-agnostic sampler with live x₀ previews and a job queue; measurement tools (JVP sensitivity, an output-space atlas, metrics); then steering and interpretability (σ-timeline modulation, ConceptAttention, attribute directions) and an experiment-designing agent.

## Contributing

Contributions are welcome. Use GitHub Issues for bug reports and feature requests. Please run `make check` before opening a pull request.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details. Models downloaded through GRADIENT keep their own licences (shown in the catalog); several are non-commercial.

## Citation

```bibtex
@software{Sandner_GRADIENT_2025,
  author = {Sandner, Daniel},
  title = {{GRADIENT: Generative Research \& Architectural Diffusion Interactive Exploration Navigation Toolkit}},
  year = {2025},
  url = {https://github.com/sandner-art/gradient}
}
```
