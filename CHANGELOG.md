# Changelog

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
