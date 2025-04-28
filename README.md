# GRADIENT: Generative Research Lab

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/sandner-art/gradient) <!-- Placeholder -->
[![Docs Status](https://img.shields.io/badge/docs-in%20progress-orange.svg)](https://github.com/sandner-art/gradient/wiki) <!-- Placeholder -->

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

## Key Features

*   **Interactive Latent Space Navigator:** Visualize and interact with model latent spaces (UMAP/t-SNE) for exploration, analysis, and novel generation techniques.
*   **Structural Diffusion Modulation:** Directly influence diffusion dynamics (e.g., noise schedules, guidance vectors, UNet layer weighting) via UI controls.
*   **Parameter Sensitivity Visualization:** Estimate the spatial impact of parameter changes (e.g., CFG, LoRA weight) using gradient-based or empirical methods.
*   **Advanced Model/Adapter Handling:** Unified support for SD 1.5/XL, Flux.1, LoRAs, PEFT, VAEs, ControlNets, etc.
*   **Dynamic Prompt Weaving:** Inject or modify prompts/embeddings mid-generation, integrating ELLA principles.
*   **Agentic Art Director:** Utilize LLM agents for autonomous experimentation and batch generation based on high-level goals.
*   **Research Toolkit:** Native XYZ+ grids, automated experiment logging (parameters, outputs, metadata), integrated data plotting, and potential Jupyter Notebook export.
*   **Visualization Suite:** Live diffusion process visualization, semantic labeling of latent space, prompt-based highlighting.
*   **Advanced Workflows:** Cross-model/seed latent grafting, robust prompt/workflow management.
*   **API Access:** Allow external control and integration via a documented Python API.
*   **Modular Architecture:** Built for extensibility and maintainability.

## Visuals

*(Placeholder: Screenshots/GIFs demonstrating the Latent Space Navigator, Structural Modulation UI, Sensitivity Heatmaps, and overall interface will be added here.)*

## Installation

GRADIENT is designed to run locally. Prerequisites include Python 3.10+, PyTorch (with CUDA support recommended), and Git.

```bash
# 1. Clone the repository
git clone https://github.com/sandner-art/gradient.git
cd gradient

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install PyTorch with CUDA support (example - adapt for your system)
# See: https://pytorch.org/get-started/locally/
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Build llama-cpp-python (if using GGUF models with GPU)
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
# (See llama-cpp-python documentation for details and prerequisites)

# 6. Download models (Stable Diffusion checkpoints, VAEs, GGUF files, etc.)
# Place them in the appropriate './models' subdirectories (create if needed).
```

*(Detailed dependency installation, especially for CUDA and llama.cpp, may require additional system-specific steps.)*

## Hardware-Specific PyTorch Builds

GRADIENT is configured by default to install PyTorch with CUDA 12.1 support, suitable for NVIDIA GPUs like the RTX 30x0/40x0 series and newer cards supporting CUDA 12.x.

If you have **different hardware**, you will need to manually install the correct PyTorch build *after* running `poetry install`:

1.  **Activate the virtual environment:** `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows).
2.  **Uninstall the default torch:** `pip uninstall torch torchvision torchaudio -y`
3.  **Find your build:** Go to the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).
4.  **Select your OS, 'Pip', and your Compute Platform** (e.g., CUDA 11.8, ROCm for AMD, CPU Only, MPS for Apple Silicon).
5.  **Copy the generated `pip install` command.**
6.  **Run the command** in your activated venv.

This will replace the default CUDA 12.1 build with the one appropriate for your system. Note that your `poetry.lock` file will still reference the default source, but the manually installed version will take precedence in your environment.

## Quick Start

```bash
# Ensure your virtual environment is active
source venv/bin/activate

# Launch the GRADIENT application
python main.py # Or specific launch script

# Access the UI via your web browser (typically http://127.0.0.1:7860 or similar).
# Follow UI prompts to load models and start generating/experimenting.
```

*(Detailed usage documentation will be available in the project Wiki.)*

## Core Concepts

*   **Latent Space Navigation:** Instead of relying solely on text prompts, GRADIENT allows direct visual exploration and interaction within the high-dimensional latent space where images are formed.
*   **Structural Modulation:** GRADIENT exposes controls to modify aspects of the diffusion model's *architecture* or *sampling dynamics* during generation, enabling unique artistic effects unachievable through standard parameter tuning.
*   **Experiment-Centric Workflow:** Research tasks are first-class citizens, with integrated tools for batching, systematic parameter sweeps, and automated logging for reproducibility.

## Architecture Overview

GRADIENT utilizes a modern Python backend (FastAPI) with asynchronous processing for performance and responsiveness. AI model interactions are handled via libraries like `diffusers` and `llama-cpp-python`. The architecture is designed to be modular, allowing for easier updates and extensions. Communication with the frontend leverages WebSockets for real-time interaction. *(Further details in docs/ARCHITECTURE.md)*

## Roadmap

GRADIENT is under active development. Key areas for near-term focus include:

*   Robust implementation of the Interactive Latent Space Navigator.
*   Refinement of Structural Diffusion Modulation controls.
*   Implementation of the Agentic Art Director prototype.
*   Performance optimization and latency hiding techniques.
*   UI framework finalization and UX improvements.
*   Expansion of supported models and techniques (e.g., ELLA integration).

## Contributing

Contributions are welcome! Please read our `CONTRIBUTING.md` guidelines (Placeholder) for details on reporting issues, submitting pull requests, and coding standards. Key areas for contribution include feature development, UI/UX refinement, performance optimization, and documentation.

Use GitHub Issues for bug reports and feature requests.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.

## Citation

If you use GRADIENT in your research or artistic scholarly work, please cite it as follows (Placeholder BibTeX):

```bibtex
@software{Sandner_GRADIENT_2025,
  author = {Sandner, Daniel},
  title = {{GRADIENT: Generative Research \& Architectural Diffusion Interactive Exploration Navigation Toolkit}},
  month = {6},
  year = {2025},
  url = {https://github.com/sandner-art/gradient} # Placeholder URL
}
```

## Contact

*   **Issues:** [GitHub Issues](https://github.com/sandner-art/gradient/issues) <!-- Placeholder -->
*   **Community/Discussions:** [GitHub Discussions](https://github.com/sandner-art/gradient/discussions) *(Optional - Placeholder)*
