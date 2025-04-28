**GRADIENT Development Plan (Revised & Detailed)**

**Overall Principles:**

*   **Iterative Development:** Build, test, refine in cycles. Prioritize functional core features.
*   **Modular Design:** Ensure components (API, AI modules, UI parts) are as independent as possible for easier updates and future expansion.
*   **Core First:** Implement essential generation, interaction, and management features to create a usable foundation quickly.
*   **User-Centric:** Focus on workflows that empower practical use, experimentation, and research.
*   **Metadata & Reproducibility:** Build in experiment tracking and metadata handling from the start.
*   **Strategic Research:** Integrate research spikes logically after core functionalities are stable.

---

**Phase 0: Foundation & Setup (COMPLETE)**

*   **Goal:** Establish core project structure, backend infrastructure, basic UI shell, dependencies, and essential CI/CD.
*   **Key Deliverables:**
    *   ✅ Project repository, structure, license.
    *   ✅ Basic CI/CD pipeline (linting, testing placeholders).
    *   ✅ Core backend framework (FastAPI, Uvicorn).
    *   ✅ Configuration loading (`config/`, `.env`).
    *   ✅ Logging setup.
    *   ✅ Basic API endpoints (`/`, `/status`).
    *   ✅ Basic WebSocket endpoint placeholder.
    *   ✅ Core service placeholders (`TaskScheduler`, `ResourceManager`).
    *   ✅ Core AI dependencies installed (PyTorch+CUDA, `diffusers`, `llama-cpp`, etc.).
    *   ✅ Basic UI framework setup (SvelteKit, Vite proxy).
    *   ✅ Development tooling (`pyproject.toml`, `Makefile`, Poetry setup).

**Phase 1: Core AI Functionality & Management Foundations (Current Focus)**

*   **Goal:** Implement essential SD & LLM generation, basic visualization backend, foundational experiment tracking, and initial UI components. Make the tool usable for core tasks.
*   **Key Actions:**
    *   **(Backend) SD Loading & Text-to-Image:**
        *   ✅ Load SD 1.5/XL (diffusers/checkpoints) (`diffusion/loading.py`).
        *   ✅ Basic VAE handling (extracted from pipeline or specified).
        *   ✅ Implement `run_text_to_image` pipeline (`diffusion/pipelines.py`).
        *   ✅ Expose via `/api/v1/generate/text-to-image` (`generate.py`).
        *   ✅ Basic caching & Resource Manager integration (VRAM tracking).
    *   **(Backend) LLM Loading & Chat:**
        *   ✅ Load GGUF models (`llm/loading.py`).
        *   ✅ Implement `run_chat_completion` (streaming & non-streaming) (`llm/interaction.py`).
        *   ✅ Expose via `/api/v1/llm/chat/completions` (`llm.py`).
        *   ✅ Basic caching & Resource Manager integration (VRAM tracking for GPU layers).
    *   **(UI) Basic Text-to-Image Form:**
        *   ✅ Create API client utility (`lib/apiClient.ts`).
        *   ✅ Create `TextToImageForm.svelte` component with core parameters (prompt, neg, steps, cfg, seed, model_id, width, height, num_outputs).
        *   ✅ Implement API call and basic image result display.
        *   ✅ Integrate form into main page (`+page.svelte`).
    *   **(Backend) Experiment DB & Metadata (v1):**
        *   Setup basic SQLite database (using SQLAlchemy ORM).
        *   Define initial DB schema (`ExperimentRun`, `GeneratedImage`, storing parameters, paths, timestamps, seeds).
        *   Modify `run_text_to_image` (and later other generators) to log run details and output paths to the database.
        *   Implement metadata injection (saving parameters as JSON chunk or EXIF) into generated PNGs.
    *   **(Backend) Basic File Management:**
        *   Establish default save locations (`experiments/` subfolders based on date/run_id).
        *   Ensure output paths are logged correctly.
    *   **(Backend) XYZ Grids:**
        *   ✅ Define Pydantic models (`XYZGridRequest`, `XYZGridResponse`).
        *   ✅ Implement `run_xyz_grid` logic (`diffusion/xyz_grid.py`).
        *   ✅ Expose via `/api/v1/generate/xyz-grid` endpoint (`generate.py`).
        *   Modify grid logic to log each cell generation to the Experiment DB.
    *   **(Backend) Latent Space Viz Backend:**
        *   ✅ Define Pydantic models (`LatentSpaceVizRequest`, `LatentSpaceVizResponse`).
        *   ✅ Implement `generate_latent_space_visualization` (sampling, VAE decode, UMAP) (`viz/latent_space.py`).
        *   ✅ Expose via `/api/v1/visualize/latent-space` endpoint (`visualize.py`).
*   **Focus Areas:** Core generation pipelines (T2I, Chat, Grid), Backend DB/Metadata logging, Basic UI for T2I.

**Phase 2: UI Integration & Core Workflow Refinement**

*   **Goal:** Build out the UI for core features, implement essential workflows like Img2Img, add basic ControlNet/LoRA support, and refine backend systems. Make the tool feel more integrated and practically useful.
*   **Key Actions:**
    *   **(UI) Implement LLM Chat Interface:** Frontend component(s) for chat history display and interaction with the chat API (including handling SSE streams).
    *   **(UI) Implement XYZ Grid Display:** Frontend component to fetch and display grid results visually (grid layout, image previews, parameter display).
    *   **(UI) Implement Latent Space Viz (Basic Viewer):** Frontend component using a plotting library to display `/visualize/latent-space` data (scatter plot, hover previews).
    *   **(Backend) Implement Img2Img Pipeline:** Models, `run_image_to_image` logic, API endpoint. Log to Experiment DB.
    *   **(Backend) Implement Inpainting Pipeline:** Models (incl. mask), `run_inpainting` logic, API endpoint. Log to Experiment DB.
    *   **(UI) Add Img2Img/Inpainting Forms:** UI components for these workflows (image upload, mask drawing for inpainting).
    *   **(Backend) Scheduler Selection:** API/Logic to allow choosing different diffusers schedulers.
    *   **(UI) Scheduler Selector:** Dropdown in generation forms.
    *   **(Backend/UI) Parameter Presets:** Backend API and DB table for saving/loading parameter sets. UI for managing presets.
    *   **(Backend/UI) Basic LoRA/Embedding Handling:**
        *   Modify loading/pipelines to accept list of LoRAs + weights.
        *   API update.
        *   UI multi-select or tag input for selecting LoRAs and setting weights.
    *   **(Backend/UI) Basic ControlNet Handling:**
        *   Models, loading logic, pipeline modification (using `StableDiffusionControlNetPipeline` etc.).
        *   API update.
        *   UI for uploading control image, selecting ControlNet model, setting weight.
    *   **(Backend) Task Queue & Batching (v1):**
        *   Implement basic background task queue using `TaskScheduler` (e.g., using `asyncio` queues or `arq`).
        *   Modify API endpoints (optional flag?) to submit long tasks (grids, multiple images) to the queue instead of running synchronously.
        *   Basic API endpoint to check task status.
    *   **(UI) Basic Queue Monitor:** Simple display of queued/running tasks.
    *   **(Refinement) Resource Management:** Implement actual unloading logic coordinated between `ResourceManager` and AI loaders (`unload_pipeline`, `unload_llm` callbacks). Improve VRAM estimates.
    *   **(Refinement) Metadata Extraction:** Implement "Send to..." functionality by adding an API endpoint/UI feature to read parameters from image metadata.

**Phase 3: Advanced Features & Research Spike Initiation**

*   **Goal:** Introduce advanced/experimental features (Sensitivity Viz, Modulation, VAE Hub), mature research tools, and start dedicated research spikes.
*   **Key Actions:**
    *   **(Backend/UI) Parameter Sensitivity Viz (Display):**
        *   ✅ Backend API providing statistics (`/visualize/parameter-sensitivity`).
        *   (UI) Component to trigger analysis and display results (stats, potentially basic heatmap if backend adds image generation).
    *   **(Backend/UI) Structural Diffusion Modulation (Noise Schedule v1.5):**
        *   (Backend - Research) Investigate methods to *actually use* `custom_sigmas` values with common schedulers (e.g., custom Karras conversion, direct sigma injection if possible). Implement for 1-2 schedulers.
        *   (UI) Visual editor/input for designing noise schedules.
    *   **(Backend/UI) VAE Analysis Hub (v1):**
        *   (Backend) Refined VAE loading/caching. API endpoint to run multiple specified VAEs on a given latent/image and return comparative metrics (PSNR, SSIM - requires image comparison library like `scikit-image` or `piq`).
        *   (UI) Basic "VAE Cockpit" UI to select an image/latent, choose multiple VAEs, run comparison, and display results side-by-side with metrics.
    *   **(UI) Experiment Browser:** Develop UI for browsing, searching, filtering, tagging, and viewing details from the Experiment DB. Allow resubmitting runs ("Send to T2I").
    *   **(Research Spike) Novel Samplers:** Setup, literature review, implement/compare 1-2 novel samplers. Document results.
    *   **(Research Spike) Novel ControlNets:** Setup, literature review, initial experiments. Document results.
    *   **(Feature) Agentic Art Director (v1):** Basic implementation with simple goal processing (e.g., "Generate 10 variations of prompt X with CFG from 5-10"). Integrate with Task Queue. UI for defining agent tasks.
    *   **(Feature) Prompt Experimentation Tools (v1):** Extend XYZ grid UI/backend to better handle prompt variations (e.g., using wildcards/placeholders replaced by axis values).

**Phase 4: Experimental Deep Dive & Platform Maturation**

*   **Goal:** Develop more advanced experimental features based on Phase 3 research, refine core systems significantly, and improve platform robustness.
*   **Key Actions:**
    *   **(Feature) Structural Diffusion Modulation (v2):** Implement more modulation types based on research (UNet weighting, guidance manipulation).
    *   **(Feature) Cross-Model / Latent Grafting (v1):** Backend logic and basic UI for visual latent manipulation/merging.
    *   **(Feature) VAE Hub & Research Spike (v2):** Tools for VAE development acceleration (parameter exploration, A/B testing UI). Integrate findings from novel VAE research spike (if pursued).
    *   **(Feature) Advanced Latent Space Navigator:** Incorporate Semantic Navigation features (text-based nudging). Add "Latent Painting" PoC.
    *   **(Refinement) Agentic Art Director (v2):** More complex goal handling, potentially using visual feedback.
    *   **(Refinement) Performance Paradigms:** Implement selected techniques (latency hiding, speculative execution, adaptive allocation).
    *   **(Refinement) Error Handling & Stability:** Comprehensive testing, improved error reporting, resource leak checks.
    *   **(Refinement) API & Documentation:** Finalize v1 API, improve documentation significantly.
    *   **(Packaging/Deployment):** Investigate robust packaging solutions (PyInstaller, Docker).

**Phase 5: Polish, Release Candidate, Community (Ongoing)**

*   **Goal:** Stabilize, polish, optimize for release, foster community.
*   **Key Actions:**
    *   UI/UX overhaul based on feedback.
    *   Performance profiling & final optimization pass.
    *   Extensive testing (unit, integration, user).
    *   Bug fixing blitz.
    *   Final documentation, tutorials.
    *   Release Candidate builds.
    *   Community features (sharing presets/workflows).
    *   Ongoing maintenance, SOTA integration.

---
**Future Research & Feature Notes (Post Phase 4/5 or Parallel Spikes):**

*   **Gradient Flow Visualization:** Deep visualization of gradient propagation through UNet layers.
*   **Cross-Attention Map Visualization & Control:** Inspecting and manipulating attention mechanisms.
*   **Concept Algebra:** Latent vector arithmetic for conceptual blending.
*   **Embedding/TI Training UI:** Interface for training custom concepts.
*   **Style/Artist Database & Analysis:** Tools for style identification and replication.
*   **Node-Based Workflow Editor:** Visual pipeline construction alternative/addition.
*   **Real-time Collaboration / Federated Features:** Advanced multi-user or distributed concepts.
*   **Multimodal Inputs:** Using audio/image as direct inputs or control signals.

---

This revised plan prioritizes getting a usable tool with core generation and management features first (Phase 1 & 2), then layering the more advanced visualization, modulation, and research components (Phase 3 & 4), while clearly earmarking highly experimental ideas for later stages or dedicated research efforts.