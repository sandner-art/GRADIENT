**GRADIENT Development Plan (Updated 04292025)**

**Overall Principles:**

*   **Iterative Development:** Build, test, refine in cycles. Prioritize functional core features.
*   **Modular Design:** Ensure components (API, AI modules, UI parts) are as independent as possible for easier updates and future expansion.
*   **Core First:** Implement essential generation, interaction, and management features to create a usable foundation quickly.
*   **User-Centric:** Focus on workflows that empower practical use, experimentation, and research.
*   **Metadata & Reproducibility:** Build in experiment tracking and metadata handling from the start.
*   **Strategic Research:** Integrate research spikes logically after core functionalities are stable.

---

**Phase 0: Foundation & Setup (COMPLETE)**

*   ✅ Project structure, backend/frontend frameworks, config, logging, core dependencies, basic API/WebSocket, dev tooling, CI placeholders.

---

**Phase 1: Core AI Functionality & Management Foundations (Mostly COMPLETE)**

*   **Goal:** Implement essential SD & LLM generation, basic visualization backend, foundational experiment tracking, and initial UI components.
*   **Key Actions:**
    *   **(Backend) SD Loading & T2I:**
        *   ✅ Load SD (diffusers/checkpoints) (`diffusion/loading.py`).
        *   ✅ Flexible Path Handling (Configured search dirs, absolute paths, Hub IDs).
        *   ✅ Basic VAE handling.
        *   ✅ Implement `run_text_to_image` pipeline (`diffusion/pipelines.py`).
        *   ✅ Expose via `/api/v1/generate/text-to-image`.
        *   ✅ Basic model caching & VRAM tracking (`ResourceManager`).
    *   **(Backend) LLM Loading & Chat:**
        *   ✅ Load GGUF models.
        *   ✅ Implement `run_chat_completion` (streaming & non-streaming).
        *   ✅ Expose via `/api/v1/llm/chat/completions`.
        *   ✅ Basic caching & VRAM tracking.
    *   **(UI) Basic Text-to-Image Form:**
        *   ✅ Create `apiClient.ts`.
        *   ✅ Create `TextToImageForm.svelte` (initial version).
        *   ✅ Implement basic API call and result display.
        *   ✅ Integrate into main page (`+page.svelte`).
    *   **(Backend) Experiment DB & Metadata:**
        *   ✅ Setup SQLite database & SQLAlchemy ORM.
        *   ✅ Define `ExperimentRun`, `GeneratedOutput` schema.
        *   ✅ Integrate DB logging into T2I pipeline (`run_text_to_image`).
        *   ✅ Implement PNG metadata injection (`run_text_to_image`).
    *   **(Backend) Basic File Management:**
        *   ✅ Save outputs to `experiments/<run_uuid>/`.
        *   ✅ Log correct file paths to DB.
    *   **(Backend) XYZ Grids:**
        *   ✅ Define Pydantic models.
        *   ✅ Implement `run_xyz_grid` logic.
        *   ✅ Expose via `/api/v1/generate/xyz-grid`.
        *   ✅ Integrate DB logging via T2I calls.
    *   **(Backend) Latent Space Viz Backend:**
        *   ✅ Define Pydantic models.
        *   ✅ Implement `generate_latent_space_visualization` (UMAP, optional VAE decode).
        *   ✅ Expose via `/api/v1/visualize/latent-space`.
    *   **(Backend) Parameter Sensitivity Backend:**
        *   ✅ Define Pydantic models.
        *   ✅ Basic backend logic (`analyze_parameter_sensitivity`).
        *   ✅ Expose via `/api/v1/visualize/parameter-sensitivity`.

---

**Phase 2: UI Integration & Core Workflow Refinement (In Progress / Current Focus)**

*   **Goal:** Build out the UI for core features, implement essential workflows (Img2Img, Inpainting), add basic ControlNet/LoRA support, and refine backend systems.
*   **Key Actions:**
    *   **(Backend) Implement Img2Img Pipeline:**
        *   ✅ Define API models (`ImageToImageRequest/Response`).
        *   ✅ Implement `run_image_to_image` logic (using specific pipeline, handling strength).
        *   ✅ Add `/api/v1/generate/image-to-image` endpoint.
        *   ✅ Integrate DB logging.
    *   **(Backend) Scheduler Selection:**
        *   ✅ Add `scheduler_id` to base request models.
        *   ✅ Implement `setup_scheduler_and_timesteps` helper in `pipelines.py` to load/switch schedulers.
        *   ✅ Create `/api/v1/system/schedulers` endpoint to list available schedulers.
    *   **(UI) Add Img2Img Form & Tab:**
        *   ✅ Create `ImageToImageForm.svelte`.
        *   ✅ Implement image upload, strength slider, API call, result display.
        *   ✅ Add tab to `+page.svelte`.
    *   **(UI) Refactor Forms & State:**
        *   ✅ Create shared components (`ModelSelector`, `SchedulerSelector`, `SeedInput`, `SubmitButton`).
        *   ✅ Implement Svelte stores (`t2iFormStore`, `i2iFormStore`) for persistent form state.
        *   ✅ Integrate shared components and stores into T2I/I2I forms.
        *   ✅ Implement "Last Result" indicators in UI for T2I/I2I/Grid.
    *   **(UI) Dynamic Selectors:**
        *   ✅ Create backend endpoints (`/system/models/...`) to list models, LoRAs, embeddings.
        *   ✅ Update forms to fetch and use dynamic lists for Model & Scheduler dropdowns.
    *   **(Backend/UI) Parameter Presets:**
        *   ✅ Define `ModelPreset` DB model & update DB schema.
        *   ✅ Implement `/api/v1/presets` CRUD endpoints.
        *   ✅ Add Preset selector, apply logic, and save functionality to T2I/I2I forms.
    *   **(UI) Basic LoRA/Embedding Handling:**
        *   ✅ Add UI sections to T2I/I2I forms for selecting/adding/removing LoRAs (with weights) and Embeddings.
        *   ✅ Fetch available LoRAs/Embeddings dynamically.
        *   ✅ Include selected items in API payload (backend processing TBD).
    *   --- **NEXT TASKS WITHIN PHASE 2** ---
    *   **(Backend) LoRA/Embedding Pipeline Integration:**
        *   Modify `load_diffusion_pipeline` and/or pipeline execution (`run_text_to_image`, `run_image_to_image`) to dynamically load and apply selected LoRAs (using `load_lora_weights`) and potentially handle embedding triggers based on the request payload.
    *   **(UI) Implement XYZ Grid Display:**
        *   ✅ Basic `XYZGridDisplay.svelte` component created.
        *   (Refinement Needed) Enhance display (parameter visibility, Z-axis handling?), improve styling.
        *   (Refinement Needed) Ensure `handleRerun` logic correctly populates the chosen form (T2I/I2I).
    *   **(UI) Implement LLM Chat Interface:** Create `LLMChat.svelte` (currently placeholder), implement chat history display, message input, API calls (including handling SSE stream), and response rendering.
    *   **(Backend) Implement Inpainting Pipeline:** Define `StableDiffusionInpaintPipeline` logic, models, API endpoint (`run_inpainting` handling image+mask). Integrate DB logging.
    *   **(UI) Add Inpainting Form:** Create `InpaintingForm.svelte`, add image upload, **implement/integrate mask drawing component** (or mask upload), hook up API call. Add Inpainting tab.
    *   **(Backend/UI) Basic ControlNet Handling:** Backend loading/pipeline integration (`StableDiffusionControlNet...Pipeline`), API updates, Frontend UI for control image upload, model selection, weight setting.
    *   **(UI) Implement Latent Space Viz (Basic Viewer):** Create component using a JS plotting library (e.g., Plotly.js, Chart.js) to display data from `/visualize/latent-space`. Add basic interaction (hover).
    *   **(Refinement) Metadata Extraction:** Implement "Send to..." (e.g., reading PNG metadata and populating a form). Requires backend endpoint to parse metadata from uploaded file/base64.
    *   **(Refinement) Resource Management:** Implement actual model unloading callbacks in `ResourceManager` coordinated with loaders. Improve VRAM estimation.

---

**Phase 3: Advanced Features & Research Spike Initiation**

*   **(UI) Experiment Browser:** UI for browsing/searching/filtering/tagging/rerunning experiments from the DB.
*   **(Backend/UI) Parameter Sensitivity Viz (Display):** UI component to trigger/display results from `/visualize/parameter-sensitivity` and its backend.
*   **(Backend) Advanced Scheduler/Sampler Integration:** Implement backend support for scheduled CFG/parameters (from Phase 2 ideas). Add support for 1-2 novel samplers.
*   **(UI) Advanced Modulation UI:** Visual editor for noise schedules or other scheduled parameters.
*   **(Backend/UI) VAE Analysis Hub (v1):** Backend comparison endpoint, basic UI cockpit.
*   **(Backend) Task Queue & Batching (v1):** Implement basic background task queue (`TaskScheduler`). Modify API endpoints for async submission. Add status check endpoint.
*   **(UI) Basic Queue Monitor:** Simple UI to view task queue.
*   **(Research Spike) Novel ControlNets / Samplers:** Focused investigation.
*   **(Feature) Agentic Art Director (v1):** Basic implementation.
*   **(Feature) Prompt Experimentation Tools (v1):** Extend XYZ grid for better prompt handling (wildcards?).

---


**Phase 4: Experimental Deep Dive & Platform Maturation**

*   **Goal:** Develop more advanced experimental features based on Phase 3 research, refine core systems significantly, and improve platform robustness.
*   Structural Diffusion v2, Latent Grafting, VAE Hub v2, Latent Navigator, Agent v2, Performance Paradigms, Stability/Error Handling, API/Docs v1, Packaging.
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


---

**Phase 5: Polish, Release Candidate, Community** (Goals mostly unchanged)

*   UI/UX Polish, Optimization, Testing, Docs, Release, Community.

---

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

This updated plan reflects the completion of many foundational backend and UI tasks, the significant progress on core workflows (T2I/I2I) and management features (Presets, Dynamic Selectors), and sets the stage for finishing Phase 2 (LoRA/ControlNet backend integration, Inpainting, Viz UI, Chat UI) before moving to more advanced features.