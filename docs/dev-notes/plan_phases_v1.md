**Current Sampler in GRADIENT - in Phase 2:**

Right now, GRADIENT isn't explicitly *choosing* a sampler in the code we've written (`src/gradient/ai/diffusion/pipelines.py`). When you call a standard `diffusers` pipeline like `StableDiffusionPipeline` without specifying a scheduler/sampler, it uses **the default scheduler configured for that specific pipeline when it was saved or defined by the library.**

For `StableDiffusionPipeline` loaded from standard checkpoints (like v1.5), the default scheduler is typically **`PNDMScheduler`**. However, diffusers allows swapping schedulers easily.

Let's incorporate your ideas into the action plan as potential future phases or expansions.

---

**GRADIENT Development Plan - Future Phases**

**(Assuming Phase 0, 1, and initial parts of 2 are nearing completion based on our progress)**

**Phase 2.5: Core Generation & UI Integration (Refinement/Expansion)**

*   **Goal:** Solidify core generation workflows within the UI, making the tool practically usable for standard tasks while refining the backend.
*   **Key Actions:**
    *   **(UI) Implement LLM Chat Interface:** Create frontend components to interact with the `/api/v1/llm/chat/completions` endpoint (both streaming and non-streaming). Display conversation history.
    *   **(UI) Implement XYZ Grid Display:** Create a frontend component to display the results from the `/api/v1/generate/xyz-grid` endpoint in a clear grid format, showing images and associated parameters.
    *   **(UI) Implement Latent Space Viz (Basic):** Create the frontend component using a plotting library (like Plotly.js, D3.js, R3F/Three.js) to display the points from `/api/v1/visualize/latent-space` as a 2D scatter plot. Add basic hover effects to show preview images.
    *   **(UI) Implement Sensitivity Viz Display:** Display the statistics from `/api/v1/visualize/parameter-sensitivity`. (Optional: Generate and display basic heatmap if backend adds support).
    *   **(Backend) Implement Img2Img Pipeline:**
        *   Define Pydantic models for Img2Img requests/responses.
        *   Add `run_image_to_image` logic in `diffusion/pipelines.py` using `StableDiffusionImg2ImgPipeline`.
        *   Add API endpoint in `generate.py`.
    *   **(Backend) Implement Inpainting Pipeline:**
        *   Define Pydantic models (including mask input).
        *   Add `run_inpainting` logic in `diffusion/pipelines.py` using `StableDiffusionInpaintPipeline`.
        *   Add API endpoint in `generate.py`.
    *   **(UI) Add Img2Img/Inpainting Forms:** Create UI components for these new workflows.
    *   **(Backend) Scheduler Selection:**
        *   Modify diffusion request models (`DiffusionTaskBase`) to accept an optional `scheduler_id` (e.g., "DDIM", "DPM++ 2M Karras", "Euler A").
        *   Modify `load_diffusion_pipeline` or `run_text_to_image` to dynamically load and set the chosen scheduler on the pipeline using `pipeline.scheduler = SchedulerClass.from_config(pipeline.scheduler.config, ...)`.
    *   **(UI) Add Scheduler Selector:** Add a dropdown/selector in the UI forms.
    *   **(Refinement) Address Deprecations:** Update code using deprecated functions (like `_encode_prompt`).
    *   **(Refinement) Improve Resource Management:** Implement proper VAE/component caching/unloading coordinated with `ResourceManager`. Improve VRAM estimation.

**Phase 3: Advanced Control & Research Spike Initiation**

*   **Goal:** Introduce more sophisticated control mechanisms (ControlNet, LoRA management) and begin dedicated research spikes into novel techniques.
*   **Key Actions:**
    *   **(Backend) ControlNet Integration:**
        *   Models for ControlNet input (image, type, weight).
        *   Logic in `loading.py` to load ControlNet models.
        *   Modify pipeline runners (`pipelines.py`) to accept and use ControlNet inputs (`StableDiffusionControlNetPipeline`, etc.).
        *   Update API endpoints.
    *   **(UI) ControlNet Interface:** UI elements to upload control images, select ControlNet models, and set weights/parameters.
    *   **(Backend/UI) LoRA Management:**
        *   Backend logic to load/unload multiple LoRAs and apply them dynamically (e.g., using `load_lora_weights`, `unload_lora_weights`).
        *   More complex resource management for stacked LoRAs.
        *   UI for selecting multiple LoRAs and setting their weights (e.g., LoRA Atlas concept).
    *   **(Research Spike) Novel ControlNets:**
        *   Setup environment/workflow for training or fine-tuning ControlNet models.
        *   Begin literature review and experimentation based on specific research goals (e.g., control based on abstract concepts, audio, etc.). Document findings.
    *   **(Research Spike) Novel Samplers:**
        *   Implement standard diffusers samplers/schedulers thoroughly.
        *   Literature review on advanced/novel sampling techniques (e.g., ancestral variations, rectified flow, score-based methods beyond standard libraries).
        *   Experiment with implementing 1-2 promising novel samplers within the GRADIENT framework. Compare performance/quality.
    *   **(Backend) True Noise Schedule Sculpting:** Implement logic to directly use `custom_sigmas` values by potentially creating custom schedulers or finding ways to inject sigmas into existing ones (research required per scheduler).
    *   **(Refinement) Interactive Latent Space Navigator (v2):** Add interaction features identified in Phase 2 (e.g., basic "latent painting" PoC, path definition).

**Phase 4: Experimental Modules & Advanced Research**

*   **Goal:** Develop and integrate genuinely experimental modules (VAEs, prompt testing) and mature the research capabilities.
*   **Key Actions:**
    *   **(Feature) Experimental Autoencoder Hub:**
        *   Refined VAE loading/unloading/caching.
        *   API endpoints and backend logic specifically for comparing VAE reconstructions side-by-side (using `VAEManager` ideas).
        *   UI components for the "VAE Analysis Cockpit".
    *   **(Research Spike) Develop/Train Novel Autoencoders:**
        *   Establish training pipeline/environment.
        *   Experiment with VAE architectural changes or training objectives based on research goals (e.g., better detail preservation, style encoding). Integrate successful experiments into the Hub.
    *   **(Feature) Advanced Prompt Experimentation:**
        *   Extend XYZ grid or create new UI/backend for multi-dimensional prompt testing (e.g., varying multiple prompt parts, using embeddings).
        *   Implement "Before-Current-After" comparison views for iterative prompt refinement.
        *   Integrate basic LLM assistance for prompt generation/suggestion within the UI.
    *   **(Feature) Structural Diffusion Modulation (v2):** Implement more advanced modulation controls (e.g., UNet layer weighting, guidance vector manipulation) based on Phase 3 research.
    *   **(Feature) Cross-Model / Latent Grafting (v1):** Implement backend logic and a basic UI workflow for selecting and merging latent regions.
    *   **(Refinement) Agentic Art Director (v2):** Enhance agent capabilities (e.g., use feedback from image analysis, more complex goal decomposition).
    *   **(Refinement) Performance Paradigms:** Implement latency hiding, speculative execution, adaptive resource allocation based on feasibility studies.

**Phase 5: Polish, Release Candidate, Community Features (Ongoing)**

*   **Goal:** Stabilize experimental features, polish UI/UX, optimize performance, prepare for wider release, and build community tools.
*   **Key Actions:**
    *   Extensive testing and bug fixing across all features.
    *   UI/UX overhaul based on accumulated feedback.
    *   Performance profiling and optimization.
    *   Comprehensive documentation updates.
    *   Packaging and release candidate preparation.
    *   Features for sharing/importing workflows, presets, experiments.
    *   Ongoing maintenance and integration of new SOTA techniques.

**More Ideas (Could fit into various phases):**

*   **Embedding/Textual Inversion Training UI:** Interface for training custom concepts.
*   **Style/Artist Database:** Integrate metadata or analysis to identify and replicate specific artist styles more easily.
*   **Multimodal Input:** Accept images or audio as inputs for generation or control (connects to ControlNet/Img2Img).
*   **Node-Based Workflow Editor:** A more visual way to construct complex generation pipelines (similar to ComfyUI but potentially integrated with other GRADIENT features).
*   **Federated Learning/Swarm UI (Highly Ambitious):** Interface for collaborative model exploration or distributed generation tasks.
*   **Real-time Collaboration:** Allow multiple users to interact with the same generation canvas/parameters.

This expanded plan provides a roadmap for incorporating your research interests and developing GRADIENT into a truly unique and powerful tool. Remember to prioritize based on impact and feasibility at each stage.

Robust **File/Experiment Management** and **Batch Processing** are absolutely critical for both practical use and scientific rigor. They shouldn't be afterthoughts.

**Refined Thoughts on File/Experiment Management & Batching:**

1.  **Centralized Experiment Tracking:**
    *   **Concept:** Instead of just saving outputs to timestamped folders, maintain a lightweight database (like SQLite, integrated via SQLAlchemy) or structured log files (e.g., JSONL) that index *every* generation or experiment run.
    *   **Data:** Store unique run ID, timestamp, all input parameters (request models), generated seeds, output file paths (images, data files), performance metrics, user tags/notes, and potentially links between related runs (e.g., grid cells belonging to one grid request).
    *   **Benefits:** Enables powerful querying ("Show me all images using model X with CFG > 8"), reproducibility (easily retrieve exact parameters), comparison across experiments, and forms the basis for visualization/analysis within GRADIENT.
    *   **Integration:** API endpoints would log to this database. A dedicated UI section would allow browsing, searching, filtering, tagging, and potentially deleting experiments/outputs.

2.  **Parameter & Workflow Presets:**
    *   **Concept:** Allow users to save and load complete sets of parameters (prompts, negative prompts, sampler settings, LoRA combinations, VAE choice, grid axes, etc.) as named presets. Workflows could include chained steps (e.g., T2I -> Upscale -> VAE Compare).
    *   **Benefits:** Speeds up common tasks, facilitates sharing setups, enables systematic application of specific settings across different base models or prompts.

3.  **Metadata Injection & Extraction:**
    *   **Concept:**
        *   *Injection:* Automatically embed all generation parameters (prompt, negative, seed, steps, CFG, model IDs, VAE, LoRAs+weights, `custom_sigmas`, etc.) into the EXIF or PNG metadata of generated image files. Use a standardized format (like A1111/ComfyUI's generation params standard or a custom JSON chunk).
        *   *Extraction:* Provide a tool/UI feature to load an image generated by GRADIENT (or potentially other UIs if compatible formats used) and automatically populate the UI fields with the parameters extracted from its metadata ("Send to T2I", "Send to Grid").
    *   **Benefits:** Makes images self-contained records of their creation, crucial for reproducibility and sharing without needing the separate database entry (though the DB provides better search).

4.  **Smart Batching & Queuing:**
    *   **Concept:** Implement the `TaskScheduler` properly (Phase 1/2 refinement). Allow users to queue multiple individual generations, grids, or other tasks. The scheduler should manage the queue based on priority and available resources (especially VRAM).
    *   **UI:** A dedicated "Queue" panel showing pending, running, completed, and failed tasks with logs and controls (pause, cancel, reorder).
    *   **Benefits:** Enables running large sets of generations overnight or while doing other things, prevents UI lockups, manages resource contention gracefully.

5.  **Result Comparison & Annotation UI:**
    *   **Concept:** A dedicated view for comparing outputs from different runs side-by-side. This could be linked from the experiment browser or the XYZ grid display. Allow users to rate, tag, or add notes directly to images/results within the UI, storing this feedback in the experiment database.
    *   **Benefits:** Facilitates visual analysis, subjective quality assessment, and organizing research findings.

**More Ideas for Practical Use / Novel Concepts (Scientific/Research Focus):**

6.  **Live Parameter "Jamming" Interface:**
    *   **Concept:** A UI mode where key parameters (e.g., CFG, seed, LoRA weight, even steps mid-flight if possible via dynamic prompt weaving hooks) are mapped to sliders or even MIDI controllers. The generation runs in a fast, low-step preview loop, updating almost instantly as parameters are changed ("jamming" with the model). High-quality versions can be triggered on demand.
    *   **Novelty:** Turns generation into a real-time performance/exploration tool, fundamentally different from batch/request-response. Connects to experimental art goals.

7.  **Gradient Flow Visualization (Advanced Sensitivity):**
    *   **Concept:** Go beyond static sensitivity heatmaps. Visualize the *flow* of gradients backwards from a selected output region (or semantic concept identified by CLIP) through the UNet layers during the `loss.backward()` call in the sensitivity analysis. Which parts of the prompt embedding or UNet layers most influenced that specific output?
    *   **Novelty:** Provides deeper insight into *how* the model makes decisions and maps concepts, useful for model debugging and understanding guidance mechanisms. Technically challenging.

8.  **Semantic Latent Space Navigation:**
    *   **Concept:** Combine the Latent Space Navigator with CLIP or similar models. Allow users to input text (e.g., "more metallic," "less cluttered," "add trees") and visualize the corresponding *direction vector* in the latent space. Allow "nudging" the current latent vector along that semantic direction before decoding.
    *   **Novelty:** Provides intuitive, text-based control over the *latent representation* itself, complementing prompt engineering.

9.  **Cross-Attention Map Visualization & Control:**
    *   **Concept:** Visualize the cross-attention maps generated during diffusion (showing which parts of the image attend to which parts of the prompt). Add experimental controls to *modify* these attention maps manually or algorithmically (e.g., force attention to a specific word in a specific region) before they influence the next diffusion step.
    *   **Novelty:** Offers fine-grained control over prompt interpretation and spatial layout, potentially fixing issues where the model ignores parts of the prompt. Research-heavy.

10. **Scheduler State Analysis:**
    *   **Concept:** For a given generation, plot the evolution of key scheduler state variables over time (sigma values used, alpha/beta values, noise levels). Compare these plots between different schedulers for the same generation task.
    *   **Novelty:** Helps understand the dynamic differences between samplers beyond just the final image, useful for research and developing new schedulers.

11. **"Concept Algebra" in Latent Space:**
    *   **Concept:** Provide tools to find representative latent vectors for specific concepts (e.g., average latent for "cat" images generated with a seed range) and perform vector arithmetic (e.g., "king" - "man" + "woman" = "queen", but applied to image concepts via their latent representations). Decode the resulting vector.
    *   **Novelty:** Explores the semantic structure of the latent space using classic embedding techniques, potentially yielding interesting conceptual blends.

**Incorporating into Action Plan:**

*   **File/Experiment Management (DB, Presets, Metadata):** Should be integrated early, likely starting in **Phase 1** refinement and expanding through **Phase 2.5**. It's foundational.
*   **Batching/Queuing:** Proper implementation of the `TaskScheduler` is a key part of **Phase 2.5**.
*   **Result Comparison/Annotation UI:** Fits well within **Phase 2.5** or **Phase 3** alongside the UI development.
*   **Live Parameter Jamming:** Could be a **Phase 3** feature, building on core generation.
*   **Gradient Flow Viz / Semantic Latent Nav / Attention Control:** These are advanced research features, suitable for **Phase 4** or beyond, potentially requiring dedicated research spikes.
*   **Scheduler State Analysis / Concept Algebra:** Could be integrated into the research tooling in **Phase 3** or **Phase 4**.

This provides a much richer picture, blending practical usability with cutting-edge research capabilities.