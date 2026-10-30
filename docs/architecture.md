# GRADIENT Architecture

This document outlines the high-level architecture of the GRADIENT Workbench.

## Overview

GRADIENT is designed as a modular, web-based toolkit for interacting with and experimenting on generative AI models, primarily focusing on Diffusion and Large Language Models (LLMs). It follows a client-server architecture:

1.  **Backend (Python/FastAPI):** A Python server responsible for all core AI operations, model management, data persistence, and serving the API.
2.  **Frontend (SvelteKit):** A modern web application providing the user interface, interacting with the backend via its API.
3.  **Core Services:** Shared Python modules providing configuration, logging, resource management, and potentially background task scheduling.
4.  **Database:** A persistent store (currently SQLite) for experiment tracking, results metadata, and user configurations like presets.

```mermaid
graph LR
    A[User Browser] -- HTTP/WebSocket --> B(Frontend - SvelteKit/Vite);
    B -- API Calls --> C{Backend - FastAPI};
    C -- Uses --> D[Core Services];
    C -- Uses --> E[AI Modules];
    C -- R/W --> F[Database - SQLite];
    E -- Loads --> G[Models (Filesystem/Hub)];
    D -- Manages --> G;
    C -- Writes --> H[Experiment Outputs (Filesystem)];

    subgraph Backend Server
        C
        D(Config, Logging, Resources, Scheduler)
        E(Diffusion, LLM, Viz)
        F
    end

    subgraph Client-Side
        A
        B
    end

    subgraph Data Storage
        G
        H
    end

    style F fill:#lightgrey,stroke:#333,stroke-width:2px
    style G fill:#lightgrey,stroke:#333,stroke-width:2px
    style H fill:#lightgrey,stroke:#333,stroke-width:2px
```

## Backend (`src/gradient/`)

The backend is the core engine of GRADIENT, built using Python 3.11+ and the FastAPI framework.

### Key Components:

1.  **API (`src/gradient/api/`)**:
    *   Built with **FastAPI** for high performance and automatic OpenAPI (Swagger UI) documentation.
    *   Defines endpoints grouped by functionality (e.g., `generate`, `llm`, `visualize`, `presets`, `system`).
    *   Uses **Pydantic** models for request/response validation and serialization.
    *   Handles HTTP requests from the frontend.
    *   Includes a placeholder **WebSocket** endpoint for potential real-time updates (e.g., progress, logs).
    *   `main.py`: Initializes the FastAPI app, sets up lifespan events (like DB initialization), and registers routers.

2.  **Core Services (`src/gradient/core/`)**:
    *   **Configuration (`config.py`)**: Loads settings from YAML files (`default_settings.yaml`, `config.yaml`) and environment variables (`.env`). Provides access via `get_setting()`. Defines key paths like `PROJECT_ROOT`.
    *   **Logging (`log_setup.py`)**: Configures application-wide logging based on `logging_config.yaml`.
    *   **Device (`device.py`)**: Picks the compute device (cuda > mps > xpu > cpu, or `performance.device`) and a dtype per model family (fp16 for UNet-era models, bf16 for DiT/flow-matching models).
    *   **Resource Manager (`resource_manager.py`)**: Tracks loaded models with real unload callbacks and evicts the least recently used one when needed.
    *   **Scheduling (`scheduling.py`)**: `run_gpu_task()` runs blocking model work in a worker thread behind a single GPU lock, so the event loop (API, downloads, polling) stays responsive. Also tracks step progress for `/system/status`. Foundation for the planned job queue.

3.  **AI Modules (`src/gradient/ai/`)**: Encapsulates logic for specific AI tasks.
    *   **Diffusion (`diffusion/`)**:
        *   `loading.py`: Loads any diffusers family. Diffusers folders/Hub repos use the class from `model_index.json`; single-file checkpoints are identified from their tensor names. Image-to-image pipelines are derived with `AutoPipelineForImage2Image.from_pipe` (shared weights). Applies `performance.memory_mode` and keeps an LRU cache of pipelines.
        *   `guidance.py`: Parser for scheduled CFG (`linear(a,b)`, `cosine(a,b)`, keyframes).
        *   `models.py`: Pydantic models for diffusion tasks (T2I, I2I, XYZGrid requests/responses).
        *   `pipelines.py`: `generate_images()` calls the pipeline's own `__call__` with per-image seeded generators, custom sigmas, scheduler validation, and a `callback_on_step_end` hook (scheduled CFG, progress; the extension point for modulation). Async wrappers save PNGs with metadata and log runs to the database.
        *   `xyz_grid.py`: Implements logic for generating images based on parameter variations across axes, coordinating calls to underlying pipelines.
    *   **LLM (`llm/`)**: (Placeholder/Basic Implementation)
        *   `loading.py`: Logic for loading LLMs (currently focusing on GGUF via `llama-cpp-python`).
        *   `interaction.py`: Functions to handle chat completions (streaming and non-streaming).
        *   `models.py`: Pydantic models for LLM requests/responses.
    *   **Visualization (`viz/`)**: (Placeholder/Basic Implementation)
        *   `latent_space.py`: Prototype UMAP projection of random latents (scaffolding; see the research review for the planned output-space atlas).
        *   `sensitivity.py`: Parameter sensitivity heatmaps by central finite differences on shared noise (guidance scale or a LoRA weight).
        *   `models.py`: Pydantic models for visualization requests/responses.
    *   **Agents (`agents/`)**: Placeholder for future agentic capabilities.

4.  **Model Manager (`src/gradient/model_manager/`)**:
    *   `catalog.py` loads the curated catalog (`config/model_catalog.yaml`).
    *   `inventory.py` lists, resolves and deletes local models across the configured search folders; `manifest.py` records where downloads came from (repo, licence, family, recommended defaults).
    *   `downloader.py` runs downloads in the background: Hugging Face repos/files in a child process (`hf_worker.py`, so they can be cancelled and resumed) with byte-level progress; direct URLs with HTTP Range resume. Files are staged in `.staging/` and moved into place when complete.
    *   Exposed via `/api/v1/models/*` and the `gradient models ...` CLI (`src/gradient/cli.py`).

5.  **Database (`src/gradient/database/`)**:
    *   **Models (`models.py`)**: Defines SQLAlchemy ORM models (`ExperimentRun`, `GeneratedOutput`, `ModelPreset`) representing the database schema using `declarative_base`.
    *   **Session (`session.py`)**: Configures the asynchronous SQLAlchemy engine (`create_async_engine` with `aiosqlite`), provides an async session factory (`async_sessionmaker`), and includes the `get_db_session` context manager for dependency injection in FastAPI endpoints. Also contains database initialization logic (`init_db`) and helper functions (like `update_run_status`).

### Design Principles:

*   **Asynchronous:** Uses `async`/`await` throughout FastAPI and database interactions for non-blocking I/O. AI inference tasks are typically CPU/GPU bound but wrapped in async functions for API consistency.
*   **Modularity:** Separates concerns into API endpoints, core services, AI logic, and database interactions.
*   **Dependency Injection:** Leverages FastAPI's `Depends` system, particularly for database sessions.
*   **Type Hinting:** Extensive use of Python type hints, enforced by Pydantic for API models.

## Frontend (`frontend/`)

The frontend provides the user interface for interacting with the GRADIENT backend.

### Key Technologies:

*   **SvelteKit 3:** Configured in `vite.config.ts` (no `svelte.config.js`), built with `adapter-static` as a single-page app into `frontend/build`, which the FastAPI server serves at `/`. Library code is imported through the `#lib` subpath alias.
*   **Svelte:** A reactive UI compiler that shifts work from the browser to the build step, resulting in highly performant applications.
*   **TypeScript:** Used for type safety in frontend code.
*   **Tailwind CSS:** A utility-first CSS framework for rapid UI development.
*   **Vite:** The build tool used by SvelteKit, providing fast development server startup and optimized builds.

### Structure:

*   **`src/routes/`:** Defines the pages and layouts of the application according to SvelteKit's file-based routing conventions.
    *   `+layout.svelte`: Root layout, applies global styling.
    *   `+page.svelte`: The main application page, currently hosting the tabbed interface for different generation modes. Manages high-level UI state and orchestrates child components.
*   **`src/lib/`:** Contains reusable code and components.
    *   `components/`: Svelte components for specific UI parts (e.g., `TextToImageForm`, `ImageToImageForm`, `XYZGridForm`, `XYZGridDisplay`, `LLMChat`). Includes `shared/` sub-directory for highly reusable elements (`ModelSelector`, `SeedInput`, etc.).
    *   `stores/`: Svelte stores (e.g., `t2iFormStore`, `i2iFormStore`) used for managing persistent UI state across component lifecycles (like form inputs when switching tabs).
    *   `types.ts`: TypeScript interfaces mirroring backend Pydantic models for type safety during API interactions.
    *   `apiClient.ts`: Utility functions (`get`, `post`, etc.) built on top of the browser's `fetch` API for communicating with the backend API, including basic error handling.
*   **`vite.config.ts`:** Configures Vite, notably including a proxy setup to forward `/api` requests from the frontend dev server to the backend FastAPI server during development, avoiding CORS issues.
*   **`static/`:** For static assets like favicons.

### Design Principles:

*   **Component-Based:** UI built from reusable Svelte components.
*   **Reactivity:** Leverages Svelte's built-in reactivity for state management (`$` store subscriptions, `bind:value`).
*   **State Management:** Uses Svelte stores for managing shared/persistent UI state.
*   **API Client:** Centralizes API communication logic.

## Data Flow Example (Text-to-Image)

1.  User interacts with `TextToImageForm.svelte` in the browser.
2.  Input values update the `$t2iFormStore`.
3.  User clicks "Generate".
4.  The `generateImage` function in the component reads current state from `$t2iFormStore`.
5.  It calls the `post` function in `apiClient.ts`.
6.  `apiClient.ts` makes an HTTP POST request to `/api/v1/generate/text-to-image` (proxied by Vite dev server to the FastAPI backend).
7.  FastAPI routes the request to `handle_text_to_image` in `generate.py`.
8.  The endpoint function parses the request using `Text2ImageRequest` model.
9.  It calls `run_text_to_image` in `pipelines.py`.
10. `run_text_to_image` potentially creates an `ExperimentRun` DB record.
11. It calls `load_diffusion_pipeline` (which might load from disk/cache and update `ResourceManager`).
12. It performs the diffusion steps using the pipeline, VAE, scheduler.
13. It saves the output image(s) to the `experiments/` directory.
14. It creates `GeneratedOutput` DB record(s) linking to the `ExperimentRun`.
15. It updates the `ExperimentRun` status to "completed".
16. It encodes the image(s) to base64 and returns a `Text2ImageResponse`.
17. FastAPI sends the JSON response back to the browser.
18. `apiClient.ts` receives the response.
19. The `generateImage` function updates the `results` state variable in `TextToImageForm.svelte`.
20. Svelte reactively updates the UI to display the generated image(s).

## Future Considerations

*   **Task Queue:** Implementing the `TaskScheduler` for background processing of long tasks.
*   **Resource Management:** Implementing actual model unloading based on LRU policy via callbacks between `ResourceManager` and loaders.
*   **Error Handling:** More granular error handling and user feedback on both frontend and backend.
*   **Scalability:** Database choice (beyond SQLite for multi-user/high load), potential for distributed workers.
*   **Security:** Authentication/Authorization if deployed publicly. Input sanitization.
*   **Testing:** Comprehensive unit, integration, and end-to-end tests.



