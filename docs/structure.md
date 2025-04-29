```
gradient/ # project root
│
├── .git/
├── .github/
│   └── workflows/
│       └── python-app.yml        # Basic CI workflow
│
├── config/                       # Backend configuration files
│   ├── default_settings.yaml   # Default application settings (now includes model paths)
│   ├── logging_config.yaml     # Logging configuration
│   └── config.yaml             # User overrides (Optional, -> .gitignore)
│
├── data/                         # Default location for user data (-> .gitignore)
│   ├── .gitkeep
│   ├── database/                 # <<< ADDED: Default location for SQLite DB
│   │   └── gradient_data.db      # <<< ADDED: Main database file
│   ├── models/                   # Default model storage location (SEARCHED BY BACKEND)
│   │   ├── checkpoints/          # <<< For .ckpt, .safetensors single files
│   │   ├── diffusers/            # <<< For exploded diffusers directories
│   │   ├── llm/                  # <<< For GGUF, etc.
│   │   ├── lora/                 # <<< For LoRA files
│   │   ├── vae/                  # <<< For VAE files/dirs
│   │   ├── controlnet/           # <<< For ControlNet models
│   │   └── embeddings/           # <<< For Textual Inversion embeddings
│   └── vector_db/                # Default vector database storage (Placeholder)
│       └── .gitkeep
│
├── docs/                         # Project documentation
│   ├── index.md
│   ├── architecture.md           # <<< ADDED/UPDATED: High-level design doc
│   ├── user_guide.md
│   ├── api.md                    # <<< ADDED/UPDATED: API endpoint overview
│   └── contributing.md
│
├── experiments/                  # User-generated experiment outputs (e.g., images) (-> .gitignore)
│   └── .gitkeep                  # <<< Output images saved here in subfolders named by run_uuid
│
├── frontend/                     # Frontend UI code (SvelteKit)
│   ├── .git/                     # (Ignore nested .git)
│   ├── .gitignore                # Frontend specific ignores
│   ├── .prettierrc               # Prettier config
│   ├── .prettierignore           # Prettier ignore rules
│   ├── .eslintrc.cjs             # ESLint config
│   ├── node_modules/             # (-> .gitignore)
│   ├── package.json              # Frontend dependencies
│   ├── package-lock.json         # Locked frontend dependencies
│   ├── postcss.config.js         # PostCSS config
│   ├── README.md                 # (Optional: Frontend specific README)
│   ├── svelte.config.js          # SvelteKit config
│   ├── static/                   # Static assets for frontend
│   │   └── favicon.png
│   ├── src/                      # Frontend source code
│   │   ├── app.css               # Global CSS / Tailwind directives
│   │   ├── app.d.ts              # Global TS declarations
│   │   ├── app.html              # Main HTML template
│   │   ├── hooks.server.ts       # SvelteKit server hooks (Optional)
│   │   ├── lib/                  # Reusable frontend modules
│   │   │   ├── index.ts          # (Optional: Barrel file)
│   │   │   ├── apiClient.ts      # <<< UPDATED: Utility for backend API calls
│   │   │   ├── types.ts          # <<< ADDED/UPDATED: TypeScript interfaces matching backend models (T2I, I2I, Grid, Presets)
│   │   │   ├── stores/           # <<< ADDED: Svelte stores for persistent UI state
│   │   │   │   ├── t2iFormStore.ts # <<< ADDED: Store for Text-to-Image form state
│   │   │   │   └── i2iFormStore.ts # <<< ADDED: Store for Image-to-Image form state
│   │   │   └── components/       # Svelte components
│   │   │       ├── TextToImageForm.svelte  # <<< ADDED/UPDATED: T2I form, uses store, shared components, presets, LoRA/Emb UI
│   │   │       ├── ImageToImageForm.svelte # <<< ADDED/UPDATED: I2I form, uses store, shared components, presets, LoRA/Emb UI
│   │   │       ├── XYZGridForm.svelte      # <<< ADDED/UPDATED: Form for XYZ Grid configuration
│   │   │       ├── XYZGridDisplay.svelte   # <<< ADDED/UPDATED: Component to display grid results
│   │   │       ├── LLMChat.svelte          # <<< ADDED: Placeholder/Component for LLM chat interface
│   │   │       └── shared/                 # <<< ADDED: Reusable sub-components
│   │   │           ├── ModelSelector.svelte    # <<< ADDED: Dropdown for model selection
│   │   │           ├── SchedulerSelector.svelte# <<< ADDED: Dropdown for scheduler selection
│   │   │           ├── SeedInput.svelte        # <<< ADDED: Input group for seed (number, random, clear)
│   │   │           └── SubmitButton.svelte     # <<< ADDED: Button with loading state
│   │   └── routes/               # SvelteKit page routes
│   │       ├── +layout.svelte    # Root layout (basic styling)
│   │       └── +page.svelte      # <<< UPDATED: Main page, uses tabs, manages grid state, hosts forms/displays
│   ├── tailwind.config.js        # Tailwind CSS config
│   ├── tsconfig.json             # Frontend TypeScript config
│   └── vite.config.ts            # Vite build tool config (includes /api proxy)
│
├── logs/                         # Application runtime logs (-> .gitignore)
│   ├── gradient.log
│   └── .gitkeep
│
├── notebooks/                    # Jupyter notebooks (Optional)
│   └── .gitkeep
│
├── scripts/                      # Utility scripts
│   └── download_models.py        # Example placeholder
│
├── src/                          # Main Python source code package
│   ├── gradient/                 # The actual Python package
│   │   ├── __init__.py
│   │   ├── assets/
│   │   │   └── .gitkeep
│   │   ├── api/                  # FastAPI endpoints
│   │   │   ├── __init__.py
│   │   │   ├── main.py           # <<< UPDATED: FastAPI app, registers all routers
│   │   │   ├── endpoints/        # Route definitions
│   │   │   │   ├── __init__.py
│   │   │   │   ├── generate.py   # <<< UPDATED: T2I, I2I, XYZ Grid endpoints
│   │   │   │   ├── llm.py        # Basic LLM endpoint
│   │   │   │   ├── status.py     # Basic status endpoint
│   │   │   │   ├── visualize.py  # Basic Latent Viz, Sensitivity endpoints
│   │   │   │   ├── presets.py    # <<< ADDED: API endpoints for managing ModelPresets
│   │   │   │   └── system.py     # <<< ADDED: API endpoints for dynamic lists (models, schedulers, etc.)
│   │   │   └── websocket.py      # WebSocket endpoint placeholder
│   │   ├── core/                 # Core services
│   │   │   ├── __init__.py
│   │   │   ├── config.py         # <<< UPDATED: Configuration loading (handles model paths)
│   │   │   ├── log_setup.py      # Logging setup
│   │   │   ├── resource_manager.py # Basic VRAM/Resource tracking
│   │   │   └── scheduling.py     # Task scheduler placeholder
│   │   ├── ai/                   # AI model interaction modules
│   │   │   ├── __init__.py
│   │   │   ├── diffusion/        # Diffusion model code
│   │   │   │   ├── __init__.py
│   │   │   │   ├── loading.py    # <<< UPDATED: Handles flexible paths, Hub IDs, VAEs, uses config
│   │   │   │   ├── models.py     # <<< UPDATED: Pydantic models (T2I, I2I, XYZ, includes scheduler_id)
│   │   │   │   ├── pipelines.py  # <<< UPDATED: T2I/I2I logic, scheduler handling, DB logging, metadata injection
│   │   │   │   └── xyz_grid.py   # <<< UPDATED: XYZ Grid logic, logs to DB via pipeline calls
│   │   │   ├── llm/              # LLM code (Basic)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── interaction.py
│   │   │   │   ├── loading.py
│   │   │   │   └── models.py
│   │   │   ├── viz/              # Visualization backend code (Basic)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── latent_space.py # <<< UPDATED: Uses find_model_path for VAE
│   │   │   │   ├── models.py
│   │   │   │   └── sensitivity.py
│   │   │   └── agents/           # Placeholder
│   │   │       └── __init__.py
│   │   ├── database/             # <<< ADDED: Database interaction layer
│   │   │   ├── __init__.py
│   │   │   ├── models.py         # <<< ADDED/UPDATED: SQLAlchemy ORM models (ExperimentRun, GeneratedOutput, ModelPreset)
│   │   │   └── session.py        # <<< ADDED/UPDATED: Async DB session setup, init_db, helpers
│   │   ├── experiments/          # Placeholder for experiment running CODE (distinct from output folder)
│   │   │   ├── __init__.py
│   │   │   ├── plotting.py
│   │   │   └── runner.py
│   │   ├── ui/                   # Placeholder
│   │   │   └── __init__.py
│   │   ├── utils/                # Common utilities (Placeholder)
│   │   │   └── __init__.py
│   │   └── __main__.py           # Enables `python -m gradient`
│
├── tests/                        # Unit and integration tests (Placeholders)
│   ├── __init__.py
│   ├── conftest.py
│   ├── gradient/
│   │   # ... (Mirror src structure) ...
│   └── integration/
│       └── __init__.py
│
├── .env                          # Local environment variables (-> .gitignore)
├── .env.example                  # Example environment variables file
├── .gitignore                    # Main project gitignore (Updated for DB file)
├── LICENSE                       # Apache 2.0 License file
├── Makefile                      # Convenience commands
├── README.md                     # Project overview
├── main.py                       # Main application entry point script
├── poetry.lock                   # Locked Python dependencies
└── pyproject.toml                # Project metadata and dependencies
```

**Summary of Additions/Modifications Noted:**

*   **Frontend:** Added entire SvelteKit structure, including specific components (`TextToImageForm.svelte`), utilities (`apiClient.ts`), config files (`vite.config.ts`, `tailwind.config.js`, etc.), modified layout/page (`+layout.svelte`, `+page.svelte`), and type declarations (`app.d.ts`).
Installed also frontend\.storybook for UI update.
*   **Backend (`src/gradient`):**
    *   **API:** Added `generate.py`, `llm.py`, `visualize.py` endpoints. Modified `main.py` to register routers with correct prefixes.
    *   **AI:** Added `diffusion/xyz_grid.py`, `llm/` subdirectory with `loading.py`, `interaction.py`, `models.py`. Added `viz/` subdirectory with `latent_space.py`, `sensitivity.py`, `models.py`. Modified `diffusion/loading.py`, `diffusion/pipelines.py`, `diffusion/models.py`.
    *   **Core:** Modified `resource_manager.py`, `log_setup.py`, `config.py`.
*   **Root:** Modified `pyproject.toml`, `poetry.lock`, `Makefile`, `.gitignore`, `.env`.

This reflects the significant progress made through Phase 0 and the initial stages of Phase 1 & 2 backend implementation, plus the frontend setup.