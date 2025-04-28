```
gradient/ # project root
│
├── .git/
├── .github/
│   └── workflows/
│       └── python-app.yml        # Basic CI workflow
│
├── config/                       # Backend configuration files
│   ├── default_settings.yaml   # Default application settings
│   ├── logging_config.yaml     # Logging configuration (Updated)
│   └── config.yaml             # User overrides (Optional, -> .gitignore)
│
├── data/                         # Default location for user data (-> .gitignore)
│   ├── .gitkeep
│   ├── models/                   # Default model storage location
│   │   ├── checkpoints/
│   │   ├── diffusers/
│   │   ├── llm/
│   │   ├── lora/
│   │   ├── vae/
│   │   ├── controlnet/
│   │   └── embeddings/
│   └── vector_db/                # Default vector database storage
│       └── .gitkeep
│
├── docs/                         # Project documentation
│   ├── index.md
│   ├── architecture.md
│   ├── user_guide.md
│   ├── api.md
│   └── contributing.md
│
├── experiments/                  # User-generated experiment outputs (-> .gitignore)
│   └── .gitkeep
│
├── frontend/                     # Frontend UI code (SvelteKit)
│   ├── .git/                     # (Possibly nested git if initialized separately, usually not needed)
│   ├── .gitignore                # Frontend specific ignores (node_modules, build, .svelte-kit, etc.)
│   ├── .prettierrc               # Prettier config (Added/Generated)
│   ├── .prettierignore           # Prettier ignore rules (Added/Generated)
│   ├── .eslintrc.cjs             # ESLint config (Added/Generated)
│   ├── node_modules/             # (-> .gitignore)
│   ├── package.json              # Frontend dependencies (Added/Generated)
│   ├── package-lock.json         # Locked frontend dependencies (Added/Generated)
│   ├── postcss.config.js         # PostCSS config (Added/Generated)
│   ├── README.md                 # (Optional: Frontend specific README)
│   ├── svelte.config.js          # SvelteKit config (Added/Generated)
│   ├── static/                   # Static assets for frontend
│   │   └── favicon.png           # Example
│   ├── src/                      # Frontend source code
│   │   ├── app.css               # Global CSS / Tailwind directives (Added/Generated)
│   │   ├── app.d.ts              # Global TS declarations (Added/Modified)
│   │   ├── app.html              # Main HTML template (Added/Generated)
│   │   ├── hooks.server.ts       # SvelteKit server hooks (Optional)
│   │   ├── lib/                  # Reusable frontend modules
│   │   │   ├── index.ts          # (Optional: Barrel file for lib exports)
│   │   │   ├── apiClient.ts      # <<< ADDED: API client utility
│   │   │   └── components/       # <<< ADDED: Directory for components
│   │   │       └── TextToImageForm.svelte # <<< ADDED: T2I Form component
│   │   └── routes/               # SvelteKit page routes
│   │       ├── +layout.svelte    # <<< MODIFIED: Root layout
│   │       └── +page.svelte      # <<< MODIFIED: Main page using T2I Form
│   ├── tailwind.config.js        # Tailwind CSS config (Added/Generated)
│   ├── tsconfig.json             # Frontend TypeScript config (Added/Modified)
│   └── vite.config.ts            # Vite build tool config (Added/Modified for proxy)
│
├── logs/                         # Application runtime logs (-> .gitignore)
│   ├── gradient.log
│   └── .gitkeep
│
├── notebooks/                    # Jupyter notebooks (Optional, -> .gitignore?)
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
│   │   │   ├── main.py           # <<< MODIFIED: Main FastAPI app, router registration
│   │   │   ├── endpoints/        # Route definitions
│   │   │   │   ├── __init__.py
│   │   │   │   ├── generate.py   # <<< ADDED/MODIFIED: Text2Image, XYZ Grid endpoints
│   │   │   │   ├── llm.py        # <<< ADDED: Chat Completion endpoints
│   │   │   │   ├── status.py     # Basic status endpoint
│   │   │   │   └── visualize.py  # <<< ADDED: Latent Viz, Sensitivity endpoints
│   │   │   └── websocket.py      # WebSocket endpoint placeholder
│   │   ├── core/                 # Core services
│   │   │   ├── __init__.py
│   │   │   ├── config.py         # Configuration loading (Modified slightly)
│   │   │   ├── log_setup.py      # Logging setup (Modified slightly)
│   │   │   ├── resource_manager.py # <<< MODIFIED: Basic VRAM/Resource tracking
│   │   │   └── scheduling.py     # Task scheduler placeholder
│   │   ├── ai/                   # AI model interaction modules
│   │   │   ├── __init__.py
│   │   │   ├── diffusion/        # Diffusion model code
│   │   │   │   ├── __init__.py
│   │   │   │   ├── loading.py    # <<< ADDED/MODIFIED: Pipeline/Checkpoint loading
│   │   │   │   ├── models.py     # <<< ADDED/MODIFIED: Pydantic diffusion models (T2I, XYZ)
│   │   │   │   ├── pipelines.py  # <<< ADDED/MODIFIED: Text-to-Image pipeline runner logic
│   │   │   │   └── xyz_grid.py   # <<< ADDED: XYZ Grid generation logic
│   │   │   ├── llm/              # LLM code
│   │   │   │   ├── __init__.py
│   │   │   │   ├── interaction.py # <<< ADDED: Chat interaction logic
│   │   │   │   ├── loading.py    # <<< ADDED: GGUF loading logic
│   │   │   │   └── models.py     # <<< ADDED: Pydantic LLM models
│   │   │   ├── viz/              # Visualization backend code
│   │   │   │   ├── __init__.py
│   │   │   │   ├── latent_space.py # <<< ADDED: Latent space sampling/projection
│   │   │   │   ├── models.py     # <<< ADDED: Pydantic viz models (Latent, Sensitivity)
│   │   │   │   └── sensitivity.py  # <<< ADDED: Parameter sensitivity analysis
│   │   │   └── agents/           # Placeholder for agentic features
│   │   │       └── __init__.py
│   │   ├── experiments/          # CODE for running/logging experiments
│   │   │   ├── __init__.py
│   │   │   ├── plotting.py       # Placeholder
│   │   │   └── runner.py         # Placeholder
│   │   ├── ui/                   # Backend logic supporting UI state (Placeholder)
│   │   │   └── __init__.py
│   │   ├── utils/                # Common utilities
│   │   │   └── __init__.py
│   │   └── __main__.py           # Enables `python -m gradient` (Modified slightly)
│
├── tests/                        # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── gradient/                 # Mirror src structure
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   └── __init__.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── endpoints/
│   │   │       └── __init__.py
│   │   ├── ai/
│   │   │   ├── __init__.py
│   │   │   └── diffusion/
│   │   │       └── __init__.py
│   │   # ... other test subdirs ...
│   └── integration/
│       └── __init__.py
│
├── .env                          # Local environment variables (-> .gitignore, Modified)
├── .env.example                  # Example environment variables file
├── .gitignore                    # Main project gitignore (Modified)
├── LICENSE                       # Apache 2.0 License file
├── Makefile                      # Convenience commands (Modified)
├── README.md                     # Project overview
├── main.py                       # Main application entry point script (Modified)
├── poetry.lock                   # Locked Python dependencies
└── pyproject.toml                # Project metadata and dependencies (Modified)
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