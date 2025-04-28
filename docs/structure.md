gradient/ # project root
│
├── .git/
├── .github/
│   └── workflows/
│       └── python-app.yml
│
├── config/                   # Configuration files (defaults, logging)
│   ├── default_settings.yaml # Default application settings
│   ├── logging_config.yaml   # Logging configuration
│   └── config.yaml           # User overrides (-> .gitignore)
│
├── data/                     # Default location for user-managed AI models, local DBs, etc. (-> .gitignore)
│   │                         # NOTE: App configuration allows adding paths outside this structure.
│   ├── .gitkeep
│   ├── models/               # Default model storage location
│   │   ├── checkpoints/
│   │   ├── diffusers/
│   │   ├── llm/
│   │   ├── lora/
│   │   ├── vae/
│   │   ├── controlnet/
│   │   └── embeddings/
│   └── vector_db/            # Default vector database storage
│       └── .gitkeep
├── docs/                     # Project documentation
│   ├── index.md              # Main documentation page
│   ├── architecture.md       # Detailed architecture description
│   ├── user_guide.md         # How to use the application
│   ├── api.md                # API documentation
│   └── contributing.md       # Contribution guidelines
├── experiments/              # User-generated experiment outputs (-> .gitignore)
│   └── .gitkeep
│
├── frontend/                 # Frontend UI code (React/Vue/Svelte/etc.)
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── vite.config.js        # Example build config
│   └── ...
│
├── logs/                     # Application runtime logs (-> .gitignore)
│   ├── gradient.log
│   └── .gitkeep
│
├── notebooks/                # Jupyter notebooks for exploration/analysis (Optional, -> .gitignore?)
│   └── .gitkeep
│
├── scripts/                  # Utility scripts
│   └── download_models.py
│
├── src/                      # Main Python source code package
│   ├── gradient/             # The actual Python package
│   │   ├── __init__.py
│   │   ├── assets/           # Optional: Static assets shipped WITH the app (e.g., default icons)
│   │   │   └── .gitkeep
│   │   ├── api/              # FastAPI endpoints, models, WebSockets
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/
│   │   │   ├── models.py     # Pydantic models
│   │   │   └── websocket.py
│   │   ├── core/             # Core services (scheduling, resources, sessions, config loading)
│   │   │   ├── __init__.py
│   │   │   ├── scheduling.py
│   │   │   └── ...
│   │   ├── ai/               # AI model interaction modules
│   │   │   ├── __init__.py
│   │   │   ├── diffusion/
│   │   │   ├── llm/
│   │   │   └── agents/
│   │   ├── experiments/      # CODE for running/logging experiments
│   │   │   ├── __init__.py
│   │   │   ├── runner.py
│   │   │   └── plotting.py
│   │   ├── ui/               # Backend logic specifically supporting UI state/views
│   │   │   └── __init__.py
│   │   ├── utils/            # Common utilities
│   │   │   ├── __init__.py
│   │   └── __main__.py       # Enables `python -m gradient`
│
├── tests/                    # Unit and integration tests (mirrors `src/gradient` structure)
│   ├── __init__.py
│   ├── conftest.py
│   ├── gradient/
│   │   ├── core/
│   │   └── api/
│   └── integration/
│
├── .env.example              # Example environment variables file
├── .env                      # Local environment variables (API keys, paths) (-> .gitignore)
├── .gitignore                # Specifies untracked files (venv, __pycache__, data/, experiments/, logs/, config.yaml, .env, etc.)
├── LICENSE                   # Apache 2.0 License file
├── Makefile                  # Convenience commands for development tasks (test, lint, run, etc.)
├── README.md                 # Project overview
├── main.py                   # Main application entry point script (imports from `src/gradient`)
└── pyproject.toml            # Project metadata and dependencies (Poetry, PDM, Hatch, etc.)