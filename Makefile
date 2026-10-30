# gradient-project-root/Makefile
# Thin wrappers around install.sh / uv / npm. Run `make help`.

.PHONY: help install dev-install run dev frontend frontend-dev test test-frontend lint format check clean

VENV := .venv
BIN := $(VENV)/bin

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install GRADIENT (GPU auto-detected) and build the UI
	./install.sh

dev-install: ## Install with test/lint tools
	./install.sh --dev

run: ## Start API + UI on http://127.0.0.1:8000
	$(BIN)/gradient serve

dev: ## Start the API with auto-reload (use with `make frontend-dev`)
	$(BIN)/gradient serve --reload

frontend: ## Build the web UI into frontend/build
	cd frontend && npm ci && npm run build

frontend-dev: ## Vite dev server on :5173 (proxies /api to :8000)
	cd frontend && npm run dev

test: ## Backend tests (offline, CPU)
	$(BIN)/python -m pytest

test-frontend: ## Frontend type check + unit tests
	cd frontend && npm run check && npm test

lint: ## Ruff lint + format check
	$(BIN)/ruff check . && $(BIN)/ruff format --check .

format: ## Format Python code
	$(BIN)/ruff format . && $(BIN)/ruff check --fix .

check: lint test test-frontend ## Everything CI runs

clean: ## Remove caches and build output (keeps .venv, models and data)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache frontend/build frontend/.svelte-kit frontend/storybook-static
