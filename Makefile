# gradient-project-root/Makefile

.PHONY: help setup install dev-install run lint format test clean docs freeze

# --- Configuration ---
VENV_DIR := venv
PYTHON_EXECUTABLE ?= python3 # Use python3 by default, can be overridden (e.g., PYTHON_EXECUTABLE=python3.10)

# Check if venv directory exists
VENV_EXISTS := $(shell test -d $(VENV_DIR) && echo 1 || echo 0)

# Define Python/Pip executables based on OS and venv existence
ifeq ($(OS),Windows_NT)
	PYTHON := $(VENV_DIR)/Scripts/python
	PIP := $(VENV_DIR)/Scripts/pip
	ACTIVATE := $(VENV_DIR)/Scripts/activate
else
	PYTHON := $(VENV_DIR)/bin/python
	PIP := $(VENV_DIR)/bin/pip
	ACTIVATE := source $(VENV_DIR)/bin/activate
endif

# Detect Poetry
POETRY := $(shell command -v poetry 2> /dev/null)

# Helper message if venv doesn't exist
define VENV_MISSING_MSG
@echo "Virtual environment '$(VENV_DIR)' not found."
@echo "Please run 'make setup' first."
endef

help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- Setup & Installation ---
setup: $(VENV_DIR)/pyvenv.cfg ## Create the Python virtual environment
	@echo "Virtual environment setup complete in $(VENV_DIR)/"

$(VENV_DIR)/pyvenv.cfg:
	@echo "Creating Python virtual environment using $(PYTHON_EXECUTABLE)..."
	$(PYTHON_EXECUTABLE) -m venv $(VENV_DIR)

install: setup ## Install production dependencies into the venv
	@echo "Installing production dependencies..."
ifdef POETRY
	$(POETRY) install --no-dev --no-interaction --no-ansi --no-root
else
	$(PIP) install -r requirements.txt
endif

dev-install: setup ## Install all dependencies including dev into the venv
	@echo "Installing development dependencies..."
ifdef POETRY
	$(POETRY) install --no-interaction --no-ansi
else
	$(PIP) install -e ".[dev]"
endif
	@echo "Use 'source $(VENV_DIR)/bin/activate' (Linux/macOS) or '.\\$(VENV_DIR)\\Scripts\\activate' (Windows) to activate."


# --- Development ---
# Note: These targets assume the venv is active OR they prepend the venv path.
# Running directly via venv path is more robust for Makefile usage.
run: ## Run the GRADIENT server with Uvicorn inside the venv
ifeq ($(VENV_EXISTS), 1)
	$(PYTHON) main.py
else
	$(VENV_MISSING_MSG)
endif

lint: ## Run Ruff linter inside the venv
ifeq ($(VENV_EXISTS), 1)
	$(PYTHON) -m ruff check src tests *.py
else
	$(VENV_MISSING_MSG)
endif

format: ## Run Ruff formatter inside the venv
ifeq ($(VENV_EXISTS), 1)
	$(PYTHON) -m ruff format src tests *.py
else
	$(VENV_MISSING_MSG)
endif

test: ## Run pytest tests inside the venv
ifeq ($(VENV_EXISTS), 1)
	$(PYTHON) -m pytest
else
	$(VENV_MISSING_MSG)
endif

# --- Cleanup ---
clean-pyc: ## Remove Python bytecode files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-build: ## Remove build artifacts
	rm -rf build/ dist/ .eggs/ *.egg-info/

clean-test: ## Remove test artifacts
	rm -rf .pytest_cache/ htmlcov/ .coverage

clean-venv: ## Remove the virtual environment directory
	rm -rf $(VENV_DIR)

clean: clean-pyc clean-build clean-test ## Run all clean targets (excluding venv)
	@echo "Cleaned Python artifacts, build files, and test cache."

# --- Utility ---
freeze: ## Generate requirements.txt from current environment (useful if not using Poetry)
ifeq ($(VENV_EXISTS), 1)
ifdef POETRY
	$(POETRY) export --without-hashes -f requirements.txt --output requirements.txt
	$(POETRY) export --without-hashes -f requirements.txt --output requirements-dev.txt --with dev
else
	$(PIP) freeze > requirements.txt
endif
	@echo "requirements.txt generated from venv."
else
	$(VENV_MISSING_MSG)
endif