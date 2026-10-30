#!/usr/bin/env bash
# GRADIENT installer for Linux and macOS.
#
#   ./install.sh                 auto-detect GPU, install everything into ./.venv, build the UI
#   ./install.sh --cpu           force a CPU-only PyTorch
#   ./install.sh --legacy-gpu    NVIDIA Maxwell/Pascal/Volta (GTX 9xx/10xx, Titan V): PyTorch 2.14 + CUDA 12.6
#   ./install.sh --with-llm      also install llama-cpp-python (GGUF chat; compiles, needs a C++ toolchain)
#   ./install.sh --dev           also install test/lint tools
#   ./install.sh --no-frontend   skip building the web UI
#
# Override the PyTorch wheel index (e.g. for AMD ROCm) with:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/<variant> ./install.sh
# (pick <variant> at https://pytorch.org/get-started/locally/)

set -euo pipefail
cd "$(dirname "$0")"

MODE=auto
EXTRAS=""
BUILD_FRONTEND=1
for arg in "$@"; do
  case "$arg" in
    --cpu) MODE=cpu ;;
    --legacy-gpu) MODE=legacy ;;
    --with-llm) EXTRAS="${EXTRAS:+$EXTRAS,}llm" ;;
    --dev) EXTRAS="${EXTRAS:+$EXTRAS,}dev" ;;
    --no-frontend) BUILD_FRONTEND=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)"; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*"; }

# --- 1. uv (fast Python package manager) -----------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# --- 2. virtual environment ------------------------------------------------------
if [ ! -d .venv ]; then
  say "Creating .venv with Python 3.12"
  uv venv --python 3.12 .venv
fi
export VIRTUAL_ENV="$PWD/.venv"

# --- 3. PyTorch for this machine -------------------------------------------------
TORCH_SPEC="torch>=2.15"
INDEX="${TORCH_INDEX_URL:-}"
OS="$(uname -s)"
if [ -z "$INDEX" ]; then
  if [ "$MODE" = cpu ]; then
    INDEX="https://download.pytorch.org/whl/cpu"
  elif [ "$OS" = Darwin ]; then
    INDEX=""  # PyPI wheels include Apple Silicon (MPS) support
  elif [ "$MODE" = legacy ]; then
    INDEX="https://download.pytorch.org/whl/cu126"; TORCH_SPEC="torch>=2.14,<2.15"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -n1)"
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
    GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    say "NVIDIA GPU: ${GPU:-unknown} (compute ${CC:-?}, driver supports CUDA ${DRIVER_CUDA:-?})"
    CUDA_MAJOR="${DRIVER_CUDA%%.*}"
    CC_NUM="$(echo "${CC:-0}" | tr -d '.')"
    if [ -n "$CC" ] && [ "${CC_NUM:-0}" -lt 75 ]; then
      warn "This GPU predates Turing; using PyTorch 2.14 + CUDA 12.6 (the last build supporting it)."
      INDEX="https://download.pytorch.org/whl/cu126"; TORCH_SPEC="torch>=2.14,<2.15"
    elif [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -lt 13 ]; then
      warn "Your NVIDIA driver supports CUDA $DRIVER_CUDA; PyTorch 2.15 needs a CUDA 13 driver (R580+)."
      warn "Installing PyTorch 2.14 + CUDA 12.6 instead. Update the driver later and re-run for 2.15."
      INDEX="https://download.pytorch.org/whl/cu126"; TORCH_SPEC="torch>=2.14,<2.15"
    else
      INDEX="https://download.pytorch.org/whl/cu132"
    fi
  elif command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then
    warn "AMD ROCm detected. ROCm wheel names change with every release, so please choose one at"
    warn "https://pytorch.org/get-started/locally/ and re-run:  TORCH_INDEX_URL=<url> ./install.sh"
    warn "Continuing with CPU PyTorch for now."
    INDEX="https://download.pytorch.org/whl/cpu"
  else
    say "No GPU detected: installing CPU PyTorch"
    INDEX="https://download.pytorch.org/whl/cpu"
  fi
fi

say "Installing ${TORCH_SPEC} ${INDEX:+from $INDEX}"
if [ -n "$INDEX" ]; then
  uv pip install "$TORCH_SPEC" --index-url "$INDEX"
else
  uv pip install "$TORCH_SPEC"
fi

# --- 4. GRADIENT and its dependencies (torch stays as installed above) -----------
say "Installing GRADIENT${EXTRAS:+ with extras [$EXTRAS]}"
if [[ ",$EXTRAS," == *",llm,"* ]] && command -v nvidia-smi >/dev/null 2>&1 && [ "$MODE" != cpu ]; then
  export CMAKE_ARGS="${CMAKE_ARGS:--DGGML_CUDA=on}"
  say "llama-cpp-python will be compiled with CUDA (CMAKE_ARGS=$CMAKE_ARGS)"
fi
uv pip install -e ".${EXTRAS:+[$EXTRAS]}"

# --- 5. Web UI ------------------------------------------------------------------
if [ "$BUILD_FRONTEND" = 1 ]; then
  if command -v npm >/dev/null 2>&1 && node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=17)?0:1)'; then
    say "Building the web UI"
    (cd frontend && npm ci --no-audit --no-fund && npm run build)
  else
    warn "Node.js >= 22.17 not found, so the web UI was not built."
    warn "Install Node.js (https://nodejs.org) and run: cd frontend && npm ci && npm run build"
    warn "The API (and its docs at /docs) works without it."
  fi
fi

# --- 6. Summary -------------------------------------------------------------------
.venv/bin/gradient info || true
cat <<'EOF'

GRADIENT is installed.
  Start:            ./run.sh            (or: .venv/bin/gradient serve --open)
  Then open:        http://127.0.0.1:8000  ->  "Models" tab to download a model
  CLI downloads:    .venv/bin/gradient models catalog
                    .venv/bin/gradient models download sdxl-turbo
  Gated models (FLUX.1, SD 3.5, ...): accept the licence on huggingface.co, then
                    export HF_TOKEN=hf_...   (or: .venv/bin/hf auth login)
EOF
