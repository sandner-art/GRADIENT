# GRADIENT installer for Windows (PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File install.ps1              auto-detect GPU
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Cpu         CPU-only PyTorch
#   powershell -ExecutionPolicy Bypass -File install.ps1 -LegacyGpu   GTX 9xx/10xx, Titan V (PyTorch 2.14 + CUDA 12.6)
#   ... -WithLlm      also install llama-cpp-python (needs Visual Studio Build Tools to compile)
#   ... -Dev          also install test/lint tools
#   ... -NoFrontend   skip building the web UI
#
# Override the PyTorch wheel index with $env:TORCH_INDEX_URL (see https://pytorch.org/get-started/locally/).

param(
    [switch]$Cpu,
    [switch]$LegacyGpu,
    [switch]$WithLlm,
    [switch]$Dev,
    [switch]$NoFrontend
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }

# --- 1. uv ---------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv (https://docs.astral.sh/uv/)"
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# --- 2. virtual environment ----------------------------------------------------
if (-not (Test-Path ".venv")) {
    Say "Creating .venv with Python 3.12"
    uv venv --python 3.12 .venv
}
$env:VIRTUAL_ENV = (Resolve-Path ".venv").Path

# --- 3. PyTorch ----------------------------------------------------------------
$torchSpec = "torch>=2.15"
$index = $env:TORCH_INDEX_URL
if (-not $index) {
    if ($Cpu) {
        $index = "https://download.pytorch.org/whl/cpu"
    } elseif ($LegacyGpu) {
        $index = "https://download.pytorch.org/whl/cu126"; $torchSpec = "torch>=2.14,<2.15"
    } elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $smi = (nvidia-smi) -join "`n"
        $driverCuda = if ($smi -match "CUDA Version:\s*([0-9]+)\.([0-9]+)") { [int]$Matches[1] } else { 0 }
        $cc = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
        $gpu = (nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1).Trim()
        Say "NVIDIA GPU: $gpu (compute $cc, driver CUDA major $driverCuda)"
        $ccNum = [int]($cc -replace "\.", "")
        if ($ccNum -gt 0 -and $ccNum -lt 75) {
            Warn "This GPU predates Turing; using PyTorch 2.14 + CUDA 12.6."
            $index = "https://download.pytorch.org/whl/cu126"; $torchSpec = "torch>=2.14,<2.15"
        } elseif ($driverCuda -gt 0 -and $driverCuda -lt 13) {
            Warn "Your NVIDIA driver is older than CUDA 13 (R580+). Installing PyTorch 2.14 + CUDA 12.6; update the driver and re-run for 2.15."
            $index = "https://download.pytorch.org/whl/cu126"; $torchSpec = "torch>=2.14,<2.15"
        } else {
            $index = "https://download.pytorch.org/whl/cu132"
        }
    } else {
        Say "No NVIDIA GPU detected: installing CPU PyTorch"
        $index = "https://download.pytorch.org/whl/cpu"
    }
}
Say "Installing $torchSpec from $index"
uv pip install $torchSpec --index-url $index
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }

# --- 4. GRADIENT ---------------------------------------------------------------
$extras = @()
if ($WithLlm) { $extras += "llm" }
if ($Dev) { $extras += "dev" }
$target = if ($extras.Count) { ".[" + ($extras -join ",") + "]" } else { "." }
if ($WithLlm -and -not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue) -and -not $env:CMAKE_ARGS) {
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
}
Say "Installing GRADIENT $target"
uv pip install -e $target
if ($LASTEXITCODE -ne 0) { throw "GRADIENT installation failed" }

# --- 5. Web UI -----------------------------------------------------------------
if (-not $NoFrontend) {
    $node = Get-Command node -ErrorAction SilentlyContinue
    $nodeOk = $false
    if ($node) {
        $v = (node -p "process.versions.node").Split(".")
        $nodeOk = ([int]$v[0] -gt 22) -or ([int]$v[0] -eq 22 -and [int]$v[1] -ge 17)
    }
    if ($nodeOk) {
        Say "Building the web UI"
        Push-Location frontend
        npm ci --no-audit --no-fund
        npm run build
        Pop-Location
    } else {
        Warn "Node.js >= 22.17 not found, so the web UI was not built. Install it from https://nodejs.org and run: cd frontend; npm ci; npm run build"
    }
}

& .venv\Scripts\gradient.exe info
Write-Host ""
Write-Host "GRADIENT is installed."
Write-Host "  Start:          run.bat   (or: .venv\Scripts\gradient.exe serve --open)"
Write-Host "  Then open:      http://127.0.0.1:8000  ->  Models tab to download a model"
Write-Host "  CLI downloads:  .venv\Scripts\gradient.exe models catalog"
Write-Host "  Gated models:   accept the licence on huggingface.co, then set HF_TOKEN or run .venv\Scripts\hf.exe auth login"
