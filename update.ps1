# Update GRADIENT to the latest version (Windows PowerShell).
#
#   .\update.ps1              latest release (newest v* tag)
#   .\update.ps1 -Main        latest development version (main branch)
#   .\update.ps1 -Check       only report whether an update is available
#
# Install options are passed through: -Cpu, -LegacyGpu, -WithLlm, -Dev, -NoFrontend.
# Your models, database and outputs in data\ and experiments\ are not touched.
# If scripts are blocked: powershell -ExecutionPolicy Bypass -File update.ps1

param(
    [switch]$Main,
    [switch]$Check,
    [switch]$Cpu,
    [switch]$LegacyGpu,
    [switch]$WithLlm,
    [switch]$Dev,
    [switch]$NoFrontend
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Die($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Invoke-GitChecked { & git @args; if ($LASTEXITCODE -ne 0) { Die "git $($args -join " ") failed" } }

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die "git is not installed." }
if (-not (Test-Path ".git")) { Die "This folder is not a git checkout; download the new release instead." }

$current = (& git describe --tags --always 2>$null)
if (-not $current) { $current = (& git rev-parse --short HEAD) }
Say "Current version: $current"
Say "Fetching updates"
Invoke-GitChecked fetch --tags --quiet origin

if ($Main) {
    $target = "origin/main"
    $targetRev = (& git rev-parse origin/main)
} else {
    $target = (& git tag -l "v*" --sort=-v:refname | Select-Object -First 1)
    if (-not $target) { Die "No release tags found. Use .\update.ps1 -Main for the development version." }
    $targetRev = (& git rev-list -n1 $target)
}

if ((& git rev-parse HEAD) -eq $targetRev) { Say "Already up to date ($target)."; exit 0 }
if ($Check) { Say "Update available: $current -> $target"; exit 0 }

if (& git status --porcelain --untracked-files=no) {
    & git status --short --untracked-files=no
    Die "These tracked files have local changes. Commit or stash them first (git stash), then run this again."
}

Say "Updating to $target"
if ($Main) {
    $branch = (& git symbolic-ref --quiet --short HEAD 2>$null)
    if ($branch -ne "main") {
        & git checkout --quiet main 2>$null
        if ($LASTEXITCODE -ne 0) { Invoke-GitChecked checkout --quiet -b main origin/main }
    }
    & git merge --ff-only --quiet origin/main
    if ($LASTEXITCODE -ne 0) { Die "main has diverged from origin/main; update it manually." }
} else {
    Invoke-GitChecked -c advice.detachedHead=false checkout --quiet $target
}

Say "Updating dependencies and rebuilding the web UI"
$installArgs = @{}
foreach ($name in "Cpu", "LegacyGpu", "WithLlm", "Dev", "NoFrontend") {
    if ((Get-Variable $name).Value) { $installArgs[$name] = $true }
}
& "$PSScriptRoot\install.ps1" @installArgs
$now = (& git describe --tags --always)
Say "Updated: $current -> $now"
if (Test-Path CHANGELOG.md) { Say "What's new: see CHANGELOG.md" }
