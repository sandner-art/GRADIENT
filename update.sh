#!/usr/bin/env bash
# Update GRADIENT to the latest version (Linux / macOS).
#
#   ./update.sh              latest release (newest v* tag)
#   ./update.sh --main       latest development version (main branch)
#   ./update.sh --check      only report whether an update is available
#
# Other options are passed to install.sh (e.g. --cpu, --legacy-gpu, --with-llm, --dev).
# Your models, database and outputs in data/ and experiments/ are not touched.

set -euo pipefail
cd "$(dirname "$0")"

CHANNEL=release
CHECK_ONLY=0
INSTALL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --main) CHANNEL=main ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) INSTALL_ARGS+=("$arg") ;;
  esac
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is not installed."
[ -d .git ] || die "This folder is not a git checkout; download the new release instead."

CURRENT="$(git describe --tags --always 2>/dev/null || git rev-parse --short HEAD)"
say "Current version: $CURRENT"
say "Fetching updates"
git fetch --tags --quiet origin

if [ "$CHANNEL" = release ]; then
  TARGET="$(git tag -l 'v*' --sort=-v:refname | head -n1)"
  [ -n "$TARGET" ] || die "No release tags found. Use ./update.sh --main for the development version."
  TARGET_REV="$(git rev-list -n1 "$TARGET")"
else
  TARGET="origin/main"
  TARGET_REV="$(git rev-parse origin/main)"
fi

if [ "$(git rev-parse HEAD)" = "$TARGET_REV" ]; then
  say "Already up to date ($TARGET)."
  exit 0
fi
if [ "$CHECK_ONLY" = 1 ]; then
  say "Update available: $CURRENT -> $TARGET"
  exit 0
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  git status --short --untracked-files=no >&2
  die "These tracked files have local changes. Commit or stash them first (git stash), then run this again."
fi

say "Updating to $TARGET"
if [ "$CHANNEL" = release ]; then
  git -c advice.detachedHead=false checkout --quiet "$TARGET"
else
  BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
  if [ "$BRANCH" = main ]; then
    git merge --ff-only --quiet origin/main || die "main has diverged from origin/main; update it manually."
  else
    git checkout --quiet main 2>/dev/null || git checkout --quiet -b main origin/main
    git merge --ff-only --quiet origin/main || die "main has diverged from origin/main; update it manually."
  fi
fi

say "Updating dependencies and rebuilding the web UI"
./install.sh ${INSTALL_ARGS[@]+"${INSTALL_ARGS[@]}"}
say "Updated: $CURRENT -> $(git describe --tags --always)"
[ -f CHANGELOG.md ] && say "What's new: see CHANGELOG.md"
