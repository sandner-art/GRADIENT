#!/usr/bin/env bash
# Start GRADIENT (API + web UI) and open the browser. Extra args go to `gradient serve`.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/gradient ]; then
  echo "GRADIENT is not installed yet. Run ./install.sh first." >&2
  exit 1
fi
exec .venv/bin/gradient serve --open "$@"
