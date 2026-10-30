# scripts/download_models.py
"""Download models from the command line (wrapper around `gradient models download`).

Examples:
    python scripts/download_models.py --list
    python scripts/download_models.py sdxl-base flux1-schnell
    python scripts/download_models.py stabilityai/sdxl-turbo
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gradient.cli import main  # noqa: E402


def run(argv: list[str]) -> int:
    if not argv or argv[0] in ("-l", "--list"):
        return main(["models", "catalog"])
    status = 0
    for ref in argv:
        if ref.startswith("-"):
            continue
        status |= main(["models", "download", ref])
    return status


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
