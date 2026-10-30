# gradient-project-root/main.py
"""Convenience launcher: `python main.py [serve|models|info ...]` (same as the `gradient` command)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gradient.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
