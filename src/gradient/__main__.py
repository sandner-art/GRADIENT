# gradient-project-root/src/gradient/__main__.py
"""`python -m gradient ...` is equivalent to the `gradient` command."""

import sys

from gradient.cli import main

if __name__ == "__main__":
    sys.exit(main())
