# gradient-project-root/src/gradient/__main__.py

# This allows running the app using 'python -m gradient'
# For Phase 0, it can just replicate main.py's uvicorn call.
# Later, it can be expanded using Typer or Click for CLI commands.

import uvicorn
import logging
import sys
from pathlib import Path

# If running as module, 'src' should already be discoverable, but defensive check
project_root = Path(__file__).parent.parent.parent # Assumes src/gradient/__main__.py
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from gradient.core.config import get_setting, PROJECT_ROOT
    from gradient.core.log_setup import setup_logging
    # Note: app is imported implicitly by uvicorn using the string "gradient.api.main:app"
except ImportError as e:
    print(f"Error importing GRADIENT modules: {e}", file=sys.stderr)
    print("Ensure you have installed the package ('pip install -e .') and dependencies.", file=sys.stderr)
    sys.exit(1)


def run_server():
    """Sets up logging and runs the Uvicorn server."""
    setup_logging() # Setup logging first
    logger = logging.getLogger(__name__) # Get logger for this module

    host = get_setting("app.host", "127.0.0.1")
    port = get_setting("app.port", 8000)
    reload_dev = get_setting("app.reload", False)

    logger.info(f"(Module Execution) Starting GRADIENT server on {host}:{port}")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Reload mode: {'Enabled' if reload_dev else 'Disabled'}")

    try:
        uvicorn.run(
            "gradient.api.main:app",
            host=host,
            port=port,
            reload=reload_dev,
            log_config=None, # Use our logging setup
        )
    except Exception as e:
        logger.critical(f"Failed to start Uvicorn server via module: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Could add CLI arguments here using argparse, Typer, or Click later
    # For now, just run the server.
    run_server()