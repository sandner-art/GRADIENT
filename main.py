# gradient-project-root/main.py

import uvicorn
import logging
import sys
from pathlib import Path

# Ensure src is in path if running script directly from root
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import AFTER potentially modifying sys.path
try:
    from src.gradient.core.config import get_setting, PROJECT_ROOT # Use PROJECT_ROOT from config
    from src.gradient.core.log_setup import setup_logging
    from src.gradient.api.main import app # Import the FastAPI app instance
except ImportError as e:
    print(f"Error importing GRADIENT modules: {e}", file=sys.stderr)
    print("Ensure you have installed the package ('pip install -e .') or are running from the project root.", file=sys.stderr)
    sys.exit(1)

# --- Setup Logging ---
# Must be called before the app starts using logging
setup_logging()
logger = logging.getLogger(__name__) # Get logger for this script

# --- Main Execution ---
if __name__ == "__main__":
    host = get_setting("app.host", "127.0.0.1")
    port = get_setting("app.port", 8000)
    reload_dev = get_setting("app.reload", False) # Add 'reload: true' to config.yaml for dev

    logger.info(f"Starting GRADIENT server on {host}:{port}")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Reload mode: {'Enabled' if reload_dev else 'Disabled'}")

    try:
        uvicorn.run(
            "gradient.api.main:app", # Path to the FastAPI app instance
            host=host,
            port=port,
            reload=reload_dev, # Enable auto-reload for development
            log_config=None, # Disable uvicorn's default logging config, use ours
            # Use reload_dirs=[str(src_path)] if reload doesn't pick up src changes
        )
    except Exception as e:
        logger.critical(f"Failed to start Uvicorn server: {e}", exc_info=True)
        sys.exit(1)