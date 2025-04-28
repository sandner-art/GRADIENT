# gradient-project-root/src/gradient/core/log_setup.py

import logging.config
from .config import logging_config # Import the loaded config dict

def setup_logging():
    """Applies the loaded logging configuration."""
    if logging_config:
        try:
            logging.config.dictConfig(logging_config)
            logging.info("Logging configured successfully.")
        except Exception as e:
            # Fallback to basic config if dictConfig fails
            logging.basicConfig(level=logging.WARNING)
            logging.exception("Failed to apply logging configuration from file. Using basic config.", exc_info=e)
    else:
        # Fallback if config wasn't loaded at all
        logging.basicConfig(level=logging.INFO)
        logging.warning("Logging configuration is empty. Using basic config.")