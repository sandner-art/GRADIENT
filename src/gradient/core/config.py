# gradient-project-root/src/gradient/core/config.py

import yaml
from pathlib import Path
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Load .env file first to potentially override paths
load_dotenv()

# Define base paths relative to this file's location or project root
# Assuming this file is src/gradient/core/config.py
# Project root is 3 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "default_settings.yaml"
USER_SETTINGS_PATH = CONFIG_DIR / "config.yaml" # User override file
LOGGING_CONFIG_PATH = CONFIG_DIR / "logging_config.yaml"

# Placeholder for loaded settings
settings: Dict[str, Any] = {}
logging_config: Dict[str, Any] = {}

def load_yaml_config(file_path: Path) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if file_path.is_file():
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {file_path}: {e}")
        except IOError as e:
            logging.error(f"Error reading file {file_path}: {e}")
    else:
        logging.warning(f"Configuration file not found: {file_path}")
    return {}

def deep_update(source: Dict, overrides: Dict) -> Dict:
    """Recursively update a dictionary."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = deep_update(source[key], value)
        else:
            source[key] = value
    return source

def load_settings():
    """Loads default settings and merges user overrides."""
    global settings
    default_settings = load_yaml_config(DEFAULT_SETTINGS_PATH)
    user_settings = load_yaml_config(USER_SETTINGS_PATH)

    # Start with defaults and update with user settings
    settings = deep_update(default_settings.copy(), user_settings)

    # Allow overrides from environment variables (optional, prefix e.g., GRADIENT_)
    # Example: GRADIENT_APP_HOST would override settings['app']['host']
    app_host_env = os.getenv('GRADIENT_APP_HOST')
    if app_host_env:
        settings.setdefault('app', {})['host'] = app_host_env

    app_port_env = os.getenv('GRADIENT_APP_PORT')
    if app_port_env:
        try:
            settings.setdefault('app', {})['port'] = int(app_port_env)
        except ValueError:
            logging.warning(f"Invalid GRADIENT_APP_PORT environment variable: {app_port_env}. Must be an integer.")

    logging.info("Application settings loaded.")
    # print(f"Debug - Loaded Settings: {settings}") # Uncomment for debugging

def load_logging_config():
    """Loads the logging configuration."""
    global logging_config
    logging_config = load_yaml_config(LOGGING_CONFIG_PATH)
    if not logging_config:
        logging.error("Failed to load logging configuration. Using basic logging.")
        # Setup basic logging as fallback?
    else:
        # Ensure log directory exists - logging config might need it
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_filepath = log_dir / "gradient.log"
        # Dynamically update log file path if specified relative in config
        # This example assumes the filename is fixed in the yaml for simplicity,
        # but you could parse/update the path here if needed.
        logging.info("Logging configuration loaded.")
        # print(f"Debug - Loaded Logging Config: {logging_config}") # Uncomment for debugging


# Load configurations when the module is imported
load_settings()
load_logging_config()

# Convenience function to get settings
def get_setting(key_path: str, default: Any = None) -> Any:
    """
    Get a setting using dot notation, e.g., 'app.host'.
    """
    keys = key_path.split('.')
    value = settings
    try:
        for key in keys:
            if isinstance(value, dict):
                 value = value[key]
            else:
                # Handle case where intermediate key is not a dictionary
                logging.warning(f"Invalid key path '{key_path}' at segment '{key}'.")
                return default
        return value
    except KeyError:
        logging.warning(f"Setting '{key_path}' not found.")
        return default
    except Exception as e:
        logging.error(f"Error accessing setting '{key_path}': {e}")
        return default


if __name__ == "__main__":
    # Example usage/test
    print(f"App Host: {get_setting('app.host', 'default_host')}")
    print(f"App Port: {get_setting('app.port', 'default_port')}")
    print(f"Non-existent: {get_setting('app.nonexistent.key', 'fallback')}")