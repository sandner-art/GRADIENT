# gradient-project-root/src/gradient/api/main.py

from fastapi import FastAPI, APIRouter # Ensure APIRouter is imported if used in checks
import logging
from contextlib import asynccontextmanager
from typing import List, Union # For route checking types

# Import routers from endpoint modules
# Ensure these relative imports work correctly from this location
from .endpoints import status as status_endpoints
from .endpoints import generate as generate_endpoints
from .endpoints import llm as llm_endpoints
from .endpoints import visualize as visualize_endpoints
from .endpoints import presets as presets_endpoints
from . import websocket as websocket_endpoints

# Import DB initializer
# Ensure this path is correct relative to this file
from gradient.database.session import init_db

# --- Get Logger ---
# Setup logger for this specific module
logger = logging.getLogger(__name__)

# --- Lifespan Event Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup, before requests are served
    print("\nGRADIENT API starting up...") # Use print for initial startup message
    logger.info("Application startup sequence initiated.")

    # Initialize Database
    await init_db()
    logger.info("Database initialization complete.")

    # --- Print final routes AFTER includes and startup --- vvv
    # This is the best place to check the fully configured routes
    print("\n--- Registered Routes (After Startup in Lifespan) ---")
    # Define route types for checking (adjust imports if needed)
    from starlette.routing import Route, WebSocketRoute, Mount

    def format_route(route: Union[Route, WebSocketRoute, Mount], prefix: str = ""):
        """Helper to format route information."""
        full_path = prefix + getattr(route, 'path', '')
        if isinstance(route, (Route, WebSocketRoute)):
            methods = getattr(route, 'methods', None)
            name = getattr(route, 'name', 'N/A')
            methods_str = str(methods) if methods else "[WebSocket]" if isinstance(route, WebSocketRoute) else "[Unknown]"
            print(f"  Path: {full_path}, Methods: {methods_str}, Name: {name}")
        elif isinstance(route, Mount):
            mount_path = getattr(route, 'path', 'N/A')
            print(f"  Mounted App/Router at: {full_path}")
            # Recursively print routes within the mount if possible
            if hasattr(route, 'routes'):
                for sub_route in route.routes:
                    format_route(sub_route, prefix=full_path)
        else:
             print(f"  Unknown route type found at prefix '{prefix}': {type(route)}")

    for route in app.routes:
        format_route(route)

    print("--- End Final Route Check ---\n")
    # --- Print final routes AFTER startup --- ^^^

    yield # Application runs after yield

    # Code to run on shutdown
    print("\nGRADIENT API shutting down...") # Use print for shutdown message
    logger.info("Application shutdown sequence initiated.")
    # Perform any cleanup here if needed (e.g., close DB engine pool?)

# --- Create FastAPI App Instance ---
app = FastAPI(
    title="GRADIENT Workbench API",
    description="API for the Generative Research & Architectural Diffusion Interactive Exploration Navigation Toolkit",
    version="0.1.0",
    lifespan=lifespan # Register the lifespan context manager
)

# --- Routers Registration ---
# Include routers AFTER the app object is created

# Status Endpoint
app.include_router(status_endpoints.router, prefix="/api/v1", tags=["Status"])
logger.debug("Registered status_endpoints router with prefix /api/v1")

# Generation Endpoints (Text2Image, XYZGrid)
app.include_router(generate_endpoints.router, prefix="/api/v1/generate", tags=["Generation"])
logger.debug("Registered generate_endpoints router with prefix /api/v1/generate")

# LLM Endpoints (Chat Completions)
app.include_router(llm_endpoints.router, prefix="/api/v1/llm", tags=["LLM"])
logger.debug("Registered llm_endpoints router with prefix /api/v1/llm")

# Visualization Endpoints (Latent Space, Sensitivity)
app.include_router(visualize_endpoints.router, prefix="/api/v1/visualize", tags=["Visualization"])
logger.debug("Registered visualize_endpoints router with prefix /api/v1/visualize")

# --- Add Presets Router --- vvv
app.include_router(presets_endpoints.router, prefix="/api/v1", tags=["Presets"])
logger.debug("Registered presets_endpoints router with prefix /api/v1")
# --- Add Presets Router --- ^^^

# WebSocket Endpoint (Usually at root or specific path like /ws)
# Ensure websocket_endpoints.router defines paths like "/ws"
app.include_router(websocket_endpoints.router, tags=["WebSocket"])
logger.debug("Registered websocket_endpoints router")


# --- Root Endpoint (Optional) ---
@app.get("/", tags=["Root"], include_in_schema=False) # Keep hidden from docs
async def read_root():
    """Simple root endpoint to confirm the API is running."""
    # Log using the logger instance for this module
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the GRADIENT API. See /docs for details."}

# Note: No 'if __name__ == "__main__":' block here.
# This file defines the 'app' object to be imported by the runner (e.g., main.py).