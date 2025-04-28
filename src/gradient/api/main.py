# gradient-project-root/src/gradient/api/main.py

from fastapi import FastAPI, APIRouter
import logging

# Import routers for different parts of the API
# We create these routers in separate files/modules
from .endpoints import status as status_endpoints
from . import websocket as websocket_endpoints # Assuming websocket.py is in the same dir

# Create the main FastAPI application instance
app = FastAPI(
    title="GRADIENT Workbench API",
    description="API for the Generative Research & Architectural Diffusion Interactive Exploration Navigation Toolkit",
    version="0.1.0",
    # Add other FastAPI options like lifespan context managers later if needed
)

# --- Routers ---
# Include routers from different modules
# Status/Health checks
app.include_router(status_endpoints.router, prefix="/api/v1", tags=["Status"])

# WebSocket communication
app.include_router(websocket_endpoints.router) # Websocket usually at root path /ws

# Add more routers here later for:
# - Image Generation
# - LLM Interaction
# - Experiment Management
# - Model Management
# - etc.

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """Simple root endpoint to confirm the API is running."""
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the GRADIENT API"}

# Note: Logging is configured externally before the app starts (e.g., in main.py)
# You can get logger instances within endpoints: logger = logging.getLogger(__name__)