# src/gradient/api/main.py

from fastapi import FastAPI, APIRouter
import logging # <--- IMPORT LOGGING
from contextlib import asynccontextmanager

# Import routers
from .endpoints import status as status_endpoints
from .endpoints import generate as generate_endpoints
from .endpoints import llm as llm_endpoints
from .endpoints import visualize as visualize_endpoints
from . import websocket as websocket_endpoints

# Import DB initializer
from gradient.database.session import init_db

# --- Get Logger --- vvv
# Get a logger instance specific to this module
logger = logging.getLogger(__name__)
# --- Get Logger --- ^^^

# --- Lifespan Event Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("GRADIENT API starting up...") # Print is okay here for very initial startup
    logger.info("Application startup sequence initiated.") # Use logger now
    await init_db()
    logger.info("Database initialization complete.")
    yield
    # Code to run on shutdown
    print("GRADIENT API shutting down...") # Print is okay here
    logger.info("Application shutdown sequence initiated.") # Use logger now

# --- Create FastAPI App Instance ---
app = FastAPI(
    title="GRADIENT Workbench API",
    description="API for the Generative Research & Architectural Diffusion Interactive Exploration Navigation Toolkit",
    version="0.1.0",
    lifespan=lifespan # Register lifespan manager
)

# --- Routers Registration ---
app.include_router(status_endpoints.router, prefix="/api/v1", tags=["Status"])
print("--- Registered status_endpoints router with prefix /api/v1 ---")

app.include_router(generate_endpoints.router, prefix="/api/v1/generate", tags=["Generation"])
print("--- Registered generate_endpoints router with prefix /api/v1/generate ---")

app.include_router(llm_endpoints.router, prefix="/api/v1/llm", tags=["LLM"])
print("--- Registered llm_endpoints router with prefix /api/v1/llm ---")

app.include_router(visualize_endpoints.router, prefix="/api/v1/visualize", tags=["Visualization"])
print("--- Registered visualize_endpoints router with prefix /api/v1/visualize ---")

app.include_router(websocket_endpoints.router)
print("--- Registered websocket_endpoints router ---")


# --- Root Endpoint (Optional) ---
@app.get("/", tags=["Root"], include_in_schema=False) # Hidden from docs
async def read_root():
    """Simple root endpoint to confirm the API is running."""
    logger.info("Root endpoint '/' accessed.") # Use logger now
    return {"message": "Welcome to the GRADIENT API. See /docs for details."}


# --- Final Route Check function (not called by default) ---
def print_final_routes():
    # This function doesn't use the logger directly but could if needed
    print("\n--- Registered Routes (Final Check) ---")
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
             print(f"Path: {route.path}, Methods: {route.methods}, Name: {getattr(route, 'name', 'N/A')}")
        elif hasattr(route, 'routes'):
             print(f"Mounted Router/App found at: {getattr(route, 'path', 'N/A')}")
    print("--- End Final Route Check ---\n")