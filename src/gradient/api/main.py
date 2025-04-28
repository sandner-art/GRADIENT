# gradient-project-root/src/gradient/api/main.py

from fastapi import FastAPI, APIRouter # Import APIRouter here as well for the check
import logging

# Import routers
# Make sure these imports resolve correctly based on file structure
from .endpoints import status as status_endpoints
from .endpoints import generate as generate_endpoints
from .endpoints import llm as llm_endpoints
from .endpoints import visualize as visualize_endpoints
from . import websocket as websocket_endpoints

# --- Create FastAPI App Instance ---
app = FastAPI(
    title="GRADIENT Workbench API",
    description="API for the Generative Research & Architectural Diffusion Interactive Exploration Navigation Toolkit",
    version="0.1.0",
    # Lifespan events can be added later for startup/shutdown tasks
    # lifespan=lifespan,
)

# --- Temporary Debugging: Print registered routes (Add this section) ---
# This code will run when the module is imported during application startup.
print("\n--- Registered Routes (Before Explicit Includes) ---")
# Note: Routes added via app.get/post directly might show here,
# but included routers often register fully later in the startup cycle.
# This initial print might be empty or incomplete.
for route in app.routes:
    if hasattr(route, "path") and hasattr(route, "methods"):
            print(f"Path: {route.path}, Methods: {route.methods}, Name: {getattr(route, 'name', 'N/A')}")
    elif isinstance(route, APIRouter):
            print(f"Router object found: {route}") # Less useful here
print("--- End Initial Route Check ---\n")
# --- End Temporary Debugging ---


# --- Routers Registration ---
# Registering routers adds their defined routes to the main app

# Status Endpoint
app.include_router(status_endpoints.router, prefix="/api/v1", tags=["Status"])
print("--- Registered status_endpoints router with prefix /api/v1 ---") # Confirmation log

# Generation Endpoints (Text2Image, XYZGrid)
app.include_router(generate_endpoints.router, prefix="/api/v1/generate", tags=["Generation"]) # <-- Corrected prefix!
print("--- Registered generate_endpoints router with prefix /api/v1/generate ---") # Confirmation log

# LLM Endpoints (Chat Completions)
app.include_router(llm_endpoints.router, prefix="/api/v1/llm", tags=["LLM"])
print("--- Registered llm_endpoints router with prefix /api/v1/llm ---") # Confirmation log

# Visualization Endpoints (Latent Space, Sensitivity)
app.include_router(visualize_endpoints.router, prefix="/api/v1/visualize", tags=["Visualization"])
print("--- Registered visualize_endpoints router with prefix /api/v1/visualize ---") # Confirmation log

# WebSocket Endpoint (Usually at root or specific path like /ws)
app.include_router(websocket_endpoints.router) # No prefix needed if endpoint is defined as "/ws"
print("--- Registered websocket_endpoints router ---") # Confirmation log


# --- Root Endpoint (Optional) ---
@app.get("/", tags=["Root"], include_in_schema=False) # Hidden from docs
async def read_root():
    """Simple root endpoint to confirm the API is running."""
    logging.info("Root endpoint '/' accessed.")
    # It's better practice to return a more informative message or redirect
    return {"message": "Welcome to the GRADIENT API. See /docs for details."}

# --- Lifespan Events (Optional - Example for later) ---
# from contextlib import asynccontextmanager
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Code to run on startup
#     print("GRADIENT API starting up...")
#     # Load ML models? Connect to DB?
#     yield
#     # Code to run on shutdown
#     print("GRADIENT API shutting down...")

# --- Final Route Check (Might be more accurate after includes) ---
# This won't run automatically here, but you could call it if needed
def print_final_routes():
    print("\n--- Registered Routes (Final Check) ---")
    for route in app.routes:
        # This might still not show all routes if they are nested/mounted differently
        if hasattr(route, "path") and hasattr(route, "methods"):
             print(f"Path: {route.path}, Methods: {route.methods}, Name: {getattr(route, 'name', 'N/A')}")
        elif hasattr(route, 'routes'): # Check if it's a router/mount with sub-routes
             print(f"Mounted Router/App found at: {getattr(route, 'path', 'N/A')}")
             # You might need to recursively check route.routes here for full detail
    print("--- End Final Route Check ---\n")

# If you want to see final routes on startup, maybe call inside lifespan startup event