# gradient-project-root/src/gradient/api/main.py

from fastapi import FastAPI, APIRouter
import logging

# Import routers
from .endpoints import status as status_endpoints
from .endpoints import generate as generate_endpoints # <-- IMPORT NEW ROUTER
from . import websocket as websocket_endpoints

app = FastAPI(
    title="GRADIENT Workbench API",
    description="API for the Generative Research & Architectural Diffusion Interactive Exploration Navigation Toolkit",
    version="0.1.0",
)

# --- Routers ---
app.include_router(status_endpoints.router, prefix="/api/v1", tags=["Status"])
app.include_router(generate_endpoints.router, prefix="/api/v1", tags=["Generation"]) # <-- REGISTER NEW ROUTER
app.include_router(websocket_endpoints.router) # Websocket usually at root path /ws

# --- Root Endpoint ---
@app.get("/", tags=["Root"], include_in_schema=False) # Hide basic root from OpenAPI
async def read_root():
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the GRADIENT API"}