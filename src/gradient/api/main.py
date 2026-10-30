# gradient-project-root/src/gradient/api/main.py

import logging
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from gradient.core.config import PROJECT_ROOT, get_setting
from gradient.database.session import init_db

from . import websocket as websocket_endpoints
from .endpoints import generate as generate_endpoints
from .endpoints import llm as llm_endpoints
from .endpoints import models as models_endpoints
from .endpoints import presets as presets_endpoints
from .endpoints import status as status_endpoints
from .endpoints import system as system_endpoints
from .endpoints import visualize as visualize_endpoints

logger = logging.getLogger(__name__)

try:
    APP_VERSION = version("gradient-workbench")
except PackageNotFoundError:  # running from a source checkout without install
    APP_VERSION = "0.2.0"

FRONTEND_BUILD = PROJECT_ROOT / "frontend" / "build"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GRADIENT API %s starting up", APP_VERSION)
    await init_db()
    if (FRONTEND_BUILD / "index.html").is_file():
        logger.info("Serving web UI from %s", FRONTEND_BUILD)
    else:
        logger.info(
            "Web UI build not found (run `npm run build` in frontend/ or use the Vite dev server)"
        )
    yield
    logger.info("GRADIENT API shutting down")
    from gradient.model_manager.downloader import download_manager

    await download_manager.shutdown()


app = FastAPI(
    title="GRADIENT Workbench API",
    description="API for the Generative Research & Architectural Diffusion Interactive Exploration Navigation Toolkit",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.include_router(status_endpoints.router, prefix="/api/v1", tags=["Status"])
app.include_router(generate_endpoints.router, prefix="/api/v1/generate", tags=["Generation"])
app.include_router(llm_endpoints.router, prefix="/api/v1/llm", tags=["LLM"])
app.include_router(visualize_endpoints.router, prefix="/api/v1/visualize", tags=["Visualization"])
app.include_router(presets_endpoints.router, prefix="/api/v1", tags=["Presets"])
app.include_router(system_endpoints.router, prefix="/api/v1", tags=["System Info"])
app.include_router(models_endpoints.router, prefix="/api/v1", tags=["Models"])
app.include_router(websocket_endpoints.router, tags=["WebSocket"])


# --- Web UI (static SvelteKit build) ---
if get_setting("app.serve_frontend", True) and (FRONTEND_BUILD / "index.html").is_file():
    app.mount("/_app", StaticFiles(directory=FRONTEND_BUILD / "_app"), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def serve_frontend(path: str):
        if path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (FRONTEND_BUILD / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(FRONTEND_BUILD.resolve()):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_BUILD / "index.html")  # SPA fallback
else:

    @app.get("/", include_in_schema=False)
    async def read_root():
        return {
            "message": "GRADIENT API is running. API docs: /docs. "
            "Build the web UI with `cd frontend && npm install && npm run build` to serve it here."
        }
