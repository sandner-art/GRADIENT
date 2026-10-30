# gradient-project-root/src/gradient/api/main.py

import logging
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from gradient.core.config import PROJECT_ROOT, get_setting
from gradient.database.session import init_db

from . import websocket as websocket_endpoints
from .endpoints import generate as generate_endpoints
from .endpoints import lab as lab_endpoints
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
    APP_VERSION = "0.4.0"

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
app.include_router(lab_endpoints.router, prefix="/api/v1", tags=["Lab"])
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
    _NO_UI_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>GRADIENT</title>
<style>body{font-family:system-ui,sans-serif;max-width:46rem;margin:3rem auto;padding:0 1rem;line-height:1.5;color:#1f2937}
code,pre{background:#f3f4f6;border-radius:4px;padding:.1rem .3rem}pre{padding:.8rem}</style></head><body>
<h1>GRADIENT is running</h1>
<p>The API works (<a href="/docs">API docs</a>), but the web interface has not been built yet.
It needs <a href="https://nodejs.org" target="_blank" rel="noreferrer">Node.js</a> 22.17 or newer.</p>
<ol><li>Install Node.js (LTS), e.g. <code>winget install OpenJS.NodeJS.LTS</code> on Windows.</li>
<li>Open a <b>new</b> terminal in the GRADIENT folder and run:
<pre>cd frontend
npm ci
npm run build</pre></li>
<li>Restart GRADIENT and reload this page.</li></ol>
<p>Models can already be downloaded from the terminal: <code>gradient models catalog</code>.</p>
</body></html>"""

    @app.get("/", include_in_schema=False)
    async def read_root():
        return HTMLResponse(_NO_UI_PAGE)
