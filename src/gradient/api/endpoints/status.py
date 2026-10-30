# gradient-project-root/src/gradient/api/endpoints/status.py

import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel


# Use Pydantic models for response structure and validation
class StatusResponse(BaseModel):
    status: str
    timestamp: float
    version: str  # Placeholder for app version


router = APIRouter()


def _app_version() -> str:
    from gradient.api.main import APP_VERSION

    return APP_VERSION


logger = logging.getLogger(__name__)  # Get logger specific to this module


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Provides the current operational status of the API."""
    logger.info("Status endpoint called.")
    return StatusResponse(
        status="OK",
        timestamp=time.time(),
        version=_app_version(),
    )


# Add other status-related endpoints here if needed (e.g., dependency checks)
