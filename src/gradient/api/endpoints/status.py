# gradient-project-root/src/gradient/api/endpoints/status.py

from fastapi import APIRouter
import logging
import time
from pydantic import BaseModel

# Use Pydantic models for response structure and validation
class StatusResponse(BaseModel):
    status: str
    timestamp: float
    version: str # Placeholder for app version

router = APIRouter()
logger = logging.getLogger(__name__) # Get logger specific to this module

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Provides the current operational status of the API."""
    logger.info("Status endpoint called.")
    return StatusResponse(
        status="OK",
        timestamp=time.time(),
        version="0.1.0" # Replace with dynamic version later
    )

# Add other status-related endpoints here if needed (e.g., dependency checks)