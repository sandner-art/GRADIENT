# gradient-project-root/src/gradient/api/websocket.py

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


# Basic connection manager (can be expanded later)
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connection established: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed: {websocket.client}")

    async def broadcast(self, message: str):
        # Example: Send message to all connected clients
        # In reality, you'll likely send targeted messages
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to {connection.client}: {e}")
                # Optionally remove broken connections here


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections."""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {data}")

            # Basic Echo Logic (Replace with actual logic)
            try:
                # Attempt to parse as JSON, echo back structured data
                message_data = json.loads(data)
                response = {
                    "status": "received",
                    "original_message": message_data,
                    "reply": "Message processed.",
                }
                await websocket.send_text(json.dumps(response))
            except json.JSONDecodeError:
                # If not JSON, just echo back the text
                await websocket.send_text(f"Echo: {data}")

            # Example broadcasting (uncomment to test)
            # await manager.broadcast(f"Client {websocket.client} sent: {data}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected cleanly: {websocket.client}")
    except Exception as e:
        # Log unexpected errors during WebSocket communication
        logger.exception(f"WebSocket error for {websocket.client}: {e}", exc_info=e)
        # Ensure disconnect is called even on errors if possible
        if websocket in manager.active_connections:
            manager.disconnect(websocket)
