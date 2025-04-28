# gradient-project-root/src/gradient/api/endpoints/llm.py (NEW FILE)

import logging
import json
from fastapi import APIRouter, HTTPException, Body, Depends
from fastapi.responses import StreamingResponse # For SSE streaming
from typing import Generator, AsyncGenerator

# Import models and interaction functions
from gradient.ai.llm.models import ChatCompletionRequest, ChatCompletionResponse
from gradient.ai.llm.interaction import run_chat_completion, run_chat_completion_stream

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/llm/chat/completions",
    # Define response model based on stream parameter later, or use Union
    # For now, specify non-streaming response, handle stream separately
    response_model=ChatCompletionResponse,
    summary="Create a chat completion using a GGUF model",
    tags=["LLM"]
)
async def handle_chat_completion(
    request: ChatCompletionRequest = Body(...)
):
    """
    Processes a list of chat messages and returns a model-generated completion.
    Supports both streaming and non-streaming responses based on the `stream` flag.
    """
    logger.info(f"Received LLM chat request: model={request.llm_config.model_id}, stream={request.stream}")

    if request.stream:
        # --- Handle Streaming Response ---
        logger.debug("Handling stream request.")
        try:
            # Define an async generator that FastAPI/Starlette can use for SSE
            async def event_stream() -> AsyncGenerator[str, None]:
                async for chunk_json in run_chat_completion_stream(request):
                    # SSE format: data: <json_string>\n\n
                    yield f"data: {chunk_json}\n\n"
                # Signal stream end (optional, depends on client handling)
                # yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        except Exception as e:
            # Log error that happened *before* streaming started (e.g., model load fail)
            logger.exception(f"Error setting up chat stream: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to start stream: {e}")

    else:
        # --- Handle Non-Streaming Response ---
        logger.debug("Handling non-stream request.")
        try:
            result = await run_chat_completion(request)
            if result is None:
                # Error logged internally
                raise HTTPException(status_code=500, detail="Chat completion failed internally.")
            return result
        except MemoryError as e:
             logger.error(f"Out of memory error during chat completion: {e}", exc_info=True)
             raise HTTPException(status_code=507, detail=f"Insufficient memory for completion: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during chat completion API call: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")