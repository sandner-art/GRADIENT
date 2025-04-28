# gradient-project-root/src/gradient/ai/llm/interaction.py

import logging
import time
import uuid
import json # Import json for error handling in stream
from typing import List, Optional, Dict, AsyncGenerator, Any

# Use TYPE_CHECKING for conditional import of Llama for static analysis
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llama_cpp import Llama, LlamaGrammar
else:
    try:
        from llama_cpp import Llama, LlamaGrammar # Optional: for grammar-based sampling later
    except ImportError:
        Llama = None # Set to None if import fails
        LlamaGrammar = None
        # Error logged during loading module import now

from .loading import load_llm
from .models import (
    ChatCompletionRequest, ChatMessage, GenerationParams,
    ChatCompletionResponse, ChatCompletionChoice,
    ChatCompletionStreamResponse, ChatCompletionChoiceDelta
)

logger = logging.getLogger(__name__)

async def run_chat_completion(
    request: ChatCompletionRequest
) -> Optional[ChatCompletionResponse]:
    """Handles non-streaming chat completion requests."""
    # Check if Llama class is available (import might have failed)
    if Llama is None:
        logger.error("Cannot run chat completion: llama-cpp-python is not available.")
        return None

    start_time = time.time()
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    # Use renamed field llm_config here
    logger.info(f"Received chat completion request {request_id}: model={request.llm_config.model_id}")

    # 1. Load LLM using the renamed field
    llm = await load_llm(request.llm_config)
    if not llm:
        logger.error(f"Failed to load LLM for request {request_id}")
        return None # Or raise appropriate exception

    # 2. Prepare parameters for llama_cpp create_chat_completion
    gen_params = request.generation_params
    llm_params: Dict[str, Any] = { # Ensure type hint for clarity
        "temperature": gen_params.temperature,
        "top_k": gen_params.top_k,
        "top_p": gen_params.top_p,
        "repeat_penalty": gen_params.repeat_penalty,
        "max_tokens": gen_params.max_tokens,
        "stop": gen_params.stop,
        # llama.cpp uses -1 for random seed by default if seed is None
        "seed": gen_params.seed if gen_params.seed is not None else -1,
        # Add grammar later if needed: "grammar": grammar_object
    }

    # Filter out None values for stop sequences if provided but empty
    if llm_params["stop"] is not None and not llm_params["stop"]:
         llm_params["stop"] = None # Pass None if list is empty

    # Filter out parameters that might not be expected by all llama_cpp versions or are default
    # Example: max_tokens=None means infinity in llama_cpp, check if 0 means error or default
    if llm_params["max_tokens"] is None or llm_params["max_tokens"] <= 0:
         llm_params["max_tokens"] = None # Let llama_cpp use its default potentially

    # 3. Call llama_cpp create_chat_completion
    # Note: The messages format should be compatible with create_chat_completion
    try:
        logger.info(f"Running create_chat_completion for {request_id} with params: {llm_params}")
        # Convert Pydantic models to dicts for the API call
        messages_dict_list = [msg.model_dump(exclude_unset=True) for msg in request.messages]

        completion = llm.create_chat_completion(
            messages=messages_dict_list,
            stream=False, # Force non-streaming for this function
            **llm_params
        )
        logger.info(f"Chat completion successful for {request_id}.")

        # 4. Format the response
        # Ensure completion structure matches expectations
        if not completion or 'choices' not in completion or not completion['choices']:
             logger.error(f"Invalid completion structure received from llama_cpp for {request_id}: {completion}")
             return None

        # Extract choices, ensuring message content is present
        choices = []
        for choice_data in completion.get('choices', []):
             message_data = choice_data.get('message', {})
             # Check if message content exists, provide empty string otherwise
             content = message_data.get('content', None)
             if content is None:
                  logger.warning(f"Choice message content is missing in completion for {request_id}. Using empty string.")
                  content = ""

             choices.append(
                  ChatCompletionChoice(
                       message=ChatMessage(
                           role=message_data.get('role', 'assistant'),
                           content=content
                       ),
                       finish_reason=choice_data.get('finish_reason'),
                       index=choice_data.get('index', 0)
                  )
             )

        usage = completion.get('usage') # Extract token usage info if available

        response = ChatCompletionResponse(
            id=request_id,
            # Use model name from llama_cpp completion if available, else from request
            model=completion.get('model', request.llm_config.model_id),
            choices=choices,
            usage=usage # Can be None if not provided by llama_cpp
        )
        return response

    except Exception as e:
        logger.error(f"Error during chat completion call for {request_id}: {e}", exc_info=True)
        return None

async def run_chat_completion_stream(
    request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """
    Handles streaming chat completion requests.
    Yields JSON strings formatted as Server-Sent Events (SSE).
    """
    # Check if Llama class is available
    if Llama is None:
        logger.error("Cannot run streaming chat completion: llama-cpp-python is not available.")
        error_resp = {"id": "stream_error", "error": "llama-cpp-python is not available."}
        try:
            # Yield error as JSON string - consuming endpoint needs to handle this
            yield json.dumps(error_resp)
        except Exception as e:
             logger.error(f"Failed to yield initial stream error: {e}")
        return # Stop generation

    start_time = time.time()
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    # Use renamed field llm_config here
    logger.info(f"Received streaming chat completion request {request_id}: model={request.llm_config.model_id}")

    # 1. Load LLM using the renamed field
    llm = await load_llm(request.llm_config)
    if not llm:
        logger.error(f"Failed to load LLM for streaming request {request_id}")
        error_resp = {"id": request_id, "error": "Failed to load LLM"}
        try:
             yield json.dumps(error_resp)
        except Exception as e:
             logger.error(f"Failed to yield LLM load stream error: {e}")
        return

    # 2. Prepare parameters
    gen_params = request.generation_params
    llm_params: Dict[str, Any] = {
        "temperature": gen_params.temperature,
        "top_k": gen_params.top_k,
        "top_p": gen_params.top_p,
        "repeat_penalty": gen_params.repeat_penalty,
        "max_tokens": gen_params.max_tokens,
        "stop": gen_params.stop,
        "seed": gen_params.seed if gen_params.seed is not None else -1,
    }
    # Filter out None values for stop sequences if provided but empty
    if llm_params["stop"] is not None and not llm_params["stop"]:
         llm_params["stop"] = None
    # Handle max_tokens = 0 or None
    if llm_params["max_tokens"] is None or llm_params["max_tokens"] <= 0:
         llm_params["max_tokens"] = None # Let llama_cpp use its default

    # 3. Call llama_cpp create_chat_completion with stream=True
    try:
        logger.info(f"Starting create_chat_completion stream for {request_id}...")
        # Convert Pydantic models to dicts
        messages_dict_list = [msg.model_dump(exclude_unset=True) for msg in request.messages]

        streamer = llm.create_chat_completion(
            messages=messages_dict_list,
            stream=True, # Enable streaming
            **llm_params
        )

        # 4. Iterate through the stream and format/yield results
        for chunk in streamer:
            # Format chunk into OpenAI-compatible streaming format
            if not chunk or 'choices' not in chunk or not chunk['choices']:
                logger.debug(f"Skipping empty or invalid chunk: {chunk}")
                continue # Skip empty or invalid chunks

            # Check if delta exists and has content
            delta_data = chunk['choices'][0].get('delta', {})
            delta_content = delta_data.get('content')

            # Only yield if there is actual content in the delta
            if delta_content is not None:
                delta = ChatCompletionChoiceDelta(content=delta_content)
                stream_resp = ChatCompletionStreamResponse(
                    id=request_id,
                    model=chunk.get('model', request.llm_config.model_id),
                    choices=[{"delta": delta}] # Structure matches OpenAI format
                )
                # Yield as JSON string for SSE/WebSocket
                yield stream_resp.model_dump_json()
            # else: # Handle other delta types like role if needed
            #    logger.debug(f"Received non-content delta chunk: {chunk}")


        logger.info(f"Stream finished for {request_id}.")
        # TODO: Optionally yield a final message with usage stats if llama_cpp provides it post-stream
        # Might need to capture usage from the streamer object if possible, or run a separate token count.

    except Exception as e:
        logger.error(f"Error during streaming chat completion for {request_id}: {e}", exc_info=True)
        # Yield specific error JSON if possible
        error_resp = {"id": request_id, "error": f"Stream error: {str(e)}"}
        try:
             yield json.dumps(error_resp)
        except Exception as yield_err: # Handle case where stream might already be closed
             logger.error(f"Failed to yield final stream error: {yield_err}")