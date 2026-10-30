# gradient-project-root/src/gradient/ai/llm/models.py


from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for loading an LLM."""

    model_id: str = Field(..., description="Identifier for the GGUF model (filename)")
    n_ctx: int = Field(default=2048, description="Context window size")
    n_gpu_layers: int = Field(
        default=0, description="Number of layers to offload to GPU (-1 for all)"
    )
    # Add other llama.cpp params later if needed


class GenerationParams(BaseModel):
    """Common generation parameters."""

    temperature: float = Field(default=0.7, ge=0.0)
    top_k: int = Field(default=40, ge=0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repeat_penalty: float = Field(default=1.1, ge=0.0)
    max_tokens: int = Field(default=512, gt=0)
    stop: list[str] | None = None
    seed: int | None = None


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""

    role: str = Field(..., description="'system', 'user', or 'assistant'")
    content: str


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    # --- RENAME THIS FIELD --- vvv
    llm_config: LLMConfig = Field(
        ..., description="Configuration specific to the LLM model to load/use"
    )
    # --- RENAME THIS FIELD --- ^^^
    messages: list[ChatMessage] = Field(..., min_length=1)
    generation_params: GenerationParams = Field(default_factory=GenerationParams)
    stream: bool = Field(default=False, description="Whether to stream the response token by token")


# ... (rest of the models remain the same: ChatCompletionChoiceDelta, etc.) ...


class ChatCompletionChoiceDelta(BaseModel):
    content: str | None = None
    role: str | None = "assistant"


class ChatCompletionStreamResponse(BaseModel):
    id: str
    model: str
    choices: list[dict[str, ChatCompletionChoiceDelta]]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    finish_reason: str | None = None
    index: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict[str, int] | None = None
