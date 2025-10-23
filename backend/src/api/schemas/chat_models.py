"""
Request/Response models for chat API endpoints.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from ...models.chat import Chat, UIState
from ...models.message import Message, MessageMetadata

# ===== Request Models =====


class ChatRequest(BaseModel):
    """Chat request from user."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message or analysis results",
    )
    session_id: str | None = Field(
        None, description="Session ID for continuing conversation (legacy)"
    )
    chat_id: str | None = Field(
        None, description="Chat ID for persistent conversation (new MongoDB-based)"
    )
    title: str | None = Field(
        None,
        min_length=1,
        max_length=200,
        description="Optional title for new chat (defaults to 'New Chat')",
    )
    role: Literal["user", "assistant", "system"] = Field(
        "user",
        description="Message role: 'user', 'assistant', or 'system'",
    )
    source: Literal["user", "llm", "tool"] = Field(
        "user",
        description="Message source: 'user' (user input, calls LLM), 'tool' (tool output, skip LLM), 'llm' (LLM response). Use metadata.selected_tool to identify specific tool.",
    )
    metadata: MessageMetadata | dict[str, Any] | None = Field(
        None,
        description="Analysis metadata for overlays (Fibonacci levels, Stochastic signals, etc.)",
    )
    # Agent Configuration
    agent_version: Literal["v2", "v3"] = Field(
        "v3",
        description="Agent version: 'v2' (simple ChatAgent), 'v3' (SDK ReAct Agent with tool chaining)",
    )
    # LLM Configuration
    model: str = Field(
        "qwen-plus",
        description="Model ID: qwen-plus, qwen3-max, deepseek-v3, deepseek-v3.2-exp",
    )
    thinking_enabled: bool = Field(
        False,
        description="Enable thinking mode (4x cost for qwen-plus, not supported on qwen3-max/deepseek-v3)",
    )
    max_tokens: int = Field(
        3000,
        ge=500,
        le=32768,
        description="Maximum output tokens (500-32768)",
    )


class UpdateUIStateRequest(BaseModel):
    """Request to update chat UI state."""

    ui_state: UIState


# ===== Response Models =====


class ChatListResponse(BaseModel):
    """Response for listing chats."""

    chats: list[Chat]
    total: int
    page: int
    page_size: int


class ChatDetailResponse(BaseModel):
    """Response for getting chat with messages."""

    chat: Chat
    messages: list[Message]
