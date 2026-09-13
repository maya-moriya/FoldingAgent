"""LLM backends: one module per provider, behind a shared LLMBackend interface."""

from foldingagent.backends.anthropic_backend import AnthropicBackend
from foldingagent.backends.base import (
    ExecutedToolResult,
    LLMBackend,
    LLMResponse,
    TextContentBlock,
    ToolCallRequest,
)
from foldingagent.backends.gemini_backend import GeminiBackend
from foldingagent.backends.openai_backend import OpenAIBackend

__all__ = [
    "AnthropicBackend",
    "ExecutedToolResult",
    "GeminiBackend",
    "LLMBackend",
    "LLMResponse",
    "OpenAIBackend",
    "TextContentBlock",
    "ToolCallRequest",
]
