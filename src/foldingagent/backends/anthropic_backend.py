"""Anthropic Claude backend (tool_use API)."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from foldingagent import config
from foldingagent.tools import AGENT_TOOLS
from foldingagent.backends.base import (
    ExecutedToolResult,
    LLMBackend,
    LLMResponse,
    TextContentBlock,
    ToolCallRequest,
    _load_dotenv_into_env,
    _mime_type_for_path,
    _resolve_api_key,
)
from foldingagent.logger import _save_llm_call

logger = logging.getLogger(__name__)


class AnthropicBackend(LLMBackend):
    """Claude tool-use via the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        max_tokens: int = config.BACKEND_MAX_TOKENS,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "Install the Anthropic SDK:  pip install anthropic"
            ) from exc
        _load_dotenv_into_env()
        _api_key = _resolve_api_key("anthropic")
        self._client = _anthropic.Anthropic(api_key=_api_key)
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, messages: list[Any], system_prompt: str) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "tools": AGENT_TOOLS,
            "messages": messages,
        }
        response = self._client.messages.create(**kwargs)
        _call_id = _save_llm_call(self.log_dir, kwargs, response, self.log_label) if self.log_dir is not None else None

        text_parts = []
        thinking_parts = []
        tool_calls = []
        ordered_content: list[TextContentBlock | ToolCallRequest] = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                is_thought = getattr(block, "type", "") == "thinking"
                if not is_thought:
                    text_parts.append(block.text)
                else:
                    thinking_parts.append(block.text)
                ordered_content.append(
                    TextContentBlock(text=block.text, is_thought=is_thought)
                )
            elif block.type == "tool_use":
                tc = ToolCallRequest(id=block.id, name=block.name, input=block.input)
                tool_calls.append(tc)
                ordered_content.append(tc)
        return LLMResponse(
            raw=response,
            text=" ".join(text_parts) if text_parts else None,
            thinking=" ".join(thinking_parts) if thinking_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            ordered_content=ordered_content,
            llm_call_id=_call_id,
        )

    def init_messages(self, task_description: str) -> list[Any]:
        return [{"role": "user", "content": task_description}]

    def append_turn(
        self,
        messages: list[Any],
        response: LLMResponse,
        results: list[ExecutedToolResult],
    ) -> list[Any]:
        messages = list(messages)
        messages.append({"role": "assistant", "content": response.raw.content})

        tool_result_content: list[dict[str, Any]] = []
        for executed in results:
            content_blocks: list[dict[str, Any]] = [
                {"type": "text", "text": executed.result.to_json()}
            ]
            for img_path in executed.result.images:
                try:
                    b64 = base64.standard_b64encode(img_path.read_bytes()).decode()
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _mime_type_for_path(img_path),
                            "data": b64,
                        },
                    })
                except OSError as exc:
                    logger.warning("Could not encode image %s: %s", img_path, exc)

            tool_result_content.append({
                "type": "tool_result",
                "tool_use_id": executed.call.id,
                "content": content_blocks,
            })

        messages.append({"role": "user", "content": tool_result_content})
        return messages

    def append_user_text(
        self,
        messages: list[Any],
        response: LLMResponse,
        text: str,
    ) -> list[Any]:
        messages = list(messages)
        messages.append({"role": "assistant", "content": response.raw.content})
        messages.append({"role": "user", "content": text})
        return messages
