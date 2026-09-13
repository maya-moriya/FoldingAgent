"""OpenAI GPT backend (chat.completions tool_calls API)."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from foldingagent import config
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
from foldingagent.tools import AGENT_TOOLS

logger = logging.getLogger(__name__)


def _build_openai_tools() -> list[dict[str, Any]]:
    """Convert AGENT_TOOLS (Anthropic format) to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in AGENT_TOOLS
    ]


class OpenAIBackend(LLMBackend):
    """GPT tool-use via the OpenAI SDK (chat.completions)."""

    def __init__(
        self,
        model: str = "gpt-5.5",
        max_tokens: int = config.BACKEND_MAX_TOKENS,
    ) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "Install the OpenAI SDK:  pip install openai"
            ) from exc
        _load_dotenv_into_env()
        _api_key = _resolve_api_key("openai")
        self._client = _openai.OpenAI(api_key=_api_key)
        self.model = model
        self.max_tokens = max_tokens
        self._tools = _build_openai_tools()

    def chat(self, messages: list[Any], system_prompt: str) -> LLMResponse:
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": full_messages,
            "tools": self._tools,
        }
        response = self._client.chat.completions.create(**kwargs)

        _call_id = _save_llm_call(self.log_dir, kwargs, response, self.log_label) if self.log_dir is not None else None

        message = response.choices[0].message
        text = message.content or None
        tool_calls: list[ToolCallRequest] = []
        ordered_content: list[TextContentBlock | ToolCallRequest] = []
        if text:
            ordered_content.append(TextContentBlock(text=text, is_thought=False))
        for tc in message.tool_calls or []:
            call = ToolCallRequest(
                id=tc.id,
                name=tc.function.name,
                input=json.loads(tc.function.arguments or "{}"),
            )
            tool_calls.append(call)
            ordered_content.append(call)

        finish_reason = response.choices[0].finish_reason or "stop"
        stop_reason = "tool_use" if (finish_reason == "tool_calls" or tool_calls) else "end_turn"

        return LLMResponse(
            raw=response,
            text=text,
            thinking=None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            ordered_content=ordered_content,
            llm_call_id=_call_id,
        )

    def init_messages(self, task_description: str) -> list[Any]:
        return [{"role": "user", "content": task_description}]

    def _assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        raw_message = response.raw.choices[0].message
        msg: dict[str, Any] = {"role": "assistant", "content": response.text}
        if raw_message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in raw_message.tool_calls
            ]
        return msg

    def append_turn(
        self,
        messages: list[Any],
        response: LLMResponse,
        results: list[ExecutedToolResult],
    ) -> list[Any]:
        messages = list(messages)
        messages.append(self._assistant_message(response))

        # The "tool" role only supports text content, so tool results are
        # appended as text and any images follow in a separate user turn.
        image_content: list[dict[str, Any]] = []
        for executed in results:
            messages.append({
                "role": "tool",
                "tool_call_id": executed.call.id,
                "content": executed.result.to_json(),
            })
            for img_path in executed.result.images:
                try:
                    b64 = base64.standard_b64encode(img_path.read_bytes()).decode()
                    image_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{_mime_type_for_path(img_path)};base64,{b64}"},
                    })
                except OSError as exc:
                    logger.warning("Could not encode image %s: %s", img_path, exc)

        if image_content:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Images returned by the tool calls above:"},
                    *image_content,
                ],
            })
        return messages

    def append_user_text(
        self,
        messages: list[Any],
        response: LLMResponse,
        text: str,
    ) -> list[Any]:
        messages = list(messages)
        messages.append(self._assistant_message(response))
        messages.append({"role": "user", "content": text})
        return messages
