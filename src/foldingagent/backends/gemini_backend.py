"""Google Gemini backend (function-calling API)."""

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
    _resolve_api_key,
)
from foldingagent.logger import _save_llm_call
from foldingagent.tools import AGENT_TOOLS

logger = logging.getLogger(__name__)


def _anthropic_to_gemini_schema(props: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert JSON Schema type names to Gemini's UPPER_CASE format."""
    type_map = {
        "object": "OBJECT",
        "array": "ARRAY",
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
    }
    result: dict[str, Any] = {}
    for key, value in props.items():
        if key == "type" and isinstance(value, str):
            result[key] = type_map.get(value, value.upper())
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                k: _anthropic_to_gemini_schema(v) for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            result[key] = _anthropic_to_gemini_schema(value)
        elif key == "enum":
            result[key] = [str(v) for v in value]
            result["type"] = "STRING"
        else:
            result[key] = value
    return result


def _build_gemini_tools() -> Any:
    """Convert AGENT_TOOLS to google.genai FunctionDeclarations."""
    from google.genai import types as gtypes  # type: ignore[import]

    declarations = []
    for tool in AGENT_TOOLS:
        schema_dict = _anthropic_to_gemini_schema(tool["input_schema"])
        declarations.append(
            gtypes.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=schema_dict if schema_dict.get("properties") else None,
            )
        )
    return gtypes.Tool(function_declarations=declarations)


class GeminiBackend(LLMBackend):
    """Gemini function-calling via the google-genai SDK."""

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        thinking_budget: int = config.THINKING_BUDGET,
    ) -> None:
        try:
            from google import genai as _genai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install the Google GenAI SDK:  pip install google-genai"
            ) from exc
        _api_key = _resolve_api_key("gemini")
        self._client = _genai.Client(api_key=_api_key)
        self.model = model
        self.thinking_budget = thinking_budget
        self._tools = _build_gemini_tools()

    def chat(self, messages: list[Any], system_prompt: str) -> LLMResponse:
        from google.genai import types as gtypes  # type: ignore[import]

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "tools": [self._tools],
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
            "thinking_config": gtypes.ThinkingConfig(
                thinking_budget=self.thinking_budget
            ),
        }
        response = self._client.models.generate_content(
            model=self.model,
            contents=messages,
            config=gtypes.GenerateContentConfig(**config_kwargs),
        )
        _call_id = _save_llm_call(
            self.log_dir,
            {"model": self.model, "contents": messages, "config": config_kwargs},
            response,
            self.log_label,
        ) if self.log_dir is not None else None

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        ordered_content: list[TextContentBlock | ToolCallRequest] = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            parts = (getattr(content, "parts", None) or []) if content else []
            for part in parts:
                if hasattr(part, "text") and part.text:
                    is_thought = bool(getattr(part, "thought", False))
                    if not is_thought:
                        text_parts.append(part.text)
                    else:
                        thinking_parts.append(part.text)
                    ordered_content.append(
                        TextContentBlock(
                            text=part.text,
                            is_thought=is_thought,
                            thought_signature_present=(
                                getattr(part, "thought_signature", None) is not None
                            ),
                        )
                    )
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tc = ToolCallRequest(
                        id=fc.id or fc.name,
                        name=fc.name,
                        input=dict(fc.args or {}),
                    )
                    tool_calls.append(tc)
                    ordered_content.append(tc)

        finish_reason = "end_turn"
        if response.candidates:
            raw_reason = str(
                getattr(response.candidates[0], "finish_reason", "")
            ).lower()
            if "tool" in raw_reason or tool_calls:
                finish_reason = "tool_use"

        return LLMResponse(
            raw=response,
            text=" ".join(text_parts) or None,
            thinking=" ".join(thinking_parts) or None,
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            ordered_content=ordered_content,
            llm_call_id=_call_id,
        )

    def init_messages(self, task_description: str) -> list[Any]:
        return [{"role": "user", "parts": [{"text": task_description}]}]

    def append_turn(
        self,
        messages: list[Any],
        response: LLMResponse,
        results: list[ExecutedToolResult],
    ) -> list[Any]:
        from google.genai import types as gtypes  # type: ignore[import]

        messages = list(messages)

        # Append the model's turn by passing the Content object directly so the
        # SDK receives exactly what it produced (avoids repr-stringification of
        # Part objects when they are embedded inside a plain dict).
        for candidate in response.raw.candidates or []:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                messages.append(content)
                break

        # Build a single user turn containing all function responses + images.
        user_parts: list[Any] = []
        for executed in results:
            user_parts.append(
                gtypes.Part.from_function_response(
                    name=executed.call.name,
                    response=json.loads(executed.result.to_json()),
                )
            )
            for img_path in executed.result.images:
                try:
                    b64 = base64.standard_b64encode(img_path.read_bytes()).decode()
                    user_parts.append(
                        gtypes.Part.from_bytes(
                            data=base64.standard_b64decode(b64),
                            mime_type="image/png",
                        )
                    )
                except OSError as exc:
                    logger.warning("Could not encode image %s: %s", img_path, exc)

        if user_parts:
            messages.append({"role": "user", "parts": user_parts})
        return messages

    def append_user_text(
        self,
        messages: list[Any],
        response: LLMResponse,
        text: str,
    ) -> list[Any]:
        messages = list(messages)

        for candidate in response.raw.candidates or []:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                messages.append(content)
                break

        messages.append({"role": "user", "parts": [{"text": text}]})
        return messages
