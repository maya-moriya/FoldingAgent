"""The LLMBackend interface, its shared response types, and provider/key resolution."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foldingagent.config import PROVIDER_API_KEY_ENV_VARS

if TYPE_CHECKING:
    from foldingagent.controller import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRequest:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalised response from either backend."""

    raw: Any
    text: str | None
    tool_calls: list[ToolCallRequest]
    stop_reason: str
    # Ordered interleaving of text strings and tool calls, preserving the
    # original sequence so thinking between tool calls can be displayed.
    ordered_content: list["TextContentBlock | ToolCallRequest"] = field(default_factory=list)
    llm_call_id: str | None = None
    thinking: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class TextContentBlock:
    """A text block emitted by the model, optionally marked as reasoning."""

    text: str
    is_thought: bool = False
    thought_signature_present: bool = False


@dataclass
class ExecutedToolResult:
    """A tool call paired with its result."""

    call: ToolCallRequest
    result: ToolResult

class LLMBackend(ABC):
    """Minimal interface that both Anthropic and Gemini backends must satisfy."""

    log_dir: Path | None = None
    log_label: str | None = None

    @abstractmethod
    def chat(self, messages: list[Any], system_prompt: str) -> LLMResponse:
        """Send messages and return a normalised response."""

    @abstractmethod
    def init_messages(self, task_description: str) -> list[Any]:
        """Build the initial message list for a new conversation."""

    @abstractmethod
    def append_turn(
        self,
        messages: list[Any],
        response: LLMResponse,
        results: list[ExecutedToolResult],
    ) -> list[Any]:
        """Append the assistant response and tool results to the message list."""

    @abstractmethod
    def append_user_text(
        self,
        messages: list[Any],
        response: LLMResponse,
        text: str,
    ) -> list[Any]:
        """Append the model response and a plain-text user follow-up."""

@dataclass
class BackendError:
    """A backend failure, identified well enough to log it and show it to the user.

    Every provider SDK raises its own exception hierarchy, so nothing here is
    matched by type: the fields are read off whatever object arrives, and an
    attribute that is absent simply stays ``None``.
    """

    backend: str
    model: str | None
    type: str
    status_code: int | None
    message: str

    def summary(self) -> str:
        """One line naming what failed and where, e.g. for a log or a header."""
        status = f" [{self.status_code}]" if self.status_code is not None else ""
        return f"{self.backend} raised {self.type}{status}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "error_type": self.type,
            "status_code": self.status_code,
            "error_message": self.message,
        }


#: Attributes the provider SDKs use to carry an HTTP status: ``status_code``
#: (anthropic, openai, httpx responses) and ``code`` (google-genai APIError).
_STATUS_CODE_ATTRS = ("status_code", "code")


def _status_code_of(exc: BaseException) -> int | None:
    """Read an HTTP status off *exc*, or off the response it carries, if any."""
    for source in (exc, getattr(exc, "response", None)):
        if source is None:
            continue
        for attr in _STATUS_CODE_ATTRS:
            value = getattr(source, attr, None)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def describe_backend_error(exc: BaseException, *, backend: LLMBackend) -> BackendError:
    """Identify any exception a backend raised, without knowing its provider.

    The qualified type name (``google.genai.errors.ServerError``) already says
    which SDK failed, so no provider guessing is needed.
    """
    module = type(exc).__module__
    qualified = type(exc).__qualname__
    if module and module != "builtins":
        qualified = f"{module}.{qualified}"
    message = str(exc).strip() or repr(exc)
    return BackendError(
        backend=type(backend).__name__,
        model=getattr(backend, "model", None),
        type=qualified,
        status_code=_status_code_of(exc),
        message=message,
    )


def _mime_type_for_path(path: Path) -> str:
    """Infer the image MIME type from a file's extension."""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"

def _load_dotenv_into_env() -> None:
    """Populate ``os.environ`` from the nearest ``.env`` file for any unset keys."""
    import os as _os
    import re as _re

    _search = Path(__file__).resolve()
    for _parent in [_search.parent, *_search.parents]:
        _dotenv = _parent / ".env"
        if _dotenv.is_file():
            for _line in _dotenv.read_text().splitlines():
                _m = _re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"?([^"#\n]*)"?', _line)
                if _m and _m.group(1) not in _os.environ:
                    _os.environ[_m.group(1)] = _m.group(2).strip()
            break

_PROVIDER_API_KEY_ENV_VARS = PROVIDER_API_KEY_ENV_VARS

def _resolve_api_key(provider: str) -> str:
    """Return the API key for *provider*, read from its standard env var.

    ``anthropic`` -> ``ANTHROPIC_API_KEY``, ``openai`` -> ``OPENAI_API_KEY``,
    ``gemini`` -> ``GOOGLE_API_KEY``. Values may come from the environment or
    from the nearest ``.env`` file.
    """
    _load_dotenv_into_env()
    import os as _os

    env_var_name = _PROVIDER_API_KEY_ENV_VARS[provider]
    api_key = _os.environ.get(env_var_name)
    if not api_key:
        raise ValueError(
            f"API key not found: set the {env_var_name} environment variable "
            f"(or add it to a .env file) to use {provider} models."
        )
    return api_key


def _provider_for_model(model: str) -> str:
    """Identify the API provider for a critic model id."""
    normalized = model.strip().lower()
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "gemini"
