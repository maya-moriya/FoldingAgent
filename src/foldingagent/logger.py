"""Run event log (agent_log.jsonl) and on-disk recording of every LLM call."""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foldingagent.controller import ToolResult
    from foldingagent.backends.base import BackendError, ToolCallRequest


def _serialize_for_log(obj: Any) -> Any:
    """Recursively convert SDK objects to JSON-serializable dicts."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, dict):
        return {k: _serialize_for_log(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_log(item) for item in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _serialize_for_log(obj.model_dump(exclude_none=True))
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return _serialize_for_log(obj.to_dict())
        except Exception:
            pass
    return repr(obj)

def _save_llm_call(log_dir: Path, request: dict, response: Any, label: str | None = None) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    call_id = uuid.uuid4().hex[:8]
    label_part = f"_{label}" if label else ""
    path = log_dir / f"{ts}{label_part}_{call_id}.json"
    path.write_text(json.dumps(
        {"request": _serialize_for_log(request), "response": _serialize_for_log(response)},
        indent=2,
    ))
    return call_id

def _save_critic_call(
    log_dir: Path,
    request: dict[str, Any],
    response: Any,
    label: str,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    call_id = uuid.uuid4().hex[:8]
    path = log_dir / f"{ts}_{label}_{call_id}.json"
    path.write_text(json.dumps(
        {"request": request, "response": _serialize_for_log(response)},
        indent=2,
    ))

def _strip_images(obj: Any) -> Any:
    """Recursively replace image data in messages with a placeholder string."""
    if isinstance(obj, dict):
        # Anthropic base64 image source
        if obj.get("type") == "image" and isinstance(obj.get("source"), dict):
            source = obj["source"]
            if source.get("type") == "base64":
                return {**obj, "source": {**source, "data": "<image_data_stripped>"}}
        return {k: _strip_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_images(item) for item in obj]
    # Gemini Part objects and other non-serialisable types
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Try to extract useful fields before falling back to repr
        part_repr: dict[str, Any] = {}
        if hasattr(obj, "text") and obj.text:
            part_repr["text"] = obj.text
        if hasattr(obj, "function_call") and obj.function_call:
            fc = obj.function_call
            part_repr["function_call"] = {"name": fc.name, "args": dict(fc.args or {})}
        if hasattr(obj, "function_response") and obj.function_response:
            fr = obj.function_response
            part_repr["function_response"] = {"name": fr.name, "response": dict(fr.response or {})}
        if hasattr(obj, "inline_data") and obj.inline_data:
            part_repr["inline_data"] = "<image_data_stripped>"
        if hasattr(obj, "thought"):
            part_repr["thought"] = obj.thought
        return part_repr if part_repr else repr(obj)

def _get_resume_iteration_offset(log_path: Path) -> int:
    """Return one past the highest ``iteration`` recorded in an existing log,
    so a resumed run continues numbering instead of restarting at 0."""
    if not log_path.exists():
        return 0
    last_iteration = -1
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            iteration = event.get("iteration")
            if isinstance(iteration, int) and iteration > last_iteration:
                last_iteration = iteration
    return last_iteration + 1

class AgentLogger:
    """Appends a JSONL event log and prints a human-readable trace."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("a", encoding="utf-8")

    def log_llm_text(
        self,
        text: str | None,
        iteration: int,
        *,
        is_thought: bool = False,
        thought_signature_present: bool = False,
    ) -> None:
        if text:
            label = "THOUGHT" if is_thought else "MODEL"
            print(f"\n{'━' * 60}")
            print(f"[iter {iteration}] {label}:")
            print(text)
            self._write(
                {
                    "event": "llm_text",
                    "iteration": iteration,
                    "text": text,
                    "is_thought": is_thought,
                    "thought_signature_present": thought_signature_present,
                }
            )

    def log_tool_call(
        self,
        call: ToolCallRequest,
        result: ToolResult,
        iteration: int,
    ) -> None:
        symbol = "✓" if result.data.get("success") else "✗"
        print(f"\n  >> CALL  {call.name}")
        print(f"     INPUT  {json.dumps(call.input, ensure_ascii=False)}")
        print(f"  {symbol}  RESULT {json.dumps(result.data, ensure_ascii=False)}")
        if result.images:
            print(f"     IMAGES {len(result.images)} image(s) returned")
        self._write({
            "event": "tool_call",
            "iteration": iteration,
            "tool": call.name,
            "input": call.input,
            "output": result.data,
            "image_count": len(result.images),
        })

    def log_run_start(
        self,
        *,
        frame_count: int,
        backend_name: str,
        model: str | None,
        max_iterations: int,
        out_dir: Path,
        frame_paths: list[str] | None = None,
    ) -> None:
        self._write({
            "event": "run_start",
            "frame_count": frame_count,
            "backend": backend_name,
            "model": model,
            "max_iterations": max_iterations,
            "out_dir": str(out_dir),
            "frame_paths": frame_paths or [],
        })

    def log_usage(
        self,
        iteration: int,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        llm_call_id: str | None = None,
    ) -> None:
        print(
            f"[iter {iteration}] tokens: in={input_tokens} out={output_tokens} total={total_tokens}"
        )
        record: dict[str, Any] = {
            "event": "usage",
            "iteration": iteration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        if llm_call_id is not None:
            record["llm_call_id"] = llm_call_id
        self._write(record)

    def log_thinking_time(
        self,
        iteration: int,
        *,
        elapsed_seconds: float,
        cumulative_since_checkpoint_seconds: float | None = None,
    ) -> None:
        suffix = ""
        if cumulative_since_checkpoint_seconds is not None:
            suffix = (
                f"  step_total={cumulative_since_checkpoint_seconds:.3f}s"
            )
        print(f"[iter {iteration}] thinking_time={elapsed_seconds:.3f}s{suffix}")
        record: dict[str, Any] = {
            "event": "thinking_time",
            "iteration": iteration,
            "elapsed_seconds": elapsed_seconds,
        }
        if cumulative_since_checkpoint_seconds is not None:
            record["cumulative_since_checkpoint_seconds"] = (
                cumulative_since_checkpoint_seconds
            )
        self._write(record)

    def log_finish(
        self,
        reason: str,
        checkpoints: list[int],
        *,
        iterations_completed: int,
        max_iterations: int,
        total_thinking_time_seconds: float,
    ) -> None:
        print(
            "\n[done] "
            f"stop_reason={reason!r}  "
            f"iterations={iterations_completed}/{max_iterations}  "
            f"checkpoints={checkpoints}  "
            f"thinking_time={total_thinking_time_seconds:.3f}s"
        )
        self._write({
            "event": "finish",
            "stop_reason": reason,
            "iterations_completed": iterations_completed,
            "max_iterations": max_iterations,
            "checkpoints": checkpoints,
            "total_thinking_time_seconds": total_thinking_time_seconds,
        })

    def log_messages(self, iteration: int, messages: list[Any]) -> None:
        self._write({
            "event": "messages",
            "iteration": iteration,
            "messages": _strip_images(messages),
        })

    def log_comparator_selection(
        self,
        iteration: int,
        *,
        frame_index: int,
        attempts: list[str],
        candidates: list[dict[str, Any]],
        chosen_node: str,
        chosen_label: str,
        raw_response: str,
        path_to_root: list[str],
        target_frame_path: str = "",
        chosen_actions: list[str] | None = None,
    ) -> None:
        print(
            f"[iter {iteration}] comparator_selection  "
            f"frame={frame_index}  attempts={attempts}  chosen={chosen_node}"
        )
        self._write({
            "event": "comparator_selection",
            "iteration": iteration,
            "frame_index": frame_index,
            "attempts": attempts,
            "candidates": candidates,
            "chosen_node": chosen_node,
            "chosen_label": chosen_label,
            "raw_response": raw_response,
            "path_to_root": path_to_root,
            "target_frame_path": target_frame_path,
            # The chosen attempt's actions, so the log alone says what was kept.
            "chosen_actions": chosen_actions or [],
        })

    def log_no_tool_call_loop_reset(
        self,
        iteration: int,
        *,
        consecutive_no_tool_responses: int,
        rollback_count: int,
        restored_frame: int | None,
    ) -> None:
        """Record that the loop rewound its context after a run of tool-less replies.

        Emits ``no_tool_call_loop_reset``. Runs recorded before this event was
        renamed carry the old ``hallucination_reset`` name; the viewer reads both.
        """
        restored_label = restored_frame if restored_frame is not None else "initial_state"
        print(
            f"[iter {iteration}] no_tool_call_loop_reset  "
            f"consecutive_no_tool={consecutive_no_tool_responses}  "
            f"rollback={rollback_count}  restored_to={restored_label}"
        )
        self._write(
            {
                "event": "no_tool_call_loop_reset",
                "iteration": iteration,
                "consecutive_no_tool_responses": consecutive_no_tool_responses,
                "rollback_count": rollback_count,
                "restored_frame": restored_frame,
            }
        )

    def log_backend_error(self, iteration: int, error: BackendError) -> None:
        """Record the backend failure that ended the run, in full."""
        self._write({
            "event": "backend_error",
            "iteration": iteration,
            **error.to_dict(),
        })

    def close(self) -> None:
        self._file.close()

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
