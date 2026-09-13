"""
Interactive agent loop for origami sequence reconstruction.

The provider is inferred from the model id, and its API key is read from the
matching environment variable (or a ``.env`` file):

  - Anthropic Claude  (tool_use API)          ANTHROPIC_API_KEY
  - Google Gemini     (function_calling API)  GOOGLE_API_KEY
  - OpenAI GPT        (tool_calls API)        OPENAI_API_KEY

A run is described entirely by its ``keyframes.json`` manifest — the frame
directory, the keyframe indices and the paper colours. The model, the run
limits and the output location come from ``foldingagent.config``.

Usage (Python):

    from foldingagent.main import run_agent

    checkpoints = run_agent("data/heart/keyframes.json")

Usage (CLI):

    python -m foldingagent.main data/heart/keyframes.json

"""

from __future__ import annotations

import datetime
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foldingagent import config, prompt_assembler, prompts
from foldingagent.controller import FRAME_NAME_RE, OrigamiController, ToolResult
# Re-exported: `from foldingagent.main import LLMBackend, LLMResponse, ...` is the
# documented way to build a custom backend (the replay harness relies on it).
from foldingagent.backends import (
    AnthropicBackend,
    ExecutedToolResult,
    GeminiBackend,
    LLMBackend,
    LLMResponse,
    OpenAIBackend,
    TextContentBlock,
    ToolCallRequest,
)
from foldingagent.backends.base import (
    BackendError,
    _provider_for_model,
    describe_backend_error,
)
from foldingagent.comparator import OrigamiComparator, _comparator_select_best
from foldingagent.rendering import PaperColors
from foldingagent.critic import OrigamiCritic
from foldingagent.logger import AgentLogger, _get_resume_iteration_offset
from foldingagent.memory import (
    AttemptTree,
    HistoryLog,
    _get_pending_operations,
    _load_checkpoints,
    _save_checkpoints,
)
from foldingagent.overview_critic import OrigamiOverviewCritic
from foldingagent.prompt_assembler import (
    build_critic_prompt,
    build_overview_critic_prompt,
    build_system_prompt,
)

logger = logging.getLogger(__name__)

#: Longest error message printed to the terminal. The full text always reaches
#: ``agent_log.jsonl``; this only stops a provider's multi-kilobyte HTML error
#: page from burying the rest of the run summary.
_ERROR_MESSAGE_DISPLAY_CHARS = 1000


def _format_backend_error(error: BackendError, *, out_dir: Path) -> str:
    """Render a backend failure as a readable block for the terminal."""
    message = error.message
    if len(message) > _ERROR_MESSAGE_DISPLAY_CHARS:
        message = (
            message[:_ERROR_MESSAGE_DISPLAY_CHARS]
            + " … (truncated — the full message is in agent_log.jsonl)"
        )
    rule = "─" * 60
    lines = [rule, f"Run stopped: {error.summary()}"]
    if error.model:
        lines.append(f"Model      : {error.model}")
    lines.append("")
    lines.extend(f"    {line}" for line in message.splitlines())
    lines += [
        "",
        "Solved frames are saved. Continue this run in place with:",
        f'    run_agent(resume_run_dir="{out_dir}")',
        rule,
    ]
    return "\n" + "\n".join(lines)


# ── Run setup ─────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    """Reduce a model id or sequence name to something safe in a path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "unknown"


def _repo_relative(path: str | Path) -> str:
    """Express *path* relative to the working directory when it sits beneath it."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _resolve_out_dir(
    resume_run_dir: str | Path | None, *, model: str | None, sequence: str
) -> tuple[Path, bool]:
    """Return this run's directory and whether it continues an existing one.

    A fresh run is ``run_<timestamp>_<model>_<sequence>``, so a directory listing
    says which model folded which sequence without opening anything.
    """
    if resume_run_dir is not None:
        # In-place continuation: no new directory.
        out_dir = Path(resume_run_dir)
        if not out_dir.is_dir():
            raise ValueError(f"resume_run_dir does not exist: {out_dir}")
        return out_dir, True

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{stamp}_{_slug(model or config.DEFAULT_MODEL)}_{_slug(sequence)}"
    return Path(config.OUT_DIR) / name, False


def _build_backend(backend: LLMBackend | None) -> tuple[LLMBackend, str | None]:
    """Return the backend to drive the run, and the model it speaks to.

    Built from ``config.DEFAULT_MODEL`` unless one was passed in; the provider —
    and therefore which API key env var is read — follows from the model name.
    """
    if backend is not None:
        return backend, getattr(backend, "model", None)

    model = config.DEFAULT_MODEL
    provider = _provider_for_model(model)
    if provider == "anthropic":
        return AnthropicBackend(model=model), model
    if provider == "openai":
        return OpenAIBackend(model=model), model
    return GeminiBackend(model=model), model


def _build_reviewers(
    model: str, colors: PaperColors, llm_call_dir: Path
) -> tuple[OrigamiCritic | None, OrigamiComparator | None, OrigamiOverviewCritic | None]:
    """Build the three reviewers, each on the same model and key as the agent.

    Any of them may fail to build — a missing key, an unsupported model — and the
    run goes ahead without it rather than not at all.
    """
    critic: OrigamiCritic | None = None
    try:
        critic = OrigamiCritic(
            model=model,
            system_prompt=build_critic_prompt(
                front_color=colors.front_color, back_color=colors.back_color
            ),
        )
        critic.log_dir = llm_call_dir
        print(f"Critic  : {critic.model}")
    except Exception as exc:
        logger.warning("Could not initialise critic: %s — running without critic.", exc)

    comparator: OrigamiComparator | None = None
    if config.MAX_ATTEMPTS_PER_FRAME > 0:
        try:
            comparator = OrigamiComparator(model=model)
            comparator.log_dir = llm_call_dir
            print(f"Comparator: {comparator.model} "
                  f"(max_attempts_per_frame={config.MAX_ATTEMPTS_PER_FRAME})")
        except Exception as exc:
            logger.warning("Could not initialise comparator: %s", exc)

    overview_critic: OrigamiOverviewCritic | None = None
    try:
        overview_critic = OrigamiOverviewCritic(
            model=model,
            system_prompt=build_overview_critic_prompt(
                front_color=colors.front_color, back_color=colors.back_color
            ),
        )
        overview_critic.log_dir = llm_call_dir
        print(f"Overview : {overview_critic.model}")
    except Exception as exc:
        logger.warning("Could not initialise overview critic: %s — running without it.", exc)

    return critic, comparator, overview_critic


def _restore_previous_checkpoints(controller: OrigamiController, out_dir: Path) -> None:
    """Load a resumed run's checkpoints and put the simulator back at the last one."""
    saved = _load_checkpoints(out_dir)
    if not saved:
        return
    controller.checkpoints = saved
    last_frame = max(saved)
    controller.current_geometry = saved[last_frame].model_copy(deep=True)
    print(
        f"[resume] Loaded {len(saved)} checkpoint(s) from {out_dir}. "
        f"Continuing after frame {last_frame}."
    )


def _build_memory(
    out_dir: Path, controller: OrigamiController, is_resume: bool
) -> tuple[HistoryLog, AttemptTree]:
    """Open the run's history log and attempt tree, reloading them on a resume."""
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    history_log = HistoryLog(log_dir / "history_log.json")
    if is_resume and not history_log.load_from_file():
        history_log.load_from_checkpoints(controller)
    print(f"History : {history_log.path}")

    attempt_tree = AttemptTree(log_dir / "attempt_tree.json")
    if not (is_resume and attempt_tree.load_from_file()):
        attempt_tree.init_root_from_checkpoints(controller)
    print(f"Attempt tree: {attempt_tree.path}")

    return history_log, attempt_tree


def _opening_message(controller: OrigamiController, frame_count: int) -> str:
    """The first user turn: a fresh task, or a summary of what is already solved."""
    if not controller.checkpoints:
        return prompt_assembler.build_task_message(frame_count)
    last_solved = max(controller.checkpoints)
    return prompt_assembler.build_resume_task_message(
        num_frames=frame_count,
        solved=sorted(controller.checkpoints),
        last_solved=last_solved,
        remaining=list(range(last_solved + 1, frame_count + 1)),
    )


@dataclass
class _Run:
    """Everything a run needs, assembled once before the loop starts."""

    frames: list[str | Path]
    out_dir: Path
    backend: LLMBackend
    controller: OrigamiController
    critic: OrigamiCritic | None
    comparator: OrigamiComparator | None
    agent_logger: AgentLogger
    history_log: HistoryLog
    attempt_tree: AttemptTree
    system_prompt: str
    iteration_offset: int
    keyframes: Path
    sequence: "KeyframeSequence"
    attempt: int


def _prepare_run(
    keyframes: str | Path,
    resume_run_dir: str | Path | None,
    backend: LLMBackend | None,
) -> _Run:
    """Resolve the sequence, the output directory, the backend and the reviewers."""
    sequence = _load_keyframe_sequence(keyframes)
    frames: list[str | Path] = list(sequence.frames)
    colors = PaperColors(
        front_color=sequence.front_color, back_color=sequence.back_color
    )

    backend_obj, model = _build_backend(backend)
    out_dir, is_resume = _resolve_out_dir(
        resume_run_dir, model=model, sequence=sequence.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Record the resolved frame list alongside the run's other artefacts.
    frames_json = out_dir / "frames.json"
    if not frames_json.exists():
        frames_json.write_text(
            json.dumps([str(f) for f in frames], indent=2), encoding="utf-8"
        )

    llm_call_dir = out_dir / "llm_calls"
    backend_obj.log_dir = llm_call_dir

    critic, comparator, overview_critic = _build_reviewers(
        model or config.DEFAULT_MODEL, colors, llm_call_dir
    )

    controller = OrigamiController(
        frames=frames,
        out_dir=out_dir,
        paper_colors=colors,
        critic=critic,
        overview_critic=overview_critic,
    )
    if is_resume:
        _restore_previous_checkpoints(controller, out_dir)

    log_path = out_dir / "logs" / "agent_log.jsonl"
    # A resumed run appends to the log already in the directory.
    iteration_offset = _get_resume_iteration_offset(log_path) if is_resume else 0
    history_log, attempt_tree = _build_memory(out_dir, controller, is_resume)

    return _Run(
        frames=frames,
        out_dir=out_dir,
        backend=backend_obj,
        controller=controller,
        critic=critic,
        comparator=comparator,
        agent_logger=AgentLogger(log_path),
        history_log=history_log,
        attempt_tree=attempt_tree,
        system_prompt=build_system_prompt(
            num_frames=len(frames),
            front_color=colors.front_color,
            back_color=colors.back_color,
        ),
        iteration_offset=iteration_offset,
        keyframes=Path(keyframes),
        sequence=sequence,
        attempt=_previous_attempts(out_dir) + 1,
    )


# ── Executing a turn's tool calls ─────────────────────────────────────────────

class _TurnExecutor:
    """Runs the tool calls in one model turn, keeping the run's records in step.

    Each tool the model can call has a consequence beyond its own return value —
    a checkpoint to persist, an attempt to record, a rollback that may trigger
    the comparator. Those live here, one method per tool, rather than inline in
    the loop.
    """

    def __init__(self, run: _Run) -> None:
        self.run = run
        self.checkpoint_saved_frame: int | None = None

    def execute(self, response: LLMResponse, iteration: int) -> list[ExecutedToolResult]:
        """Dispatch every tool call in *response*, in the order the model made them."""
        self.checkpoint_saved_frame = None
        executed: list[ExecutedToolResult] = []
        for item in response.ordered_content:
            if not isinstance(item, ToolCallRequest):
                continue
            if item.name == "restore_checkpoint":
                # The node being left must be closed before the state moves.
                closed_id = self.run.attempt_tree._close_node(self.run.controller)
                if closed_id is not None:
                    self.run.attempt_tree._ensure_node_render(closed_id, self.run.controller)

            result = self.run.controller.dispatch(item)
            self._record(item, result, iteration)

            self.run.agent_logger.log_tool_call(item, result, iteration)
            executed.append(ExecutedToolResult(call=item, result=result))
        return executed

    def _record(self, call: ToolCallRequest, result: ToolResult, iteration: int) -> None:
        """Apply the bookkeeping this particular tool call implies."""
        if call.name == "save_checkpoint":
            self._after_save_checkpoint(call, result)
        elif call.name == "ask_critic":
            self._after_ask_critic(call, result)
        elif call.name == "restore_checkpoint":
            self._after_restore_checkpoint(call, result, iteration)
        elif call.name == "render_current":
            if result.data.get("success"):
                self.run.attempt_tree.on_render(result.data.get("render_path", ""), None)
        elif call.name in AttemptTree._ACTION_TOOLS:
            if result.data.get("success", True):
                self.run.attempt_tree.on_action(call.name, call.input, result.data, None)

    def _after_save_checkpoint(self, call: ToolCallRequest, result: ToolResult) -> None:
        if result.data.get("success"):
            frame_n = int(call.input["frame_n"])
            self.checkpoint_saved_frame = frame_n
            self.run.history_log.update(
                frame_n, self.run.controller, notes=call.input.get("notes")
            )
            self.run.attempt_tree.on_save_checkpoint(frame_n, self.run.controller)
        # Persisted even when the save failed, so the file matches the live state.
        try:
            _save_checkpoints(self.run.controller, self.run.out_dir)
        except Exception as exc:
            logger.warning("Could not save checkpoint: %s", exc)

    def _after_ask_critic(self, call: ToolCallRequest, result: ToolResult) -> None:
        if not result.data.get("success"):
            return
        frame_n = int(call.input["frame_n"])
        self.run.history_log.log_attempt(
            frame_n,
            attempt=result.data.get("transition_attempt", 0),
            operations=_get_pending_operations(self.run.controller, frame_n),
            verdict=result.data.get("critic_verdict"),
            reasoning=result.data.get("critic_analysis"),
            discrepancies=result.data.get("critic_discrepancies"),
        )
        self.run.attempt_tree.on_ask_critic(frame_n, result.data, self.run.controller)

    def _after_restore_checkpoint(
        self, call: ToolCallRequest, result: ToolResult, iteration: int
    ) -> None:
        if not result.data.get("success"):
            return
        controller, attempt_tree = self.run.controller, self.run.attempt_tree
        rolled_back_to = int(call.input["frame_n"])
        if controller.checkpoints and rolled_back_to < max(controller.checkpoints):
            self.run.history_log.log_rollback(max(controller.checkpoints), rolled_back_to)
        attempt_tree.on_restore_checkpoint(rolled_back_to, controller)
        self._maybe_select_best(rolled_back_to, iteration)

    def _maybe_select_best(self, rolled_back_to: int, iteration: int) -> None:
        """Hand the frame to the comparator once its attempts are used up."""
        comparator = self.run.comparator
        if comparator is None or config.MAX_ATTEMPTS_PER_FRAME <= 0:
            return
        controller, attempt_tree = self.run.controller, self.run.attempt_tree

        next_frame = rolled_back_to + 1
        attempts = attempt_tree.get_attempts_for_frame(next_frame)
        restore_count = controller._restore_counts.get(rolled_back_to, 0)
        exhausted = len(attempts) >= config.MAX_ATTEMPTS_PER_FRAME or (
            restore_count >= config.MAX_ATTEMPTS_PER_FRAME and len(attempts) > 0
        )
        if not exhausted:
            return

        selection = _comparator_select_best(
            comparator=comparator,
            attempt_tree=attempt_tree,
            controller=controller,
            frame_index=next_frame,
            frames=self.run.frames,
            out_dir=self.run.out_dir,
        )
        if selection is None:
            return

        # The chosen attempt is final: lock the source frame so the model cannot
        # roll back past it and start the same search again.
        controller._locked_checkpoints.add(rolled_back_to)
        controller._min_restore_frame = max(controller._min_restore_frame, next_frame)
        self.checkpoint_saved_frame = next_frame

        # A selection saves next_frame just as save_checkpoint does, so it is
        # recorded the same way; otherwise the history log skips the frame, and a
        # resumed run loses it.
        self.run.history_log.update(
            next_frame,
            controller,
            notes=f"selected by comparator: {selection.chosen_node} ({selection.chosen_label})",
        )
        chosen = attempt_tree.nodes.get(selection.chosen_node, {})
        self.run.agent_logger.log_comparator_selection(
            iteration,
            frame_index=next_frame,
            attempts=selection.attempts,
            candidates=selection.candidates,
            chosen_node=selection.chosen_node,
            chosen_label=selection.chosen_label,
            raw_response=selection.raw_response,
            path_to_root=selection.path_to_root,
            target_frame_path=selection.target_frame_path,
            chosen_actions=list(chosen.get("actions") or []),
        )


# ── Logging one turn ──────────────────────────────────────────────────────────

def _first_int(source: object, *names: str) -> int | None:
    """The first of *names* present on *source* as an integer."""
    for name in names:
        value = getattr(source, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _log_usage(agent_logger: AgentLogger, response: LLMResponse, iteration: int,
               totals: "_TokenTotals | None" = None) -> None:
    """Record token counts, under whichever names the responding provider uses.

    Gemini reports them on ``usage_metadata``; Anthropic and OpenAI on ``usage``,
    but with different field names — ``input_tokens``/``output_tokens`` against
    ``prompt_tokens``/``completion_tokens``.
    """
    raw = response.raw
    usage = getattr(raw, "usage_metadata", None) or getattr(raw, "usage", None)

    input_tokens = _first_int(usage, "prompt_token_count", "input_tokens", "prompt_tokens")
    output_tokens = _first_int(usage, "candidates_token_count", "output_tokens", "completion_tokens")
    total_tokens = _first_int(usage, "total_token_count", "total_tokens")
    if total_tokens is None and (input_tokens or output_tokens):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    if totals is not None:
        totals.add(input_tokens, output_tokens, total_tokens)

    agent_logger.log_usage(
        iteration,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        llm_call_id=response.llm_call_id,
    )


def _log_response_text(agent_logger: AgentLogger, response: LLMResponse, iteration: int) -> None:
    """Record every text block the model emitted, reasoning included."""
    for item in response.ordered_content:
        if isinstance(item, TextContentBlock):
            agent_logger.log_llm_text(
                item.text,
                iteration,
                is_thought=item.is_thought,
                thought_signature_present=item.thought_signature_present,
            )


class _TokenTotals:
    """Tokens billed across the run, summed from what each response reported."""

    def __init__(self) -> None:
        self.input = 0
        self.output = 0
        self.total = 0

    def add(self, input_tokens: int | None, output_tokens: int | None,
            total_tokens: int | None) -> None:
        self.input += input_tokens or 0
        self.output += output_tokens or 0
        self.total += total_tokens or ((input_tokens or 0) + (output_tokens or 0))

    def as_dict(self) -> dict[str, int]:
        return {"input": self.input, "output": self.output, "total": self.total}


def _run_metadata(
    run: "_Run",
    *,
    keyframes: Path,
    sequence: "KeyframeSequence",
    started: datetime.datetime,
    attempt: int,
) -> dict[str, Any]:
    """What this run is, before it has an outcome.

    Written at the start so a directory describes itself even if the run is
    killed, and rewritten at the end with the result.
    """
    return {
        "run": run.out_dir.name,
        "sequence": sequence.name,
        "keyframes": str(keyframes),
        "frame_count": len(run.frames),
        "model": getattr(run.backend, "model", None),
        "backend": type(run.backend).__name__,
        "attempt": attempt,
        "started": started.isoformat(timespec="seconds"),
        "finished": None,
        "elapsed_seconds": None,
        "stop_reason": None,
        "iterations": 0,
        "frames_solved": [],
        "tokens": {"input": 0, "output": 0, "total": 0},
        "thinking_time_seconds": 0.0,
        # The settings in force, so the run stays reproducible after config.py moves on.
        "config": {
            "max_iterations": config.MAX_ITERATIONS,
            "max_attempts_per_frame": config.MAX_ATTEMPTS_PER_FRAME,
            "max_consecutive_no_tool_responses": config.MAX_CONSECUTIVE_NO_TOOL_RESPONSES,
            "repeated_mismatch_attempts": config.REPEATED_MISMATCH_ATTEMPTS,
            "thinking_budget": config.THINKING_BUDGET,
            "backend_max_tokens": config.BACKEND_MAX_TOKENS,
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
            "front_color": sequence.front_color,
            "back_color": sequence.back_color,
        },
    }


#: What a resumed run must not change. The frames are already folded and the
#: checkpoints already written against them; continuing with a different
#: sequence, or a model whose output the earlier attempts never saw, produces a
#: directory whose contents contradict each other.
_RESUME_MUST_MATCH = ("sequence", "keyframes", "frame_count", "model")


def _check_resume_matches(out_dir: Path, metadata: dict[str, Any]) -> None:
    """Refuse a resume that would contradict what the directory already holds."""
    path = out_dir / "metadata.json"
    if not path.exists():
        return
    try:
        earlier = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — an unreadable summary is not a reason to stop
        logger.warning("Could not read %s to verify the resume: %s", path, exc)
        return

    mismatched = [
        f"{field}: this run has {metadata.get(field)!r}, "
        f"the directory was built with {earlier.get(field)!r}"
        for field in _RESUME_MUST_MATCH
        if field in earlier and earlier.get(field) != metadata.get(field)
    ]
    if mismatched:
        raise ValueError(
            f"Cannot resume {out_dir}: it does not match this run.\n  "
            + "\n  ".join(mismatched)
            + "\nStart a new run, or resume with the sequence and model it was built with."
        )


def _previous_attempts(out_dir: Path) -> int:
    """How many runs this directory has already held."""
    path = out_dir / "metadata.json"
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("attempt", 1))
    except Exception:  # noqa: BLE001
        return 1


def _write_metadata(out_dir: Path, metadata: dict[str, Any]) -> None:
    """Persist the run summary, best-effort — never fail a run over it.

    A resumed run continues in the same directory, so the summary it replaces is
    kept under ``previous``: the record of what the earlier attempt did, and why
    it stopped, survives the resume that follows it.
    """
    path = out_dir / "metadata.json"
    try:
        if path.exists() and "previous" not in metadata:
            earlier = json.loads(path.read_text(encoding="utf-8"))
            if earlier.get("attempt") != metadata.get("attempt"):
                metadata["previous"] = earlier
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write metadata.json: %s", exc)


class _ThinkingTime:
    """Model latency, accumulated over the run and since the last checkpoint.

    The per-checkpoint figure restarts whenever a frame is solved, so it reads
    as "how long this step has been taking" rather than "how long the run has".
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.since_checkpoint = 0.0

    def record(self, started_at: float) -> float:
        """Add the time since *started_at* to both totals, and return it."""
        elapsed = time.perf_counter() - started_at
        self.total += elapsed
        self.since_checkpoint += elapsed
        return elapsed

    def checkpoint_reached(self) -> None:
        """A frame was solved: the per-step figure starts again from zero."""
        self.since_checkpoint = 0.0


class _NoToolStreak:
    """A run of replies that called no tool, and the context from before it.

    A model that stops calling tools mid-sequence is usually stuck on something
    it cannot see its way out of. It gets nudged a couple of times; past that,
    the conversation is rewound to the turn before the streak began and the
    request is put again from there, which is more likely to break the loop
    than another nudge on top of the replies that caused it.
    """

    def __init__(self) -> None:
        self.count = 0
        self._before: tuple[list, int] | None = None

    def reset(self) -> None:
        self.count = 0
        self._before = None

    def record(self, messages: list, session_start: int) -> None:
        """Note one tool-less reply, snapshotting the context on the first."""
        if self.count == 0:
            self._before = (list(messages), session_start)
        self.count += 1

    @property
    def is_exhausted(self) -> bool:
        return self.count >= config.MAX_CONSECUTIVE_NO_TOOL_RESPONSES

    def rewind(self) -> tuple[list, int]:
        """Return the messages and session start from before the streak."""
        assert self._before is not None, "rewind before any reply was recorded"
        return list(self._before[0]), self._before[1]


def _stop_on_backend_error(
    exc: Exception,
    *,
    backend: LLMBackend,
    agent_logger: AgentLogger,
    iteration: int,
    out_dir: Path,
) -> str:
    """Identify a backend failure, record it, show it, and name the stop reason.

    Any error from any provider ends the run here rather than escaping as a
    traceback.
    """
    error = describe_backend_error(exc, backend=backend)
    logger.error("Backend error at iteration %d: %s", iteration, error.summary())
    agent_logger.log_backend_error(iteration, error)
    print(_format_backend_error(error, out_dir=out_dir))
    return f"backend_error:{type(exc).__name__}"


# ── The agent loop ────────────────────────────────────────────────────────────

def run_agent(
    keyframes: str | Path,
    *,
    resume_run_dir: str | Path | None = None,
    backend: LLMBackend | None = None,
) -> dict[int, Any]:
    """
    Run the interactive origami agent over a keyframe sequence.

    Parameters
    ----------
    keyframes:
        Path to a ``keyframes.json`` manifest: frame directory, keyframe
        indices and paper colours.
    resume_run_dir:
        Continue an existing run directory in place, instead of starting a new one.
    backend:
        Custom ``LLMBackend`` to use instead of one built from
        ``config.DEFAULT_MODEL``.

    A backend error stops the run without raising; checkpoints solved so far
    are kept, and the run can be continued with ``resume_run_dir``.

    Returns
    -------
    dict[int, GeometryState]
        The saved checkpoints keyed by frame index.
    """
    run = _prepare_run(keyframes, resume_run_dir, backend)
    controller, agent_logger, backend_obj = run.controller, run.agent_logger, run.backend
    frames = run.frames
    executor = _TurnExecutor(run)

    messages = backend_obj.init_messages(_opening_message(controller, len(frames)))
    stop_reason = "incomplete"
    t_start = time.time()

    print(
        f"\n{'─' * 60}\n"
        f"Origami Agent — {len(frames)} frames\n"
        f"Backend : {type(backend_obj).__name__}\n"
        f"Critic  : {'none' if run.critic is None else run.critic.model}\n"
        f"Work dir: {run.out_dir}\n"
        f"Log     : {agent_logger.log_path}\n"
        f"{'─' * 60}"
    )

    iterations_completed = 0
    started_at = datetime.datetime.now()
    tokens = _TokenTotals()
    # A resumed run is the next attempt in the same directory.
    metadata = _run_metadata(run, keyframes=run.keyframes, sequence=run.sequence,
                             started=started_at, attempt=run.attempt)
    if run.attempt > 1:
        _check_resume_matches(run.out_dir, metadata)
    _write_metadata(run.out_dir, metadata)
    thinking = _ThinkingTime()
    no_tool_streak = _NoToolStreak()
    session_start: int = 0        # index in messages where the current frame session began

    agent_logger.log_run_start(
        frame_count=len(frames),
        backend_name=type(backend_obj).__name__,
        model=getattr(backend_obj, "model", None),
        max_iterations=config.MAX_ITERATIONS,
        out_dir=run.out_dir,
        frame_paths=[_repo_relative(f) for f in frames],
    )

    try:
        for local_iteration in range(config.MAX_ITERATIONS):
            iteration = local_iteration + run.iteration_offset
            iterations_completed = local_iteration + 1
            agent_logger.log_messages(iteration, messages)

            controller.begin_iteration(iteration)
            current_frame = (max(controller.checkpoints) if controller.checkpoints else 0) + 1
            backend_obj.log_label = f"iter{iteration:04d}_frame{current_frame:02d}"

            thinking_started_at = time.perf_counter()
            try:
                response = backend_obj.chat(messages, run.system_prompt)
            except Exception as exc:
                agent_logger.log_thinking_time(
                    iteration,
                    elapsed_seconds=thinking.record(thinking_started_at),
                    cumulative_since_checkpoint_seconds=thinking.since_checkpoint,
                )
                stop_reason = _stop_on_backend_error(
                    exc,
                    backend=backend_obj,
                    agent_logger=agent_logger,
                    iteration=iteration,
                    out_dir=run.out_dir,
                )
                break

            agent_logger.log_thinking_time(
                iteration,
                elapsed_seconds=thinking.record(thinking_started_at),
                cumulative_since_checkpoint_seconds=thinking.since_checkpoint,
            )

            _log_usage(agent_logger, response, iteration, tokens)
            _log_response_text(agent_logger, response, iteration)

            if not response.has_tool_calls:
                if (
                    response.stop_reason != "end_turn"
                    or len(controller.checkpoints) >= len(frames)
                ):
                    stop_reason = response.stop_reason
                    break

                no_tool_streak.record(messages, session_start)
                if no_tool_streak.is_exhausted:
                    messages, session_start = no_tool_streak.rewind()
                    agent_logger.log_no_tool_call_loop_reset(
                        iteration,
                        consecutive_no_tool_responses=no_tool_streak.count,
                        rollback_count=0,
                        restored_frame=max(controller.checkpoints) if controller.checkpoints else None,
                    )
                    no_tool_streak.reset()
                    continue

                messages = backend_obj.append_user_text(
                    messages, response, prompts.CONTINUE_NUDGE
                )
                continue

            no_tool_streak.reset()

            executed = executor.execute(response, iteration)
            messages = backend_obj.append_turn(messages, response, executed)

            if executor.checkpoint_saved_frame is not None:
                thinking.checkpoint_reached()
                no_tool_streak.reset()

            if len(controller.checkpoints) == len(frames):
                stop_reason = "all_frames_solved"
                print("\n[done] All frames solved!")
                break
        else:
            stop_reason = "max_iterations"
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        raise
    except Exception as exc:
        stop_reason = f"error:{type(exc).__name__}"
        raise
    finally:
        elapsed_total = time.time() - t_start
        metadata.update(
            finished=datetime.datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=round(elapsed_total, 1),
            stop_reason=stop_reason,
            iterations=iterations_completed,
            frames_solved=sorted(controller.checkpoints.keys()),
            tokens=tokens.as_dict(),
            thinking_time_seconds=round(thinking.total, 1),
        )
        _write_metadata(run.out_dir, metadata)
        agent_logger.log_finish(
            stop_reason,
            sorted(controller.checkpoints.keys()),
            iterations_completed=iterations_completed,
            max_iterations=config.MAX_ITERATIONS,
            total_thinking_time_seconds=thinking.total,
        )
        agent_logger.close()
        print(
            f"\n{'─' * 60}\n"
            f"Finished in {elapsed_total:.1f}s  |  stop={stop_reason!r}  "
            f"|  iterations={iterations_completed}/{config.MAX_ITERATIONS}\n"
            f"Checkpoints: {sorted(controller.checkpoints.keys())}\n"
            f"{'─' * 60}\n"
        )
        try:
            _save_checkpoints(controller, run.out_dir)
        except Exception as exc:
            logger.warning("Could not save checkpoints file: %s", exc)

    return controller.checkpoints

# ── CLI ───────────────────────────────────────────────────────────────────────

@dataclass
class KeyframeSequence:
    """A sequence manifest: which frames to fold, and what colour the paper is."""

    name: str
    frames: list[str]
    front_color: str
    back_color: str


def _load_keyframe_sequence(manifest_path: str | Path) -> KeyframeSequence:
    """Load a ``keyframes.json`` manifest and resolve its keyframe paths.

    The manifest is the single source of truth for a sequence::

        {
          "name": "how-to-make-an-origami-heart",
          "frames": "frames",
          "front_color": "white",
          "back_color": "red",
          "keyframes": [4, 10, 38, ...]
        }

    ``frames`` is a directory relative to the manifest, holding images named by
    their global index alone (``000004.jpg``); each keyframe index names one.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    frames_dir = manifest_path.parent / manifest.get("frames", "frames")
    if not frames_dir.is_dir():
        raise ValueError(f"Frames directory not found: {frames_dir}")

    indices = manifest.get("keyframes")
    if not indices:
        raise ValueError(f"No keyframes listed in {manifest_path}")

    by_index: dict[int, Path] = {}
    for path in sorted(frames_dir.iterdir()):
        match = FRAME_NAME_RE.match(path.name)
        if match is not None:
            by_index[int(match.group("index"))] = path

    missing = [i for i in indices if i not in by_index]
    if missing:
        raise ValueError(f"Keyframes missing from {frames_dir}: {missing}")

    return KeyframeSequence(
        name=manifest.get("name", manifest_path.parent.name),
        frames=[_repo_relative(by_index[i]) for i in indices],
        front_color=manifest.get("front_color", config.DEFAULT_FRONT_COLOR),
        back_color=manifest.get("back_color", config.DEFAULT_BACK_COLOR),
    )


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the interactive origami agent on a keyframe sequence. "
            "The model, the run limits, the output location and the fallback "
            "paper colours are set in foldingagent/config.py, not on the "
            "command line."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "keyframes",
        help=(
            "Path to a keyframes.json manifest: the frame directory, the keyframe "
            "indices and the paper colours, e.g. 'data/heart/keyframes.json'."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Continue an existing run directory in place, e.g. "
            f"'{config.OUT_DIR}/20260902_101010'. Without it a new run "
            f"directory is created under {config.OUT_DIR}/."
        ),
    )
    args = parser.parse_args()

    # Validate the arguments here, so a bad path is a usage error rather than a
    # traceback. Anything that goes wrong once the run starts is the run's own.
    try:
        _load_keyframe_sequence(args.keyframes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.resume is not None and not Path(args.resume).is_dir():
        parser.error(f"--resume: run directory does not exist: {args.resume}")

    run_agent(args.keyframes, resume_run_dir=args.resume)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()
