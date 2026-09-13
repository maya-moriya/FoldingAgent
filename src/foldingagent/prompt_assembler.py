"""Builds the final prompt strings by filling the templates in :mod:`foldingagent.prompts`.

All substitution goes through ``str.replace`` on ``{placeholder}`` markers, never
``str.format``: the prompt text contains literal braces (JSON response examples,
set notation such as ``{'x', 'y'}``) that must reach the model untouched.
"""

from __future__ import annotations

from typing import Any

from foldingagent import prompts
from foldingagent.config import (
    DEFAULT_BACK_COLOR,
    DEFAULT_FRONT_COLOR,
    REPEATED_MISMATCH_ATTEMPTS,
)


def _fill(template: str, **values: Any) -> str:
    """Replace each ``{name}`` marker in *template* with its value."""
    for name, value in values.items():
        template = template.replace("{" + name + "}", str(value))
    return template


def _with_colors(body: str, front_color: str, back_color: str) -> str:
    return _fill(body, front_color=front_color, back_color=back_color)


# ── System prompts ───────────────────────────────────────────────────────────

def build_system_prompt(
    num_frames: int,
    front_color: str = DEFAULT_FRONT_COLOR,
    back_color: str = DEFAULT_BACK_COLOR,
) -> str:
    """Assemble the agent's system prompt for a sequence of *num_frames* frames."""
    body = _with_colors(prompts.SYSTEM_PROMPT, front_color, back_color)
    body = _fill(body, num_frames=num_frames)
    return body + _fill(prompts.SYSTEM_PROMPT_FOOTER, num_frames=num_frames)


def build_critic_prompt(
    front_color: str = DEFAULT_FRONT_COLOR,
    back_color: str = DEFAULT_BACK_COLOR,
) -> str:
    """Assemble the per-step visual critic's system prompt."""
    return _with_colors(prompts.CRITIC_PROMPT, front_color, back_color)


def build_overview_critic_prompt(
    front_color: str = DEFAULT_FRONT_COLOR,
    back_color: str = DEFAULT_BACK_COLOR,
) -> str:
    """Assemble the overview critic's system prompt."""
    return _with_colors(prompts.OVERVIEW_CRITIC_PROMPT, front_color, back_color)


# ── Conversation turns ───────────────────────────────────────────────────────

def build_task_message(num_frames: int) -> str:
    """The opening user turn for a fresh run."""
    return _fill(
        prompts.TASK_MESSAGE,
        num_frames=num_frames,
        penultimate_frame=num_frames - 1,
    )


def build_resume_task_message(
    *,
    num_frames: int,
    solved: list[int],
    last_solved: int,
    remaining: list[int],
) -> str:
    """The opening user turn when continuing a run that already has checkpoints."""
    return _fill(
        prompts.RESUME_TASK_MESSAGE,
        num_frames=num_frames,
        solved=solved,
        last_solved=last_solved,
        remaining=remaining,
        next_frame=last_solved + 1,
    )


def build_overview_instruction(frame_indices: list[int]) -> str:
    """The user turn accompanying the checkpoint overview grid."""
    return _fill(
        prompts.OVERVIEW_INSTRUCTION,
        frame_count=len(frame_indices),
        frame_indices=frame_indices,
    )


# ── Tool-result guidance ─────────────────────────────────────────────────────

def build_critic_grid_legend(*, source_frame: int, frame: int) -> dict[str, str]:
    """Label the four panels of a critic grid for the model."""
    return {
        "A": _fill(prompts.CRITIC_GRID_LEGEND_A, source_frame=source_frame),
        "B": _fill(prompts.CRITIC_GRID_LEGEND_B, frame=frame),
        "C": _fill(prompts.CRITIC_GRID_LEGEND_C, source_frame=source_frame),
        "D": prompts.CRITIC_GRID_LEGEND_D,
    }


def build_mismatch_instructions(*, attempt_n: int, source_frame: int, frame: int) -> str:
    """What to do after a MISMATCH — sterner once the transition keeps failing."""
    if attempt_n >= REPEATED_MISMATCH_ATTEMPTS:
        return _fill(
            prompts.VERDICT_MISMATCH_REPEATED,
            attempt_n=attempt_n,
            source_frame=source_frame,
            frame=frame,
        )
    return prompts.VERDICT_MISMATCH


def build_overview_recommendation(
    *, first_suspicious: int | None, last_correct: int | None
) -> str:
    """What the overview critic's reading of the whole sequence implies."""
    if first_suspicious is None:
        return prompts.OVERVIEW_RECOMMENDATION_ALL_CORRECT
    return _fill(
        prompts.OVERVIEW_RECOMMENDATION_DIVERGENCE,
        first_suspicious=first_suspicious,
        last_correct=last_correct,
    )
