"""Default constants for the origami agent — every tunable in one place."""

from __future__ import annotations

# ── Model selection ──────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-3.1-pro-preview"

PROVIDER_API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

# ── Output ───────────────────────────────────────────────────────────────────
#: Root for run output. Each run creates its own ``<OUT_DIR>/<timestamp>/``.
OUT_DIR = "out"

# ── Run limits ───────────────────────────────────────────────────────────────
MAX_ITERATIONS = 300
MAX_CONSECUTIVE_NO_TOOL_RESPONSES = 3
MAX_ATTEMPTS_PER_FRAME = 5
#: Failed attempts at one transition before the critic's guidance hardens.
REPEATED_MISMATCH_ATTEMPTS = 3

# ── Paper ────────────────────────────────────────────────────────────────────
DEFAULT_FRONT_COLOR = "white"
DEFAULT_BACK_COLOR = "lightblue"

# ── Generation ───────────────────────────────────────────────────────────────
THINKING_BUDGET = 8192          # Gemini ThinkingConfig(thinking_budget=...)
BACKEND_MAX_TOKENS = 8192       # Anthropic / OpenAI agent backends
MAX_OUTPUT_TOKENS = 4096        # Gemini agent backend + all critics
