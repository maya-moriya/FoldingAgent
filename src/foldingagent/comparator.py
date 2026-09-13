"""Comparator: picks the best candidate render among several attempts at one frame."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foldingagent import config
from foldingagent.backends.base import (
    _load_dotenv_into_env,
    _mime_type_for_path,
    _provider_for_model,
    _resolve_api_key,
)
from foldingagent.logger import _save_critic_call
from foldingagent.critic import CriticRequest
from foldingagent.simulator import GeometryState
from foldingagent.memory import AttemptTree, _save_checkpoints
from foldingagent.prompts import COMPARATOR_SYSTEM_PROMPT, COMPARATOR_USER_PROMPT

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = config.MAX_OUTPUT_TOKENS
_COMPARATOR_SYSTEM_PROMPT = COMPARATOR_SYSTEM_PROMPT


@dataclass
class ComparatorResponse:
    chosen_label: str
    raw_response: str


class OrigamiComparator:
    """Pick the best candidate render from a set of attempts for a given target frame."""

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
    ) -> None:
        self.model = model
        self.provider = _provider_for_model(model)
        self.log_dir: Path | None = None

        _load_dotenv_into_env()

        if self.provider == "anthropic":
            import anthropic as _anthropic  # type: ignore[import]
            _api_key = _resolve_api_key("anthropic")
            self._client = _anthropic.Anthropic(api_key=_api_key)
        elif self.provider == "openai":
            import openai as _openai  # type: ignore[import]
            _api_key = _resolve_api_key("openai")
            self._client = _openai.OpenAI(api_key=_api_key)
        else:
            from google import genai as _genai  # type: ignore[import]

            _api_key = _resolve_api_key("gemini")
            self._client = _genai.Client(api_key=_api_key)

    def select_best(
        self,
        target_photo: Path,
        candidates: list[tuple[str, Path]],
    ) -> ComparatorResponse:
        """Select the best candidate render matching *target_photo*.

        Parameters
        ----------
        target_photo : Path
            The real photo of the target origami state (shown as image A).
        candidates : list[tuple[str, Path]]
            List of ``(node_id, render_path)`` pairs. Each render is shown as
            B1, B2, … in order.

        Returns
        -------
        ComparatorResponse with ``chosen_label`` (e.g. ``"B2"``) and ``raw_response``.
        """
        label_to_node: dict[str, str] = {}
        image_order: list[tuple[str, Path]] = [("A", target_photo)]
        for i, (node_id, render_path) in enumerate(candidates, 1):
            label = f"B{i}"
            label_to_node[label] = node_id
            image_order.append((label, render_path))

        request = CriticRequest(
            image_order=image_order,
            final_text=COMPARATOR_USER_PROMPT,
            model=self.model,
            system_prompt=_COMPARATOR_SYSTEM_PROMPT,
        )

        if self.provider == "anthropic":
            raw_text = self._call_anthropic(request)
        elif self.provider == "openai":
            raw_text = self._call_openai(request)
        else:
            raw_text = self._call_gemini(request)

        if self.log_dir is not None:
            _save_critic_call(
                self.log_dir, request.as_serializable(),
                {"raw_text": raw_text}, "comparator",
            )

        import re
        match = re.search(r"B\d+", raw_text.strip())
        chosen_label = match.group(0) if match else "B1"

        return ComparatorResponse(chosen_label=chosen_label, raw_response=raw_text)

    def _call_gemini(self, request: CriticRequest) -> str:
        from google.genai import types as gtypes  # type: ignore[import]

        parts: list[Any] = []
        for kind, value in request.iter_content_blocks():
            if kind == "text":
                parts.append(gtypes.Part.from_text(text=value))
            else:
                parts.append(
                    gtypes.Part.from_bytes(data=value.read_bytes(), mime_type=_mime_type_for_path(value))
                )

        response = self._client.models.generate_content(
            model=request.model,
            contents=[{"role": "user", "parts": parts}],
            config=gtypes.GenerateContentConfig(
                system_instruction=request.system_prompt,
            ),
        )

        raw_text = ""
        if response and response.candidates:
            for part in response.candidates[0].content.parts or []:
                if hasattr(part, "text") and part.text:
                    raw_text += part.text
        return raw_text

    def _call_anthropic(self, request: CriticRequest) -> str:
        content: list[dict[str, Any]] = []
        for kind, value in request.iter_content_blocks():
            if kind == "text":
                content.append({"type": "text", "text": value})
            else:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _mime_type_for_path(value),
                        "data": base64.standard_b64encode(value.read_bytes()).decode(),
                    },
                })

        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": content}],
        }

        response = self._client.messages.create(**kwargs)

        raw_text = ""
        if response:
            for block in response.content:
                if hasattr(block, "text"):
                    raw_text += block.text
        return raw_text

    def _call_openai(self, request: CriticRequest) -> str:
        content: list[dict[str, Any]] = []
        for kind, value in request.iter_content_blocks():
            if kind == "text":
                content.append({"type": "text", "text": value})
            else:
                image_bytes = value.read_bytes()
                b64 = base64.standard_b64encode(image_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{_mime_type_for_path(value)};base64,{b64}"},
                })

        response = self._client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return response.choices[0].message.content or ""


@dataclass
class _ComparatorResult:
    chosen_node: str
    chosen_label: str
    raw_response: str
    attempts: list[str]
    candidates: list[dict[str, Any]]
    path_to_root: list[str]
    target_frame_path: str = ""


def _comparator_select_best(
    comparator: "OrigamiComparator",
    attempt_tree: AttemptTree,
    controller: "OrigamiController",
    frame_index: int,
    frames: list[str | Path],
    out_dir: Path,
) -> _ComparatorResult | None:
    """Use the comparator to pick the best attempt for *frame_index*, then
    restore the controller's checkpoints along that node's path to root.
    """
    attempts = attempt_tree.get_attempts_for_frame(frame_index)
    candidates: list[tuple[str, Path]] = []
    for nid in attempts:
        rp = attempt_tree.nodes[nid].get("render_path")
        if rp:
            candidates.append((nid, Path(rp)))
    if not candidates:
        logger.warning("No renders available for frame %d attempts; skipping comparator.", frame_index)
        return None

    target_photo = Path(frames[frame_index - 1])

    try:
        resp = comparator.select_best(target_photo=target_photo, candidates=candidates)
    except Exception as exc:
        logger.warning("Comparator failed for frame %d: %s", frame_index, exc)
        return None

    import re
    match = re.search(r"B(\d+)", resp.chosen_label)
    if not match or int(match.group(1)) < 1 or int(match.group(1)) > len(candidates):
        chosen_idx = 0
    else:
        chosen_idx = int(match.group(1)) - 1

    chosen_node_id = candidates[chosen_idx][0]

    path = attempt_tree.get_path_to_root(chosen_node_id)
    controller.checkpoints.clear()
    for nid in path:
        node = attempt_tree.nodes[nid]
        fi = node["frame_index"]
        geom = node.get("geometry")
        if geom is not None:
            controller.checkpoints[fi] = GeometryState(raw_representation=geom)

    if frame_index in controller.checkpoints:
        controller.current_geometry = controller.checkpoints[frame_index].model_copy(deep=True)

    _save_checkpoints(controller, out_dir)

    attempt_tree._checkpoint_node[frame_index] = chosen_node_id
    attempt_tree._current_parent_id = chosen_node_id
    attempt_tree._save()

    return _ComparatorResult(
        chosen_node=chosen_node_id,
        chosen_label=resp.chosen_label,
        raw_response=resp.raw_response,
        attempts=attempts,
        candidates=[
            {"node_id": nid, "render_path": str(rp)}
            for nid, rp in candidates
        ],
        path_to_root=path,
        target_frame_path=str(target_photo),
    )
