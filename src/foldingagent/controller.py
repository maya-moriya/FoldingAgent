"""OrigamiController — the origami simulator, critics and checkpoints, exposed as LLM tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from foldingagent import prompt_assembler, prompts
from foldingagent.rendering import render_geometry_diagram
from foldingagent.simulator import (
    GeometryFunctionCall,
    GeometryState,
    OrigamiLibraryExecutionEngine,
    ParsedActionPlan,
)
from foldingagent.backends.base import ToolCallRequest
from foldingagent.rendering import PaperColors

from foldingagent.critic import (
    CriticVerdict,
    FoldAnchor,
    OrigamiCritic,
    create_critic_grid,
)
from foldingagent.overview_critic import (
    OrigamiOverviewCritic,
    build_checkpoint_overview_image,
)

#: Frame files are named by their global index alone, e.g. ``000004.jpg``.
FRAME_NAME_RE = re.compile(
    r"^(?P<index>\d+)\.(?:png|jpe?g)$",
    re.IGNORECASE,
)


@dataclass
class ToolResult:
    """Structured response returned by every agent tool."""

    data: dict[str, Any]
    images: list[Path] = field(default_factory=list)

    _PATH_KEYS: frozenset[str] = frozenset({
        "image_path",
        "render_path",
        "grid_path",
        "filmstrip_path",
        "overview_path",
        "previous_selected_frame",
        "current_selected_frame",
        "sampled_frame_paths",
    })

    def _model_safe_data(self) -> dict[str, Any]:
        """Return tool data with local file paths removed for model-facing turns."""

        def _strip_paths(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: _strip_paths(item)
                    for key, item in value.items()
                    if key not in ToolResult._PATH_KEYS
                }
            if isinstance(value, list):
                return [_strip_paths(item) for item in value]
            return value

        sanitized = _strip_paths(self.data)
        return sanitized if isinstance(sanitized, dict) else self.data

    def to_json(self) -> str:
        return json.dumps(self._model_safe_data(), indent=2)


class OrigamiController:
    """Orchestrates the simulator, critics and checkpoint state; each method returns a ``ToolResult``."""

    def __init__(
        self,
        frames: list[str | Path],
        out_dir: str | Path,
        paper_colors: PaperColors | None = None,
        critic: OrigamiCritic | None = None,
        overview_critic: OrigamiOverviewCritic | None = None,
    ) -> None:
        if not frames:
            raise ValueError("At least one frame path is required.")
        self.frames: list[Path] = [Path(f) for f in frames]
        self.out_dir: Path = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._paper_colors: PaperColors = paper_colors or PaperColors()
        self._engine = OrigamiLibraryExecutionEngine(
            paper_colors_override=self._paper_colors
        )
        self._critic: OrigamiCritic | None = critic
        self._overview_critic: OrigamiOverviewCritic | None = overview_critic

        self.current_geometry: GeometryState = GeometryState()
        self.checkpoints: dict[int, GeometryState] = {}
        self._iteration: int = 0
        self._transition_attempts: dict[int, int] = {}
        self._restore_counts: dict[int, int] = {}
        self._rollback_events: list[tuple[int, int]] = []  # (rolled_back_from, rolled_back_to)
        self._locked_checkpoints: set[int] = set()
        self._min_restore_frame: int = 0

    # ── Navigation ─────────────────────────────────────────────────────────────

    def view_frame(self, n: int) -> ToolResult:
        """Return the real-world photo at frame n (1-based index)."""
        if n < 1 or n > len(self.frames):
            return ToolResult({
                "success": False,
                "error": f"Frame {n} is out of range. Valid range: 1-{len(self.frames)}.",
            })
        path = self.frames[n - 1]
        if not path.exists():
            return ToolResult({
                "success": False,
                "error": f"Image file for frame {n} not found.",
            })
        return ToolResult(
            {
                "success": True,
                "frame_index": n,
                "total_frames": len(self.frames),
                "image_path": str(path),
            },
            images=[path],
        )

    def get_current_state(self) -> ToolResult:
        """Return the current geometry graph as structured JSON."""
        try:
            geom = self._ensure_raw(self.current_geometry)
        except RuntimeError as exc:
            return ToolResult({"success": False, "error": str(exc)})
        ops = [fn.render() for fn in geom.operations]
        return ToolResult({
            "success": True,
            "geometry": geom.raw_representation or {},
            "operation_count": len(ops),
            "applied_operations": ops,
        })

    # ── Rendering ──────────────────────────────────────────────────────────────

    def render_current(self) -> ToolResult:
        """Render the current simulator state as a diagram PNG."""
        try:
            geom = self._ensure_raw(self.current_geometry)
        except RuntimeError as exc:
            return ToolResult({"success": False, "error": str(exc)})

        output_path = self._next_artifact("render_current")
        try:
            render_geometry_diagram(
                geom.raw_representation,
                output_path,
                paper_colors=self._paper_colors,
            )
        except Exception as exc:
            return ToolResult({"success": False, "error": f"Render failed: {exc}"})

        return ToolResult(
            {"success": True, "render_path": str(output_path)},
            images=[output_path],
        )

    def observe_movement(self, t: int) -> ToolResult:
        """
        Build a left-to-right filmstrip for the motion that occurred during step ``t``.

        Step indices are 1-based over the selected frames passed to the agent. For
        ``t >= 2``, this samples up to five evenly spaced frames from the sibling
        ``all_frames`` directory (or ``frames`` if that doesn't exist), spanning
        the previous selected step through the current one (inclusive), and
        returns the strip as a PNG.
        """
        if t < 2 or t > len(self.frames):
            return ToolResult(
                {
                    "success": False,
                    "error": (
                        f"observe_movement requires 2 <= t <= {len(self.frames)}. "
                        f"Received t={t}."
                    ),
                }
            )

        try:
            from PIL import Image
        except ImportError:
            return ToolResult(
                {
                    "success": False,
                    "error": "Pillow is required for observe_movement.",
                }
            )

        previous_selected = self.frames[t - 2]
        current_selected = self.frames[t - 1]

        try:
            previous_index = self._extract_global_frame_index(previous_selected)
            current_index = self._extract_global_frame_index(current_selected)
        except ValueError as exc:
            return ToolResult({"success": False, "error": str(exc)})

        if current_index <= previous_index:
            return ToolResult(
                {
                    "success": False,
                    "error": (
                        "Selected frames must have strictly increasing global indices. "
                        f"Received {previous_index} -> {current_index} for step {t}."
                    ),
                }
            )

        sequence_dir = current_selected.parent.parent
        all_frames_dir = sequence_dir / "all_frames"
        fallback_frames_dir = sequence_dir / "frames"
        movement_frames_dir = all_frames_dir
        if not movement_frames_dir.is_dir() and fallback_frames_dir.is_dir():
            movement_frames_dir = fallback_frames_dir
        if not movement_frames_dir.is_dir():
            return ToolResult(
                {
                    "success": False,
                    "error": (
                        "observe_movement requires a sibling all_frames or frames "
                        "directory next to the selected frames."
                    ),
                }
            )

        frame_by_index = self._index_frames_by_global_index(movement_frames_dir)
        available_indices = [
            index
            for index in range(previous_index, current_index + 1)
            if index in frame_by_index
        ]
        if len(available_indices) < 2:
            return ToolResult(
                {
                    "success": False,
                    "error": (
                        "Could not find enough all_frames images between the selected steps. "
                        f"Range: {previous_index}..{current_index}"
                    ),
                }
            )

        sampled_indices = self._sample_evenly_spaced_indices(
            available_indices,
            target_count=5,
        )
        sampled_paths = [frame_by_index[index] for index in sampled_indices]

        resized_frames: list[Image.Image] = []
        try:
            for frame_path in sampled_paths:
                with Image.open(frame_path) as image:
                    rgb = image.convert("RGB")
                    resized_frames.append(rgb.copy())

            separator = 2
            MAX_STRIP_WIDTH = 1024
            total_width = (
                sum(image.width for image in resized_frames)
                + separator * (len(resized_frames) + 1)
            )
            if total_width > MAX_STRIP_WIDTH:
                scale = MAX_STRIP_WIDTH / total_width
                scaled: list[Image.Image] = []
                for img in resized_frames:
                    w = max(1, int(img.width * scale))
                    h = max(1, int(img.height * scale))
                    scaled.append(img.resize((w, h), Image.LANCZOS))
                for img in resized_frames:
                    img.close()
                resized_frames = scaled
                total_width = (
                    sum(image.width for image in resized_frames)
                    + separator * (len(resized_frames) + 1)
                )
            total_height = max(image.height for image in resized_frames) + separator * 2
            strip = Image.new("RGB", (total_width, total_height), (255, 255, 255))

            x = separator
            for image in resized_frames:
                y = separator + (total_height - separator * 2 - image.height) // 2
                strip.paste(image, (x, y))
                x += image.width + separator

            output_path = self._next_artifact("observe_movement")
            strip.save(str(output_path), format="PNG")
        finally:
            for image in resized_frames:
                image.close()

        result_data: dict[str, Any] = {
            "success": True,
            "step_index": t,
            "previous_step_index": t - 1,
            "previous_selected_frame": str(previous_selected),
            "current_selected_frame": str(current_selected),
            "previous_global_index": previous_index,
            "current_global_index": current_index,
            "sampled_global_indices": sampled_indices,
            "sampled_frame_paths": [str(path) for path in sampled_paths],
            "returned_frame_count": len(sampled_indices),
            "requested_frame_count": 5,
            "filmstrip_path": str(output_path),
        }
        if len(sampled_indices) < 5:
            result_data["sampling_note"] = prompts.MOVEMENT_SAMPLING_NOTE

        return ToolResult(result_data, images=[output_path])

    def ask_critic(
        self,
        frame_n: int,
        action_classes: list[str] | None = None,
        fold_anchor: dict[str, str] | None = None,
    ) -> ToolResult:
        """
        Build the 2x2 critic grid, then send the four images to the dedicated
        visual critic for an independent verdict.

        Parameters
        ----------
        frame_n:
            Target frame index being verified.
        action_classes:
            High-level transition classes for this step, such as ``["fold"]`` or
            ``["rotate", "flip"]``. These are sent directly to the critic.
        fold_anchor:
            Dict with keys ``moving_element``, ``static_reference``, and
            ``relationship``. Required whenever ``action_classes`` includes
            ``"fold"``. When provided, the critic will explicitly check whether
            the anchor relationship is satisfied in D and will downgrade to
            MISMATCH if it is clearly violated.

        Returns the critic grid image together with the critic's structured analysis:
          - ``critic_verdict``      : "MATCH" | "MISMATCH" | "EXTREME DIVERGANCE"
          - ``critic_analysis``     : concise prose explanation
          - ``critic_anchor_check`` : one-sentence verdict on the fold anchor
          - ``critic_discrepancies``: list of specific geometric issues (empty for MATCH)
          - ``transition_attempt``  : retry count for this transition after critic failures

        Based on the verdict:
          MATCH          → call save_checkpoint(frame_n) and advance.
          MISMATCH            → call restore_checkpoint(frame_n - 1) and try a different approach.
          EXTREME DIVERGANCE  → assume a false assumption, roll back farther, and re-evaluate.
        """
        if frame_n < 1 or frame_n > len(self.frames):
            return ToolResult({
                "success": False,
                "error": f"frame_n={frame_n} out of range 1-{len(self.frames)}.",
            })

        if frame_n <= self._min_restore_frame and self._min_restore_frame > 0:
            return ToolResult({
                "success": False,
                "error": (
                    f"Cannot ask critic for frame {frame_n}. "
                    f"The best result for frames up to {self._min_restore_frame} "
                    f"has already been selected by the comparator. "
                    f"Continue working from frame {self._min_restore_frame} forward."
                ),
            })

        source_frame_n = frame_n - 1
        if source_frame_n >= 1 and source_frame_n not in self.checkpoints:
            return ToolResult({
                "success": False,
                "error": (
                    f"Cannot call ask_critic({frame_n}) because frame {source_frame_n} "
                    f"has not been saved yet. Call save_checkpoint({source_frame_n}) first."
                ),
            })

        target_frame_path = self.frames[frame_n - 1]
        source_frame_path = (
            self.frames[source_frame_n - 1] if source_frame_n >= 1 else self.frames[0]
        )

        source_raw_geom = self.checkpoints.get(source_frame_n, GeometryState())
        try:
            source_geom = self._ensure_raw(source_raw_geom)
            result_geom = self._ensure_raw(self.current_geometry)
        except RuntimeError as exc:
            return ToolResult({"success": False, "error": str(exc)})

        normalized_action_classes = self._normalize_action_classes(action_classes)
        requires_fold_anchor = "fold" in normalized_action_classes
        if requires_fold_anchor and not self._has_complete_fold_anchor(fold_anchor):
            return ToolResult(
                {
                    "success": False,
                    "error": (
                        "ask_critic requires fold_anchor when action_classes includes 'fold'. "
                        "Please provide moving_element, static_reference, and relationship."
                    ),
                    "requested_field": "fold_anchor",
                    "action_classes": normalized_action_classes,
                }
            )

        source_diagram = self._next_artifact("ask_critic_source")
        result_diagram = self._next_artifact("ask_critic_result")
        grid_path = self._next_artifact("ask_critic_grid")

        try:
            render_geometry_diagram(
                source_geom.raw_representation,
                source_diagram,
                paper_colors=self._paper_colors,
            )
            render_geometry_diagram(
                result_geom.raw_representation,
                result_diagram,
                paper_colors=self._paper_colors,
            )
            create_critic_grid(
                source_image=source_frame_path,
                target_image=target_frame_path,
                source_diagram=source_diagram,
                result_diagram=result_diagram,
                output_path=grid_path,
            )
        except Exception as exc:
            return ToolResult({
                "success": False,
                "error": f"Critic grid creation failed: {exc}",
            })

        data: dict[str, Any] = {
            "success": True,
            "grid_path": str(grid_path),
            "legend": prompt_assembler.build_critic_grid_legend(
                source_frame=source_frame_n, frame=frame_n
            ),
            "action_classes": normalized_action_classes,
            "source_state_json": source_geom.raw_representation,
            "result_state_json": result_geom.raw_representation,
        }

        if self._critic is not None:
            try:
                parsed_anchor: FoldAnchor | None = None
                if fold_anchor is not None:
                    parsed_anchor = FoldAnchor(
                        moving_element=fold_anchor.get("moving_element", ""),
                        static_reference=fold_anchor.get("static_reference", ""),
                        relationship=fold_anchor.get("relationship", ""),
                    )
                critic_response = self._critic.evaluate(
                    source_photo=source_frame_path,
                    target_photo=target_frame_path,
                    source_diagram=source_diagram,
                    result_diagram=result_diagram,
                    frame_n=frame_n,
                    action_classes=normalized_action_classes,
                    source_state_json=source_geom.raw_representation,
                    result_state_json=result_geom.raw_representation,
                    fold_anchor=parsed_anchor,
                )
                raw_verdict = critic_response.raw_verdict or critic_response.verdict.value
                data["critic_verdict"] = raw_verdict
                data["critic_analysis"] = critic_response.analysis
                data["critic_anchor_check"] = critic_response.anchor_check
                data["critic_discrepancies"] = critic_response.discrepancies

                if critic_response.verdict == CriticVerdict.MATCH:
                    self._transition_attempts.pop(frame_n, None)
                    data["transition_attempt"] = 0
                    data["verdict_instructions"] = prompts.VERDICT_MATCH
                elif critic_response.verdict == CriticVerdict.EXTREME_DIVERGANCE:
                    attempt_n = self._transition_attempts.get(frame_n, 0) + 1
                    self._transition_attempts[frame_n] = attempt_n
                    data["transition_attempt"] = attempt_n
                    data["verdict_instructions"] = prompts.VERDICT_EXTREME_DIVERGANCE
                else:
                    attempt_n = self._transition_attempts.get(frame_n, 0) + 1
                    self._transition_attempts[frame_n] = attempt_n
                    data["transition_attempt"] = attempt_n
                    data["verdict_instructions"] = prompt_assembler.build_mismatch_instructions(
                        attempt_n=attempt_n, source_frame=source_frame_n, frame=frame_n
                    )
            except Exception as exc:
                data["critic_verdict"] = "UNAVAILABLE"
                data["critic_error"] = str(exc)
                data["verdict_instructions"] = prompts.VERDICT_CRITIC_UNAVAILABLE
        else:
            data["critic_verdict"] = "UNAVAILABLE"
            data["verdict_instructions"] = prompts.VERDICT_NO_CRITIC

        return ToolResult(data, images=[grid_path])

    @staticmethod
    def _normalize_action_classes(action_classes: list[str] | None) -> list[str]:
        if action_classes is None:
            return []

        normalized: list[str] = []
        for action in action_classes:
            normalized_action = str(action).strip().lower()
            if not normalized_action:
                continue
            normalized.append(normalized_action)
        return normalized

    @staticmethod
    def _has_complete_fold_anchor(fold_anchor: dict[str, str] | None) -> bool:
        if not isinstance(fold_anchor, dict):
            return False
        required_fields = ("moving_element", "static_reference", "relationship")
        return all(str(fold_anchor.get(field, "")).strip() for field in required_fields)

    # ── Simulator actions ───────────────────────────────────────────────────────

    def add_vertex(self, edge: list[int], position: float) -> ToolResult:
        """
        Add a new vertex at a fractional position (0.0-1.0) along an existing edge.

        Returns the new vertex ID so you can reference it in subsequent fold calls.
        Vertex IDs start at 1; the initial square has vertices 1-4, so the first
        added vertex becomes 5, the second 6, and so on.
        """
        old_ids = self._vertex_id_set(self.current_geometry)
        result = self._execute_single("add_vertex", {"edge": edge, "position": position})
        if result.data["success"]:
            new_ids = self._vertex_id_set(self.current_geometry)
            added = new_ids - old_ids
            result.data["new_vertex_id"] = max(added) if added else None
        return result

    def fold(self, edge: list[int], direction: int) -> ToolResult:
        """
        Fold the paper along the given edge.
        direction = +1 or -1 (which side of the crease moves toward viewer)
        Non-vertical: +1 = upper side moves, -1 = lower side moves
        Vertical: +1 = right side moves, -1 = left side moves
        """
        return self._execute_single("fold", {"edge": edge, "direction": direction})

    def unfold(
        self,
        edge: list[int] | None = None,
        target: str | None = None,
    ) -> ToolResult:
        """
        Unfold a previous fold, leaving a crease mark (dashed line).

        Provide either ``edge`` (specific crease) or ``target='last'`` (most recent fold).
        """
        params: dict[str, Any] = {}
        if edge is not None:
            params["edge"] = edge
        elif target:
            params["target"] = target
        else:
            params["target"] = "last"
        return self._execute_single("unfold", params)

    def rotate(self, angle: float) -> ToolResult:
        """Rotate the entire model by angle degrees (positive = clockwise)."""
        return self._execute_single("rotate", {"angle": angle})

    def flip(self, axis: str) -> ToolResult:
        """
        Flip the model about an axis, revealing the other side of the paper.

        axis: 'x' | 'y' | 'y=x' | 'y=-x'
        """
        return self._execute_single("flip", {"axis": axis})

    # ── Checkpoints ────────────────────────────────────────────────────────────

    def save_checkpoint(self, frame_n: int, notes: str | None = None) -> ToolResult:
        """Save the current geometry as the verified solution for frame_n."""
        if frame_n <= self._min_restore_frame and self._min_restore_frame > 0:
            return ToolResult({
                "success": False,
                "error": (
                    f"Cannot save checkpoint for frame {frame_n}. "
                    f"The best result for frames up to {self._min_restore_frame} "
                    f"has already been selected by the comparator. "
                    f"Continue working from frame {self._min_restore_frame} forward."
                ),
            })
        self.checkpoints[frame_n] = self.current_geometry.model_copy(deep=True)
        result: dict[str, Any] = {
            "success": True,
            "message": f"Checkpoint saved for frame {frame_n}.",
            "saved_frames": sorted(self.checkpoints.keys()),
        }
        if notes:
            result["notes"] = notes
        return ToolResult(result)

    def restore_checkpoint(self, frame_n: int) -> ToolResult:
        """Restore the simulator to a previously saved checkpoint."""
        if frame_n not in self.checkpoints:
            return ToolResult({
                "success": False,
                "error": (
                    f"No checkpoint for frame {frame_n}. "
                    f"Available: {sorted(self.checkpoints.keys())}."
                ),
            })
        if frame_n < self._min_restore_frame:
            return ToolResult({
                "success": False,
                "error": (
                    f"Cannot restore to frame {frame_n}. "
                    f"Frame {self._min_restore_frame} is already saved with the best "
                    f"available result. You can only restore to frame "
                    f"{self._min_restore_frame} or later. "
                    f"Available checkpoints: "
                    f"{sorted(f for f in self.checkpoints if f >= self._min_restore_frame)}."
                ),
            })
        if frame_n in self._locked_checkpoints:
            return ToolResult({
                "success": False,
                "error": (
                    f"Checkpoint {frame_n} is locked: you exceeded the maximum "
                    f"number of restore_checkpoint calls for this frame. "
                    f"The best attempt for frame {frame_n + 1} has already been "
                    f"selected automatically. Continue from the current state."
                ),
            })
        if self.checkpoints and frame_n < max(self.checkpoints):
            self._rollback_events.append((max(self.checkpoints), frame_n))
        self.current_geometry = self.checkpoints[frame_n].model_copy(deep=True)
        self._restore_counts[frame_n] = self._restore_counts.get(frame_n, 0) + 1
        restore_count = self._restore_counts[frame_n]
        next_transition_target_frame = frame_n + 1 if frame_n < len(self.frames) else None
        return ToolResult({
            "success": True,
            "message": f"Restored simulator to frame {frame_n} checkpoint.",
            "operation_count": len(self.current_geometry.operations),
            "restored_frame": frame_n,
            "restore_count": restore_count,
            "next_transition_source_frame": frame_n,
            "next_transition_target_frame": next_transition_target_frame,
        })

    def get_checkpoint_state(self, frame_n: int) -> ToolResult:
        """Return the saved geometry JSON for a checkpointed frame without changing current state."""
        if frame_n not in self.checkpoints:
            return ToolResult({
                "success": False,
                "error": (
                    f"No checkpoint for frame {frame_n}. "
                    f"Available: {sorted(self.checkpoints.keys())}."
                ),
            })
        try:
            geom = self._ensure_raw(self.checkpoints[frame_n])
        except RuntimeError as exc:
            return ToolResult({"success": False, "error": str(exc)})
        ops = [fn.render() for fn in geom.operations]
        return ToolResult({
            "success": True,
            "frame_n": frame_n,
            "geometry": geom.raw_representation or {},
            "operation_count": len(ops),
            "applied_operations": ops,
        })

    def generate_checkpoint_overview(self) -> ToolResult:
        """
        Build a three-row overview PNG of all saved checkpoints.

        Layout (one column per checkpointed frame, left=earliest, right=latest):
          Row 0 (label bar): frame index
          Row 1:             real-world photo
          Row 2:             rendered diagram

        Columns are separated by a 2-px black grid.  Use this image to decide
        which checkpoint to roll back to when things look off.
        """
        if not self.checkpoints:
            return ToolResult({"success": False, "error": "No checkpoints saved yet."})

        try:
            from PIL import Image
        except ImportError:
            return ToolResult({"success": False, "error": "Pillow is required for generate_checkpoint_overview."})

        sorted_frames = sorted(self.checkpoints.keys())

        # Render diagram for every checkpoint
        diagram_paths: list[Path] = []
        for frame_n in sorted_frames:
            geom = self.checkpoints[frame_n]
            try:
                geom = self._ensure_raw(geom)
            except RuntimeError as exc:
                return ToolResult({"success": False, "error": str(exc)})
            diag_path = self._next_artifact(f"generate_checkpoint_overview_f{frame_n}")
            try:
                render_geometry_diagram(
                    geom.raw_representation,
                    diag_path,
                    paper_colors=self._paper_colors,
                )
            except Exception as exc:
                return ToolResult({"success": False, "error": f"Render failed for frame {frame_n}: {exc}"})
            diagram_paths.append(diag_path)

        overview_path = self._next_artifact("generate_checkpoint_overview")
        build_checkpoint_overview_image(
            sorted_frames=sorted_frames,
            frame_paths=[
                self.frames[frame_n - 1] if 1 <= frame_n <= len(self.frames) else None
                for frame_n in sorted_frames
            ],
            diagram_paths=diagram_paths,
            out_path=overview_path,
        )

        result_data: dict[str, Any] = {
            "success": True,
            "checkpointed_frames": sorted_frames,
            "overview_path": str(overview_path),
            "layout": prompts.OVERVIEW_LAYOUT,
        }

        if self._overview_critic is not None:
            try:
                ov = self._overview_critic.evaluate(overview_path, sorted_frames)
                result_data["overview_analysis"] = {
                    "last_correct_frame": ov.last_correct_frame,
                    "first_suspicious_frame": ov.first_suspicious_frame,
                    "analysis": ov.analysis,
                    "per_frame_notes": ov.per_frame_notes,
                }
                result_data["overview_recommendation"] = (
                    prompt_assembler.build_overview_recommendation(
                        first_suspicious=ov.first_suspicious_frame,
                        last_correct=ov.last_correct_frame,
                    )
                )
            except Exception as exc:
                result_data["overview_analysis_error"] = str(exc)

        return ToolResult(result_data, images=[overview_path])

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _execute_single(self, name: str, parameters: dict[str, Any]) -> ToolResult:
        """Run a single-function plan through the execution engine."""
        plan = ParsedActionPlan(functions=[
            GeometryFunctionCall(name=name, parameters=parameters)
        ])
        result = self._engine.execute(plan, self.current_geometry)
        if result.success and result.updated_geometry is not None:
            self.current_geometry = result.updated_geometry
            return ToolResult({
                "success": True,
                "applied": [fn.render() for fn in result.applied_functions],
                "operation_count": len(self.current_geometry.operations),
            })
        return ToolResult({
            "success": False,
            "error": result.error_message or "Execution failed with no error details.",
        })

    def _ensure_raw(self, geometry: GeometryState) -> GeometryState:
        """
        Return a geometry that is guaranteed to have ``raw_representation`` populated.

        If the geometry was built only from operation replay (e.g. initial square)
        this materialises it through the engine first.
        """
        if geometry.raw_representation is not None:
            return geometry
        origami, error = OrigamiLibraryExecutionEngine._materialize_origami(
            geometry,
            paper_colors=self._paper_colors,
        )
        if error or origami is None:
            raise RuntimeError(f"Cannot materialise geometry: {error}")
        raw = OrigamiLibraryExecutionEngine._serialize_origami(origami)
        return GeometryState(
            base_description=geometry.base_description,
            raw_representation=raw,
            operations=list(geometry.operations),
            paper_colors=geometry.paper_colors,
        )

    def _vertex_id_set(self, geometry: GeometryState) -> set[int]:
        """Return the set of current vertex IDs (1-indexed)."""
        rep = geometry.raw_representation
        if rep is None:
            # Initial square has vertices 1-4 before any operation has been applied.
            return {1, 2, 3, 4}
        vertices = rep.get("vertices", {})
        if isinstance(vertices, dict):
            return {int(k) for k in vertices}
        # Fallback: vertices stored as a list of IDs or index-based
        return {i + 1 for i in range(len(vertices))}

    def _extract_global_frame_index(self, frame_path: Path) -> int:
        match = FRAME_NAME_RE.match(frame_path.name)
        if match is None:
            raise ValueError(
                f"Could not extract a global frame index from filename: {frame_path.name}"
            )
        return int(match.group("index"))

    def _index_frames_by_global_index(self, directory: Path) -> dict[int, Path]:
        frame_by_index: dict[int, Path] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            match = FRAME_NAME_RE.match(path.name)
            if match is None:
                continue
            frame_by_index[int(match.group("index"))] = path
        return frame_by_index

    @staticmethod
    def _sample_evenly_spaced_indices(
        available_indices: list[int],
        target_count: int,
    ) -> list[int]:
        if len(available_indices) <= target_count:
            return available_indices
        last_position = len(available_indices) - 1
        positions = [
            round((step * last_position) / (target_count - 1))
            for step in range(target_count)
        ]
        return [available_indices[position] for position in positions]

    def _next_artifact(self, name: str) -> Path:
        """Path for a figure this turn produces, named for the tool that makes it.

        A tool may write more than one figure per turn — ask_critic writes three —
        so *name* carries the tool and, where needed, which of its figures this is.
        The iteration comes from the loop via :meth:`begin_iteration`; before the
        first turn (or in a bare controller) it is 0, which keeps the name valid.
        """
        figs = self.out_dir / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        return figs / f"iter_{self._iteration:04d}_{name}.png"

    def begin_iteration(self, iteration: int) -> None:
        """Tell the controller which turn its figures belong to."""
        self._iteration = iteration

    def dispatch(self, call: ToolCallRequest) -> ToolResult:
        """Route one tool call from the model to the method that implements it."""
        inp = call.input
        name = call.name
        try:
            if name == "view_frame":
                return self.view_frame(n=int(inp["n"]))
            if name == "observe_movement":
                return self.observe_movement(t=int(inp["t"]))
            if name == "get_current_state":
                return self.get_current_state()
            if name == "render_current":
                return self.render_current()
            if name == "add_vertex":
                return self.add_vertex(
                    edge=list(inp["edge"]),
                    position=float(inp["position"]),
                )
            if name == "fold":
                return self.fold(
                    edge=list(inp["edge"]),
                    direction=int(inp["direction"]),
                )
            if name == "unfold":
                edge = [int(v) for v in inp["edge"]] if "edge" in inp else None
                target = str(inp["target"]) if "target" in inp else None
                return self.unfold(edge=edge, target=target)
            if name == "rotate":
                return self.rotate(angle=float(inp["angle"]))
            if name == "flip":
                return self.flip(axis=str(inp["axis"]))

            if name == "save_checkpoint":
                return self.save_checkpoint(
                    frame_n=int(inp["frame_n"]),
                    notes=inp.get("notes"),
                )
            if name == "restore_checkpoint":
                return self.restore_checkpoint(frame_n=int(inp["frame_n"]))
            if name == "get_checkpoint_state":
                return self.get_checkpoint_state(frame_n=int(inp["frame_n"]))
            if name == "generate_checkpoint_overview":
                return self.generate_checkpoint_overview()
            if name == "ask_critic":
                return self.ask_critic(
                    frame_n=int(inp["frame_n"]),
                    action_classes=inp.get("action_classes", inp.get("actions_classes")),
                    fold_anchor=inp.get("fold_anchor"),
                )
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult({
                "success": False,
                "error": f"Bad arguments for tool '{name}': {exc}",
            })
        return ToolResult({
            "success": False,
            "error": f"Unknown tool: '{name}'",
        })
