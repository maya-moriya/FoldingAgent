"""Run state that outlives a single iteration: checkpoints, history log, attempt tree."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foldingagent.controller import OrigamiController

logger = logging.getLogger(__name__)


@dataclass
class FrameHistoryEntry:
    frame_index: int
    geometry: dict[str, Any]    # raw_representation (same as get_current_state returns)
    status: str                 # "Resolved" | "Pending Update"
    notes: str | None = None


class HistoryLog:
    """Maintains a structured JSON record of every saved frame checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[int, FrameHistoryEntry] = {}
        self.current_frame: int | None = None
        self.history: list[dict[str, Any]] = []

    def update(self, frame_n: int, controller: "OrigamiController", notes: str | None = None) -> None:
        self.entries[frame_n] = FrameHistoryEntry(
            frame_index=frame_n,
            geometry=_get_frame_geometry(controller, frame_n),
            status="Resolved",
            notes=notes or None,
        )
        self.current_frame = frame_n + 1
        for fn in list(self.entries.keys()):
            if fn > frame_n and self.entries[fn].status == "Resolved":
                e = self.entries[fn]
                self.entries[fn] = FrameHistoryEntry(
                    frame_index=fn, geometry=e.geometry,
                    status="Pending Update", notes=e.notes,
                )
        self._save()

    def log_attempt(
        self,
        frame_n: int,
        attempt: int,
        operations: list[str],
        verdict: str | None,
        reasoning: str | None,
        discrepancies: list[str] | None = None,
    ) -> None:
        """Append a chronological record of one ask_critic call."""
        entry: dict[str, Any] = {
            "event": "attempt",
            "frame": frame_n,
            "attempt": attempt,
            "operations": operations,
            "verdict": verdict,
            "reasoning": _truncate(reasoning),
        }
        if discrepancies:
            entry["discrepancies"] = [_truncate(d, 150) for d in discrepancies]
        self.history.append(entry)
        self._save()

    def log_rollback(self, from_frame: int, to_frame: int) -> None:
        """Append a chronological record of a restore_checkpoint call."""
        self.history.append({"event": "rollback", "from_frame": from_frame, "to_frame": to_frame})
        self._save()

    def load_from_checkpoints(self, controller: "OrigamiController") -> None:
        """Rebuild history from existing checkpoints (all set to 'Resolved')."""
        for frame_n in sorted(controller.checkpoints.keys()):
            self.entries[frame_n] = FrameHistoryEntry(
                frame_index=frame_n,
                geometry=_get_frame_geometry(controller, frame_n),
                status="Resolved",
                notes=None,
            )
        if self.entries:
            self.current_frame = max(self.entries) + 1

    def load_from_file(self) -> bool:
        """Load history from disk. Returns True if successful."""
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.current_frame = data.get("current_frame")
            self.history = data.get("history", [])
            for item in data.get("checkpoints", []):
                fn = int(item["frame_index"])
                self.entries[fn] = FrameHistoryEntry(
                    frame_index=fn,
                    geometry=item.get("geometry", {}),
                    status=item.get("status", "Resolved"),
                    notes=item.get("notes"),
                )
            return True
        except Exception as exc:
            logger.warning("Could not load history_log.json: %s", exc)
            return False

    def to_json_str(self) -> str:
        return json.dumps({
            "current_frame": self.current_frame,
            "checkpoints": [
                {
                    "frame_index": e.frame_index,
                    "status": e.status,
                    "geometry": e.geometry,
                    "notes": e.notes,
                }
                for e in sorted(self.entries.values(), key=lambda x: x.frame_index)
            ],
            "history": self.history,
        }, indent=2)

    def _save(self) -> None:
        self.path.write_text(self.to_json_str(), encoding="utf-8")


def _get_frame_geometry(controller: "OrigamiController", frame_n: int) -> dict[str, Any]:
    """Return the raw_representation dict for a checkpointed frame (same as get_current_state)."""
    geom = controller.checkpoints.get(frame_n)
    if geom is None:
        return {}
    try:
        mat = controller._ensure_raw(geom)
        return mat.raw_representation or {}
    except Exception:
        return {}



# ── Attempt Tree ─────────────────────────────────────────────────────────────

class AttemptTree:
    """Records the full tree of attempts during a run_agent execution.

    Each node represents a paper state snapshot. Edges are parent→child
    relationships formed by actions between checkpoints.
    """

    _ACTION_TOOLS = frozenset({"fold", "unfold", "rotate", "flip", "add_vertex"})

    def __init__(self, path: Path) -> None:
        self.path = path
        self.nodes: dict[str, dict[str, Any]] = {}
        self.root_id: str | None = None
        self._current_parent_id: str | None = None
        # The node currently being built; closed by save/restore_checkpoint.
        self._open_node_id: str | None = None
        # frame_n → node_id for the committed checkpoint at that frame
        self._checkpoint_node: dict[int, str] = {}
        self._next_id: int = 0

    # ── helpers ────────────────────────────────────────────────────────────

    def _new_id(self) -> str:
        nid = f"n{self._next_id}"
        self._next_id += 1
        return nid

    def _make_node(self, frame_index: int) -> str:
        nid = self._new_id()
        self.nodes[nid] = {
            "id": nid,
            "frame_index": frame_index,
            "parent": self._current_parent_id,
            "children": [],
            "actions": [],
            "reasoning": None,
            "geometry": None,
            "render_path": None,
            "critic_grid_path": None,
            "critic_query": None,
            "critic_verdict": None,
            "critic_analysis": None,
            "critic_discrepancies": None,
        }
        if self._current_parent_id is not None and self._current_parent_id in self.nodes:
            self.nodes[self._current_parent_id]["children"].append(nid)
        return nid

    def _ensure_open_node(self, target_frame: int) -> str:
        if self._open_node_id is None:
            self._open_node_id = self._make_node(target_frame)
        return self._open_node_id

    def _close_node(self, controller: "OrigamiController") -> str | None:
        """Close the open node by saving its result geometry. Returns the closed node id."""
        nid = self._open_node_id
        if nid is not None:
            node = self.nodes[nid]
            node["geometry"] = self._capture_geometry(controller)
            self._open_node_id = None
        return nid

    def _capture_geometry(self, controller: "OrigamiController") -> dict[str, Any] | None:
        try:
            geom = controller._ensure_raw(controller.current_geometry)
            return geom.raw_representation
        except Exception:
            return None

    # ── public hooks (called from the main loop) ──────────────────────────

    def on_save_checkpoint(self, frame_n: int, controller: "OrigamiController") -> None:
        """Close the open node, mark it as the checkpoint for frame_n."""
        closed_id = self._close_node(controller)
        if closed_id is not None:
            committed_id = closed_id
            self.nodes[committed_id]["frame_index"] = frame_n
            self.nodes[committed_id]["geometry"] = self._capture_geometry(controller)
        else:
            committed_id = self._make_node(frame_n)
            self.nodes[committed_id]["geometry"] = self._capture_geometry(controller)

        self._checkpoint_node[frame_n] = committed_id
        self._current_parent_id = committed_id

        if self.root_id is None:
            self.root_id = committed_id

        self._save()

    def on_restore_checkpoint(self, frame_n: int, controller: "OrigamiController") -> None:
        """Close the open node (failed attempt), branch from the restored checkpoint."""
        closed_id = self._close_node(controller)

        if closed_id is not None:
            self._ensure_node_render(closed_id, controller)

        if frame_n in self._checkpoint_node:
            self._current_parent_id = self._checkpoint_node[frame_n]
        self._save()

    def on_action(self, tool_name: str, tool_input: dict[str, Any],
                  result_data: dict[str, Any], target_frame: int | None) -> None:
        """Called after a geometry-mutating tool (fold, unfold, etc.)."""
        if tool_name not in self._ACTION_TOOLS:
            return
        tf = target_frame or self._guess_target_frame()
        nid = self._ensure_open_node(tf)
        action_desc = f"{tool_name}({json.dumps(tool_input, separators=(',', ':'))})"
        self.nodes[nid]["actions"].append(action_desc)
        self._save()

    def on_render(self, render_path: str, target_frame: int | None) -> None:
        """Called after a successful render_current."""
        if self._open_node_id is None:
            return
        self.nodes[self._open_node_id]["render_path"] = render_path
        self._save()

    def on_ask_critic(self, frame_n: int, result_data: dict[str, Any],
                      controller: "OrigamiController") -> None:
        """Save critic results on the open node. Node stays open until save/restore."""
        nid = self._ensure_open_node(frame_n)
        node = self.nodes[nid]
        node["frame_index"] = frame_n
        node["geometry"] = self._capture_geometry(controller)
        node["critic_grid_path"] = result_data.get("grid_path")
        node["critic_verdict"] = result_data.get("critic_verdict")
        node["critic_analysis"] = result_data.get("critic_analysis")
        node["critic_discrepancies"] = result_data.get("critic_discrepancies")
        node["reasoning"] = result_data.get("critic_analysis")
        self._save()

    def _ensure_node_render(self, node_id: str, controller: "OrigamiController") -> None:
        """Render the node's geometry if no render_path exists yet."""
        node = self.nodes.get(node_id)
        if node is None or node.get("render_path"):
            return
        geom = node.get("geometry")
        if geom is None:
            return
        try:
            from foldingagent.simulator import GeometryState
            from foldingagent.rendering import render_geometry_diagram

            state = GeometryState(raw_representation=geom)
            materialized = controller._ensure_raw(state)
            # Beside the run's other figures, not beside attempt_tree.json.
            figs = self.path.parent.parent / "figs"
            figs.mkdir(parents=True, exist_ok=True)
            render_path = figs / f"attempt_{node_id}_render.png"
            render_geometry_diagram(
                materialized.raw_representation,
                render_path,
                paper_colors=controller._paper_colors,
            )
            node["render_path"] = str(render_path)
        except Exception as exc:
            logger.warning("Could not render node %s: %s", node_id, exc)

    def _guess_target_frame(self) -> int:
        if self._current_parent_id is not None:
            return self.nodes[self._current_parent_id]["frame_index"] + 1
        return 1

    # ── persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        data = {
            "root": self.root_id,
            "current_parent": self._current_parent_id,
            "nodes": self.nodes,
            "_checkpoint_node": self._checkpoint_node,
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_from_file(self) -> bool:
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.root_id = data.get("root")
            self._current_parent_id = data.get("current_parent")
            self.nodes = data.get("nodes", {})
            self._next_id = 0
            for nid in self.nodes:
                if nid.startswith("n"):
                    try:
                        self._next_id = max(self._next_id, int(nid[1:]) + 1)
                    except ValueError:
                        pass
            self._checkpoint_node = data.get("_checkpoint_node", {})
            self._open_node_id = None
            return True
        except Exception as exc:
            logger.warning("Could not load attempt_tree.json: %s", exc)
            return False

    def get_attempts_for_frame(self, frame_index: int) -> list[str]:
        """Return all node ids whose frame_index matches *frame_index*."""
        return [
            nid for nid, node in self.nodes.items()
            if node["frame_index"] == frame_index
        ]

    def get_path_to_root(self, node_id: str) -> list[str]:
        """Return the ordered path from the root to *node_id* (inclusive)."""
        path: list[str] = []
        cur = node_id
        while cur is not None:
            path.append(cur)
            cur = self.nodes[cur]["parent"]
        path.reverse()
        return path


    def init_root_from_checkpoints(self, controller: "OrigamiController") -> None:
        """Bootstrap tree from existing checkpoints (e.g. on resume)."""
        if not controller.checkpoints:
            return
        parent_id = None
        for fn in sorted(controller.checkpoints):
            nid = self._new_id()
            self.nodes[nid] = {
                "id": nid,
                "frame_index": fn,
                "parent": parent_id,
                "children": [],
                "actions": [],
                "reasoning": None,
                "geometry": _get_frame_geometry(controller, fn),
                "render_path": None,
                "critic_grid_path": None,
                "critic_query": None,
                "critic_verdict": None,
                "critic_analysis": None,
                "critic_discrepancies": None,
            }
            if parent_id is not None:
                self.nodes[parent_id]["children"].append(nid)
            self._checkpoint_node[fn] = nid
            parent_id = nid
        self.root_id = list(self.nodes.keys())[0] if self.nodes else None
        self._current_parent_id = parent_id
        self._save()


def _get_all_frame_operations(controller: "OrigamiController", frame_n: int) -> list[str]:
    """Return the full cumulative operation list for a checkpointed frame."""
    geom = controller.checkpoints.get(frame_n)
    if geom is None:
        return []
    try:
        mat = controller._ensure_raw(geom)
        return [op.render() for op in mat.operations]
    except Exception:
        try:
            return [op.render() for op in geom.operations]
        except Exception:
            return []


def _get_pending_operations(controller: "OrigamiController", frame_n: int) -> list[str]:
    """Return the ops applied to current_geometry since the source checkpoint for frame_n."""
    source_frame_n = frame_n - 1
    try:
        mat = controller._ensure_raw(controller.current_geometry)
        current_ops = [op.render() for op in mat.operations]
    except Exception:
        current_ops = []
    base_ops = _get_all_frame_operations(controller, source_frame_n) if source_frame_n in controller.checkpoints else []
    return current_ops[len(base_ops):]

def _truncate(text: str | None, limit: int = 300) -> str | None:
    """Truncate a string to keep history entries compact."""
    if text is None:
        return None
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"

def _load_checkpoints(run_dir: Path) -> dict[int, Any]:
    """Load checkpoints from a previous run directory, or return {} if not found."""
    from foldingagent.simulator import GeometryState, GeometryFunctionCall, PaperColors as _PC

    cp_path = run_dir / "checkpoints.json"
    if not cp_path.exists():
        logger.warning("No checkpoints.json found in %s", run_dir)
        return {}
    try:
        data = json.loads(cp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read %s: %s", cp_path, exc)
        return {}

    checkpoints: dict[int, Any] = {}
    for key, entry in data.items():
        try:
            geom = GeometryState(
                raw_representation=entry.get("raw_representation"),
                paper_colors=_PC.model_validate(entry["paper_colors"]),
                operations=[
                    GeometryFunctionCall.model_validate(op)
                    for op in entry.get("operations", [])
                ],
            )
            checkpoints[int(key)] = geom
        except Exception as exc:
            logger.warning("Could not reconstruct geometry for frame %s: %s", key, exc)

    return checkpoints

def _save_checkpoints(controller: "OrigamiController", out_dir: Path) -> None:
    """Serialize controller checkpoints to checkpoints.json."""
    data: dict[str, Any] = {}
    for frame_n, geom in sorted(controller.checkpoints.items()):
        try:
            materialized = controller._ensure_raw(geom)
        except RuntimeError:
            materialized = geom
        data[str(frame_n)] = {
            "raw_representation": materialized.raw_representation,
            "paper_colors": materialized.paper_colors.model_dump(mode="json"),
            "operations": [op.model_dump(mode="json") for op in materialized.operations],
        }
    output_path = out_dir / "checkpoints.json"
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Checkpoints saved → {output_path}")
