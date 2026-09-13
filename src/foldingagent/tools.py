"""The agent's tool schema, authored in Anthropic tool-use format.

This is the canonical definition: AnthropicBackend sends it verbatim, and the
Gemini and OpenAI backends each translate it into their own shape.
"""

from __future__ import annotations

from typing import Any


# ── Tool schema definitions (Anthropic tool-use format) ────────────────────────

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "view_frame",
        "description": (
            "View the real-world origami photo at frame n (1-based). "
            "Returns the image so you can see the physical state of the paper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Frame index (1-based). Frame 1 is the initial flat square.",
                }
            },
            "required": ["n"],
        },
    },
    {
        "name": "observe_movement",
        "description": (
            "For step t>=2, returns a PNG filmstrip showing the motion between "
            "selected frames t-1 and t. Use this when the action between those "
            "frames is ambiguous."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "t": {
                    "type": "integer",
                    "description": (
                        "Selected-frame step index to inspect (1-based). "
                        "Use t>=2 because the strip spans steps t-1 to t."
                    ),
                }
            },
            "required": ["t"],
        },
    },
    {
        "name": "get_current_state",
        "description": (
            "Get the current origami geometry as structured JSON with vertices, edges, "
            "faces, and layers. Use this to inspect vertex IDs before folding."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "render_current",
        "description": (
            "Render the current simulator state as a diagram image. "
            "Use this for a quick visual preview of your progress."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_vertex",
        "description": (
            "Add a new vertex at a fractional position (0.0-1.0) along an existing edge. "
            "Use this when a fold crease endpoint does not coincide with an existing vertex. "
            "Returns new_vertex_id (starting from 5 for the first added vertex) which you "
            "can use immediately in a subsequent fold call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edge": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Edge to subdivide, as [vertex_id_1, vertex_id_2].",
                },
                "position": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Fractional position along the edge: "
                        "0.0 = start vertex, 0.5 = midpoint, 1.0 = end vertex."
                    ),
                },
            },
            "required": ["edge", "position"],
        },
    },
    {
        "name": "fold",
        "description": (
            "Fold the paper along an edge (crease line). "
            "direction = +1 or -1 (which side of the crease moves toward viewer)"
            "Non-vertical crease: +1 = upper side moves, -1 = lower side moves"
            "Vertical crease: +1 = right side moves, -1 = left side moves"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edge": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Crease line as [vertex_id_1, vertex_id_2].",
                },
                "direction": {
                    "type": "integer",
                    "enum": [1, -1],
                    "description": "1=upper side moves, -1=lower side moves.",
                },
            },
            "required": ["edge", "direction"],
        },
    },
    {
        "name": "unfold",
        "description": (
            "Unfold a previous fold, leaving a crease mark (dashed line in diagrams). "
            "Provide either 'edge' (specific crease) or target='last' (most recent fold)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edge": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Specific edge to unfold (optional).",
                },
                "target": {
                    "type": "string",
                    "enum": ["last"],
                    "description": "Use 'last' to unfold the most recent fold.",
                },
            },
        },
    },
    {
        "name": "rotate",
        "description": "Rotate the entire origami model by angle degrees (positive = clockwise).",
        "input_schema": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "number",
                    "description": "Rotation angle in degrees. Positive = clockwise.",
                }
            },
            "required": ["angle"],
        },
    },
    {
        "name": "flip",
        "description": (
            "Turn the model overacross a specific axis, revealing the back side of the paper. "
            "Note: The axis refers to the coordinate being inverted. "
            "axis: 'x' (Reflect horizontally: left and right swap, x -> -x), "
            "axis: 'y' (Reflect vertically: top and bottom swap, y -> -y), "
            "axis: 'y=x' (Reflect across main diagonal), "
            "axis: 'y=-x' (Reflect across anti-diagonal)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "axis": {
                    "type": "string",
                    "enum": ["x", "y", "y=x", "y=-x"],
                    "description": "Flip axis.",
                }
            },
            "required": ["axis"],
        },
    },
    {
        "name": "save_checkpoint",
        "description": (
            "Save the current geometry as the verified solution for frame_n. "
            "Call this after ask_critic returns MATCH. "
            "Use the optional 'notes' field to record anything important about this checkpoint "
            "(it will appear in the history log)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "frame_n": {
                    "type": "integer",
                    "description": "Frame index to associate with the current state.",
                },
                "notes": {
                    "type": "string",
                    "description": (
                        "Optional memory note for this checkpoint. "
                        "Record anything you want to remember about this frame for future reference."
                    ),
                },
            },
            "required": ["frame_n"],
        },
    },
    {
        "name": "restore_checkpoint",
        "description": (
            "Restore the simulator to a previously saved checkpoint. "
            "Use this to backtrack when an attempt fails. "
            "The tool result includes restore_count: the total number of times this checkpoint "
            "has been restored across the entire session (never resets). If restore_count is high (e.g. >3), "
            "your current approach is not working — call generate_checkpoint_overview(), restore the last "
            "known-good checkpoin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "frame_n": {
                    "type": "integer",
                    "description": "Frame checkpoint to restore.",
                }
            },
            "required": ["frame_n"],
        },
    },
    {
        "name": "get_checkpoint_state",
        "description": (
            "Get the saved geometry JSON for a specific checkpointed frame without changing the current simulator state. "
            "Use this to inspect what a checkpoint looks like (vertices, edges, faces, layers) before deciding to restore it. "
            "Unlike restore_checkpoint, this does NOT change the current state — it is read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "frame_n": {
                    "type": "integer",
                    "description": "Frame index of the checkpoint to inspect.",
                }
            },
            "required": ["frame_n"],
        },
    },
    {
        "name": "generate_checkpoint_overview",
        "description": (
            "Generate a two-row overview PNG of ALL saved checkpoints, then ask a dedicated overview "
            "specialist to identify the last frame where the diagram still matches the real photo. "
            "Row 1 shows the real-world photos; row 2 shows the rendered diagrams. "
            "Columns are ordered left (earliest) → right (latest) with frame indices labeled. "
            "The tool returns overview_analysis with last_correct_frame and first_suspicious_frame, "
            "and overview_recommendation with a suggested rollback target. "
            "Use this whenever you need to decide which checkpoint to roll back to."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_critic",
        "description": (
            "Ask the dedicated visual critic to evaluate your current simulator state "
            "against target frame_n. "
            "The critic builds a 2x2 grid (A=source photo, B=target photo, C=source diagram, "
            "D=your result) and returns a structured verdict: "
            "MATCH, MISMATCH, or EXTREME DIVERGANCE — together with a prose analysis, an "
            "anchor_check sentence, and a list of specific discrepancies. "
            "Always pass action_classes, such as ['fold'] or ['rotate', 'flip'], so the critic "
            "understands the intended transition. For folds, pass fold_anchor "
            "(moving_element, static_reference, relationship) so the critic can explicitly "
            "verify the anchor relationship in D. "
            "Trust the critic's verdict: MATCH → save_checkpoint; "
            "MISMATCH → restore_checkpoint and retry; "
            "EXTREME DIVERGANCE → treat it as a false assumption and reconsider earlier steps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "frame_n": {
                    "type": "integer",
                    "description": "The target frame index you are trying to reach.",
                },
                "action_classes": {
                    "type": "array",
                    "description": (
                        "High-level action classes for this transition. Always include all of them, "
                        "for example ['fold'], ['rotate'], or ['rotate', 'flip']."
                    ),
                    "items": {
                        "type": "string",
                        "enum": ["fold", "unfold", "rotate", "flip"],
                    },
                    "minItems": 1,
                },
                "fold_anchor": {
                    "type": "object",
                    "description": (
                        "Required when action_classes includes 'fold'. Describe the anchor you established "
                        "while inspecting the transition. The critic will explicitly check this relationship in D."
                    ),
                    "properties": {
                        "moving_element": {
                            "type": "string",
                            "description": "The part of the paper that moved (e.g. 'upper-left triangular flap').",
                        },
                        "static_reference": {
                            "type": "string",
                            "description": "The part that stayed fixed (e.g. 'bottom-right region').",
                        },
                        "relationship": {
                            "type": "string",
                            "description": (
                                "The geometric relationship that should hold between them in B and D "
                                "(e.g. 'top edge of moving element aligns with the right boundary edge')."
                            ),
                        },
                    },
                    "required": ["moving_element", "static_reference", "relationship"],
                },
            },
            "required": ["frame_n", "action_classes"],
        },
    },
]
