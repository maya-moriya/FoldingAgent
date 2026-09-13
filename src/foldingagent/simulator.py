"""Low-level execution engine: applies fold/unfold operations to the origami mesh."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from origami.origami import Origami
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from foldingagent.rendering import PaperColors


# ── The geometry model ────────────────────────────────────────────────────────



def normalize_origami_representation(
    representation: object,
) -> dict[str, Any] | None:
    """Normalize an origami representation from a dict or JSON object string."""
    if representation is None:
        return None

    if isinstance(representation, dict):
        return json.loads(json.dumps(representation, ensure_ascii=True))

    if isinstance(representation, str):
        stripped = representation.strip()
        if not stripped:
            return None

        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("Origami geometry JSON must be an object.")
        return normalize_origami_representation(parsed)

    raise TypeError(
        f"Unsupported origami representation type: {type(representation).__name__}."
    )




def render_origami_representation(
    representation: dict[str, Any] | None,
) -> str:
    """Render an origami representation to text suitable for prompts and logs."""
    if representation is None:
        return ""
    return _render_compact_json(representation)




def _render_compact_json(
    value: Any,
    *,
    indent: int = 2,
    level: int = 0,
) -> str:
    """Render JSON with readable objects and compact arrays."""
    if isinstance(value, dict):
        if not value:
            return "{}"

        padding = " " * (level * indent)
        child_padding = " " * ((level + 1) * indent)
        rendered_items = [
            (
                f"{child_padding}{json.dumps(key, ensure_ascii=True)}: "
                f"{_render_compact_json(value[key], indent=indent, level=level + 1)}"
            )
            for key in sorted(value)
        ]
        return "{\n" + ",\n".join(rendered_items) + "\n" + padding + "}"

    if isinstance(value, list):
        if not value:
            return "[]"
        rendered_items = [
            _render_compact_json(item, indent=indent, level=0) for item in value
        ]
        return "[" + ", ".join(rendered_items) + "]"

    return json.dumps(value, ensure_ascii=True)




class GeometryFunctionCall(BaseModel):
    """Structured geometric function call."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    name: str = Field(
        validation_alias=AliasChoices("function", "name"),
        serialization_alias="function",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("params", "parameters"),
        serialization_alias="params",
    )
    assign_to: str | None = None

    _binding_name_pattern = re.compile(r"^\$?[A-Za-z_][A-Za-z0-9_]*$")

    @field_validator("assign_to", mode="before")
    @classmethod
    def _normalize_numeric_assign_to(cls, value: object) -> object:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int):
            return None
        if isinstance(value, float) and value.is_integer():
            return None
        if isinstance(value, str) and value.strip().isdigit():
            return None
        return value

    @model_validator(mode="after")
    def _validate_assign_to(self) -> "GeometryFunctionCall":
        if self.assign_to and not self._binding_name_pattern.match(self.assign_to):
            raise ValueError(
                "assign_to must be an identifier such as 'mid_14' or '$V1'."
            )
        return self

    def render(self) -> str:
        """Render a canonical text representation for comparison and logging."""
        if not self.parameters:
            rendered_call = f"{self.name}()"
            return f"{self.assign_to} = {rendered_call}" if self.assign_to else rendered_call

        rendered_params = ",".join(
            f"{key}={self.parameters[key]}" for key in sorted(self.parameters)
        )
        rendered_call = f"{self.name}({rendered_params})"
        return f"{self.assign_to} = {rendered_call}" if self.assign_to else rendered_call




class GeometryState(BaseModel):
    """Current origami geometry represented as a base shape plus operations."""

    base_description: str = "square sheet"
    raw_representation: dict[str, Any] | None = None
    operations: list[GeometryFunctionCall] = Field(default_factory=list)
    paper_colors: PaperColors = Field(default_factory=PaperColors)

    @model_validator(mode="after")
    def _normalize_raw_representation(self) -> "GeometryState":
        self.raw_representation = normalize_origami_representation(self.raw_representation)
        return self

    @computed_field
    @property
    def description(self) -> str:
        """Human-readable geometry description."""
        if self.raw_representation:
            rendered_representation = render_origami_representation(
                self.raw_representation
            )
            if not self.operations:
                return rendered_representation

            rendered_steps = "\n".join(
                f"{index + 1}. {operation.render()}"
                for index, operation in enumerate(self.operations)
            )
            return (
                f"{rendered_representation}\n\n"
                "Applied operations:\n"
                f"{rendered_steps}"
            )

        if not self.operations:
            return self.base_description

        rendered_steps = " -> ".join(operation.render() for operation in self.operations)
        return f"{self.base_description} -> {rendered_steps}"


# ── The engine's own signature ────────────────────────────────────────────────

class ParsedActionPlan(BaseModel):
    """Structured execution plan produced by the parser."""

    functions: list[GeometryFunctionCall] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Outcome of attempting to apply a parsed action plan."""

    success: bool
    updated_geometry: GeometryState | None = None
    error_message: str | None = None
    applied_functions: list[GeometryFunctionCall] = Field(default_factory=list)


class OrigamiExecutionEngine(ABC):
    """Abstract interface for applying geometric functions."""

    @abstractmethod
    def execute(
        self,
        plan: ParsedActionPlan,
        current_geometry: GeometryState,
        ) -> ExecutionResult:
        """Attempt to apply the action plan to the current geometry."""


class OrigamiLibraryExecutionEngine(OrigamiExecutionEngine):
    """Execution engine backed by the external origami simulator library."""

    def __init__(self, paper_colors_override: PaperColors | None = None) -> None:
        self._paper_colors_override = (
            paper_colors_override.model_copy(deep=True)
            if paper_colors_override is not None
            else None
        )

    def execute(
        self,
        plan: ParsedActionPlan,
        current_geometry: GeometryState,
    ) -> ExecutionResult:
        if not plan.functions:
            return ExecutionResult(
                success=False,
                error_message="No functions were provided for execution.",
            )

        resolved_paper_colors = self._paper_colors_override or current_geometry.paper_colors
        materialized_origami, load_error = self._materialize_origami(
            current_geometry,
            paper_colors=resolved_paper_colors,
        )
        if load_error:
            return ExecutionResult(success=False, error_message=load_error)

        next_operations = list(current_geometry.operations)
        applied_functions: list[GeometryFunctionCall] = []
        symbol_table: dict[str, object] = {}

        for function_call in plan.functions:
            resolved_call, resolution_error = self._resolve_symbols(
                function_call,
                symbol_table,
            )
            if resolution_error:
                return ExecutionResult(
                    success=False,
                    error_message=resolution_error,
                    applied_functions=applied_functions,
                )

            validation_error = self._validate(resolved_call)
            if validation_error:
                return ExecutionResult(
                    success=False,
                    error_message=validation_error,
                    applied_functions=applied_functions,
                )

            execution_result, execution_error = self._apply(
                materialized_origami,
                resolved_call,
                next_operations,
            )
            if execution_error:
                return ExecutionResult(
                    success=False,
                    error_message=execution_error,
                    applied_functions=applied_functions,
                )

            if resolved_call.assign_to:
                if resolved_call.assign_to in symbol_table:
                    return ExecutionResult(
                        success=False,
                        error_message=(
                            f"Symbol '{resolved_call.assign_to}' was assigned more "
                            "than once in the same plan."
                        ),
                        applied_functions=applied_functions,
                    )
                if execution_result is None:
                    return ExecutionResult(
                        success=False,
                        error_message=(
                            f"Function '{resolved_call.name}' does not return a value "
                            f"for assignment to '{resolved_call.assign_to}'."
                        ),
                        applied_functions=applied_functions,
                    )
                symbol_table[resolved_call.assign_to] = execution_result

            applied_functions.append(resolved_call)

        updated_geometry = GeometryState(
            base_description=current_geometry.base_description,
            raw_representation=self._serialize_origami(materialized_origami),
            operations=next_operations,
            paper_colors=resolved_paper_colors.model_copy(deep=True),
        )
        return ExecutionResult(
            success=True,
            updated_geometry=updated_geometry,
            applied_functions=applied_functions,
        )

    @staticmethod
    def _materialize_origami(
        current_geometry: GeometryState,
        *,
        paper_colors: PaperColors,
    ) -> tuple[Origami | None, str | None]:
        if current_geometry.raw_representation:
            try:
                loaded = Origami(
                    current_geometry.raw_representation,
                    **paper_colors.as_kwargs(),
                )
            except Exception as exc:
                return None, f"Failed to load origami structure from the previous round: {exc}"
            return loaded, None

        materialized_origami = Origami(**paper_colors.as_kwargs())
        for function_call in current_geometry.operations:
            _, execution_error = OrigamiLibraryExecutionEngine._apply(
                materialized_origami,
                function_call,
                None,
            )
            if execution_error:
                return (
                    None,
                    "Failed to reconstruct origami structure from prior operations: "
                    f"{execution_error}",
                )
        return materialized_origami, None

    @staticmethod
    def _validate(function_call: GeometryFunctionCall) -> str | None:
        name = function_call.name
        parameters = function_call.parameters

        if name == "fold":
            required = {"edge"}
            if "direction" not in parameters and "side" not in parameters:
                return f"Function '{name}' is missing parameters: direction."
        elif name == "add_vertex":
            required = {"edge", "position"}
        elif name == "unfold":
            if "edge" in parameters:
                required = {"edge"}
            elif "target" in parameters:
                required = {"target"}
            else:
                return "Function 'unfold' is missing parameters: edge."
        elif name == "rotate":
            if "angle" in parameters:
                required = {"angle"}
            else:
                required = {"axis", "degrees"}
        elif name == "flip":
            required = {"axis"}
        else:
            return f"Unsupported function '{name}'."

        missing = sorted(required - parameters.keys())
        if missing:
            missing_parameters = ", ".join(missing)
            return f"Function '{name}' is missing parameters: {missing_parameters}."

        return None

    @staticmethod
    def _apply(
        materialized_origami: Origami,
        function_call: GeometryFunctionCall,
        operations: list[GeometryFunctionCall] | None,
    ) -> tuple[object | None, str | None]:
        execution_result: object | None = None
        try:
            if function_call.name == "add_vertex":
                edge = OrigamiLibraryExecutionEngine._coerce_edge(
                    function_call.parameters.get("edge")
                )
                position = float(function_call.parameters["position"])
                execution_result = materialized_origami.add_vertex(
                    edge=edge,
                    position=position,
                )
            elif function_call.name == "fold":
                edge = OrigamiLibraryExecutionEngine._coerce_edge(
                    function_call.parameters.get("edge")
                )
                direction = OrigamiLibraryExecutionEngine._coerce_direction(
                    function_call.parameters
                )
                execution_result = materialized_origami.fold(edge, direction)
            elif function_call.name == "unfold":
                edge = OrigamiLibraryExecutionEngine._resolve_unfold_edge(
                    function_call,
                    operations,
                )
                if edge is None:
                    return None, "Cannot unfold because there is no previous fold."
                execution_result = materialized_origami.unfold(edge)
            elif function_call.name == "rotate":
                angle = OrigamiLibraryExecutionEngine._coerce_angle(
                    function_call.parameters
                )
                execution_result = materialized_origami.rotate(angle)
            elif function_call.name == "flip":
                axis = OrigamiLibraryExecutionEngine._coerce_flip_axis(
                    function_call.parameters
                )
                execution_result = materialized_origami.flip(axis)
            else:
                return None, f"Unsupported function '{function_call.name}'."
        except Exception as exc:
            return None, (
                f"Origami execution failed for '{function_call.render()}': {exc}"
            )

        if operations is not None:
            operations.append(
                GeometryFunctionCall(
                    name=function_call.name,
                    parameters=function_call.parameters,
                )
            )
        return execution_result, None

    @staticmethod
    def _resolve_symbols(
        function_call: GeometryFunctionCall,
        symbol_table: dict[str, object],
    ) -> tuple[GeometryFunctionCall, str | None]:
        try:
            resolved_parameters = OrigamiLibraryExecutionEngine._resolve_symbolic_value(
                function_call.parameters,
                symbol_table,
            )
        except KeyError as exc:
            placeholder = str(exc.args[0])
            return (
                function_call,
                f"Function '{function_call.name}' references unknown symbol '{placeholder}'.",
            )

        return (
            function_call.model_copy(
                update={"parameters": resolved_parameters},
                deep=True,
            ),
            None,
        )

    @staticmethod
    def _resolve_symbolic_value(
        value: Any,
        symbol_table: dict[str, object],
    ) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            if value not in symbol_table:
                raise KeyError(value)
            return symbol_table[value]
        if isinstance(value, str) and value in symbol_table:
            return symbol_table[value]
        if isinstance(value, list):
            return [
                OrigamiLibraryExecutionEngine._resolve_symbolic_value(item, symbol_table)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                OrigamiLibraryExecutionEngine._resolve_symbolic_value(item, symbol_table)
                for item in value
            )
        if isinstance(value, dict):
            return {
                key: OrigamiLibraryExecutionEngine._resolve_symbolic_value(
                    item,
                    symbol_table,
                )
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _coerce_edge(raw_edge: object) -> tuple[int, int]:
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            raise ValueError("edge must be a pair of vertex ids.")
        return int(raw_edge[0]), int(raw_edge[1])

    @staticmethod
    def _coerce_direction(parameters: dict[str, object]) -> int:
        if "direction" in parameters:
            return int(parameters["direction"])
        if "side" in parameters:
            return int(parameters["side"])
        raise ValueError("direction must be provided for fold.")

    @staticmethod
    def _coerce_angle(parameters: dict[str, object]) -> float:
        if "angle" in parameters:
            return float(parameters["angle"])
        if "degrees" in parameters:
            return float(parameters["degrees"])
        raise ValueError("angle must be provided for rotate.")

    @staticmethod
    def _coerce_flip_axis(parameters: dict[str, object]) -> str:
        if "axis" not in parameters:
            raise ValueError("axis must be provided for flip.")
        axis = str(parameters["axis"]).strip().lower()
        if axis not in {"x", "y", "y=x", "y=-x"}:
            raise ValueError("axis must be one of 'x', 'y', 'y=x', or 'y=-x'.")
        return axis

    @staticmethod
    def _resolve_unfold_edge(
        function_call: GeometryFunctionCall,
        operations: list[GeometryFunctionCall] | None,
    ) -> tuple[int, int] | None:
        if "edge" in function_call.parameters:
            return OrigamiLibraryExecutionEngine._coerce_edge(
                function_call.parameters["edge"]
            )

        if operations is None:
            return None

        target = function_call.parameters.get("target")
        if target not in {"last", "last_fold"}:
            raise ValueError(f"Unsupported unfold target '{target}'.")

        for previous_call in reversed(operations):
            if previous_call.name == "fold":
                return OrigamiLibraryExecutionEngine._coerce_edge(
                    previous_call.parameters.get("edge")
                )

        return None

    @staticmethod
    def _serialize_origami(
        materialized_origami: Origami,
    ) -> dict[str, Any]:
        raw_representation = materialized_origami.export()
        normalized = normalize_origami_representation(raw_representation)
        if normalized is None:
            raise ValueError("Origami.export() returned no geometry.")
        return normalized
