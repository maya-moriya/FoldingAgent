"""Geometry → PNG rendering, shared by the agent, the critics and the attempt tree."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from foldingagent.config import DEFAULT_BACK_COLOR, DEFAULT_FRONT_COLOR




class PaperColors(BaseModel):
    """Front/back paper colors used for rendering origami diagrams."""

    front_color: str = DEFAULT_FRONT_COLOR
    back_color: str = DEFAULT_BACK_COLOR

    @field_validator("front_color", "back_color", mode="before")
    @classmethod
    def _normalize_color(cls, value: object, info) -> str:
        default = DEFAULT_FRONT_COLOR if info.field_name == "front_color" else DEFAULT_BACK_COLOR
        if value is None:
            return default
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            return stripped
        return str(value)

    def as_kwargs(self) -> dict[str, str]:
        """Return constructor kwargs for the origami library."""
        return {
            "front_color": self.front_color,
            "back_color": self.back_color,
        }


def _render_geometry_plot(
    geometry_representation: dict[str, Any],
    output_path: Path,
    paper_colors: PaperColors | None = None,
) -> None:
    if output_path.exists():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_cache_dir = Path(tempfile.gettempdir()) / "origamiframework-mpl"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from origami.origami import Origami
    except ImportError as exc:
        raise RuntimeError(
            "Generating the visual summary requires the 'origami' package in the "
            "active environment."
        ) from exc

    resolved_paper_colors = paper_colors or PaperColors()
    paper = Origami(
        geometry_representation,
        **resolved_paper_colors.as_kwargs(),
    )
    paper.plot(show=False, save_path=str(output_path), debug=False)


def render_geometry_diagram(
    geometry_representation: dict[str, Any],
    output_path: str | Path,
    *,
    paper_colors: PaperColors | None = None,
) -> Path:
    """Render a geometry representation to a diagram PNG."""
    destination = Path(output_path).resolve()
    _render_geometry_plot(
        geometry_representation=geometry_representation,
        output_path=destination,
        paper_colors=paper_colors,
    )
    return destination
