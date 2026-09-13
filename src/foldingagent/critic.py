"""Visual critic: builds the 2x2 comparison grid and asks a model to judge a fold."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from foldingagent import config
from foldingagent.backends.base import (
    _load_dotenv_into_env,
    _mime_type_for_path,
    _provider_for_model,
    _resolve_api_key,
)
from foldingagent.logger import _save_critic_call
from foldingagent.prompt_assembler import build_critic_prompt

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = config.MAX_OUTPUT_TOKENS


class CriticVerdict(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    EXTREME_DIVERGANCE = "EXTREME DIVERGANCE"


@dataclass
class FoldAnchor:
    moving_element: str
    static_reference: str
    relationship: str


@dataclass
class CriticResponse:
    verdict: CriticVerdict
    analysis: str
    raw_verdict: str = ""
    anchor_check: str = "no anchor provided"
    discrepancies: list[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class CriticRequest:
    image_order: list[tuple[str, Path]]
    final_text: str
    model: str
    system_prompt: str

    def iter_content_blocks(self) -> list[tuple[str, str | Path]]:
        """
        Build the ordered list of content blocks for this request.

        Each block is a ``("text", str)`` or ``("image", Path)`` tuple: the
        A/B/C/D case images followed by the final instructions.
        """
        blocks: list[tuple[str, str | Path]] = []

        for label, path in self.image_order:
            blocks.append(("text", f"[Image {label}]"))
            blocks.append(("image", path))
        blocks.append(("text", self.final_text))
        return blocks

    def as_serializable(self) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        for kind, value in self.iter_content_blocks():
            if kind == "text":
                contents.append({"type": "text", "text": value})
            else:
                path = value
                contents.append(
                    {
                        "type": "image",
                        "path": str(path),
                        "mime_type": _mime_type_for_path(path),
                    }
                )
        return {
            "model": self.model,
            "contents": [{"role": "user", "parts": contents}],
            "config": {
                "system_instruction": self.system_prompt,
            },
        }


class OrigamiCritic:
    """
    Stateless visual critic for origami fold verification.

    Each call to ``evaluate`` is an independent single-turn request — the critic
    has no memory of previous evaluations.
    """

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or build_critic_prompt()
        self.provider = _provider_for_model(model)
        self.log_dir: Path | None = None
        _load_dotenv_into_env()

        if self.provider == "anthropic":
            try:
                import anthropic as _anthropic  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "Install the Anthropic SDK:  pip install anthropic"
                ) from exc
            _api_key = _resolve_api_key("anthropic")
            self._client = _anthropic.Anthropic(api_key=_api_key)
        elif self.provider == "openai":
            try:
                import openai as _openai  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "Install the OpenAI SDK:  pip install openai"
                ) from exc
            _api_key = _resolve_api_key("openai")
            self._client = _openai.OpenAI(api_key=_api_key)
        else:
            try:
                from google import genai as _genai  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "Install the Google GenAI SDK:  pip install google-genai"
                ) from exc

            _api_key = _resolve_api_key("gemini")
            self._client = _genai.Client(api_key=_api_key)

    def evaluate(
        self,
        source_photo: Path,
        target_photo: Path,
        source_diagram: Path,
        result_diagram: Path,
        frame_n: int,
        action_classes: list[str] | None = None,
        source_state_json: dict[str, Any] | None = None,
        result_state_json: dict[str, Any] | None = None,
        fold_anchor: FoldAnchor | None = None,
    ) -> CriticResponse:
        """
        Compare the four origami images and return a structured verdict.

        Images are sent to the critic model in order A, B, C, D — matching the
        critic prompt. If ``fold_anchor`` is provided, the critic will explicitly
        verify the anchor relationship in D and downgrade to MISMATCH if it is
        not satisfied.
        """
        request = self.build_request(
            source_photo=source_photo,
            target_photo=target_photo,
            source_diagram=source_diagram,
            result_diagram=result_diagram,
            frame_n=frame_n,
            action_classes=action_classes,
            source_state_json=source_state_json,
            result_state_json=result_state_json,
            fold_anchor=fold_anchor,
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
                {"raw_text": raw_text}, f"critic_frame{frame_n:02d}",
            )

        return self._parse_response(raw_text, frame_n)

    def _call_gemini(self, request: CriticRequest) -> str:
        from google.genai import types as gtypes  # type: ignore[import]

        parts: list[Any] = []
        for kind, value in request.iter_content_blocks():
            if kind == "text":
                parts.append(gtypes.Part.from_text(text=value))
            else:
                path = value
                try:
                    parts.append(
                        gtypes.Part.from_bytes(data=path.read_bytes(), mime_type=_mime_type_for_path(path))
                    )
                except OSError as exc:
                    raise RuntimeError(f"Cannot read critic image {path}: {exc}") from exc

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
                path = value
                try:
                    image_bytes = path.read_bytes()
                except OSError as exc:
                    raise RuntimeError(f"Cannot read critic image {path}: {exc}") from exc
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _mime_type_for_path(path),
                            "data": base64.standard_b64encode(image_bytes).decode(),
                        },
                    }
                )

        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": content}],
        }

        response = self._client.messages.create(**kwargs)

        raw_text = ""
        if response is not None:
            for block in response.content:
                if getattr(block, "type", "") == "text" and getattr(block, "text", None):
                    raw_text += block.text
        return raw_text

    def _call_openai(self, request: CriticRequest) -> str:
        content: list[dict[str, Any]] = []
        for kind, value in request.iter_content_blocks():
            if kind == "text":
                content.append({"type": "text", "text": value})
            else:
                path = value
                try:
                    image_bytes = path.read_bytes()
                except OSError as exc:
                    raise RuntimeError(f"Cannot read critic image {path}: {exc}") from exc
                b64 = base64.standard_b64encode(image_bytes).decode()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{_mime_type_for_path(path)};base64,{b64}"},
                    }
                )

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": content},
        ]

        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_completion_tokens": _MAX_OUTPUT_TOKENS,
            "messages": messages,
        }

        response = self._client.chat.completions.create(**kwargs)

        raw_text = ""
        if response is not None and response.choices:
            message_content = response.choices[0].message.content
            if message_content:
                raw_text = message_content
        return raw_text

    def build_request(
        self,
        source_photo: Path,
        target_photo: Path,
        source_diagram: Path,
        result_diagram: Path,
        frame_n: int,
        action_classes: list[str] | None = None,
        source_state_json: dict[str, Any] | None = None,
        result_state_json: dict[str, Any] | None = None,
        fold_anchor: FoldAnchor | None = None,
    ) -> CriticRequest:
        image_order: list[tuple[str, Path]] = [
            ("A — source real photo", source_photo),
            ("B — target real photo", target_photo),
            ("C — source diagram", source_diagram),
            ("D — result diagram", result_diagram),
        ]

        final_text = f"Target frame: {frame_n}."
        if action_classes:
            rendered_action_classes = ", ".join(action_classes)
            final_text += f"\nAction classes for this transition: [{rendered_action_classes}]"
        if source_state_json is not None:
            final_text += (
                "\n\nSOURCE STATE JSON (the mesh that matches A/C):\n"
                + json.dumps(source_state_json, indent=2, sort_keys=True, ensure_ascii=True)
            )
        if result_state_json is not None:
            final_text += (
                "\n\nRESULT STATE JSON (the mesh rendered as D):\n"
                + json.dumps(result_state_json, indent=2, sort_keys=True, ensure_ascii=True)
            )
        if fold_anchor is not None:
            final_text += (
                f"\n\nFOLD ANCHOR to verify:"
                f"\n  Moving element  : {fold_anchor.moving_element}"
                f"\n  Static reference: {fold_anchor.static_reference}"
                f"\n  Expected relationship in B (and D): {fold_anchor.relationship}"
                f"\n\nCheck this anchor explicitly in your anchor_check field."
            )
        final_text += "\n\nCompare image D against image B and return a JSON verdict."

        return CriticRequest(
            image_order=image_order,
            final_text=final_text,
            model=self.model,
            system_prompt=self.system_prompt,
        )

    def _parse_response(self, raw_text: str, frame_n: int) -> CriticResponse:
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(text)
            raw_verdict = str(data.get("verdict", "MISMATCH")).strip().upper()
            verdict_str = raw_verdict
            verdict_aliases = {
                "EXACT_MATCH": CriticVerdict.MATCH.value,
                "EXTREME_DIVERGENCE": CriticVerdict.EXTREME_DIVERGANCE.value,
                "EXTREME_DIVERGANCE": CriticVerdict.EXTREME_DIVERGANCE.value,
            }
            verdict_str = verdict_aliases.get(verdict_str, verdict_str)
            verdict = (
                CriticVerdict(verdict_str)
                if verdict_str in CriticVerdict._value2member_map_
                else CriticVerdict.MISMATCH
            )
            response = CriticResponse(
                verdict=verdict,
                analysis=str(data.get("analysis", raw_text)),
                raw_verdict=raw_verdict,
                anchor_check=str(data.get("anchor_check", "no anchor provided")),
                discrepancies=[str(d) for d in data.get("discrepancies", [])],
                raw_response=raw_text,
            )
            return self._apply_match_sanity_checks(response)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Critic JSON parse failed for frame %d: %s", frame_n, exc)
            return CriticResponse(
                verdict=CriticVerdict.MISMATCH,
                analysis=raw_text or "Critic returned no response.",
                raw_verdict="MISMATCH",
                anchor_check="no anchor provided",
                discrepancies=["Could not parse critic response as JSON."],
                raw_response=raw_text,
            )

    @staticmethod
    def _apply_match_sanity_checks(response: CriticResponse) -> CriticResponse:
        """
        Downgrade internally inconsistent MATCH verdicts.

        The main failure mode we guard against is when the model explicitly notes
        that photo B contains a visible gap/separation, but still approves a D
        diagram whose own description says the corresponding edges are coincident
        or perfectly touching.
        """
        if response.verdict != CriticVerdict.MATCH:
            return response

        analysis = " ".join(response.analysis.lower().split())
        anchor_check = " ".join(response.anchor_check.lower().split())

        gap_markers = (
            "small gap",
            "visible gap",
            "narrow gap",
            "gap between",
            "separation between",
            "space between",
            "slit between",
            "do not touch",
            "does not touch",
            "not touching",
            "offset between",
        )
        target_markers = ("photo b", "b shows", "in b", "target")
        closed_markers = (
            "coincident",
            "touching",
            "touches exactly",
            "meet exactly",
            "perfect alignment",
            "perfectly aligned",
            "closed seam",
        )
        idealization_markers = (
            "idealized",
            "idealised",
            "mathematical transformation",
            "common in real paper folding",
        )

        mentions_gap_in_target = any(marker in analysis for marker in gap_markers) and (
            any(marker in analysis for marker in target_markers)
            or any(marker in analysis for marker in idealization_markers)
        )
        claims_closed_contact = any(marker in analysis for marker in closed_markers) or any(
            marker in anchor_check for marker in closed_markers
        )

        if not (mentions_gap_in_target and claims_closed_contact):
            return response

        discrepancies = list(response.discrepancies)
        sanity_discrepancy = (
            "Photo B shows a visible gap/separation, but D was described as closed or coincident."
        )
        if sanity_discrepancy not in discrepancies:
            discrepancies.append(sanity_discrepancy)

        analysis_text = response.analysis.strip()
        if analysis_text:
            analysis_text += " "
        analysis_text += (
            "Downgraded to MISMATCH because visible separations in B must remain visibly separated in D."
        )

        return CriticResponse(
            verdict=CriticVerdict.MISMATCH,
            analysis=analysis_text,
            raw_verdict="MISMATCH",
            anchor_check=response.anchor_check,
            discrepancies=discrepancies,
            raw_response=response.raw_response,
        )


def create_critic_grid(
    source_image: str | Path,
    target_image: str | Path,
    source_diagram: str | Path,
    result_diagram: str | Path,
    output_path: str | Path,
    *,
    cell_width: int | None = None,
    border_width: int = 5,
) -> Path:
    """Create a labeled 2x2 grid with black borders between the quadrants."""
    image_paths = [
        Path(source_image).resolve(),
        Path(target_image).resolve(),
        Path(source_diagram).resolve(),
        Path(result_diagram).resolve(),
    ]
    labels = ["A", "B", "C", "D"]
    destination = Path(output_path).resolve()

    opened_images = [_load_rgb_image(path) for path in image_paths]
    resized_images: list[Image.Image] = []
    cells: list[Image.Image] = []
    try:
        MAX_COMPOSITE_WIDTH = 1024
        max_cell_width = (MAX_COMPOSITE_WIDTH - border_width * 3) // 2
        target_width = min(
            cell_width or max(image.width for image in opened_images),
            max_cell_width,
        )
        resized_images = [
            _resize_to_width(image=image, width=target_width)
            for image in opened_images
        ]
        cell_height = max(image.height for image in resized_images)
        cells = [
            _create_labeled_cell(
                image=image,
                label=label,
                cell_width=target_width,
                cell_height=cell_height,
            )
            for image, label in zip(resized_images, labels)
        ]

        composite = Image.new(
            "RGB",
            (
                target_width * 2 + border_width * 3,
                cell_height * 2 + border_width * 3,
            ),
            "black",
        )
        paste_positions = [
            (border_width, border_width),
            (target_width + border_width * 2, border_width),
            (border_width, cell_height + border_width * 2),
            (
                target_width + border_width * 2,
                cell_height + border_width * 2,
            ),
        ]
        for cell, position in zip(cells, paste_positions):
            composite.paste(cell, position)

        destination.parent.mkdir(parents=True, exist_ok=True)
        composite.save(destination, format="PNG")
    finally:
        for image in cells:
            image.close()
        for image in resized_images:
            image.close()
        for image in opened_images:
            image.close()

    return destination


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGBA", image.size, "white")
            flattened = Image.alpha_composite(background, image.convert("RGBA"))
            return flattened.convert("RGB")
        return image.convert("RGB")


def _resize_to_width(*, image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image.copy()
    scaled_height = max(1, round(image.height * width / image.width))
    return image.resize((width, scaled_height), Image.Resampling.LANCZOS)


def _create_labeled_cell(
    *,
    image: Image.Image,
    label: str,
    cell_width: int,
    cell_height: int,
) -> Image.Image:
    panel = Image.new("RGB", (cell_width, cell_height), "white")
    offset = (
        (cell_width - image.width) // 2,
        (cell_height - image.height) // 2,
    )
    panel.paste(image, offset)

    draw = ImageDraw.Draw(panel)
    font_size = max(24, round(min(cell_width, cell_height) * 0.11))
    font = _load_label_font(font_size)
    label_margin = max(12, round(min(cell_width, cell_height) * 0.04))
    text_bbox = draw.textbbox((0, 0), label, font=font)
    draw.text(
        (
            label_margin - text_bbox[0],
            label_margin - text_bbox[1],
        ),
        label,
        fill="black",
        font=font,
    )
    return panel


def _load_label_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
