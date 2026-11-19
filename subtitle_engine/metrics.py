"""Deterministic text metrics used by wrapping and layout."""

from __future__ import annotations

import math
import unicodedata

from .spec import Style


def glyph_scale(font_size: float) -> int:
    """Integer bitmap-font scale derived only from font size."""

    return max(1, int(round(float(font_size) / 8.0)))


def advance_width(character: str, font_size: float) -> int:
    scale = glyph_scale(font_size)
    if character.isspace():
        return 3 * scale
    width = unicodedata.east_asian_width(character)
    if width in {"W", "F"}:
        return 12 * scale
    return 6 * scale


def text_width(text: str, font_size: float) -> int:
    return sum(advance_width(character, font_size) for character in text)


def line_pixel_height(style: Style) -> int:
    return max(1, int(round(style.line_height)))


def available_text_width(style: Style, safe_width: int) -> int:
    return int(math.floor(safe_width - 2.0 * style.padding))


def _split_long_token(token: str, font_size: float, max_width: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for character in token:
        candidate = current + character
        if current and text_width(candidate, font_size) > max_width:
            pieces.append(current)
            current = character
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def wrap_text(text: str, style: Style, safe_width: int) -> tuple[list[str], list[int]]:
    """Wrap at word boundaries without dropping text.

    A token wider than the available width is split at character boundaries,
    which is the only possible deterministic fallback.  If the resulting line
    count exceeds ``max_lines`` the caller receives a layout overflow error.
    """

    max_width = available_text_width(style, safe_width)
    if max_width <= 0:
        from .errors import EngineError

        raise EngineError(
            "E_LAYOUT",
            "Padding leaves no usable text width",
            details={"safe_width": safe_width, "padding": style.padding},
        )

    lines: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph == "":
            lines.append("")
            continue
        current = ""
        for token in paragraph.split():
            candidate = token if not current else f"{current} {token}"
            if text_width(candidate, style.font_size) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            if text_width(token, style.font_size) <= max_width:
                current = token
                continue
            pieces = _split_long_token(token, style.font_size, max_width)
            lines.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""
        if current or not lines or paragraph == "":
            lines.append(current)

    if not lines:
        lines = [""]
    widths = [text_width(line, style.font_size) for line in lines]
    return lines, widths
