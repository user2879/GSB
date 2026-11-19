"""Deterministic word-boundary wrapping and font-size adaptation.

Widths come from a deterministic per-glyph metric table (see
:data:`GLYPH_WIDTHS`) instead of host fonts, so results are reproducible
byte-for-byte across machines, including headless CI. CJK characters count as
wide glyphs and may be broken anywhere; latin runs are kept whole whenever a
word boundary exists.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import WrapError
from .models import FRAME_WIDTH

# Deterministic metric table: width units per glyph at font size 100.
WIDE_WIDTH = 100      # CJK ideographs, full-width punctuation
DIGIT_WIDTH = 55
LATIN_WIDTH = 56
SPACE_WIDTH = 28
FALLBACK_WIDTH = 80   # any unmapped glyph (treated as wide, never under-sized)

FONT_TIERS = (48, 42, 36, 32, 28)  # px, largest first
DEFAULT_MAX_LINES = 2


def glyph_width(ch: str) -> int:
    if ch == " ":
        return SPACE_WIDTH
    code = ord(ch)
    if ch.isdigit():
        return DIGIT_WIDTH
    if "一" <= ch <= "鿿" or "　" <= ch <= "〿" or "＀" <= ch <= "￯":
        return WIDE_WIDTH
    if code < 0x2E80 and ch.isascii() and ch.isalpha():
        return LATIN_WIDTH
    if code >= 0x2E80:  # other CJK ranges / symbols: conservative wide width
        return WIDE_WIDTH
    return FALLBACK_WIDTH


def text_width_at(text: str, font_size: int) -> float:
    return sum(glyph_width(ch) for ch in text) * font_size / 100.0


@dataclass(frozen=True)
class WrapResult:
    lines: tuple[str, ...]
    font_size: int
    max_line_width: float
    hard_broken: bool

    def to_dict(self) -> dict:
        return {
            "lines": list(self.lines),
            "font_size": self.font_size,
            "max_line_width": round(self.max_line_width, 4),
            "hard_broken": self.hard_broken,
        }


def _tokenize(text: str) -> list[str]:
    """Split into tokens: latin/digit words, spaces, and single CJK glyphs."""
    tokens: list[str] = []
    buffer = ""
    for ch in text:
        if ch == " ":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(" ")
        elif glyph_width(ch) == WIDE_WIDTH:
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(ch)
        else:
            buffer += ch
    if buffer:
        tokens.append(buffer)
    return tokens


def _pack(tokens: list[str], font_size: int, max_width: float) -> list[str]:
    """Greedy packing with word-boundary priority.

    Returns the lines; a line whose last token still overflows is marked by a
    trailing ``\\x00`` so the caller can detect a mandatory hard break.
    """
    lines: list[str] = []
    current = ""
    for token in tokens:
        if token == " ":
            candidate = current + " "
            if current and text_width_at(candidate, font_size) <= max_width:
                current = candidate
            continue
        candidate = current + token
        if text_width_at(candidate, font_size) <= max_width or not current:
            if text_width_at(candidate, font_size) <= max_width:
                current = candidate
            else:
                # Single token wider than the line: hard-break glyph by glyph.
                head, rest = _hard_split(token, font_size, max_width)
                current = head
                lines.append(current + "\x00")
                current = rest
            continue
        lines.append(current)
        current = token
    if current:
        if text_width_at(current, font_size) > max_width:
            head, rest = _hard_split(current, font_size, max_width)
            lines.append(head + "\x00")
            if rest:
                lines.append(rest)
        else:
            lines.append(current)
    return lines


def _hard_split(token: str, font_size: int, max_width: float) -> tuple[str, str]:
    head = ""
    for index, ch in enumerate(token):
        if text_width_at(head + ch, font_size) > max_width:
            return head, token[index:]
        head += ch
    return head, ""


def wrap_text(text: str,
              safe_width: int = FRAME_WIDTH,
              side_margin: int = 120,
              max_lines: int = DEFAULT_MAX_LINES,
              font_tiers: tuple[int, ...] = FONT_TIERS) -> WrapResult:
    """Wrap ``text`` and adapt font size under the declared layout contract.

    Tiers are tried largest-first. The first tier whose greedy word-boundary
    packing fits in ``max_lines`` wins. If every tier exceeds ``max_lines`` the
    smallest tier is still used but hard glyph breaks are applied; if even the
    smallest tier cannot keep every line inside the safe width,
    :class:`WrapError` (``E_WRAP_IMPOSSIBLE``) is raised.
    """
    max_width = float(safe_width - 2 * side_margin)
    tokens = _tokenize(text)
    last_result: WrapResult | None = None

    for font_size in font_tiers:
        raw_lines = _pack(tokens, font_size, max_width)
        hard_broken = any(line.endswith("\x00") for line in raw_lines)
        lines = tuple(line.rstrip("\x00") for line in raw_lines)
        widths = [text_width_at(line, font_size) for line in lines]
        widest = max(widths, default=0.0)
        if widest > max_width + 1e-9:
            last_result = WrapResult(lines, font_size, widest, hard_broken)
            continue
        if len(lines) <= max_lines:
            return WrapResult(lines, font_size, widest, hard_broken)
        last_result = WrapResult(lines, font_size, widest, hard_broken)

    if last_result is not None and last_result.max_line_width <= max_width + 1e-9:
        return last_result  # smallest tier, possibly hard-broken but in-width
    raise WrapError(f"text cannot fit safe width {max_width:.0f}px even at "
                    f"tier {font_tiers[-1]}px")
