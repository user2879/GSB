"""Deterministic grayscale frame composition."""

from __future__ import annotations

import math
import unicodedata
from typing import Iterable

from ._numpy import np
from .font5x7 import glyph
from .layout import EventLayout
from .metrics import advance_width, glyph_scale, line_pixel_height
from .spec import FrameConfig, Rect, Style


def _wide(character: str) -> bool:
    return unicodedata.east_asian_width(character) in {"W", "F"}


def _draw_glyph(
    frame: "np.ndarray",
    character: str,
    x: int,
    y: int,
    scale: int,
    value: int,
    clip: Rect,
    outline: bool,
) -> None:
    pattern = glyph(character)
    offsets = [(0, 0)]
    if outline:
        offsets.extend([(-1, 0), (1, 0), (0, -1), (0, 1)])
    for offset_x, offset_y in offsets:
        color = 0 if outline and (offset_x or offset_y) else value
        for row, bits in enumerate(pattern):
            for col, bit in enumerate(bits):
                if bit != "1":
                    continue
                x0 = x + offset_x + col * scale
                y0 = y + offset_y + row * scale
                x1 = min(x0 + scale, clip.right)
                y1 = min(y0 + scale, clip.bottom)
                x0 = max(x0, clip.x)
                y0 = max(y0, clip.y)
                if x1 > x0 and y1 > y0:
                    frame[y0:y1, x0:x1] = color


def _draw_text(
    frame: "np.ndarray",
    layout: EventLayout,
    style: Style,
    *,
    text_value: int,
    outline: bool,
) -> None:
    scale = glyph_scale(style.font_size)
    clip = layout.bbox
    x = layout.bbox.x + int(math.ceil(style.padding))
    y = layout.bbox.y + int(math.ceil(style.padding))
    line_height = line_pixel_height(style)
    for line_index, line in enumerate(layout.lines):
        cursor = x
        baseline = y + line_index * line_height
        for character in line:
            cell_width = advance_width(character, style.font_size)
            if character != " ":
                if _wide(character):
                    # Wide characters occupy a 2-cell deterministic block.
                    block = frame
                    bx0 = max(clip.x, cursor)
                    by0 = max(clip.y, baseline)
                    bx1 = min(clip.right, cursor + cell_width)
                    by1 = min(clip.bottom, baseline + min(line_height, 7 * scale))
                    if bx1 > bx0 and by1 > by0:
                        if outline:
                            block[max(clip.y, by0 - 1) : min(clip.bottom, by1 + 1), max(clip.x, bx0 - 1) : min(clip.right, bx1 + 1)] = 0
                        block[by0:by1, bx0:bx1] = text_value
                        # Stable inner cutout distinguishes wide glyphs while preserving contrast.
                        inner_x0 = min(bx1, bx0 + max(1, scale))
                        inner_y0 = min(by1, by0 + max(1, scale))
                        inner_x1 = max(inner_x0, bx1 - max(1, scale))
                        inner_y1 = max(inner_y0, by1 - max(1, scale))
                        if inner_x1 > inner_x0 and inner_y1 > inner_y0:
                            block[inner_y0:inner_y1, inner_x0:inner_x1] = 0
                else:
                    _draw_glyph(frame, character, cursor, baseline, scale, text_value, clip, outline)
            cursor += cell_width


def _draw_critical_marker(frame: "np.ndarray", bbox: Rect, style: Style) -> None:
    cell = max(1, glyph_scale(style.font_size))
    size = cell * 3
    x0 = bbox.x + max(1, int(math.floor(style.padding / 2.0)))
    y0 = bbox.y + max(1, int(math.floor(style.padding / 2.0)))
    x1 = min(bbox.right, x0 + size)
    y1 = min(bbox.bottom, y0 + size)
    if x1 <= x0 or y1 <= y0:
        return
    marker = np.fromfunction(
        lambda yy, xx: ((xx // cell + yy // cell) % 2).astype(np.uint8) * 255,
        (y1 - y0, x1 - x0),
        dtype=int,
    ).astype(np.uint8)
    frame[y0:y1, x0:x1] = marker


def compose_frame(
    frame_config: FrameConfig,
    style: Style,
    layouts: Iterable[EventLayout],
    frame_index: int,
) -> "np.ndarray":
    """Compose one frame from event layers in deterministic id/time order."""

    frame = np.full(
        (frame_config.height, frame_config.width), frame_config.background, dtype=np.uint8
    )
    ordered = sorted(layouts, key=lambda item: (item.event.start_ms, item.event.id))
    effective_opacity = style.mask_opacity + (0.15 if style.contrast_enhance else 0.0)
    effective_opacity = min(1.0, max(0.0, effective_opacity))
    mask_value = round(frame_config.background * (1.0 - effective_opacity))
    text_value = 255 if style.contrast_enhance else 224

    for layout in ordered:
        bbox = layout.bbox
        frame[bbox.y : bbox.bottom, bbox.x : bbox.right] = mask_value
        if style.contrast_enhance:
            frame[bbox.y, bbox.x : bbox.right] = 0
            frame[max(bbox.y, bbox.bottom - 1), bbox.x : bbox.right] = 0
            frame[bbox.y : bbox.bottom, bbox.x] = 0
            frame[bbox.y : bbox.bottom, max(bbox.x, bbox.right - 1)] = 0
        _draw_text(
            frame,
            layout,
            style,
            text_value=text_value,
            outline=style.contrast_enhance,
        )
        if layout.event.critical_sound:
            _draw_critical_marker(frame, bbox, style)
    return frame
