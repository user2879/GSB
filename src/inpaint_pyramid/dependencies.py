"""Cross-layer / cross-tile dependency geometry."""

from __future__ import annotations

from .model import PyramidSpec, Rect
from .pyramid import axis_support


def lower_tile_refs(spec: PyramidSpec, level: int, row: int, col: int) -> tuple[tuple[int, int, int], ...]:
    if level == 0:
        return ()
    x0, y0, x1, y1 = spec.tile_bounds(level, row, col)
    lower_width, lower_height = spec.sizes[level - 1]
    current_width, current_height = spec.sizes[level]
    lower_x0, lower_x1 = axis_support(lower_width, current_width, x0, x1)
    lower_y0, lower_y1 = axis_support(lower_height, current_height, y0, y1)
    first_col = lower_x0 // spec.tile_size
    last_col = lower_x1 // spec.tile_size
    first_row = lower_y0 // spec.tile_size
    last_row = lower_y1 // spec.tile_size
    return tuple(
        (level - 1, lower_row, lower_col)
        for lower_row in range(first_row, last_row + 1)
        for lower_col in range(first_col, last_col + 1)
    )


def tiles_intersecting_rect(spec: PyramidSpec, level: int, rect: Rect) -> set[tuple[int, int, int]]:
    x, y, w, h = rect
    first_col = x // spec.tile_size
    last_col = (x + w - 1) // spec.tile_size
    first_row = y // spec.tile_size
    last_row = (y + h - 1) // spec.tile_size
    return {
        (level, row, col)
        for row in range(first_row, last_row + 1)
        for col in range(first_col, last_col + 1)
    }


def build_parent_map(spec: PyramidSpec) -> dict[tuple[int, int, int], tuple[tuple[int, int, int], ...]]:
    parents: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for level in range(spec.levels):
        for row in range(spec.tile_rows(level)):
            for col in range(spec.tile_cols(level)):
                key = (level, row, col)
                for child in lower_tile_refs(spec, level, row, col):
                    parents.setdefault(child, []).append(key)
    return {key: tuple(value) for key, value in parents.items()}

