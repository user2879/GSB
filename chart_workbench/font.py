"""A tiny deterministic bitmap font used by samples and the OCR stage."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


GLYPHS: Dict[str, Tuple[str, ...]] = {
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###", "..#", "###", "#..", "###"),
    "3": ("###", "..#", "###", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", "..#", "..#", "..#"),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
    "-": ("...", "...", "###", "...", "..."),
    ".": ("#", ".", ".", ".", "."),
    "e": ("...", ".##", "###", "#..", "###"),
}


def glyph_rows(char: str) -> Tuple[str, ...]:
    return GLYPHS.get(char, GLYPHS["-"])


def text_width(text: str, scale: int, spacing: int = 2) -> int:
    if not text:
        return 0
    widths = [len(glyph_rows(char)[0]) * scale for char in text]
    return sum(widths) + spacing * scale * max(0, len(text) - 1)


def text_height(scale: int) -> int:
    return 5 * scale


def draw_text(
    draw,
    x: int,
    y: int,
    text: str,
    scale: int = 2,
    fill: int = 0,
    anchor: str = "lt",
    spacing: int = 2,
) -> Tuple[int, int, int, int]:
    """Draw text and return its integer bounding box."""

    width = text_width(text, scale, spacing)
    height = text_height(scale)
    if anchor in {"rt", "ct"}:
        x -= width
    elif anchor in {"mt", "mm", "mb"}:
        x -= width // 2
    if anchor in {"mb", "mm", "cb"}:
        y -= height
    elif anchor in {"ct", "cm"}:
        y -= height // 2

    cursor = x
    for char in text:
        rows = glyph_rows(char)
        for row_index, row in enumerate(rows):
            for col_index, bit in enumerate(row):
                if bit != "#":
                    continue
                px = cursor + col_index * scale
                py = y + row_index * scale
                draw.rectangle((px, py, px + scale - 1, py + scale - 1), fill=fill)
        cursor += len(rows[0]) * scale + spacing * scale
    return (x, y, x + width - 1, y + height - 1)


def ink_templates() -> List[Tuple[str, List[List[bool]]]]:
    templates: List[Tuple[str, List[List[bool]]]] = []
    for char, rows in GLYPHS.items():
        points = [(x, y) for y, row in enumerate(rows) for x, bit in enumerate(row) if bit == "#"]
        min_x = min(x for x, _ in points)
        max_x = max(x for x, _ in points)
        min_y = min(y for _, y in points)
        max_y = max(y for _, y in points)
        matrix = [
            [rows[y][x] == "#" for x in range(min_x, max_x + 1)]
            for y in range(min_y, max_y + 1)
        ]
        templates.append((char, matrix))
    return templates


def bit_rows_from_points(points: Iterable[Tuple[int, int]]) -> List[List[bool]]:
    points = list(points)
    if not points:
        return []
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    matrix = [[False for _ in range(max_x - min_x + 1)] for _ in range(max_y - min_y + 1)]
    for x, y in points:
        matrix[y - min_y][x - min_x] = True
    return matrix
