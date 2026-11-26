from __future__ import annotations

import random
from functools import lru_cache

GLYPH_SIZE = 16


def _plot(bmp: list[list[bool]], x0: int, y0: int, x1: int, y1: int) -> None:
    """Integer Bresenham line into a 16x16 dot matrix."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        for ox, oy in ((0, 0), (1, 0), (0, 1)):
            xx, yy = x + ox, y + oy
            if 0 <= xx < GLYPH_SIZE and 0 <= yy < GLYPH_SIZE:
                bmp[yy][xx] = True
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


@lru_cache(maxsize=512)
def builtin_glyph(character: str) -> tuple[tuple[bool, ...], ...]:
    """Return a stable built-in dot-matrix glyph.

    The font is algorithmic so the repository needs no system font or binary
    asset.  Every Unicode character maps to the same pseudo-ideograph every run.
    """
    seed = (0x9E3779B97F4A7C15 ^ ord(character) * 2654435761) & 0xFFFFFFFF
    rng = random.Random(seed)
    bmp = [[False] * GLYPH_SIZE for _ in range(GLYPH_SIZE)]

    _plot(bmp, 4, 3, 11, 3)
    _plot(bmp, 7, 3, 7, 7)
    horizontal_rows = rng.sample([6, 8, 10, 12], k=3)
    for row in horizontal_rows:
        left = rng.choice((3, 4, 5))
        right = rng.choice((10, 11, 12))
        _plot(bmp, left, row, right, row + rng.choice((-1, 0, 1)))

    vertical_cols = rng.sample([4, 6, 9, 11], k=2)
    for col in vertical_cols:
        top = rng.choice((5, 6, 7))
        bottom = rng.choice((11, 12, 13))
        _plot(bmp, col, top, col + rng.choice((-1, 0, 1)), bottom)

    if rng.random() < 0.85:
        _plot(bmp, 3, 13, 7, 9)
    if rng.random() < 0.85:
        _plot(bmp, 9, 9, 13, 13)
    if ord(character) % 3 == 0:
        _plot(bmp, 5, 5, 12, 12)
    elif ord(character) % 3 == 1:
        _plot(bmp, 11, 5, 4, 12)

    return tuple(tuple(row) for row in bmp)


def glyph_to_rows(glyph: tuple[tuple[bool, ...], ...]) -> list[str]:
    return ["".join("█" if cell else "·" for cell in row) for row in glyph]
