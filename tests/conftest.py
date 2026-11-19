from __future__ import annotations

import pytest

from glyphforge.model import Contour, Glyph, Point, Segment


def rectangle(
    x0: float, y0: float, x1: float, y1: float, *, reverse: bool = False
) -> Contour:
    points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    if reverse:
        points = tuple(reversed(points))
    return Contour(
        Point(*points[0]),
        tuple(Segment("line", Point(*point)) for point in points[1:])
        + (Segment("line", Point(*points[0])),),
    )


@pytest.fixture
def square_glyph() -> Glyph:
    return Glyph("square", (rectangle(100, 100, 900, 900),), (12, 16, 24, 32, 48))


@pytest.fixture
def overlap_glyph() -> Glyph:
    return Glyph(
        "overlap",
        (rectangle(50, 50, 950, 950), rectangle(250, 250, 750, 750)),
        (12, 16, 24, 32, 48),
    )


@pytest.fixture
def empty_glyph() -> Glyph:
    return Glyph("empty", (), (12, 16, 24, 32, 48))
