"""Programmatically constructed glyph samples; no system font dependency."""

from __future__ import annotations

from pathlib import Path

from .model import (
    SUPPORTED_SIZES,
    Contour,
    Glyph,
    GlyphDocument,
    Point,
    Segment,
    dumps_glyphs,
)

KAPPA = 0.5522847498307936


def _line_to(x: float, y: float) -> Segment:
    return Segment("line", Point(x, y))


def _quad_to(cx: float, cy: float, x: float, y: float) -> Segment:
    return Segment("quadratic", Point(x, y), control1=Point(cx, cy))


def _cubic_to(
    c1x: float, c1y: float, c2x: float, c2y: float, x: float, y: float
) -> Segment:
    return Segment(
        "cubic",
        Point(x, y),
        control1=Point(c1x, c1y),
        control2=Point(c2x, c2y),
    )


def _rectangle(
    x0: float, y0: float, x1: float, y1: float, *, reverse: bool = False
) -> Contour:
    points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    if reverse:
        points = tuple(reversed(points))
    return Contour(
        start=Point(*points[0]),
        segments=tuple(_line_to(*point) for point in points[1:])
        + (_line_to(*points[0]),),
        closed=True,
    )


def _ellipse(
    cx: float, cy: float, rx: float, ry: float, *, reverse: bool = False
) -> Contour:
    kx = rx * KAPPA
    ky = ry * KAPPA
    segments = (
        _cubic_to(cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry),
        _cubic_to(cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy),
        _cubic_to(cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry),
        _cubic_to(cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy),
    )
    if reverse:
        # Reversing requires a matching start point and control order.  Build
        # the path again from the opposite starting quadrant.
        segments = (
            _cubic_to(cx - kx, cy - ry, cx - rx, cy - ky, cx - rx, cy),
            _cubic_to(cx - rx, cy + ky, cx - kx, cy + ry, cx, cy + ry),
            _cubic_to(cx + kx, cy + ry, cx + rx, cy + ky, cx + rx, cy),
            _cubic_to(cx + rx, cy - ky, cx + kx, cy - ry, cx, cy - ry),
        )
        return Contour(start=Point(cx, cy - ry), segments=segments, closed=True)
    return Contour(start=Point(cx + rx, cy), segments=segments, closed=True)


def _sample_a() -> Glyph:
    # An asymmetric A with a curved crossbar and quadratic shoulder.
    outer = Contour(
        start=Point(90, 40),
        segments=(
            _line_to(390, 920),
            _quad_to(520, 920, 620, 920),
            _line_to(900, 40),
            _quad_to(820, 20, 750, 40),
            _line_to(690, 250),
            _line_to(315, 250),
            _line_to(250, 40),
            _quad_to(170, 20, 90, 40),
            _line_to(90, 40),
        ),
    )
    hole = Contour(
        start=Point(370, 390),
        segments=(
            _line_to(640, 390),
            _line_to(505, 735),
            _line_to(370, 390),
        ),
    )
    return Glyph("A", (outer, hole), SUPPORTED_SIZES)


def _sample_o() -> Glyph:
    # Outer and inner contours have opposite winding: both fill rules leave a
    # hole, while nonzero and even-odd still share antialiased edge coverage.
    outer = _ellipse(500, 490, 420, 430)
    inner = _ellipse(500, 490, 245, 270, reverse=True)
    return Glyph("O", (outer, inner), SUPPORTED_SIZES)


def _sample_b() -> Glyph:
    outer = Contour(
        start=Point(120, 50),
        segments=(
            _line_to(120, 900),
            _line_to(520, 900),
            _cubic_to(760, 900, 820, 745, 760, 600),
            _quad_to(700, 520, 590, 500),
            _cubic_to(780, 470, 830, 300, 660, 120),
            _quad_to(560, 30, 350, 50),
            _line_to(120, 50),
        ),
    )
    upper_hole = _ellipse(505, 690, 145, 120)
    lower_hole = _ellipse(505, 290, 155, 130)
    return Glyph("B", (outer, upper_hole, lower_hole), SUPPORTED_SIZES)


def _sample_i() -> Glyph:
    stem = _rectangle(420, 40, 610, 610)
    tittle = _ellipse(515, 820, 105, 105)
    return Glyph("i", (stem, tittle), (12, 16, 24, 32))


def _sample_overlap() -> Glyph:
    # Same-winding nested rectangles make nonzero solid and even-odd hollow.
    outer = _rectangle(90, 80, 910, 900)
    inner_same_winding = _rectangle(280, 270, 720, 710)
    return Glyph("overlap", (outer, inner_same_winding), SUPPORTED_SIZES)


def _sample_degenerate() -> Glyph:
    point = Point(500, 500)
    degenerate = Contour(
        start=point,
        segments=(
            Segment("line", point),
            Segment("quadratic", point, control1=point),
            Segment("cubic", point, control1=point, control2=point),
            Segment("line", point),
        ),
        closed=True,
    )
    return Glyph("degenerate", (degenerate,), (12, 16, 24))


def _sample_dense() -> Glyph:
    contours: list[Contour] = []
    # 7 x 9 cell hints at a dense small-text bitmap while remaining only
    # programmatic geometry.  Gaps are deliberately sub-pixel at 12 px.
    for row in range(9):
        for column in range(7):
            x = 40 + column * 132
            y = 40 + row * 98
            width = 86 if (row + column) % 3 else 58
            height = 58 if (row * 2 + column) % 4 else 40
            contours.append(_rectangle(x, y, x + width, y + height))
    return Glyph("dense", tuple(contours), SUPPORTED_SIZES)


def build_sample_document() -> GlyphDocument:
    return GlyphDocument(
        glyphs=(
            _sample_a(),
            _sample_o(),
            _sample_b(),
            _sample_i(),
            _sample_overlap(),
            _sample_degenerate(),
            _sample_dense(),
        )
    )


def sample_glyphs() -> dict[str, Glyph]:
    document = build_sample_document()
    return document.by_id


def write_sample_json(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        dumps_glyphs(build_sample_document()), encoding="utf-8", newline="\n"
    )
    return destination
