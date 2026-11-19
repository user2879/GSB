"""Programmatic local glyph corpus with five required edge-case categories."""

from __future__ import annotations

import math
import random

from .model import (
    Contour,
    GlyphCase,
    GlyphSuite,
    OutlineNode,
    Point,
    cubic_node,
    line_contour,
    line_node,
    move_node,
    open_line_path,
    quadratic_node,
)

SIZES = (12, 16, 24, 32, 48)
KAPPA = 0.5522847498307936


def _p(x: float, y: float) -> Point:
    return Point(float(x), float(y))


def _quadratic_ellipse(cx: float, cy: float, rx: float, ry: float, *, clockwise: bool = False) -> Contour:
    direction = -1.0 if clockwise else 1.0
    nodes: list[OutlineNode] = [move_node(_p(cx + rx, cy))]
    angles = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0, 2.0 * math.pi]
    if clockwise:
        angles = [0.0, -math.pi / 2.0, -math.pi, -3.0 * math.pi / 2.0, -2.0 * math.pi]
    for idx in range(4):
        a0 = angles[idx]
        a1 = angles[idx + 1]
        start = (cx + rx * math.cos(a0), cy + ry * math.sin(a0))
        end = (cx + rx * math.cos(a1), cy + ry * math.sin(a1))
        c1 = (
            start[0] + direction * KAPPA * (-rx * math.sin(a0)),
            start[1] + direction * KAPPA * (ry * math.cos(a0)),
        )
        c2 = (
            end[0] - direction * KAPPA * (-rx * math.sin(a1)),
            end[1] - direction * KAPPA * (ry * math.cos(a1)),
        )
        nodes.append(cubic_node(_p(*end), _p(*c1), _p(*c2), closed=idx == 3))
    return Contour(tuple(nodes))


def _rounded_rect(x0: float, y0: float, x1: float, y1: float, radius: float) -> Contour:
    r = min(radius, (x1 - x0) * 0.25, (y1 - y0) * 0.25)
    k = r * KAPPA
    nodes = [
        move_node(_p(x0 + r, y0)),
        line_node(_p(x1 - r, y0)),
        cubic_node(_p(x1, y0 + r), _p(x1 - r + k, y0), _p(x1, y0 + r - k)),
        line_node(_p(x1, y1 - r)),
        cubic_node(_p(x1 - r, y1), _p(x1, y1 - r + k), _p(x1 - r + k, y1)),
        line_node(_p(x0 + r, y1)),
        cubic_node(_p(x0, y1 - r), _p(x0 + r - k, y1), _p(x0, y1 - r + k)),
        line_node(_p(x0, y0 + r)),
        cubic_node(_p(x0 + r, y0), _p(x0, y0 + r - k), _p(x0 + r - k, y0), closed=True),
    ]
    return Contour(tuple(nodes))


def _leaf(cx: float, cy: float, rx: float, ry: float) -> Contour:
    nodes = [
        move_node(_p(cx, cy + ry)),
        quadratic_node(_p(cx + rx, cy), _p(cx + rx * 0.72, cy + ry * 0.72)),
        quadratic_node(_p(cx, cy - ry), _p(cx + rx * 0.72, cy - ry * 0.72)),
        quadratic_node(_p(cx - rx, cy), _p(cx - rx * 0.72, cy - ry * 0.72)),
        quadratic_node(_p(cx, cy + ry), _p(cx - rx * 0.72, cy + ry * 0.72), closed=True),
    ]
    return Contour(tuple(nodes))


def _jittered(points: list[tuple[float, float]], rng: random.Random, amount: float = 0.08) -> list[tuple[float, float]]:
    return [(x + rng.uniform(-amount, amount), y + rng.uniform(-amount, amount)) for x, y in points]


def generate_suite(seed: int = 20261119) -> GlyphSuite:
    rng = random.Random(seed)

    simple_a = (
        _rounded_rect(9.0, 8.0, 53.0, 56.0, 7.0).transformed(
            _translate(rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04))
        ),
        _quadratic_ellipse(31.0, 32.0, 11.0, 15.0, clockwise=True),
    )
    simple_b = (_leaf(37.0 + rng.uniform(-0.05, 0.05), 33.0, 20.0, 24.0),)

    self_points = _jittered([(10.0, 12.0), (52.0, 50.0), (10.0, 50.0), (52.0, 12.0)], rng, 0.05)
    self_a = (line_contour(self_points), _leaf(31.0, 29.0, 17.0, 20.0))
    self_b = (
        _quadratic_ellipse(35.0, 34.0, 21.0, 18.0),
        line_contour([(18.0, 5.0), (53.0, 42.0), (30.0, 59.0)]),
    )

    overlap_a = (
        line_contour([(8.0, 10.0), (34.0, 10.0), (34.0, 48.0), (8.0, 48.0)]),
        line_contour([(15.0, 18.0), (26.0, 18.0), (26.0, 34.0), (15.0, 34.0)]),
    )
    overlap_b = (
        line_contour([(22.0, 10.0), (50.0, 10.0), (50.0, 48.0), (22.0, 48.0)]),
        line_contour([(25.0, 40.0), (25.0, 54.0), (44.0, 54.0), (44.0, 40.0)]),
    )

    open_case_a = (line_contour([(8.0, 12.0), (28.0, 12.0), (28.0, 50.0), (8.0, 50.0)]),)
    open_case_b = (_quadratic_ellipse(31.0, 31.0, 17.0, 16.0),)
    open_path = Contour(
        (
            move_node(_p(7.0, 20.0)),
            line_node(_p(20.0, 45.0)),
            cubic_node(_p(40.0, 33.0), _p(25.0, 53.0), _p(34.0, 13.0)),
            quadratic_node(_p(54.0, 21.0), _p(48.0, 30.0)),
            line_node(_p(56.0, 43.0)),
        )
    )

    degenerate_a = (
        line_contour([(8.0, 8.0), (8.0, 8.0), (8.0, 48.0), (48.0, 48.0), (48.0, 8.0)]),
        _quadratic_ellipse(27.0, 28.0, 8.0, 9.0, clockwise=True),
    )
    degenerate_b_nodes = (
        move_node(_p(20.0, 18.0)),
        cubic_node(_p(20.0, 18.0), _p(20.0, 18.0), _p(20.0, 18.0)),
        line_node(_p(54.0, 18.0)),
        line_node(_p(54.0, 50.0)),
        line_node(_p(20.0, 50.0)),
        line_node(_p(20.0, 18.0), closed=True),
    )
    degenerate_b = (
        Contour(degenerate_b_nodes),
        _quadratic_ellipse(44.0, 16.0, 7.0, 7.0),
    )

    cases = (
        GlyphCase(
            "G01_SIMPLE",
            "simple",
            simple_a,
            simple_b,
            open_line_path([(10.0, 20.0), (25.0, 39.0), (42.0, 18.0), (54.0, 35.0)]),
            SIZES,
            4.0,
            "round",
            "round",
        ),
        GlyphCase(
            "G02_SELF_INTERSECT",
            "self_intersect",
            self_a,
            self_b,
            open_line_path([(8.0, 10.0), (35.0, 35.0), (56.0, 12.0), (29.0, 52.0)]),
            SIZES,
            3.5,
            "butt",
            "miter",
        ),
        GlyphCase(
            "G03_OVERLAP_COINCIDENT",
            "overlap_coincident",
            overlap_a,
            overlap_b,
            open_line_path([(10.0, 18.0), (32.0, 18.0), (32.0, 44.0), (54.0, 44.0)]),
            SIZES,
            4.5,
            "square",
            "bevel",
        ),
        GlyphCase(
            "G04_OPEN_PATH",
            "open_path",
            open_case_a,
            open_case_b,
            open_path,
            SIZES,
            5.0,
            "round",
            "round",
        ),
        GlyphCase(
            "G05_DEGENERATE",
            "degenerate",
            degenerate_a,
            degenerate_b,
            open_line_path([(9.0, 14.0), (9.0, 14.0), (35.0, 28.0), (55.0, 48.0)]),
            SIZES,
            3.0,
            "butt",
            "bevel",
        ),
    )
    suite = GlyphSuite(seed, cases)
    suite.validate()
    return suite


def _translate(tx: float, ty: float):
    from .model import Affine

    return Affine(tx=tx, ty=ty)
