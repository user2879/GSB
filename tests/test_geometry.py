from __future__ import annotations

from glyphforge.geometry import (
    flatten_contour,
    flatten_glyph,
    flatten_tolerance_for_size,
)
from glyphforge.model import Contour, Point, Segment
from glyphforge.samples import sample_glyphs


def _curve_contour() -> Contour:
    return Contour(
        start=Point(0, 0),
        segments=(
            Segment(
                "cubic",
                Point(1000, 0),
                control1=Point(220, 650),
                control2=Point(780, -650),
            ),
            Segment("quadratic", Point(0, 0), control1=Point(500, 500)),
        ),
    )


def test_flattening_is_point_for_point_deterministic():
    contour = _curve_contour()
    first = flatten_contour(contour, scale=0.032, tolerance=0.25)
    second = flatten_contour(contour, scale=0.032, tolerance=0.25)
    assert first.points == second.points
    assert [segment.points for segment in first.segments] == [
        segment.points for segment in second.segments
    ]
    assert first.to_dict(include_points=True) == second.to_dict(include_points=True)


def test_tighter_tolerance_never_reduces_vertex_count():
    contour = _curve_contour()
    counts = [
        flatten_contour(contour, scale=0.032, tolerance=tolerance).vertex_count
        for tolerance in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625)
    ]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_larger_size_derives_tighter_glyph_unit_tolerance():
    assert flatten_tolerance_for_size(48) < flatten_tolerance_for_size(12)
    glyph = sample_glyphs()["O"]
    small = flatten_glyph(glyph, 12)
    large = flatten_glyph(glyph, 48)
    assert large.tolerance_pixels == small.tolerance_pixels
    assert large.vertex_count >= small.vertex_count


def test_each_curve_segment_records_error_and_subdivision_count():
    flattened = flatten_glyph(sample_glyphs()["B"], 48)
    curve_records = [
        segment for segment in flattened.segment_records if segment.kind != "line"
    ]
    assert curve_records
    for record in curve_records:
        assert record.actual_error_estimate >= 0.0
        assert record.actual_error_estimate <= flattened.tolerance_pixels + 1e-9
        assert record.points
