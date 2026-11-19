from __future__ import annotations

import json

import pytest

from glyphforge.errors import ErrorCode, GlyphForgeError
from glyphforge.model import (
    Glyph,
    GlyphDocument,
    Point,
    dumps_glyphs,
    loads_glyphs,
    validate_glyphs,
)
from glyphforge.samples import build_sample_document, sample_glyphs


def test_canonical_json_is_deterministic_and_round_trips():
    document = build_sample_document()
    first = dumps_glyphs(document)
    second = dumps_glyphs(document)
    assert first == second
    assert first.endswith("\n")
    loaded = loads_glyphs(first)
    assert dumps_glyphs(loaded) == first
    assert [glyph.glyph_id for glyph in loaded.glyphs] == sorted(sample_glyphs())


def test_invalid_segment_type_reports_glyph_and_segment_index():
    payload = {
        "format": "glyphforge-glyph-v1",
        "glyphs": [
            {
                "id": "bad",
                "sizes": [12],
                "contours": [
                    {
                        "closed": True,
                        "start": {"x": 0, "y": 0},
                        "segments": [
                            {"type": "arc", "to": {"x": 10, "y": 10}},
                        ],
                    }
                ],
            }
        ],
    }
    with pytest.raises(GlyphForgeError) as caught:
        loads_glyphs(json.dumps(payload))
    assert caught.value.code is ErrorCode.OUTLINE_INVALID
    assert caught.value.glyph_id == "bad"
    assert caught.value.segment_index == 0
    assert "bad" in str(caught.value)


def test_unclosed_contour_and_nonfinite_points_are_rejected():
    unclosed = {
        "format": "glyphforge-glyph-v1",
        "glyphs": [
            {
                "id": "open",
                "contours": [
                    {
                        "closed": False,
                        "start": {"x": 0, "y": 0},
                        "segments": [{"type": "line", "to": {"x": 1, "y": 1}}],
                    }
                ],
            }
        ],
    }
    with pytest.raises(GlyphForgeError) as caught:
        loads_glyphs(json.dumps(unclosed))
    assert caught.value.code is ErrorCode.OUTLINE_INVALID
    assert caught.value.glyph_id == "open"

    with pytest.raises(GlyphForgeError) as caught:
        Point(float("inf"), 0)
    assert caught.value.code is ErrorCode.OUTLINE_INVALID


def test_malformed_json_and_duplicate_glyph_ids_are_invalid():
    with pytest.raises(GlyphForgeError) as caught:
        loads_glyphs("{not json")
    assert caught.value.code is ErrorCode.OUTLINE_INVALID

    glyph = Glyph("x", (), (12,))
    with pytest.raises(GlyphForgeError) as caught:
        GlyphDocument((glyph, glyph))
    assert caught.value.code is ErrorCode.OUTLINE_INVALID


def test_point_edit_creates_a_new_immutable_glyph():
    glyph = sample_glyphs()["A"]
    edited = glyph.with_point(
        contour_index=0,
        segment_index=1,
        role="control1",
        point=Point(510, 950),
    )
    assert glyph is not edited
    assert glyph.contours[0].segments[1].control1 == Point(520, 920)
    assert edited.contours[0].segments[1].control1 == Point(510, 950)
    assert validate_glyphs([edited])["A"] == edited
