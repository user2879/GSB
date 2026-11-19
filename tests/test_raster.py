from __future__ import annotations

import numpy as np
import pytest

from glyphforge.errors import ErrorCode, GlyphForgeError
from glyphforge.raster import (
    bitmap_hash,
    normalize_fill_rule,
    rasterize_glyph,
    verify_bitmap_hash,
)


def test_empty_glyph_renders_zero_bitmap(empty_glyph):
    result = rasterize_glyph(empty_glyph, 24, "nonzero")
    assert result.is_empty_glyph
    assert result.bitmap.shape == (1, 1)
    assert result.bitmap.dtype == np.uint8
    assert not np.any(result.bitmap)
    assert result.content_hash == bitmap_hash(1, 1, result.pixels)


def test_fill_rules_are_distinguishable_on_same_winding_nesting(overlap_glyph):
    nonzero = rasterize_glyph(overlap_glyph, 32, "nonzero")
    evenodd = rasterize_glyph(overlap_glyph, 32, "evenodd")
    assert nonzero.content_hash != evenodd.content_hash
    assert nonzero.non_zero_pixels > evenodd.non_zero_pixels
    assert nonzero.bitmap.max() == 255
    assert np.any((nonzero.bitmap > 0) & (nonzero.bitmap < 255))


def test_rasterization_and_hash_are_deterministic(square_glyph):
    first = rasterize_glyph(square_glyph, 48, "evenodd")
    second = rasterize_glyph(square_glyph, 48, "even-odd")
    assert first.pixels == second.pixels
    assert first.content_hash == second.content_hash
    assert first.width == second.width
    assert first.height == second.height
    verify_bitmap_hash(first, first.content_hash)


def test_invalid_fill_rule_and_hash_mismatch_have_business_codes(square_glyph):
    with pytest.raises(GlyphForgeError) as caught:
        rasterize_glyph(square_glyph, 12, "winding")
    assert caught.value.code is ErrorCode.FILL_RULE
    assert normalize_fill_rule("non-zero") == "nonzero"

    result = rasterize_glyph(square_glyph, 12, "nonzero")
    with pytest.raises(GlyphForgeError) as caught:
        verify_bitmap_hash(result, "0" * 64)
    assert caught.value.code is ErrorCode.BITMAP_HASH


def test_bitmap_is_read_only(square_glyph):
    result = rasterize_glyph(square_glyph, 12, "nonzero")
    with pytest.raises(ValueError):
        result.bitmap[0, 0] = 1
