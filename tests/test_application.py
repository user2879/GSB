from __future__ import annotations

from glyphforge.application import GlyphApplication, RenderKey
from glyphforge.atlas import AtlasBudget
from glyphforge.model import Point
from glyphforge.samples import sample_glyphs


def _app() -> GlyphApplication:
    glyphs = sample_glyphs()
    selected = {key: glyphs[key] for key in ("A", "O", "overlap", "degenerate")}
    return GlyphApplication(
        selected,
        active_sizes={"A": (12, 16), "O": (12,), "overlap": (16,), "degenerate": (12,)},
        atlas_budget=AtlasBudget(
            page_size=256, max_pages=2, max_occupied_pixels=65536, padding=1
        ),
        worker_budget=3,
        memory_budget_bytes=4_000_000,
    )


def _region_signature(app: GlyphApplication, key: RenderKey):
    region = app.region_for(key)
    assert region is not None
    manager = app.atlas_budget_manager.manager
    page = manager._page_by_index(region.page_index)
    return (
        region.page_index,
        region.x,
        region.y,
        region.width,
        region.height,
        region.content_hash,
        page.bitmap[
            region.y : region.y + region.height,
            region.x : region.x + region.width,
        ].tobytes(),
    )


def test_edit_recomputes_only_affected_glyph_and_preserves_other_regions():
    app = _app()
    app.render_all()
    before_o = _region_signature(app, RenderKey("O", 12, "nonzero"))
    before_o_hash = app.bitmap_cache[RenderKey("O", 12, "nonzero")].content_hash

    report = app.edit_point(
        "A",
        contour_index=0,
        segment_index=1,
        role="control1",
        point=Point(540, 950),
    )
    assert report.recomputed_glyph_ids == ("A",)
    assert {key.glyph_id for key in report.recomputed_keys} == {"A"}
    assert report.bitmap_hashes
    assert app.bitmap_cache[RenderKey("O", 12, "nonzero")].content_hash == before_o_hash
    assert _region_signature(app, RenderKey("O", 12, "nonzero")) == before_o


def test_fill_and_size_switches_have_exact_change_closures():
    app = _app()
    app.render_all()
    o_before = _region_signature(app, RenderKey("O", 12, "nonzero"))

    fill = app.set_fill_rule("overlap", "evenodd")
    assert fill.recomputed_glyph_ids == ("overlap",)
    assert _region_signature(app, RenderKey("O", 12, "nonzero")) == o_before

    size = app.switch_size("A", 24)
    assert size.recomputed_glyph_ids == ("A",)
    assert tuple(key.size for key in size.recomputed_keys) == (24,)
    assert {key.size for key in size.removed_keys} == {12, 16}
    assert _region_signature(app, RenderKey("O", 12, "nonzero")) == o_before


def test_report_closure_equals_actual_recomputed_key_set():
    app = _app()
    app.render_all()
    report = app.set_global_fill_rule("evenodd")
    actual_glyphs = tuple(sorted({key.glyph_id for key in report.recomputed_keys}))
    assert report.recomputed_glyph_ids == actual_glyphs
    assert set(actual_glyphs) == set(app.glyphs)


def test_unpublish_then_republish_keeps_pixels_identical():
    app = _app()
    app.render_all()
    key = RenderKey("A", 16, "nonzero")
    before = _region_signature(app, key)
    app.unpublish_all()
    assert not app.atlas_budget_manager.manager.pages
    report = app.render_keys([key])
    after = _region_signature(app, key)
    assert report.recomputed_glyph_ids == ("A",)
    assert before[5] == after[5]
    assert before[6] == after[6]
