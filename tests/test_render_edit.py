from __future__ import annotations

import hashlib

import numpy as np
import pytest

from npr_editor.editing import Editor
from npr_editor.errors import NprError
from npr_editor.png import encode_png
from npr_editor.render import RenderConfig, RenderEngine


def test_three_modes_share_geometry_and_skeleton(sample_project):
    config = RenderConfig(width=180, height=120, supersample=1)
    engine = RenderEngine(sample_project, config)
    first = engine.render("lineart")
    shared_before = {
        shape.id: engine.geometry.ensure(shape.id).vertices.copy()
        for shape in sample_project.shapes
    }
    engine.geometry.reset_stats()
    engine.reset_stats()
    sketch = engine.render("sketch")
    toon = engine.render("toon")
    assert first.mode == "lineart"
    assert sketch.mode == "sketch"
    assert toon.mode == "toon"
    assert sketch.flatten_builds == 0
    assert sketch.skeleton_builds == 0
    assert toon.flatten_builds == 0
    assert toon.skeleton_builds == 0
    for shape_id, vertices in shared_before.items():
        assert np.array_equal(engine.geometry.ensure(shape_id).vertices, vertices)


def test_mode_switch_does_not_reflatten_after_warmup(sample_project):
    engine = RenderEngine(sample_project, RenderConfig(width=160, height=100, supersample=1))
    engine.render("toon")
    engine.geometry.reset_stats()
    engine.render("lineart")
    engine.render("sketch")
    engine.render("toon")
    assert engine.geometry.flatten_builds == 0
    assert engine.geometry.skeleton_builds == 0


def test_line_width_change_only_invalidates_dependent_shape(sample_project):
    config = RenderConfig(width=220, height=150, supersample=1)
    editor = Editor(sample_project, config=config)
    before_render = editor.render("lineart")
    before_geometry = editor.geometry_hashes()
    editor.set_shape_stroke_width("tree_crown", 4.5)
    editor.renderer.reset_stats()
    after_render = editor.render("lineart")
    after_geometry = editor.geometry_hashes()
    changed = {
        shape_id
        for shape_id in before_render.shape_hashes
        if before_render.shape_hashes[shape_id] != after_render.shape_hashes[shape_id]
    }
    assert changed == {"tree_crown"}
    assert after_render.render_builds == 1
    assert after_render.render_cache_hits == len(sample_project.shapes) - 1
    assert before_geometry == after_geometry


def test_irrelevant_style_parameter_keeps_lineart_shape_hashes(sample_project):
    editor = Editor(sample_project, config=RenderConfig(width=180, height=120, supersample=1))
    before = editor.render("lineart")
    editor.set_style_parameter("sketch_hatch_spacing", 13.5)
    editor.renderer.reset_stats()
    after = editor.render("lineart")
    assert before.shape_hashes == after.shape_hashes
    assert after.render_builds == 0
    assert after.render_cache_hits == len(sample_project.shapes)
    assert before.geometry_hashes == after.geometry_hashes


def test_unused_toon_palette_entry_does_not_invalidate_shape(sample_project):
    editor = Editor(sample_project, config=RenderConfig(width=180, height=120, supersample=1))
    before = editor.render("toon")
    palette = dict(sample_project.style.toon_palette)
    palette["unused_color"] = (1, 200, 250)
    editor.set_style_parameter("toon_palette", palette)
    after = editor.render("toon")
    assert before.shape_hashes == after.shape_hashes
    assert before.pixel_sha256 == after.pixel_sha256


def test_edit_recomputes_only_affected_shape_and_undo_restores_hashes(sample_project):
    editor = Editor(sample_project, config=RenderConfig(width=220, height=150, supersample=1))
    before_render = editor.render("lineart")
    before_geometry = editor.geometry_hashes()
    before_arrays = {
        shape.id: editor.geometry.flattened(shape.id).vertices.copy()
        for shape in sample_project.shapes
    }
    editor.render("sketch")
    editor.render("toon")
    editor.edit_control_point("tree_crown", 0, 0, 1, 151.0, 83.0)
    after_render = editor.render("lineart")
    after_geometry = editor.geometry_hashes()
    after_arrays = {
        shape.id: editor.geometry.flattened(shape.id).vertices.copy()
        for shape in sample_project.shapes
    }
    changed_geometry = {
        shape_id
        for shape_id in before_geometry
        if before_geometry[shape_id] != after_geometry[shape_id]
    }
    changed_render = {
        shape_id
        for shape_id in before_render.shape_hashes
        if before_render.shape_hashes[shape_id] != after_render.shape_hashes[shape_id]
    }
    assert changed_geometry == {"tree_crown"}
    assert changed_render == {"tree_crown"}
    for shape_id, vertices in before_arrays.items():
        if shape_id == "tree_crown":
            assert not np.array_equal(vertices, after_arrays[shape_id])
        else:
            assert np.array_equal(vertices, after_arrays[shape_id])
    undo = editor.undo()
    assert undo.restored_version_stamp == sample_project.version_stamp
    restored = editor.render("lineart")
    assert editor.project.content_hash() == sample_project.content_hash()
    assert editor.geometry_hashes() == before_geometry
    assert restored.shape_hashes == before_render.shape_hashes
    assert restored.pixel_sha256 == before_render.pixel_sha256


def test_undo_stack_lifo_and_empty_error(sample_project):
    editor = Editor(sample_project, config=RenderConfig(width=160, height=100, supersample=1))
    original = editor.project.content_hash()
    editor.edit_control_point("bird", 0, 0, 2, 75.0, 38.0)
    first = editor.project.content_hash()
    editor.edit_control_point("bird", 0, 1, 2, 93.0, 37.0)
    assert editor.project.content_hash() not in (original, first)
    editor.undo()
    assert editor.project.content_hash() == first
    editor.undo()
    assert editor.project.content_hash() == original
    with pytest.raises(NprError) as caught:
        editor.undo()
    assert caught.value.code == "E_UNDO_EMPTY"


def test_invalid_stroke_sample_edit_raises_business_error(sample_project):
    editor = Editor(sample_project)
    before = editor.project.content_hash()
    with pytest.raises(NprError) as caught:
        editor.set_stroke_sample(curve_steps=0)
    assert caught.value.code == "E_STROKE_SAMPLE"
    assert editor.project.content_hash() == before


def test_cartoon_fill_and_lineart_use_identical_contour_coordinates(sample_project):
    engine = RenderEngine(sample_project, RenderConfig(width=180, height=120, supersample=1))
    engine.render("lineart")
    engine.render("toon")
    for shape in sample_project.shapes:
        skeleton = engine.geometry.ensure(shape.id)
        flattened = engine.geometry.flattened(shape.id)
        for path, contour in zip(skeleton.paths(), flattened.contours):
            boundary = path[:-1] if contour.closed else path
            difference = np.abs(boundary.astype(np.float32) - contour.vertices).max()
            assert difference == 0.0


def test_png_output_is_deterministic(sample_project):
    engine = RenderEngine(sample_project, RenderConfig(width=120, height=90, supersample=1))
    first = engine.render("toon")
    second = RenderEngine(sample_project, engine.config).render("toon")
    assert first.pixel_sha256 == second.pixel_sha256
    assert first.png_sha256 == second.png_sha256
    assert hashlib.sha256(encode_png(first.image)).hexdigest() == first.png_sha256
