"""Editing, undo, display modes and pipeline consistency."""
import numpy as np

from cadapp import GEOMETRY_TOLERANCE
from cadapp.display import DisplayManager, DisplayMode
from cadapp.editor import EditorSession
from cadapp.geometry import evaluate_surface_point, parameter_domain
from cadapp.model import content_hash
from cadapp.samples import make_sample_model


def test_edit_recomputes_only_affected_patch():
    model = make_sample_model()
    display = DisplayManager(model)
    editor = EditorSession(model, display)
    for p in model.patches:
        display.geometry(p.patch_id)
    normals_before = {p.patch_id: display.geometry(p.patch_id).normals.copy()
                      for p in model.patches}
    rebuilds = display.rebuild_count

    editor.edit_control_point("patch_dome", 2, 2, [0.6, 0.0, 1.9])
    assert display.rebuild_count == rebuilds  # invalidation is lazy

    dome_after = display.geometry("patch_dome")
    assert display.rebuild_count == rebuilds + 1  # exactly one patch rebuilt
    assert not np.array_equal(dome_after.normals, normals_before["patch_dome"])
    for pid in ("patch_wave", "patch_saddle"):
        assert np.array_equal(display.geometry(pid).normals, normals_before[pid])


def test_mode_switch_never_rebuilds():
    model = make_sample_model()
    display = DisplayManager(model)
    for p in model.patches:
        display.geometry(p.patch_id)
    rebuilds = display.rebuild_count
    for mode in DisplayMode.ALL * 2:
        display.set_mode(mode)
        for p in model.patches:
            display.buffers_for_mode(p.patch_id)
    assert display.rebuild_count == rebuilds


def test_undo_restores_hash_and_geometry():
    model = make_sample_model()
    display = DisplayManager(model)
    editor = EditorSession(model, display)
    pre_hash = content_hash(model)
    pre_normals = display.geometry("patch_wave").normals.copy()

    editor.edit_control_point("patch_wave", 0, 0, [9.0, 9.0, 9.0])
    assert content_hash(model) != pre_hash
    editor.undo()
    assert content_hash(model) == pre_hash
    assert np.array_equal(display.geometry("patch_wave").normals, pre_normals)
    assert editor.undo_depth == 0


def test_isocurve_and_mesh_share_evaluation_pipeline():
    model = make_sample_model()
    for patch in model.patches:
        u0, u1 = parameter_domain(patch.knots_u, patch.degree_u)
        v0, v1 = parameter_domain(patch.knots_v, patch.degree_v)
        for fu in (0.13, 0.5, 0.87):
            for fv in (0.21, 0.5, 0.79):
                u = u0 + fu * (u1 - u0)
                v = v0 + fv * (v1 - v0)
                a = evaluate_surface_point(patch, u, v)  # isocurve sampler
                b = evaluate_surface_point(patch, u, v)  # mesh sampler
                assert np.abs(a - b).max() <= GEOMETRY_TOLERANCE


def test_software_render_deterministic():
    hashes = []
    for _ in range(2):
        model = make_sample_model()
        display = DisplayManager(model)
        per_mode = {}
        for mode in DisplayMode.ALL:
            display.set_mode(mode)
            per_mode[mode] = display.render(128, 128)
        hashes.append(per_mode)
    for mode in DisplayMode.ALL:
        assert hashes[0][mode] == hashes[1][mode]
    # Modes produce distinct images.
    assert len({hashes[0][m] for m in DisplayMode.ALL}) == 3
