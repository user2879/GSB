from __future__ import annotations

import copy

import pytest

import vgcover.engine as engine_module
from vgcover import (
    CoverageError,
    ErrorCode,
    IncrementalRenderer,
    assemble_ledger,
    generate_scene,
    ledger_hash,
    render_scene,
    scene_hash,
    verify_ledger,
)


def _ledger_entry(ledger, region_id):
    return next(entry for entry in ledger["regions"] if entry["region_id"] == region_id)


def test_ledger_links_components_and_image_layers_one_to_one():
    scene = generate_scene()
    render = render_scene(scene)
    ledger = render.ledger
    assert ledger["complete"] is True
    assert ledger["scene_hash"] == scene_hash(scene)
    assert ledger["image"]["complete"] is True
    assert ledger["image"]["layer_sequence"] == [10, 20, 30, 40]
    assert len(ledger["regions"]) == len(ledger["image"]["layer_hashes"]) == 4

    for entry in ledger["regions"]:
        assert entry["complete"] is True
        assert entry["total_error_bound"] is not None
        assert set(entry["components"]) == {"geometry", "quantization", "composite"}
        assert all(component["present"] for component in entry["components"].values())
        assert entry["layer_hash"].startswith(tuple("0123456789abcdef"))
        link = next(
            item for item in ledger["image"]["layer_hashes"] if item["region_id"] == entry["region_id"]
        )
        assert link["ledger_layer_hash"] == entry["layer_hash"]
        assert link["layer_content_hash"] == entry["layer_content_hash"]

    assert verify_ledger(scene, ledger)["ok"] is True
    assert ledger_hash(ledger) == ledger_hash(render.ledger)


def test_missing_error_component_marks_ledger_incomplete_without_zero_fill():
    scene = generate_scene()
    render = render_scene(scene)
    records, image_record = render.ledger_inputs()
    target = next(record for record in records if record["region_id"] == 20)
    del target["components"]["quantization"]
    ledger = assemble_ledger(
        scene,
        records,
        image_record,
        scene_hash=scene_hash(scene),
    )
    entry = _ledger_entry(ledger, 20)
    assert entry["complete"] is False
    assert entry["total_error_bound"] is None
    assert "quantization" in entry["missing_components"]
    assert ledger["complete"] is False
    assert ledger["image"]["complete"] is False
    assert ledger["image"]["total_error_bound"] is None
    assert any("region:20:quantization" == item for item in ledger["image"]["missing_components"])


def test_tampered_ledger_raises_mismatch_with_region_locator():
    scene = generate_scene()
    ledger = render_scene(scene).ledger
    tampered = copy.deepcopy(ledger)
    _ledger_entry(tampered, 10)["layer_hash"] = "0" * 64
    with pytest.raises(CoverageError) as caught:
        verify_ledger(scene, tampered)
    assert caught.value.code == ErrorCode.LEDGER_MISMATCH
    assert caught.value.region_id == 10


def test_incremental_tolerance_closure_is_exact_and_outside_is_stable():
    scene = generate_scene()
    engine = IncrementalRenderer(scene)
    before = engine.ledger
    plan = engine.plan_changes([{"region_id": 10, "field": "tolerance", "value": 0.06}])
    expected = {
        "geometry:r10",
        "coverage_nonzero:r10",
        "coverage_evenodd:r10",
        "coverage_selected:r10",
        "quantize:r10",
        "layer:r10",
        "ledger_region:r10",
        "composite",
        "ledger_image",
    }
    assert set(plan.closure) == expected
    receipt = engine.apply_changes(
        [{"region_id": 10, "field": "tolerance", "value": 0.06}],
        expected_closure=plan.closure,
    )
    assert set(receipt.recomputed) == expected
    assert receipt.outside_layer_hashes_preserved is True
    assert receipt.outside_ledger_entries_preserved is True
    for region_id in (20, 30, 40):
        assert _ledger_entry(engine.ledger, region_id) == _ledger_entry(before, region_id)
    assert verify_ledger(engine.scene, engine.ledger)["ok"] is True


def test_incremental_fill_rule_alpha_and_control_point_closures():
    engine = IncrementalRenderer(generate_scene())
    old_star = engine.states[20]
    assert old_star.dual.difference_pixel_count > 0

    rule_change = [{"region_id": 20, "field": "fill_rule", "value": "nonzero"}]
    rule_plan = engine.plan_changes(rule_change)
    assert set(rule_plan.closure) == {
        "coverage_selected:r20",
        "quantize:r20",
        "layer:r20",
        "ledger_region:r20",
        "composite",
        "ledger_image",
    }
    rule_receipt = engine.apply_changes(rule_change, expected_closure=rule_plan.closure)
    assert set(rule_receipt.recomputed) == set(rule_plan.closure)
    assert engine.states[20].selected_coverage.tobytes() == engine.states[20].dual.nonzero.coverage.tobytes()
    assert engine.states[20].layer_hash != old_star.layer_hash

    alpha_change = [{"region_id": 30, "field": "alpha", "value": 0.40}]
    before_alpha = engine.states[30]
    alpha_plan = engine.plan_changes(alpha_change)
    assert set(alpha_plan.closure) == {
        "layer:r30",
        "ledger_region:r30",
        "composite",
        "ledger_image",
    }
    alpha_receipt = engine.apply_changes(alpha_change, expected_closure=alpha_plan.closure)
    assert set(alpha_receipt.recomputed) == set(alpha_plan.closure)
    assert engine.states[30].geometry_hash == before_alpha.geometry_hash
    assert engine.states[30].quantization.gray_hash == before_alpha.quantization.gray_hash
    assert engine.states[30].layer_hash != before_alpha.layer_hash

    scene_now = engine.scene
    region10 = next(region for region in scene_now["regions"] if region["region_id"] == 10)
    points = copy.deepcopy(region10["contours"][0]["segments"][1]["control_points"])
    points[1][0] = round(points[1][0] + 0.75, 6)
    points[1][1] = round(points[1][1] - 0.5, 6)
    old_geometry_hash = engine.states[10].geometry_hash
    control_change = [
        {
            "region_id": 10,
            "field": "control_points",
            "contour_index": 0,
            "segment_index": 1,
            "value": points,
        }
    ]
    control_plan = engine.plan_changes(control_change)
    assert "geometry:r10" in control_plan.closure
    assert "coverage_nonzero:r10" in control_plan.closure
    assert "coverage_evenodd:r10" in control_plan.closure
    control_receipt = engine.apply_changes(control_change, expected_closure=control_plan.closure)
    assert set(control_receipt.recomputed) == set(control_plan.closure)
    assert engine.states[10].geometry_hash != old_geometry_hash
    assert verify_ledger(engine.scene, engine.ledger)["ok"] is True


def test_incremental_alpha_change_does_not_invoke_discretization(monkeypatch):
    engine = IncrementalRenderer(generate_scene())
    calls = []
    original = engine_module.discretize_region

    def counted(region):
        calls.append(region["region_id"])
        return original(region)

    def forbidden_full_render(scene):
        raise AssertionError("incremental changes must not call render_scene")

    monkeypatch.setattr(engine_module, "discretize_region", counted)
    monkeypatch.setattr(engine_module, "render_scene", forbidden_full_render)
    plan = engine.plan_changes([{"region_id": 10, "field": "alpha", "value": 0.55}])
    engine.apply_changes(
        [{"region_id": 10, "field": "alpha", "value": 0.55}],
        expected_closure=plan.closure,
    )
    assert calls == []


def test_wrong_or_inconsistent_closure_raises_closure_mismatch():
    engine = IncrementalRenderer(generate_scene())
    change = [{"region_id": 10, "field": "tolerance", "value": 0.07}]
    plan = engine.plan_changes(change)
    wrong = [node for node in plan.closure if node != "quantize:r10"]
    with pytest.raises(CoverageError) as caught:
        engine.apply_changes(change, expected_closure=wrong)
    assert caught.value.code == ErrorCode.CLOSURE_MISMATCH
    assert caught.value.region_id == 10
    assert "quantize:r10" in caught.value.details["missing"]


def test_incremental_invalid_changes_retain_specific_error_codes():
    engine = IncrementalRenderer(generate_scene())
    with pytest.raises(CoverageError) as alpha_error:
        engine.plan_changes([{"region_id": 10, "field": "alpha", "value": 2.0}])
    assert alpha_error.value.code == ErrorCode.ALPHA_RANGE
    assert alpha_error.value.region_id == 10

    with pytest.raises(CoverageError) as tolerance_error:
        engine.plan_changes([{"region_id": 10, "field": "tolerance", "value": 0.0}])
    assert tolerance_error.value.code == ErrorCode.TOLERANCE_RANGE

    with pytest.raises(CoverageError) as rule_error:
        engine.plan_changes([{"region_id": 10, "field": "fill_rule", "value": "bad"}])
    assert rule_error.value.code == ErrorCode.FILL_RULE
