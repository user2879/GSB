from __future__ import annotations

import copy
import math

import pytest

from vgcover import (
    CATEGORIES,
    CoverageError,
    ErrorCode,
    discretize_contour,
    discretize_region,
    generate_category_scenes,
    generate_scene,
    geometry_point_sequence,
    geometry_vertex_count,
    scene_hash,
    scene_json_bytes,
    validate_scene,
)


def _region(scene, region_id):
    return next(region for region in scene["regions"] if region["region_id"] == region_id)


def test_scene_generation_is_byte_deterministic_and_covers_all_categories():
    first = generate_scene(seed=123456, width=192, height=128)
    second = generate_scene(seed=123456, width=192, height=128)
    assert scene_json_bytes(first) == scene_json_bytes(second)
    assert scene_hash(first) == scene_hash(second)
    assert {region["category"] for region in first["regions"]} == set(CATEGORIES)

    another = generate_scene(seed=123457, width=192, height=128)
    assert scene_hash(another) != scene_hash(first)

    flagged = [region for region in first["regions"] if region["degenerate"]["flagged"]]
    assert len(flagged) == 1
    near = flagged[0]
    assert near["category"] == "near_degenerate_strip"
    assert near["degenerate"]["retained"] is True
    assert near["degenerate"]["minimum_feature_width_px"] < near["degenerate"]["threshold_px"]
    assert near["contours"]

    for region in first["regions"]:
        assert region["degenerate"]["retained"] is True
        for contour in region["contours"]:
            assert contour["closed"] is True
            assert contour["segments"]
            first_start = contour["segments"][0]["control_points"][0]
            last_end = contour["segments"][-1]["control_points"][-1]
            assert first_start == last_end


def test_category_catalogue_is_deterministic_and_isolated():
    first = generate_category_scenes(seed=77)
    second = generate_category_scenes(seed=77)
    assert list(first) == list(CATEGORIES)
    for category in CATEGORIES:
        assert scene_json_bytes(first[category]) == scene_json_bytes(second[category])
        assert len(first[category]["regions"]) == 1
        assert first[category]["regions"][0]["category"] == category


def test_discretization_is_deterministic_and_tolerance_monotone():
    scene = generate_scene(seed=2026)
    base_region = _region(scene, 10)
    first = discretize_region(base_region)
    second = discretize_region(copy.deepcopy(base_region))
    assert first["geometry_hash"] == second["geometry_hash"]
    assert geometry_point_sequence(first) == geometry_point_sequence(second)

    loose_region = copy.deepcopy(base_region)
    loose_region["tolerance"] = 0.30
    tight_region = copy.deepcopy(base_region)
    tight_region["tolerance"] = 0.05
    loose = discretize_region(loose_region)
    tight = discretize_region(tight_region)
    assert geometry_vertex_count(tight) >= geometry_vertex_count(loose)
    assert tight["summary"]["max_analytic_error_bound_px"] <= 0.05 + 1e-12
    assert loose["summary"]["max_analytic_error_bound_px"] <= 0.30 + 1e-12

    for contour in tight["contours"]:
        for segment in contour["segments"]:
            assert segment["analytic_error_bound"] + 1e-12 >= segment["measured_chord_height"]


def test_single_contour_discretization_api_matches_region_report():
    scene = generate_scene()
    region = _region(scene, 10)
    contour = region["contours"][0]
    single = discretize_contour(
        contour,
        region["tolerance"],
        region_id=10,
        contour_index=0,
    )
    whole = discretize_region(region)["contours"][0]
    assert single["points"] == whole["points"]
    assert single["segments"] == whole["segments"]


def test_line_segments_have_zero_geometric_error_and_near_strip_is_retained():
    scene = generate_scene()
    strip = _region(scene, 40)
    geometry = discretize_region(strip)
    assert geometry["contours"]
    assert geometry["summary"]["max_analytic_error_bound_px"] == 0.0
    assert all(
        segment["type"] == "line" and segment["analytic_error_bound"] == 0.0
        for segment in geometry["contours"][0]["segments"]
    )


@pytest.mark.parametrize("tolerance", [0, -0.1, float("nan")])
def test_tolerance_range_error_has_region_locator(tolerance):
    scene = generate_scene()
    region = copy.deepcopy(_region(scene, 10))
    region["tolerance"] = tolerance
    with pytest.raises(CoverageError) as caught:
        validate_scene({**scene, "regions": [region]})
    assert caught.value.code == ErrorCode.TOLERANCE_RANGE
    assert caught.value.region_id == 10


def test_invalid_control_point_and_unclosed_contour_have_indices():
    scene = generate_scene()
    invalid = copy.deepcopy(scene)
    invalid["regions"][0]["contours"][0]["segments"][1]["control_points"][1][0] = math.nan
    with pytest.raises(CoverageError) as caught:
        validate_scene(invalid)
    assert caught.value.code == ErrorCode.SCENE_INVALID
    assert caught.value.region_id == 10
    assert caught.value.contour_index == 0
    assert caught.value.segment_index == 1

    unclosed = copy.deepcopy(scene)
    strip = _region(unclosed, 40)
    last_segment = strip["contours"][0]["segments"][-1]
    last_segment["control_points"][1][0] += 0.25
    with pytest.raises(CoverageError) as caught:
        validate_scene(unclosed)
    assert caught.value.code == ErrorCode.CONTOUR_UNCLOSED
    assert caught.value.region_id == 40
    assert caught.value.contour_index == 0
    assert caught.value.segment_index == 3


def test_fill_rule_and_alpha_errors_are_not_silently_defaulted():
    scene = generate_scene()
    invalid_rule = copy.deepcopy(scene)
    invalid_rule["regions"][0]["fill_rule"] = "winding-ish"
    with pytest.raises(CoverageError) as caught_rule:
        validate_scene(invalid_rule)
    assert caught_rule.value.code == ErrorCode.FILL_RULE
    assert caught_rule.value.region_id == 10

    invalid_alpha = copy.deepcopy(scene)
    invalid_alpha["regions"][0]["alpha"] = 1.25
    with pytest.raises(CoverageError) as caught_alpha:
        validate_scene(invalid_alpha)
    assert caught_alpha.value.code == ErrorCode.ALPHA_RANGE
    assert caught_alpha.value.region_id == 10
