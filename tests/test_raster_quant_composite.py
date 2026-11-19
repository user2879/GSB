from __future__ import annotations

import copy

import numpy as np
import pytest

from vgcover import (
    COMPOSITE_POLICY,
    QUANTIZATION_POLICY,
    CoverageError,
    ErrorCode,
    build_source_layer,
    compose_layers,
    coverage_difference,
    discretize_region,
    generate_scene,
    quantize_coverage,
    rasterize_geometry_dual,
    rasterize_polylines,
    rasterize_polylines_dual,
    reconstruct_coverage,
)


def _region(scene, region_id):
    return next(region for region in scene["regions"] if region["region_id"] == region_id)


def test_pixel_centre_ray_square_coverage_and_dependency_counts():
    square = [
        [1.2, 1.2],
        [3.8, 1.2],
        [3.8, 3.8],
        [1.2, 3.8],
    ]
    nonzero = rasterize_polylines([square], 5, 5, "nonzero")
    assert nonzero.coverage.dtype == np.float32
    assert int(np.count_nonzero(nonzero.coverage)) == 9
    assert nonzero.coverage[2, 2] == 1.0
    assert nonzero.coverage[0, 0] == 0.0
    assert nonzero.coverage[4, 4] == 0.0
    assert nonzero.dependency_edge_counts[2, 2] == 1
    assert nonzero.dependency_edge_counts[2, 0] == 2
    assert nonzero.dependency_edge_counts[2, 4] == 0

    evenodd = rasterize_polylines([square], 5, 5, "evenodd")
    assert np.array_equal(nonzero.coverage, evenodd.coverage)
    assert np.count_nonzero(coverage_difference(nonzero.coverage, evenodd.coverage)) == 0


def test_invalid_fill_rule_raises_business_error():
    with pytest.raises(CoverageError) as caught:
        rasterize_polylines([[[0, 0], [1, 0], [1, 1]]], 2, 2, "default")
    assert caught.value.code == ErrorCode.FILL_RULE


def test_nonzero_and_evenodd_rule_difference_is_per_pixel_exportable():
    scene = generate_scene()
    geometry = discretize_region(_region(scene, 20))
    dual = rasterize_geometry_dual(geometry, scene["width"], scene["height"])
    assert dual.difference.dtype == np.float32
    assert dual.difference_pixel_count > 0
    assert int(np.count_nonzero(dual.difference)) == dual.difference_pixel_count
    assert np.array_equal(dual.difference, dual.nonzero.coverage - dual.evenodd.coverage)
    assert np.count_nonzero(dual.nonzero.coverage) != np.count_nonzero(dual.evenodd.coverage)


def test_quantization_rounding_reconstruction_and_determinism():
    coverage = np.asarray([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    first = quantize_coverage(coverage)
    second = quantize_coverage(coverage.copy())
    assert np.array_equal(first.gray, np.asarray([[0, 64, 128, 191, 255]], dtype=np.uint8))
    assert first.gray.dtype == np.uint8
    assert first.gray.tobytes() == second.gray.tobytes()
    assert first.gray_hash == second.gray_hash
    assert first.max_abs_error <= QUANTIZATION_POLICY["max_abs_error_bound"] + 1e-7
    assert first.over_tolerance_pixels == 0
    assert np.array_equal(first.reconstructed_coverage, reconstruct_coverage(first.gray))
    assert first.gray.min() >= 0 and first.gray.max() <= 255
    assert first.distribution["max_abs_error_bound"] == QUANTIZATION_POLICY["max_abs_error_bound"]
    assert first.distribution["reconstruction_max_abs_error"] == first.max_abs_error
    assert first.distribution["total_pixels"] == 5
    assert sum(first.distribution["abs_error_histogram"]["counts"]) == 5


def test_quantization_rejects_out_of_range_without_clamping():
    with pytest.raises(CoverageError) as caught:
        quantize_coverage(np.asarray([[0.0, 1.2]], dtype=np.float32))
    assert caught.value.code == ErrorCode.COVERAGE_RANGE


def test_premultiplied_bgra_composition_order_rounding_and_hashes():
    gray = np.asarray([[255, 0]], dtype=np.uint8)
    layer_two = build_source_layer(
        region_id=2,
        gray=gray,
        fill_color=(100, 110, 120),
        alpha=0.5,
    )
    layer_one = build_source_layer(
        region_id=1,
        gray=gray,
        fill_color=(10, 20, 30),
        alpha=1.0,
    )
    assert np.array_equal(layer_one.bgra[0, 0], np.asarray([30, 20, 10, 255], dtype=np.uint8))
    assert np.array_equal(layer_two.bgra[0, 0], np.asarray([60, 55, 50, 128], dtype=np.uint8))
    assert np.array_equal(layer_two.bgra[0, 1], np.asarray([0, 0, 0, 0], dtype=np.uint8))

    first = compose_layers([layer_two, layer_one])
    second = compose_layers([copy.deepcopy(layer_one), copy.deepcopy(layer_two)])
    assert first.layer_sequence == [1, 2]
    assert first.buffer.dtype == np.uint8
    assert np.array_equal(first.buffer[0, 0], np.asarray([75, 65, 55, 255], dtype=np.uint8))
    assert np.array_equal(first.buffer[0, 1], np.asarray([0, 0, 0, 0], dtype=np.uint8))
    assert first.buffer.tobytes() == second.buffer.tobytes()
    assert first.buffer_hash == second.buffer_hash
    assert first.layer_hashes == second.layer_hashes
    assert first.steps[0].step_rounding_error_bound_u8 == 2.0
    assert first.steps[0].cumulative_rounding_error_bound_u8 == 2.0
    assert first.steps[1].cumulative_rounding_error_bound_u8 == 4.0
    assert first.policy == COMPOSITE_POLICY


def test_alpha_range_error_has_region_locator():
    with pytest.raises(CoverageError) as caught:
        build_source_layer(
            region_id=77,
            gray=np.zeros((1, 1), dtype=np.uint8),
            fill_color=(1, 2, 3),
            alpha=1.5,
        )
    assert caught.value.code == ErrorCode.ALPHA_RANGE
    assert caught.value.region_id == 77
