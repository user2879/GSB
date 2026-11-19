from __future__ import annotations

import numpy as np

from rsseg.config import load_config
from rsseg.fusion import (
    _weighted_vote,
    assemble_global,
    build_fusion_regions,
    connected_components,
    postprocess,
)
from rsseg.tiles import load_index


def test_fusion_regions_match_overlap_grid(sample_set):
    index = load_index(sample_set / "tile_index.json")
    regions = build_fusion_regions(index)
    assert len(regions) == 12
    for region in regions:
        assert region.a < region.b
        assert region.global_x1 > region.global_x0
        assert region.global_y1 > region.global_y0


def test_tie_break_uses_lexicographic_tile_key():
    patch_a = np.full((4, 4), 2, dtype=np.uint8)
    patch_b = np.full((4, 4), 3, dtype=np.uint8)
    equal_weights = np.ones((4, 4), dtype=np.int64)
    fused = _weighted_vote(patch_a, patch_b, equal_weights, equal_weights, (0, 0), (0, 1))
    assert np.all(fused == 2)
    fused_reverse = _weighted_vote(patch_b, patch_a, equal_weights, equal_weights, (0, 1), (1, 0))
    assert np.all(fused_reverse == 3)


def test_distance_weighted_vote_prefers_closer_tile():
    patch_a = np.full((3, 3), 1, dtype=np.uint8)
    patch_b = np.full((3, 3), 4, dtype=np.uint8)
    weight_a = np.array([[3, 2, 1], [3, 2, 1], [3, 2, 1]], dtype=np.int64)
    weight_b = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=np.int64)
    fused = _weighted_vote(patch_a, patch_b, weight_a, weight_b, (0, 0), (0, 1))
    assert fused[0, 0] == 1
    assert fused[2, 2] == 4
    assert fused[1, 1] == 1  # tie resolved by lexicographic key


def test_global_assembly_is_deterministic(sample_set):
    cfg = load_config(sample_set / "segmentation_config.json")
    index = load_index(sample_set / "tile_index.json")
    class_maps = {
        tile.key: np.random.default_rng(tile.row * 7 + tile.col).integers(
            1, 5, size=(32, 32), dtype=np.uint8)
        for tile in index.tiles
    }
    first = assemble_global(index, class_maps, cfg)
    second = assemble_global(index, class_maps, cfg)
    assert np.array_equal(first, second)
    assert first.shape == (80, 80)


def test_connected_components_4_connectivity():
    plane = np.array([
        [1, 1, 2, 2],
        [1, 2, 2, 3],
        [4, 4, 3, 3],
        [4, 4, 3, 3],
    ], dtype=np.uint8)
    labels, count = connected_components(plane)
    assert count == 4
    assert labels[0, 0] == labels[1, 0]
    assert labels[0, 0] != labels[0, 2]


def test_postprocess_fills_small_components_with_neighbor_majority():
    plane = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 2, 1, 1],
        [1, 2, 2, 1, 1],
        [1, 1, 1, 3, 3],
        [1, 1, 1, 3, 3],
    ], dtype=np.uint8)
    processed, changes = postprocess(plane, min_area=5)
    assert np.all(processed == 1)
    assert changes and all(change["area"] < 5 for change in changes)
    assert {change["fill_class"] for change in changes} == {1}


def test_postprocess_records_are_deterministic():
    rng = np.random.default_rng(11)
    plane = rng.integers(1, 5, size=(12, 12)).astype(np.uint8)
    grid_a, changes_a = postprocess(plane, 3)
    grid_b, changes_b = postprocess(plane, 3)
    assert np.array_equal(grid_a, grid_b)
    assert changes_a == changes_b
