from __future__ import annotations

import numpy as np

from rsseg.config import SegmentationConfig, load_config
from rsseg.edits import copy_sample_set
from rsseg.features import classify_tile
from rsseg.fingerprint import (
    band_content_fingerprint,
    fusion_fingerprint,
    index_adjacency_version,
    tile_cache_key_with_version,
)
from rsseg.fusion import build_fusion_regions
from rsseg.tiles import load_index, load_tile_bands


def test_band_fingerprint_changes_with_pixel_content(sample_set):
    index = load_index(sample_set / "tile_index.json")
    cfg = load_config(sample_set / "segmentation_config.json")
    tile = index.tiles[0]
    bands = load_tile_bands(tile, sample_set, index)
    modified = bands.copy()
    modified[0, 0, 2] = (modified[0, 0, 2] + 1) % 256
    assert band_content_fingerprint(bands) != band_content_fingerprint(modified)
    key_a = tile_cache_key_with_version(tile, bands, cfg, index.band_order_version)
    key_b = tile_cache_key_with_version(tile, modified, cfg, index.band_order_version)
    assert key_a != key_b


def test_fusion_fingerprint_includes_strategy_and_adjacency(sample_set):
    index = load_index(sample_set / "tile_index.json")
    cfg = load_config(sample_set / "segmentation_config.json")
    regions = build_fusion_regions(index)
    region = regions[0]
    tile_keys = {tile.key: f"key-{tile.row}-{tile.col}" for tile in index.tiles}
    first = fusion_fingerprint(region, tile_keys, cfg, "adj-v1")
    other_cfg = SegmentationConfig.from_dict({**cfg.to_dict(), "fusion_strategy": "first_wins"})
    second = fusion_fingerprint(region, tile_keys, other_cfg, "adj-v1")
    third = fusion_fingerprint(region, tile_keys, cfg, "adj-v2")
    assert first != second
    assert first != third


def test_first_wins_strategy_is_deterministic(sample_set, tmp_path):
    from rsseg.engine import IncrementalEngine
    from rsseg.runner import build_inputs
    from rsseg.snapshot import SnapshotManager

    sources = tmp_path / "sources"
    work_a = tmp_path / "a"
    work_b = tmp_path / "b"
    copy_sample_set(sample_set, sources)
    override = {"fusion_strategy": "first_wins"}
    IncrementalEngine(build_inputs(sources, work_a, config_override=override)).run(force_full=True)
    IncrementalEngine(build_inputs(sources, work_b, config_override=override)).run(force_full=True)
    grid_a = np.load(SnapshotManager(work_a).latest().path / "global_grid.npy")
    grid_b = np.load(SnapshotManager(work_b).latest().path / "global_grid.npy")
    assert np.array_equal(grid_a, grid_b)


def test_adjacency_version_reacts_to_edge_changes(sample_set, tmp_path):
    from rsseg.edits import apply_edit

    sources = tmp_path / "sources"
    copy_sample_set(sample_set, sources)
    before = index_adjacency_version(load_index(sources / "tile_index.json"))
    apply_edit(sources, {"id": "edge", "type": "adjacency", "remove": [[0, 0], [0, 1]]})
    after = index_adjacency_version(load_index(sources / "tile_index.json"))
    assert before != after
