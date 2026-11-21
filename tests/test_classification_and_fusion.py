from __future__ import annotations

from dataclasses import replace

import numpy as np

from remote_seg.classify import classify_tile
from remote_seg.fusion import _vote, compute_region
from remote_seg.inputs import load_config, load_index, load_registry, load_tiles


def test_classification_is_deterministic_and_tie_uses_lowest_code(sample_workspace) -> None:
    config = load_config(sample_workspace["config"])
    registry = load_registry(sample_workspace["fusion"])
    index = load_index(sample_workspace["index"])
    loaded = load_tiles(index, config, registry)
    tile_input = next(iter(loaded.loaded.values()))
    first = classify_tile(tile_input, config)
    second = classify_tile(tile_input, config)
    assert first.cache_key == second.cache_key
    assert np.array_equal(first.classes, second.classes)
    assert np.array_equal(first.scores, second.scores)
    assert first.score_summary == second.score_summary
    assert first.windows == second.windows

    tied_weights = {
        code: {"bias": 0.0, "vegetation": 0.0, "water": 0.0, "edge": 0.0, "slope": 0.0}
        for code in config.class_codes
    }
    tied_config = replace(config, classification_weights=tied_weights)
    tied = classify_tile(tile_input, tied_config)
    assert np.all(tied.classes == 0)


def test_weighted_vote_tie_breaks_by_row_col_lexicographic() -> None:
    winner, votes, sources = _vote(
        [
            ((1, 0, "r1c0"), 2, 1.0),
            ((0, 1, "r0c1"), 1, 1.0),
            ((0, 0, "r0c0"), 3, 0.25),
        ]
    )
    assert votes == {"1": 1.0, "2": 1.0, "3": 0.25}
    assert winner == 1
    assert [source["tile_id"] for source in sources] == ["r0c0", "r0c1", "r1c0"]


def test_pair_region_records_sources_and_votes(sample_workspace) -> None:
    config = load_config(sample_workspace["config"])
    registry = load_registry(sample_workspace["fusion"])
    index = load_index(sample_workspace["index"])
    loaded = load_tiles(index, config, registry)
    strategy = registry.require(config.fusion_version)
    tile_results = {tile_id: classify_tile(value, config) for tile_id, value in loaded.loaded.items()}
    spec_a = index.tiles[0]
    spec_b = index.by_id[spec_a.neighbors[0]]
    from remote_seg.fusion import overlap_bounds

    bounds = overlap_bounds(spec_a, spec_b)
    assert bounds is not None
    region = compute_region(
        tile_results[spec_a.tile_id], spec_a, tile_results[spec_b.tile_id], spec_b, bounds, strategy
    )
    assert region.votes
    assert all(len(record["sources"]) == 2 for record in region.votes)
    assert all(sum(record["votes"].values()) > 0 for record in region.votes)
    assert region.region_version == region.fingerprint
