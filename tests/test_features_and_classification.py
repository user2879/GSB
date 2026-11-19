from __future__ import annotations

import numpy as np

from rsseg.config import load_config
from rsseg.features import (
    SCALE,
    classify_tile,
    extract_window_features,
    generate_windows,
)
from rsseg.tiles import load_index, load_tile_bands


def test_windows_partition_tile(sample_set):
    cfg = load_config(sample_set / "segmentation_config.json")
    windows = generate_windows(32, cfg)
    coverage = np.zeros((32, 32), dtype=bool)
    for window in windows:
        region = coverage[window.core_top:window.core_bottom,
                          window.core_left:window.core_right]
        assert not region.any(), "core regions must not overlap"
        region[:] = True
    assert coverage.all()


def test_features_are_integer_and_deterministic(sample_set):
    cfg = load_config(sample_set / "segmentation_config.json")
    index = load_index(sample_set / "tile_index.json")
    bands = load_tile_bands(index.tiles[0], sample_set, index)
    window = generate_windows(32, cfg)[0]
    first = extract_window_features(bands, window)
    second = extract_window_features(bands, window)
    for name, plane in first.items():
        assert np.issubdtype(plane.dtype, np.integer)
        assert np.array_equal(plane, second[name])


def test_classification_outputs_are_stable(sample_set):
    cfg = load_config(sample_set / "segmentation_config.json")
    index = load_index(sample_set / "tile_index.json")
    bands = load_tile_bands(index.tiles[0], sample_set, index)
    class_map_a, confidence_a, scores_a = classify_tile(bands, cfg)
    class_map_b, confidence_b, scores_b = classify_tile(bands, cfg)
    assert np.array_equal(class_map_a, class_map_b)
    assert np.array_equal(confidence_a, confidence_b)
    assert np.array_equal(scores_a, scores_b)
    assert class_map_a.dtype == np.uint8
    assert confidence_a.dtype == np.uint8
    assert set(np.unique(class_map_a)).issubset({1, 2, 3, 4})


def test_tie_break_prefers_lowest_class_code():
    from rsseg.config import SegmentationConfig

    cfg = SegmentationConfig.from_dict({
        "window_size": 8,
        "stride": 8,
        "context_overlap": 0,
        "feature_thresholds": {"ndvi": 10.0, "ndwi": 10.0, "edge": 10.0, "slope": 10.0},
        "class_weights": {
            "farmland": {"ndvi": 0.0, "ndwi": 0.0, "edge": 0.0, "slope": 0.0},
            "water": {"ndvi": 0.0, "ndwi": 0.0, "edge": 0.0, "slope": 0.0},
            "building": {"ndvi": 0.0, "ndwi": 0.0, "edge": 0.0, "slope": 0.0},
            "road": {"ndvi": 0.0, "ndwi": 0.0, "edge": 0.0, "slope": 0.0},
        },
        "min_component_area": 1,
        "fusion_strategy": "distance_weighted",
        "recompute_budget": 10,
        "workers": 1,
        "snapshot_commit_interval": 1,
    })
    bands = np.full((8, 8, 4), 5, dtype=np.uint8)
    class_map, _confidence, scores = classify_tile(bands, cfg)
    assert np.all(class_map == 1)
    assert np.all(scores[0] == scores[1])


def test_threshold_change_changes_fingerprint(sample_set):
    cfg = load_config(sample_set / "segmentation_config.json")
    modified = cfg.to_dict()
    modified["feature_thresholds"]["ndvi"] = 0.5
    from rsseg.config import SegmentationConfig

    other = SegmentationConfig.from_dict(modified)
    assert cfg.fingerprint() != other.fingerprint()
