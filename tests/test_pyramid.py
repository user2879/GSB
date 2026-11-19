import numpy as np

from terrain_workbench.pyramid import (
    FILTER_NAME, build_pyramid, downsample_tent, level_hashes, load_pyramid,
    save_pyramid,
)


def test_level_sizes(pyramid257):
    levels, meta = pyramid257
    assert [l.shape[0] for l in levels] == [257, 129, 65, 33, 17, 9, 5, 3]
    assert meta["filter"] == FILTER_NAME
    assert len(meta["cumulative_dev"]) == 8


def test_constant_field_preserved():
    c = np.full((257, 257), 12345, np.uint16)
    down = downsample_tent(c)
    assert down.shape == (129, 129) and (down == 12345).all()


def test_deterministic_hashes(heights257):
    h1 = level_hashes(build_pyramid(heights257)[0])
    h2 = level_hashes(build_pyramid(heights257)[0])
    assert h1 == h2


def test_deviation_bounds_recorded(pyramid257):
    _, meta = pyramid257
    for entry in meta["levels"]:
        assert entry["max_dev_vs_prev"] >= 0
        assert entry["cumulative_dev_vs_level0"] >= entry["max_dev_vs_prev"]


def test_save_load_roundtrip(pyramid257, tmp_path):
    levels, meta = pyramid257
    full = save_pyramid(levels, meta, tmp_path, "t257")
    loaded, loaded_meta = load_pyramid(tmp_path, "t257")
    assert level_hashes(levels) == level_hashes(loaded) == full["sha256"]
    assert loaded_meta["filter"] == FILTER_NAME
