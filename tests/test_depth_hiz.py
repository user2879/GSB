import numpy as np
import pytest

from terrain_culling.config import DEFAULT_CONFIG
from terrain_culling.camera import Camera, generate_track
from terrain_culling.terrain import TerrainDataset
from terrain_culling.depth import DepthRenderer
from terrain_culling.errors import HiZLevelError, DepthBiasError


@pytest.fixture(scope="module")
def dataset():
    return TerrainDataset.generate(512)


def test_hiz_max_downsample_conservative(dataset):
    cfg = DEFAULT_CONFIG
    renderer = DepthRenderer(cfg.depth_width, cfg.depth_height,
                             cfg.raymarch_steps, cfg.max_hiz_levels)
    camera = Camera(cfg.depth_width, cfg.depth_height)
    frame = generate_track(4, 512)[0]
    resident = {b.block_id for b in dataset.leaf_blocks()}
    depth = renderer.render_depth(dataset, frame, resident)
    levels, infos = renderer.build_hiz(depth)
    assert len(levels) == len(infos)
    # Every coarse pixel must equal the max of its 2x2 fine footprint.
    for fine, coarse in zip(levels[:-1], levels[1:]):
        h, w = fine.shape
        padded_h = h if h % 2 == 0 else h + 1
        padded_w = w if w % 2 == 0 else w + 1
        padded = np.full((padded_h, padded_w), -np.inf)
        padded[:h, :w] = fine
        expected = np.maximum(
            np.maximum(padded[0::2, 0::2], padded[1::2, 0::2]),
            np.maximum(padded[0::2, 1::2], padded[1::2, 1::2]))
        expected = np.where(np.isneginf(expected), np.inf, expected)
        assert np.allclose(np.nan_to_num(coarse, nan=1e9),
                           np.nan_to_num(expected, nan=1e9), equal_nan=True)


def test_hiz_error_table_recomputable(dataset):
    renderer = DepthRenderer(192, 128, 160, 6)
    levels, infos = renderer.build_hiz(np.full((128, 192), 100.0))
    for info in infos:
        assert info.depth_error_bound > 0
        assert info.texel_world_x > 0 and info.texel_world_z > 0
        assert info.width >= 1 and info.height >= 1
    # Errors are monotonic non-decreasing as resolution drops.
    bounds = [info.depth_error_bound for info in infos]
    assert bounds == sorted(bounds)


def test_hiz_level_out_of_range(dataset):
    renderer = DepthRenderer(192, 128, 160, 4)
    _, infos = renderer.build_hiz(np.full((128, 192), 1.0))
    with pytest.raises(HiZLevelError) as exc:
        renderer.check_level(99, infos)
    assert exc.value.code.value == "E_HIZ_LEVEL"


def test_depth_bias_limit():
    renderer = DepthRenderer(192, 128, 160, 4, max_depth_bias=0.05)
    renderer.check_depth_bias(0.01)
    with pytest.raises(DepthBiasError) as exc:
        renderer.check_depth_bias(0.5)
    assert exc.value.code.value == "E_DEPTH_BIAS"
    with pytest.raises(DepthBiasError):
        renderer.check_depth_bias(-0.1)


def test_depth_render_deterministic(dataset):
    cfg = DEFAULT_CONFIG
    renderer = DepthRenderer(cfg.depth_width, cfg.depth_height,
                             cfg.raymarch_steps, cfg.max_hiz_levels)
    camera = Camera(cfg.depth_width, cfg.depth_height)
    frame = generate_track(4, 512)[1]
    resident = {b.block_id for b in dataset.leaf_blocks()}
    d1 = renderer.render_depth(dataset, frame, resident)
    d2 = renderer.render_depth(dataset, frame, resident)
    assert np.array_equal(d1, d2)
    assert np.isfinite(d1).any()
