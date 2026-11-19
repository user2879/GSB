import hashlib
import json
import os

import numpy as np
import pytest

from terrain_culling.config import HEIGHT_SEED, MATERIAL_SEED
from terrain_culling.terrain import (
    TerrainDataset, fbm_heightfield, material_partition, build_quadtree,
)
from terrain_culling.errors import HeightRangeError


@pytest.mark.parametrize("size", [512, 2048])
def test_heightfield_deterministic_and_valid(size):
    a = fbm_heightfield(size)
    b = fbm_heightfield(size)
    assert a.dtype == np.uint16
    assert a.shape == (size, size)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() <= 65535


def test_material_deterministic():
    a = material_partition(512)
    b = material_partition(512)
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert set(np.unique(a)).issubset(set(range(5)))


def test_unsupported_size_raises():
    with pytest.raises(HeightRangeError) as exc:
        fbm_heightfield(256)
    assert exc.value.code.value == "E_HEIGHT_RANGE"


def test_invalid_dtype_raises():
    bad = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(HeightRangeError) as exc:
        build_quadtree(bad, np.zeros((64, 64), np.uint8))
    assert exc.value.code.value == "E_HEIGHT_RANGE"


@pytest.mark.parametrize("size,edge_leaves", [(512, 8), (2048, 32)])
def test_quadtree_structure_and_bbox(size, edge_leaves):
    ds = TerrainDataset.generate(size)
    leaves = ds.leaf_blocks()
    assert len(leaves) == edge_leaves * edge_leaves
    # Level counts form the geometric series of a quadtree.
    levels = {}
    for block in ds.blocks:
        levels.setdefault(block.level, 0)
        levels[block.level] += 1
    grid = edge_leaves
    total = 0
    level = 0
    while grid >= 1:
        assert levels[level] == grid * grid
        total += grid * grid
        grid //= 2
        level += 1
    assert total == len(ds.blocks)
    # Bounding boxes cover the full world extent and ranges are consistent.
    minx = min(b.bbox[0] for b in leaves)
    maxx = max(b.bbox[3] for b in leaves)
    assert minx == 0.0
    assert maxx == 4096.0
    for block in ds.blocks:
        assert block.height_max >= block.height_min
        assert block.height_range == block.height_max - block.height_min
        assert block.bbox[4] >= block.bbox[1]
        assert block.payload_bytes > 0 and block.vram_bytes > 0


def test_manifest_order_independent_and_byte_reproducible(tmp_path):
    ds_a = TerrainDataset.generate(512)
    info_a = ds_a.save(str(tmp_path / "run_a"))
    ds_b = TerrainDataset.generate(512)
    info_b = ds_b.save(str(tmp_path / "run_b"))
    assert info_a["manifest_sha256"] == info_b["manifest_sha256"]
    assert info_a["height_sha256"] == info_b["height_sha256"]
    raw_a = open(tmp_path / "run_a" / info_a["manifest_path"], "rb").read()
    raw_b = open(tmp_path / "run_b" / info_b["manifest_path"], "rb").read()
    assert raw_a == raw_b
    parsed = json.loads(raw_a)
    ids = [b["block_id"] for b in parsed["blocks"]]
    assert ids == sorted(ids)


def test_height_files_byte_reproducible(tmp_path):
    ds_a = TerrainDataset.generate(2048)
    info_a = ds_a.save(str(tmp_path / "a"))
    ds_b = TerrainDataset.generate(2048)
    info_b = ds_b.save(str(tmp_path / "b"))
    ha = open(tmp_path / "a" / info_a["height_path"], "rb").read()
    hb = open(tmp_path / "b" / info_b["height_path"], "rb").read()
    assert ha == hb
    assert hashlib.sha256(ha).hexdigest() == info_a["height_sha256"]


def test_seed_constants_are_fixed():
    assert HEIGHT_SEED == 20261119
    assert MATERIAL_SEED == 30261119
