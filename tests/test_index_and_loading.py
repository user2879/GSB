from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from rsseg.errors import ErrorCode
from rsseg.injection import load_injections
from rsseg.runner import validate_index_or_report
from rsseg.tiles import TileIndex, load_index, load_tile_bands


def test_index_loads_and_validates_symmetric_adjacency(sample_set):
    index = load_index(sample_set / "tile_index.json")
    assert index.band_order == ("red", "green", "nir", "elevation")
    assert len(index.tiles) == 9
    tile_map = index.tile_map()
    for tile in index.tiles:
        for neighbor in tile.neighbors:
            assert tile.key in tile_map[neighbor].neighbors


def test_duplicate_row_col_rejected(sample_set):
    raw = json.loads((sample_set / "tile_index.json").read_text())
    raw["tiles"][1]["row"] = raw["tiles"][0]["row"]
    raw["tiles"][1]["col"] = raw["tiles"][0]["col"]
    target = sample_set / "broken_index.json"
    target.write_text(json.dumps(raw))
    with pytest.raises(Exception) as exc:
        load_index(target)
    assert exc.value.code == ErrorCode.E_INDEX_INVALID


def test_asymmetric_adjacency_injection(sample_set):
    record = validate_index_or_report(sample_set, sample_set / "injections.json")
    assert record is not None
    assert record.code == ErrorCode.E_INDEX_INVALID
    assert record.injection_id == "INJ-003"


def test_pixel_integer_precision_preserved(sample_set):
    index = load_index(sample_set / "tile_index.json")
    tile = index.tiles[0]
    bands = load_tile_bands(tile, sample_set, index)
    assert bands.dtype == np.uint8
    direct = np.array(Image.open(sample_set / tile.file))
    assert np.array_equal(bands, direct)


def test_tile_shape_mismatch_detected(sample_set):
    index = load_index(sample_set / "tile_index.json")
    tile = index.tiles[0]
    wrong = np.zeros((index.tile_size - 4, index.tile_size, index.band_count), dtype=np.uint8)
    staged = sample_set / "wrong_tile.png"
    Image.fromarray(wrong, mode="RGBA").save(staged)
    from rsseg.tiles import TileRecord

    bad = TileRecord(tile.row, tile.col, "wrong_tile.png", tile.bounds, tile.neighbors)
    with pytest.raises(Exception) as exc:
        load_tile_bands(bad, sample_set, index)
    assert exc.value.code == ErrorCode.E_TILE_SHAPE_MISMATCH


def test_band_count_mismatch_detected(sample_set):
    index = load_index(sample_set / "tile_index.json")
    tile = index.tiles[0]
    wrong = np.zeros((index.tile_size, index.tile_size, 3), dtype=np.uint8)
    staged = sample_set / "three_band.png"
    Image.fromarray(wrong, mode="RGB").save(staged)
    from rsseg.tiles import TileRecord

    bad = TileRecord(tile.row, tile.col, "three_band.png", tile.bounds, tile.neighbors)
    with pytest.raises(Exception) as exc:
        load_tile_bands(bad, sample_set, index)
    assert exc.value.code == ErrorCode.E_TILE_SHAPE_MISMATCH
