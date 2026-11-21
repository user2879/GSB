from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from remote_seg.errors import (
    E_BAND_VERSION_UNKNOWN,
    E_INDEX_INVALID,
    E_TILE_DECODE,
    E_TILE_SHAPE_MISMATCH,
    SegmentationError,
)
from remote_seg.inputs import load_config, load_index, load_registry, load_tiles
from remote_seg.pngio import decode_png, encode_png
from remote_seg.samplegen import generate_sample_workspace


def test_png_roundtrip_all_channels(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    for channels in (1, 2, 3, 4):
        array = rng.integers(0, 256, size=(7, 9, channels), dtype=np.uint8)
        encoded = encode_png(array)
        decoded = decode_png(encoded)
        assert decoded.array.shape == array.shape
        assert np.array_equal(decoded.array, array)


def test_index_duplicate_cell_rejected(tmp_path: Path) -> None:
    generate_sample_workspace(tmp_path / "data", width=24, height=24, rows=2, cols=2, overlap=4)
    index_path = tmp_path / "data" / "index.json"
    index = json.loads(index_path.read_text())
    index["tiles"][1]["row"] = index["tiles"][0]["row"]
    index["tiles"][1]["col"] = index["tiles"][0]["col"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(SegmentationError) as exc:
        load_index(index_path)
    assert exc.value.code == E_INDEX_INVALID


def test_index_asymmetric_neighbors_rejected(tmp_path: Path) -> None:
    generate_sample_workspace(tmp_path / "data", width=24, height=24, rows=2, cols=2, overlap=4)
    index_path = tmp_path / "data" / "index.json"
    index = json.loads(index_path.read_text())
    index["tiles"][0]["neighbors"] = []
    index_path.write_text(json.dumps(index))
    with pytest.raises(SegmentationError) as exc:
        load_index(index_path)
    assert exc.value.code == E_INDEX_INVALID


def test_unknown_band_version_and_band_mismatch_are_tile_failures(sample_workspace) -> None:
    index_path = sample_workspace["index"]
    config_path = sample_workspace["config"]
    fusion_path = sample_workspace["fusion"]
    index = json.loads(index_path.read_text())
    index["tiles"][0]["band_order_version"] = "bov-unknown"
    index_path.write_text(json.dumps(index))
    bands_path = index_path.parent / index["tiles"][1]["path"]
    from remote_seg.pngio import read_png

    rgb = read_png(bands_path).array[:, :, :3]
    bands_path.write_bytes(encode_png(rgb))
    config = load_config(config_path)
    registry = load_registry(fusion_path)
    loaded_index = load_index(index_path)
    result = load_tiles(loaded_index, config, registry)
    codes = {failure.code for failure in result.failures}
    assert E_BAND_VERSION_UNKNOWN in codes
    assert E_TILE_SHAPE_MISMATCH in codes
    assert len(result.loaded) == len(loaded_index.tiles) - 2


def test_truncated_png_is_decode_failure(sample_workspace) -> None:
    index_path = sample_workspace["index"]
    config = load_config(sample_workspace["config"])
    registry = load_registry(sample_workspace["fusion"])
    index = load_index(index_path)
    tile = index.tiles[0]
    path = index.root / tile.path
    path.write_bytes(path.read_bytes()[:20])
    result = load_tiles(index, config, registry)
    assert any(failure.code == E_TILE_DECODE for failure in result.failures)
    assert tile.tile_id not in result.loaded
