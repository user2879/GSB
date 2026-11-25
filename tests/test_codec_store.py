from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from vscan.codec import decode_column, encode_column, read_column, read_rows
from vscan.spec import parse_spec
from vscan.store import build_store, column_path, generate_column_values


VALUES = [5, None, 2, 5, 2, 2, None, 9, 9, 9, 0, None]


def test_all_encodings_round_trip_deterministically():
    blobs = {}
    for encoding in ("raw", "dict", "rle"):
        blob = encode_column(VALUES, encoding)
        assert blob == encode_column(VALUES, encoding)
        data, actual = decode_column(blob)
        assert actual == encoding
        assert data.values.tolist() == [0 if value is None else value for value in VALUES]
        assert data.null_mask.tolist() == [value is None for value in VALUES]
        assert data.non_null_count == 9
        blobs[encoding] = blob
    assert len(set(blobs.values())) == 3


def test_dictionary_sorted_and_rle_uses_runs():
    blob = encode_column(VALUES, "dict")
    dictionary_count = int.from_bytes(blob[32:36], "little")
    assert dictionary_count == 4  # values 0, 2, 5, 9 in ascending order
    dictionary = np.frombuffer(blob, dtype="<i8", count=4, offset=40)
    assert dictionary.tolist() == [0, 2, 5, 9]

    rle_blob = encode_column(VALUES, "rle")
    run_count = int.from_bytes(rle_blob[32:40], "little")
    expected_runs = 1 + sum(VALUES[index] != VALUES[index - 1] for index in range(1, len(VALUES)))
    assert run_count == expected_runs


def test_selective_row_decode_matches_full_decode_for_all_encodings(tmp_path):
    indices = np.array([0, 2, 5, 8, 11], dtype=np.int64)
    for encoding in ("raw", "dict", "rle"):
        path = tmp_path / f"{encoding}.col"
        path.write_bytes(encode_column(VALUES, encoding))
        full, _ = decode_column(path.read_bytes())
        selected = read_rows(path, indices)
        assert selected.values.tolist() == full.values[indices].tolist()
        assert selected.null_mask.tolist() == full.null_mask[indices].tolist()
        empty = read_rows(path, np.array([], dtype=np.int64))
        assert empty.rows == 0


def test_store_is_byte_identical_regardless_of_load_order(tmp_path, spec_dict):
    first_spec = parse_spec(copy.deepcopy(spec_dict))
    reordered = copy.deepcopy(spec_dict)
    reordered["segments"][0]["columns"].reverse()
    second_spec = parse_spec(reordered)
    first = build_store(first_spec, tmp_path / "a")
    second = build_store(second_spec, tmp_path / "b")
    assert first.snapshot_digest == second.snapshot_digest
    assert first.canonical_dict() == second.canonical_dict()
    for segment in first.segments:
        for column in segment.columns:
            relative = column_path(Path("."), segment.id, column.name)
            assert column_path(tmp_path / "a", segment.id, column.name).read_bytes() == column_path(
                tmp_path / "b", segment.id, column.name
            ).read_bytes()
            assert relative.name.endswith(".col")


def test_generation_and_non_null_counts(tmp_path, spec):
    segment = spec.segments[0]
    values = generate_column_values(segment, segment.column_map()["v"])
    assert values == generate_column_values(segment, segment.column_map()["v"])
    state = build_store(spec, tmp_path)
    for column in state.segments[0].columns:
        data = read_column(column_path(tmp_path, state.segments[0].id, column.name))
        assert column.non_null_count == data.non_null_count
        assert column.rows == data.rows


def test_column_and_segment_digest_stability(tmp_path, spec):
    first = build_store(spec, tmp_path / "one")
    second = build_store(spec, tmp_path / "two")
    left = first.segment_map()["s1"]
    right = second.segment_map()["s1"]
    assert left.segment_digest == right.segment_digest
    assert [column.column_digest for column in left.columns] == [
        column.column_digest for column in right.columns
    ]
